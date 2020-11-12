# Copyright 2020 ponai Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, Union

import torch
import torch.nn.functional as F
from ponai.data.utils import compute_importance_map, dense_patch_slices, get_valid_patch_size
from ponai.utils import BlendMode, PytorchPadMode, fall_back_tuple


def sliding_window_inference(
    inputs: Union[torch.Tensor, tuple],
    roi_size,
    sw_batch_size: int,
    predictor: Callable,
    overlap: float = 0.25,
    mode: Union[BlendMode, str] = BlendMode.CONSTANT,
    padding_mode: Union[PytorchPadMode, str] = PytorchPadMode.CONSTANT,
    cval=0,
    uncertainty_flag=False
):
    """
    Sliding window inference on `inputs` with `predictor`.

    When roi_size is larger than the inputs' spatial size, the input image are padded during inference.
    To maintain the same spatial sizes, the output image will be cropped to the original input size.

    Args:
        inputs: input image to be processed (assuming NCHW[D])
        roi_size (list, tuple): the spatial window size for inferences.
            When its components have None or non-positives, the corresponding inputs dimension will be used.
            if the components of the `roi_size` are non-positive values, the transform will use the
            corresponding components of img size. For example, `roi_size=(32, -1)` will be adapted
            to `(32, 64)` if the second spatial dimension size of img is `64`.
        sw_batch_size: the batch size to run window slices.
        predictor: given input tensor `patch_data` in shape NCHW[D], `predictor(patch_data)`
            should return a prediction with the same spatial shape and batch_size, i.e. NMHW[D];
            where HW[D] represents the patch spatial size, M is the number of output channels, N is `sw_batch_size`.
        overlap: Amount of overlap between scans.
        mode: {``"constant"``, ``"gaussian"``}
            How to blend output of overlapping windows. Defaults to ``"constant"``.

            - ``"constant``": gives equal weight to all predictions.
            - ``"gaussian``": gives less weight to predictions on edges of windows.

        padding_mode: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}
            Padding mode when ``roi_size`` is larger than inputs. Defaults to ``"constant"``
            See also: https://pytorch.org/docs/stable/nn.functional.html#pad
        cval: fill value for 'constant' padding mode. Default: 0

    Raises:
        NotImplementedError: inputs must have batch_size=1.

    Note:
        - input must be channel-first and have a batch dim, support both spatial 2D and 3D.
        - currently only supports `inputs` with batch_size=1.
    """
    assert 0 <= overlap < 1, "overlap must be >= 0 and < 1."

    # determine image spatial size and batch size
    # Note: all input images must have the same image size and batch size
    inputs_type = type(inputs)
    if inputs_type == tuple:
        phys_inputs = inputs[1]
        inputs = inputs[0]
    num_spatial_dims = len(inputs.shape) - 2
    image_size_ = list(inputs.shape[2:])
    batch_size = inputs.shape[0]

    # TODO: Enable batch sizes > 1 in future
    if batch_size > 1:
        raise NotImplementedError("inputs must have batch_size=1.")

    roi_size = fall_back_tuple(roi_size, image_size_)
    # in case that image size is smaller than roi size
    image_size = tuple(max(image_size_[i], roi_size[i]) for i in range(num_spatial_dims))
    pad_size = []
    for k in range(len(inputs.shape) - 1, 1, -1):
        diff = max(roi_size[k - 2] - inputs.shape[k], 0)
        half = diff // 2
        pad_size.extend([half, diff - half])
    inputs = F.pad(inputs, pad=pad_size, mode=PytorchPadMode(padding_mode).value, value=cval)

    scan_interval = _get_scan_interval(image_size, roi_size, num_spatial_dims, overlap)

    # Store all slices in list
    slices = dense_patch_slices(image_size, roi_size, scan_interval)
    # print(f'The slices are {slices}')

    slice_batches = []
    for slice_index in range(0, len(slices), sw_batch_size):
        slice_index_range = range(slice_index, min(slice_index + sw_batch_size, len(slices)))
        input_slices = []
        for curr_index in slice_index_range:
            curr_slice = slices[curr_index]
            if len(curr_slice) == 3:
                input_slices.append(inputs[0, :, curr_slice[0], curr_slice[1], curr_slice[2]])
            else:
                input_slices.append(inputs[0, :, curr_slice[0], curr_slice[1]])
        slice_batches.append(torch.stack(input_slices))

    # Perform predictions
    if not uncertainty_flag:
        # No uncertainty, so only one prediction, so proceed normally
        output_rois = list()
        for data in slice_batches:
            if not uncertainty_flag and inputs_type == tuple:
                seg_prob, _ = predictor(data, phys_inputs)  # batched patch segmentation
                output_rois.append(seg_prob)
            elif inputs_type != tuple:
                seg_prob, _ = predictor(data)  # batched patch segmentation
                output_rois.append(seg_prob)

        # stitching output image
        output_classes = output_rois[0].shape[1]
        output_shape = [batch_size, output_classes] + list(image_size)

        # Create importance map
        importance_map = compute_importance_map(get_valid_patch_size(image_size, roi_size), mode=mode, device=inputs.device)

        # allocate memory to store the full output and the count for overlapping parts
        output_image = torch.zeros(output_shape, dtype=torch.float32, device=inputs.device)
        count_map = torch.zeros(output_shape, dtype=torch.float32, device=inputs.device)

        for window_id, slice_index in enumerate(range(0, len(slices), sw_batch_size)):
            slice_index_range = range(slice_index, min(slice_index + sw_batch_size, len(slices)))

            # store the result in the proper location of the full output. Apply weights from importance map.
            for curr_index in slice_index_range:
                curr_slice = slices[curr_index]
                if len(curr_slice) == 3:
                    # print(output_image.shape, curr_slice, importance_map.shape, output_rois[window_id].shape)
                    output_image[0, :, curr_slice[0], curr_slice[1], curr_slice[2]] += (
                        importance_map * output_rois[window_id][curr_index - slice_index, :]
                    )
                    count_map[0, :, curr_slice[0], curr_slice[1], curr_slice[2]] += importance_map
                else:
                    output_image[0, :, curr_slice[0], curr_slice[1]] += (
                        importance_map * output_rois[window_id][curr_index - slice_index, :]
                    )
                    count_map[0, :, curr_slice[0], curr_slice[1]] += importance_map

        # account for any overlapping sections
        output_image /= count_map

        if num_spatial_dims == 3:
            return output_image[
                ...,
                pad_size[4]: image_size_[0] + pad_size[4],
                pad_size[2]: image_size_[1] + pad_size[2],
                pad_size[0]: image_size_[2] + pad_size[0],
            ]
        return output_image[
            ..., pad_size[2]: image_size_[0] + pad_size[2], pad_size[0]: image_size_[1] + pad_size[0]
        ]  # 2D
    else:
        # Decide on number of histogram samples
        num_hist_samples = 20
        overall_stochastic_logits_hist = torch.empty((1, 2, 181, 217, 181, num_hist_samples))
        overall_true_seg_net_out_hist = torch.empty((1, 2, 181, 217, 181, num_hist_samples))

        output_rois = list()
        unc_output_rois = list()
        # Have uncertainty, therefore have MANY outputs, but only have ONE pass through network
        for data in slice_batches:
            if inputs_type == tuple:
                seg_prob, unc_prob, _ = predictor(data, phys_inputs)  # batched patch segmentation
                output_rois.append(seg_prob)
                unc_output_rois.append(unc_prob)
            elif inputs_type != tuple:
                seg_prob, unc_prob, _ = predictor(data)  # batched patch segmentation
                output_rois.append(seg_prob)
                unc_output_rois.append(unc_prob)
        # Get shape of logits
        logits_shape = list(seg_prob.shape)
        # Now want an array of randomly normally distributed samples size of logits x num samples
        # logits_shape.append(num_hist_samples)
        inf_ax = torch.distributions.Normal(torch.tensor(0.0).to(device=torch.device("cuda:0")),
                                            torch.tensor(1.0).to(device=torch.device("cuda:0")))
        # inf_noise_array = torch.empty(logits_shape).normal_(mean=0, std=1)
        # Loop through samples

        for infpass in range(num_hist_samples):
            true_output_rois = list()
            true_unc_output_rois = list()
            # print(f'The lengths of rois are {len(output_rois)}, {len(unc_output_rois)}')
            for roi, unc_roi in zip(output_rois, unc_output_rois):
                # output_rois = list()
                # unc_output_rois = list()m
                # Repeat steps above to get more samples
                # noise_sample = inf_noise_array[..., infpass]
                stochastic_logits = roi + unc_roi * inf_ax.sample(logits_shape)  # noise_sample
                # print(f'The sigma mean is {torch.mean(unc_roi)}, logits mean is {torch.mean(roi)}')
                # print(
                #     f'The logits, sigma, ax sizes are: {roi.shape}, {unc_roi.shape}, {inf_ax.sample(logits_shape).shape}')
                # print(
                #     f'A little ax check: {inf_ax.sample(logits_shape)[0, 0, 0, 0, 0]}, {inf_ax.sample(logits_shape)[0, 1, 0, 0, 0]}')
                true_seg_net_out = torch.softmax(stochastic_logits, dim=1)
                # print(f'The stochastic logits shapes are {stochastic_logits.shape}, {true_seg_net_out.shape}')
                true_output_rois.append(true_seg_net_out)
                true_unc_output_rois.append(stochastic_logits)

            # stitching output image
            # print(f'The true output rois tensor shapes are {true_output_rois[0].shape}')
            output_classes = true_output_rois[0].shape[1]
            output_shape = [batch_size, output_classes] + list(image_size)

            # Create importance map
            importance_map = compute_importance_map(get_valid_patch_size(image_size, roi_size), mode=mode,
                                                    device=inputs.device)

            # allocate memory to store the full output and the count for overlapping parts
            output_image = torch.zeros(output_shape, dtype=torch.float32, device=inputs.device)
            count_map = torch.zeros(output_shape, dtype=torch.float32, device=inputs.device)

            # slic_index, zero to len(slices) in increments of sw_batch_size
            for window_id, slice_index in enumerate(range(0, len(slices), sw_batch_size)):
                slice_index_range = range(slice_index, min(slice_index + sw_batch_size, len(slices)))

                # store the result in the proper location of the full output. Apply weights from importance map.
                for curr_index in slice_index_range:
                    curr_slice = slices[curr_index]
                    if len(curr_slice) == 3:
                        # print(output_image.shape, curr_slice, importance_map.shape, true_output_rois[window_id].shape)
                        output_image[0, :, curr_slice[0], curr_slice[1], curr_slice[2]] += (
                                importance_map * true_output_rois[window_id][curr_index - slice_index, :]
                        )
                        count_map[0, :, curr_slice[0], curr_slice[1], curr_slice[2]] += importance_map
                    else:
                        output_image[0, :, curr_slice[0], curr_slice[1]] += (
                                importance_map * true_output_rois[window_id][curr_index - slice_index, :]
                        )
                        count_map[0, :, curr_slice[0], curr_slice[1]] += importance_map

            # account for any overlapping sections
            output_image /= count_map

            if num_spatial_dims == 3:
                output_image = output_image[
                       ...,
                       pad_size[4]: image_size_[0] + pad_size[4],
                       pad_size[2]: image_size_[1] + pad_size[2],
                       pad_size[0]: image_size_[2] + pad_size[0],
                       ]
                overall_true_seg_net_out_hist[..., infpass] = output_image
            else:
                output_image = output_image[
                       ..., pad_size[2]: image_size_[0] + pad_size[2], pad_size[0]: image_size_[1] + pad_size[0]
                       ]  # 2D
                overall_true_seg_net_out_hist[..., infpass] = output_image

            # Uncertainty part
            # stitching output image
            output_classes = true_unc_output_rois[0].shape[1]
            output_shape = [batch_size, output_classes] + list(image_size)

            # Create importance map
            importance_map = compute_importance_map(get_valid_patch_size(image_size, roi_size), mode=mode,
                                                    device=inputs.device)

            # allocate memory to store the full output and the count for overlapping parts
            unc_output = torch.zeros(output_shape, dtype=torch.float32, device=inputs.device)
            count_map = torch.zeros(output_shape, dtype=torch.float32, device=inputs.device)

            for window_id, slice_index in enumerate(range(0, len(slices), sw_batch_size)):
                slice_index_range = range(slice_index, min(slice_index + sw_batch_size, len(slices)))

                # store the result in the proper location of the full output. Apply weights from importance map.
                for curr_index in slice_index_range:
                    curr_slice = slices[curr_index]
                    if len(curr_slice) == 3:
                        unc_output[0, :, curr_slice[0], curr_slice[1], curr_slice[2]] += (
                                importance_map * true_unc_output_rois[window_id][curr_index - slice_index, :]
                        )
                        count_map[0, :, curr_slice[0], curr_slice[1], curr_slice[2]] += importance_map
                    else:
                        unc_output[0, :, curr_slice[0], curr_slice[1]] += (
                                importance_map * true_unc_output_rois[window_id][curr_index - slice_index, :]
                        )
                        count_map[0, :, curr_slice[0], curr_slice[1]] += importance_map

            # account for any overlapping sections
            unc_output /= count_map

            if num_spatial_dims == 3:
                unc_output = unc_output[
                               ...,
                               pad_size[4]: image_size_[0] + pad_size[4],
                               pad_size[2]: image_size_[1] + pad_size[2],
                               pad_size[0]: image_size_[2] + pad_size[0],
                               ]
                overall_stochastic_logits_hist[..., infpass] = unc_output
            else:
                unc_output = unc_output[
                               ..., pad_size[2]: image_size_[0] + pad_size[2],
                               pad_size[0]: image_size_[1] + pad_size[0]
                               ]  # 2D
                overall_stochastic_logits_hist[..., infpass] = unc_output

        # Sigma part
        # stitching output image
        output_classes = unc_output_rois[0].shape[1]
        output_shape = [batch_size, output_classes] + list(image_size)

        # Create importance map
        importance_map = compute_importance_map(get_valid_patch_size(image_size, roi_size), mode=mode,
                                                device=inputs.device)

        # allocate memory to store the full output and the count for overlapping parts
        sigma_output = torch.zeros(output_shape, dtype=torch.float32, device=inputs.device)
        count_map = torch.zeros(output_shape, dtype=torch.float32, device=inputs.device)
        for window_id, slice_index in enumerate(range(0, len(slices), sw_batch_size)):
            slice_index_range = range(slice_index, min(slice_index + sw_batch_size, len(slices)))

            # store the result in the proper location of the full output. Apply weights from importance map.
            for curr_index in slice_index_range:
                curr_slice = slices[curr_index]
                if len(curr_slice) == 3:
                    sigma_output[0, :, curr_slice[0], curr_slice[1], curr_slice[2]] += (
                            importance_map * unc_output_rois[window_id][curr_index - slice_index, :]
                    )
                    count_map[0, :, curr_slice[0], curr_slice[1], curr_slice[2]] += importance_map
                else:
                    sigma_output[0, :, curr_slice[0], curr_slice[1]] += (
                            importance_map * unc_output_rois[window_id][curr_index - slice_index, :]
                    )
                    count_map[0, :, curr_slice[0], curr_slice[1]] += importance_map

        # account for any overlapping sections
        sigma_output /= count_map

        if num_spatial_dims == 3:
            sigma_output = sigma_output[
                           ...,
                           pad_size[4]: image_size_[0] + pad_size[4],
                           pad_size[2]: image_size_[1] + pad_size[2],
                           pad_size[0]: image_size_[2] + pad_size[0],
                           ]
        else:
            sigma_output = sigma_output[
                           ..., pad_size[2]: image_size_[0] + pad_size[2],
                           pad_size[0]: image_size_[1] + pad_size[0]
                           ]  # 2D
        return overall_true_seg_net_out_hist, overall_stochastic_logits_hist, sigma_output


def _get_scan_interval(image_size, roi_size, num_spatial_dims: int, overlap: float):
    assert len(image_size) == num_spatial_dims, "image coord different from spatial dims."
    assert len(roi_size) == num_spatial_dims, "roi coord different from spatial dims."

    scan_interval = []
    for i in range(num_spatial_dims):
        if roi_size[i] == image_size[i]:
            scan_interval.append(int(roi_size[i]))
        else:
            # scan interval is (1-overlap)*roi_size
            scan_interval.append(int(roi_size[i] * (1 - overlap)))
    return tuple(scan_interval)
