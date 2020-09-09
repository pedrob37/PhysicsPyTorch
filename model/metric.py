import torch
import numpy as np
import warnings
from typing import Callable, Optional, Union

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from monai.networks import one_hot
from monai.utils import LossReduction, Weight



def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


class dice(object):
    def __init__(self, indices_to_ignore=[-1]):
        self.indices_to_ignore = indices_to_ignore
        super().__init__()

    def __call__(self, scores, labels, class_to_report='mean', device=None):
        '''
        Dice overlap using hard labels.
        '''

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        with torch.no_grad():
            _ , predictions = torch.max(torch.nn.functional.softmax(scores, dim=1), dim=1)
            #predictions = torch.argmax(torch.nn.functional.softmax(scores, dim=1), dim=1)
            num_classes = scores.shape[1]
            dice_per_class = []
            for class_label in range(num_classes):
                class_labels = torch.where(labels==class_label, torch.tensor([1]).to(device), torch.tensor([0]).to(device))
                class_predictions = torch.where(predictions==class_label, torch.tensor([1]).to(device), torch.tensor([0]).to(device))
                numerator = (class_labels * class_predictions).sum()
                denominator = (class_labels + class_predictions).sum()

                dice_per_class.append((2*numerator.float() / denominator.float().clamp(min=1e-5)).item())
            indices_to_pick = [i for i in range(num_classes) if i not in self.indices_to_ignore]
        if class_to_report == 'all':
            return np.array(dice_per_class)[indices_to_pick]
        elif class_to_report == 'mean':
            return np.mean(np.array(dice_per_class)[indices_to_pick])
        else:
            return dice_per_class[class_to_report]


class dice_one_hot(object):
    '''
    Dice overlap using hard labels with one-hot encoding. Roughly 10x faster than dice, but requires more memory.
    '''
    def __call__(self, input, target, class_to_report='mean'):
        from .loss import compute_per_channel_dice
        with torch.no_grad():
            _, input = torch.max(torch.nn.functional.softmax(input, dim=1), dim=1)
            input_onehot = torch.nn.functional.one_hot(input, num_classes=151).permute([0, 4, 1, 2, 3])
            target_onehot = torch.nn.functional.one_hot(target, num_classes=151).permute([0, 4, 1, 2, 3])
            dice_per_class = compute_per_channel_dice(input_onehot, target_onehot)
            dice_per_class = dice_per_class.tolist()

            if class_to_report == 'all':
               return dice_per_class
            elif class_to_report == 'mean':
                return np.mean(dice_per_class)
            else:
                return dice_per_class[class_to_report]


class DiceLoss(_Loss):
    """
    Compute average Dice loss between two tensors. It can support both multi-classes and multi-labels tasks.
    Input logits `input` (BNHW[D] where N is number of classes) is compared with ground truth `target` (BNHW[D]).
    Axis N of `input` is expected to have logit predictions for each class rather than being image channels,
    while the same axis of `target` can be 1 or N (one-hot format). The `smooth` parameter is a value added to the
    intersection and union components of the inter-over-union calculation to smooth results and prevent divide by 0,
    this value should be small. The `include_background` class attribute can be set to False for an instance of
    DiceLoss to exclude the first category (channel index 0) which is by convention assumed to be background.
    If the non-background segmentations are small compared to the total image size they can get overwhelmed by
    the signal from the background so excluding it in such cases helps convergence.
    Milletari, F. et. al. (2016) V-Net: Fully Convolutional Neural Networks forVolumetric Medical Image Segmentation, 3DV, 2016.
    """

    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Optional[Callable] = None,
        squared_pred: bool = False,
        jaccard: bool = False,
        reduction: Union[LossReduction, str] = LossReduction.MEAN,
    ) -> None:
        """
        Args:
            include_background: if False channel index 0 (background category) is excluded from the calculation.
            to_onehot_y: whether to convert `y` into the one-hot format. Defaults to False.
            sigmoid: if True, apply a sigmoid function to the prediction.
            softmax: if True, apply a softmax function to the prediction.
            other_act: if don't want to use `sigmoid` or `softmax`, use other callable function to execute
                other activation layers, Defaults to ``None``. for example:
                `other_act = torch.tanh`.
            squared_pred: use squared versions of targets and predictions in the denominator or not.
            jaccard: compute Jaccard Index (soft IoU) instead of dice or not.
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.
                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.
        Raises:
            TypeError: When ``other_act`` is not an ``Optional[Callable]``.
            ValueError: When more than 1 of [``sigmoid=True``, ``softmax=True``, ``other_act is not None``].
                Incompatible values.
        """
        super().__init__(reduction=LossReduction(reduction).value)
        if other_act is not None and not callable(other_act):
            raise TypeError(f"other_act must be None or callable but is {type(other_act).__name__}.")
        if int(sigmoid) + int(softmax) + int(other_act is not None) > 1:
            raise ValueError("Incompatible values: more than 1 of [sigmoid=True, softmax=True, other_act is not None].")
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.other_act = other_act
        self.squared_pred = squared_pred
        self.jaccard = jaccard

    def forward(self, input: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD].
            target: the shape should be BNH[WD].
            smooth: a small constant to avoid nan.
        Raises:
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].
        """
        if self.sigmoid:
            input = torch.sigmoid(input)

        n_pred_ch = input.shape[1]
        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `softmax=True` ignored.")
            else:
                input = torch.softmax(input, 1)

        if self.other_act is not None:
            input = self.other_act(input)

        if self.to_onehot_y:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                target = one_hot(target, num_classes=n_pred_ch)

        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `include_background=False` ignored.")
            else:
                # if skipping background, removing first channel
                target = target[:, 1:]
                input = input[:, 1:]

        assert (
            target.shape == input.shape
        ), f"ground truth has differing shape ({target.shape}) from input ({input.shape})"

        # reducing only spatial dimensions (not batch nor channels)
        reduce_axis = list(range(2, len(input.shape)))
        intersection = torch.sum(target * input, dim=reduce_axis)

        if self.squared_pred:
            target = torch.pow(target, 2)
            input = torch.pow(input, 2)

        ground_o = torch.sum(target, dim=reduce_axis)
        pred_o = torch.sum(input, dim=reduce_axis)

        denominator = ground_o + pred_o

        if self.jaccard:
            denominator = 2.0 * (denominator - intersection)

        f: torch.Tensor = 1.0 - (2.0 * intersection + smooth) / (denominator + smooth)

        if self.reduction == LossReduction.MEAN.value:
            f = torch.mean(f)  # the batch and channel average
        elif self.reduction == LossReduction.SUM.value:
            f = torch.sum(f)  # sum over the batch and channel dims
        elif self.reduction == LossReduction.NONE.value:
            pass  # returns [N, n_classes] losses
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')

        return f
