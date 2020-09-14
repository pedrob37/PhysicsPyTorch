import numpy as np
import monai
import sys
# sys.path.append('/nfs/home/pedro/portio')
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd
import os
import argparse
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from model.model import nnUNet
import random
from model.metric import DiceLoss
import glob
import time

import monai.visualize.img2tensorboard as img2tensorboard
sys.path.append('/nfs/home/pedro/RangerLARS/over9000')
# from over9000 import RangerLars

os.chdir('/nfs/home/pedro/PhysicsPyTorch')
import porchio
from early_stopping import pytorchtools


class PairwiseMeasures(object):
    def __init__(self, seg_img, ref_img,
                 measures=None, num_neighbors=8, pixdim=(1, 1, 1),
                 empty=False, list_labels=None):

        self.m_dict = {
            'ref volume': (self.n_pos_ref, 'Volume (Ref)'),
            'seg volume': (self.n_pos_seg, 'Volume (Seg)'),
            'ref bg volume': (self.n_neg_ref, 'Volume (Ref bg)'),
            'seg bg volume': (self.n_neg_seg, 'Volume (Seg bg)'),
            'list_labels': (self.list_labels, 'List Labels Seg'),
            'fp': (self.fp, 'FP'),
            'fn': (self.fn, 'FN'),
            'tp': (self.tp, 'TP'),
            'tn': (self.tn, 'TN'),
            'n_intersection': (self.n_intersection, 'Intersection'),
            'n_union': (self.n_union, 'Union'),
            'sensitivity': (self.sensitivity, 'Sens'),
            'specificity': (self.specificity, 'Spec'),
            'accuracy': (self.accuracy, 'Acc'),
            'fpr': (self.false_positive_rate, 'FPR'),
            'ppv': (self.positive_predictive_values, 'PPV'),
            'npv': (self.negative_predictive_values, 'NPV'),
            'dice': (self.dice_score, 'Dice'),
            'IoU': (self.intersection_over_union, 'IoU'),
            'jaccard': (self.jaccard, 'Jaccard'),
            'informedness': (self.informedness, 'Informedness'),
            'markedness': (self.markedness, 'Markedness'),
            'vol_diff': (self.vol_diff, 'VolDiff'),
            'ave_dist': (self.measured_average_distance, 'AveDist'),
            'haus_dist': (self.measured_hausdorff_distance, 'HausDist'),
            'connected_elements': (self.connected_elements, 'TPc,FPc,FNc'),
            'outline_error': (self.outline_error, 'OER,OEFP,OEFN'),
            'detection_error': (self.detection_error, 'DE,DEFP,DEFN')
        }
        self.seg = seg_img
        self.ref = ref_img
        self.list_labels = list_labels
        self.flag_empty = empty
        self.measures = measures if measures is not None else self.m_dict
        self.neigh = num_neighbors
        self.pixdim = pixdim

    def check_binary(self):
        """
        Checks whether self.seg and self.ref are binary. This is to enable
        measurements such as 'false positives', which only have meaning in
        the binary case (what is positive/negative for multiple class?)
        """

        is_seg_binary, is_ref_binary = [((x > 0.5) == x).all()
                                        for x in [self.seg, self.ref]]
        # if (not is_ref_binary) or (not is_seg_binary):
        #     raise ValueError("The input segmentation/reference images"
        #                      " must be binary for this function.")

    def __FPmap(self):
        """
        This function calculates the false positive map from binary
        segmentation and reference maps

        :return: FP map
        """
        self.check_binary()
        return np.asarray((self.seg - self.ref) > 0.0, dtype=np.float32)

    def __FNmap(self):
        """
        This function calculates the false negative map

        :return: FN map
        """
        self.check_binary()
        return np.asarray((self.ref - self.seg) > 0.0, dtype=np.float32)

    def __TPmap(self):
        """
        This function calculates the true positive map (i.e. how many
        reference voxels are positive)

        :return: TP map
        """
        self.check_binary()
        return np.logical_and(self.ref > 0.5, self.seg > 0.5).astype(float)

    def __TNmap(self):
        """
        This function calculates the true negative map

        :return: TN map
        """
        self.check_binary()
        return np.logical_and(self.ref < 0.5, self.seg < 0.5).astype(float)

    def __union_map(self):
        """
        This function calculates the union map between segmentation and
        reference image

        :return: union map
        """
        self.check_binary()
        return np.logical_or(self.ref, self.seg).astype(float)

    def __intersection_map(self):
        """
        This function calculates the intersection between segmentation and
        reference image

        :return: intersection map
        """
        self.check_binary()
        return np.multiply(self.ref, self.seg)

    def n_pos_ref(self):
        return np.sum(self.ref)

    def n_neg_ref(self):
        self.check_binary()
        return np.sum(self.ref == 0)

    def n_pos_seg(self):
        return np.sum(self.seg)

    def n_neg_seg(self):
        return np.sum(1 - self.seg)

    def fp(self):
        return np.sum(self.__FPmap())

    def fn(self):
        return np.sum(self.__FNmap())

    def tp(self):
        return np.sum(self.__TPmap())

    def tn(self):
        return np.sum(self.__TNmap())

    def n_intersection(self):
        return np.sum(self.__intersection_map())

    def n_union(self):
        return np.sum(self.__union_map())

    def sensitivity(self):
        return self.tp() / self.n_pos_ref()

    def specificity(self):
        return self.tn() / self.n_neg_ref()

    def accuracy(self):
        return (self.tn() + self.tp()) / \
               (self.tn() + self.tp() + self.fn() + self.fp())

    def false_positive_rate(self):
        return self.fp() / self.n_neg_ref()

    def positive_predictive_values(self):
        if self.flag_empty:
            return -1
        return self.tp() / (self.tp() + self.fp())

    def negative_predictive_values(self):
        """
        This function calculates the negative predictive value ratio between
        the number of true negatives and the total number of negative elements

        :return:
        """
        return self.tn() / (self.fn() + self.tn())

    def dice_score(self):
        """
        This function returns the dice score coefficient between a reference
        and segmentation images

        :return: dice score
        """
        return 2 * self.tp() / np.sum(self.ref + self.seg)

    def intersection_over_union(self):
        """
        This function the intersection over union ratio - Definition of
        jaccard coefficient

        :return:
        """
        return self.n_intersection() / self.n_union()

    def jaccard(self):
        """
        This function returns the jaccard coefficient (defined as
        intersection over union)

        :return: jaccard coefficient
        """
        return self.intersection_over_union()

    def informedness(self):
        """
        This function calculates the informedness between the segmentation
        and the reference

        :return: informedness
        """
        return self.sensitivity() + self.specificity() - 1

    def markedness(self):
        """
        This functions calculates the markedness
        :return:
        """
        return self.positive_predictive_values() + \
               self.negative_predictive_values() - 1

    def list_labels(self):
        if self.list_labels is None:
            return ()
        return tuple(np.unique(self.list_labels))

    def vol_diff(self):
        """
        This function calculates the ratio of difference in volume between
        the reference and segmentation images.

        :return: vol_diff
        """
        return np.abs(self.n_pos_ref() - self.n_pos_seg()) / self.n_pos_ref()

    # @CacheFunctionOutput
    # def _boundaries_dist_mat(self):
    #     dist = DistanceMetric.get_metric('euclidean')
    #     border_ref = MorphologyOps(self.ref, self.neigh).border_map()
    #     border_seg = MorphologyOps(self.seg, self.neigh).border_map()
    #     coord_ref = np.multiply(np.argwhere(border_ref > 0), self.pixdim)
    #     coord_seg = np.multiply(np.argwhere(border_seg > 0), self.pixdim)
    #     pairwise_dist = dist.pairwise(coord_ref, coord_seg)
    #     return pairwise_dist

    def measured_distance(self):
        """
        This functions calculates the average symmetric distance and the
        hausdorff distance between a segmentation and a reference image

        :return: hausdorff distance and average symmetric distance
        """
        ref_border_dist, seg_border_dist, ref_border, \
            seg_border = self.border_distance()
        average_distance = (np.sum(ref_border_dist) + np.sum(
            seg_border_dist)) / (np.sum(self.ref + self.seg))
        hausdorff_distance = np.max(
            [np.max(ref_border_dist), np.max(seg_border_dist)])
        return hausdorff_distance, average_distance

    def measured_average_distance(self):
        """
        This function returns only the average distance when calculating the
        distances between segmentation and reference

        :return:
        """
        return self.measured_distance()[1]

    def measured_hausdorff_distance(self):
        """
        This function returns only the hausdorff distance when calculated the
        distances between segmentation and reference

        :return:
        """
        return self.measured_distance()[0]

    # def average_distance(self):
    #     pairwise_dist = self._boundaries_dist_mat()
    #     return (np.sum(np.min(pairwise_dist, 0)) + \
    #             np.sum(np.min(pairwise_dist, 1))) / \
    #            (np.sum(self.ref + self.seg))
    #
    # def hausdorff_distance(self):
    #     pairwise_dist = self._boundaries_dist_mat()
    #     return np.max((np.max(np.min(pairwise_dist, 0)),
    #                    np.max(np.min(pairwise_dist, 1))))

    def connected_elements(self):
        """
        This function returns the number of FP FN and TP in terms of
        connected components.

        :return: Number of true positive connected components, Number of
            false positives connected components, Number of false negatives
            connected components
        """
        blobs_ref, blobs_seg, init = self._connected_components()
        list_blobs_ref = range(1, blobs_ref[1])
        list_blobs_seg = range(1, blobs_seg[1])
        mul_blobs_ref = np.multiply(blobs_ref[0], init)
        mul_blobs_seg = np.multiply(blobs_seg[0], init)
        list_TP_ref = np.unique(mul_blobs_ref[mul_blobs_ref > 0])
        list_TP_seg = np.unique(mul_blobs_seg[mul_blobs_seg > 0])

        list_FN = [x for x in list_blobs_ref if x not in list_TP_ref]
        list_FP = [x for x in list_blobs_seg if x not in list_TP_seg]
        return len(list_TP_ref), len(list_FP), len(list_FN)

    def connected_errormaps(self):
        """
        This functions calculates the error maps from the connected components

        :return:
        """
        blobs_ref, blobs_seg, init = self._connected_components()
        list_blobs_ref = range(1, blobs_ref[1])
        list_blobs_seg = range(1, blobs_seg[1])
        mul_blobs_ref = np.multiply(blobs_ref[0], init)
        mul_blobs_seg = np.multiply(blobs_seg[0], init)
        list_TP_ref = np.unique(mul_blobs_ref[mul_blobs_ref > 0])
        list_TP_seg = np.unique(mul_blobs_seg[mul_blobs_seg > 0])

        list_FN = [x for x in list_blobs_ref if x not in list_TP_ref]
        list_FP = [x for x in list_blobs_seg if x not in list_TP_seg]
        # print(np.max(blobs_ref),np.max(blobs_seg))
        tpc_map = np.zeros_like(blobs_ref[0])
        fpc_map = np.zeros_like(blobs_ref[0])
        fnc_map = np.zeros_like(blobs_ref[0])
        for i in list_TP_ref:
            tpc_map[blobs_ref[0] == i] = 1
        for i in list_TP_seg:
            tpc_map[blobs_seg[0] == i] = 1
        for i in list_FN:
            fnc_map[blobs_ref[0] == i] = 1
        for i in list_FP:
            fpc_map[blobs_seg[0] == i] = 1
        return tpc_map, fnc_map, fpc_map

    def outline_error(self):
        """
        This function calculates the outline error as defined in Wack et al.

        :return: OER: Outline error ratio, OEFP: number of false positive
            outlier error voxels, OEFN: number of false negative outline error
            elements
        """
        TPcMap, _, _ = self.connected_errormaps()
        OEFMap = self.ref - np.multiply(TPcMap, self.seg)
        unique, counts = np.unique(OEFMap, return_counts=True)
        # print(counts)
        OEFN = counts[unique == 1]
        OEFP = counts[unique == -1]
        OEFN = 0 if len(OEFN) == 0 else OEFN[0]
        OEFP = 0 if len(OEFP) == 0 else OEFP[0]
        OER = 2 * (OEFN + OEFP) / (self.n_pos_seg() + self.n_pos_ref())
        return OER, OEFP, OEFN

    def detection_error(self):
        """
        This function calculates the volume of detection error as defined in
        Wack et al.

        :return: DE: Total volume of detection error, DEFP: Detection error
            false positives, DEFN: Detection error false negatives
        """
        TPcMap, FNcMap, FPcMap = self.connected_errormaps()
        DEFN = np.sum(FNcMap)
        DEFP = np.sum(FPcMap)
        return DEFN + DEFP, DEFP, DEFN

    def header_str(self):
        result_str = [self.m_dict[key][1] for key in self.measures]
        result_str = ',' + ','.join(result_str)
        return result_str

    def to_string(self, fmt='{:.4f}'):
        result_str = ""
        list_space = ['com_ref', 'com_seg', 'list_labels']
        for key in self.measures:
            result = self.m_dict[key][0]()
            if key in list_space:
                result_str += ' '.join(fmt.format(x) for x in result) \
                    if isinstance(result, tuple) else fmt.format(result)
            else:
                result_str += ','.join(fmt.format(x) for x in result) \
                    if isinstance(result, tuple) else fmt.format(result)
            result_str += ','
        return result_str[:-1]  # trim the last comma


class PairwiseMeasuresRegression(object):
    def __init__(self, reg_img, ref_img, measures=None):
        self.reg = reg_img
        self.ref = ref_img
        self.measures = measures

        self.m_dict = {
            'mse': (self.mse, 'MSE'),
            'rmse': (self.rmse, 'RMSE'),
            'mae': (self.mae, 'MAE'),
            'r2': (self.r2, 'R2')
        }

    def mse(self):
        return np.mean(np.square(self.reg - self.ref))

    def rmse(self):
        return np.sqrt(self.mse())

    def mae(self):
        return np.mean(np.abs(self.ref - self.reg))

    def r2(self):
        ref_var = np.sum(np.square(self.ref - np.mean(self.ref)))
        reg_var = np.sum(np.square(self.reg - np.mean(self.reg)))
        cov_refreg = np.sum(
            (self.reg - np.mean(self.reg)) * (self.ref - np.mean(
                self.ref)))
        return np.square(cov_refreg / np.sqrt(ref_var * reg_var + 0.00001))

    def header_str(self):
        result_str = [self.m_dict[key][1] for key in self.measures]
        result_str = ',' + ','.join(result_str)
        return result_str

    def to_string(self, fmt='{:.4f}'):
        result_str = ""
        for key in self.measures:
            result = self.m_dict[key][0]()
            result_str += ','.join(fmt.format(x) for x in result) \
                if isinstance(result, tuple) else fmt.format(result)
            result_str += ','
        return result_str[:-1]  # trim the last comma


def soft_dice_score(y_true, y_pred, epsilon=1e-6):
    """
    Soft dice loss calculation for arbitrary batch size, number of classes, and number of spatial dimensions.
    Assumes the `channels_last` format.

    # Arguments
        y_true: b x X x Y( x Z...) x c One hot encoding of ground truth
        y_pred: b x X x Y( x Z...) x c Network output, must sum to 1 over c channel (such as after softmax)
        epsilon: Used for numerical stability to avoid divide by zero errors

    # References
        V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation
        https://arxiv.org/abs/1606.04797
        More details on Dice loss formulation
        https://mediatum.ub.tum.de/doc/1395260/1395260.pdf (page 72)

        Adapted from https://github.com/Lasagne/Recipes/issues/99#issuecomment-347775022
    """

    # skip the batch and class axis for calculating Dice score
    axes = tuple(range(2, len(y_pred.shape)))
    numerator = 2. * np.sum(y_pred * y_true, axes)
    denominator = np.sum(np.square(y_pred) + np.square(y_true), axes)

    return np.mean((numerator + epsilon) / (denominator + epsilon))  # average over classes and batch


# Function for proper handling of bools in argparse
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description='Passing files + relevant directories')
parser.add_argument('--csv_label', type=str)
parser.add_argument('--images_dir', type=str)
parser.add_argument('--labels_dir', type=str)
parser.add_argument('--job_name', type=str)
parser.add_argument('--experiment_mode', type=str)
parser.add_argument("--physics_flag", type=str2bool, nargs='?', const=True, default=False)
parser.add_argument("--patch_size", type=int, default=80)
# parser.add_argument('--resolution', type=int)
arguments = parser.parse_args()


class BadDataset:
    def __init__(self, df, transform):
        self.df = df
        self.loader = porchio.ImagesDataset
        self.transform = transform
        self.sampler = porchio.data.UniformSampler(patch_size=80)

    def __getitem__(self, index):
        # These names are arbitrary
        MRI = 'mri'
        SEG = 'seg'
        PHYSICS = 'physics'

        subjects = []
        for (image_path, label_path, subject_physics) in zip(self.df.Filename, self.df.Label_Filename,
                                                             self.df.subject_physics):
            subject_dict = {
                MRI: porchio.ScalarImage(image_path),
                SEG: porchio.LabelMap(label_path),
                PHYSICS: subject_physics
            }
            subject = porchio.Subject(subject_dict)
            subjects.append(subject)
        this_dataset = self.loader(subjects, self.transform)

        patches_dataset = porchio.Queue(
            subjects_dataset=this_dataset,
            max_length=queue_length,
            samples_per_volume=samples_per_volume,
            sampler=porchio.sampler.UniformSampler(patch_size),
            shuffle_subjects=False,
            shuffle_patches=False,
        )

        return patches_dataset

    def __len__(self):
        return self.df.shape[0]


def BespokeDataset(df, transform, patch_size, batch_seed):
    loader = porchio.ImagesDataset
    sampler = porchio.data.UniformSampler(patch_size=patch_size, batch_seed=batch_seed)

    # These names are arbitrary
    MRI = 'mri'
    SEG = 'seg'
    PHYSICS = 'physics'

    subjects = []
    for (image_path, label_path, subject_physics) in zip(df.Filename, df.Label_Filename, df.subject_physics):
        subject_dict = {
            MRI: porchio.ScalarImage(image_path),
            SEG: porchio.LabelMap(label_path),
            PHYSICS: subject_physics
        }
        subject = porchio.Subject(subject_dict)
        subjects.append(subject)
    this_dataset = loader(subjects, transform)

    patches_dataset = porchio.Queue(
        subjects_dataset=this_dataset,
        max_length=queue_length,
        samples_per_volume=samples_per_volume,
        sampler=sampler,
        shuffle_subjects=False,
        shuffle_patches=False,
        num_workers=24,
    )

    return patches_dataset


# Not enough to shuffle batches, shuffle WITHIN batches!
# Take original csv, shuffle between subjects!
def reshuffle_csv(og_csv, batch_size):
    # Calculate some necessary variables
    batch_reshuffle_csv = pd.DataFrame({})
    num_images = len(og_csv)
    batch_numbers = list(np.array(range(num_images // batch_size)) * batch_size)
    num_unique_subjects = og_csv.subject_id.nunique()
    unique_subject_ids = og_csv.subject_id.unique()

    # First, re-order within subjects so batches don't always contain same combination of physics parameters
    for sub_ID in unique_subject_ids:
        batch_reshuffle_csv = batch_reshuffle_csv.append(og_csv[og_csv.subject_id == sub_ID].sample(frac=1).
                                                         reset_index(drop=True), ignore_index=True)

    # Set up empty lists for appending re-ordered entries
    new_subject_ids = []
    new_filenames = []
    new_label_filenames = []
    new_physics = []
    new_folds = []
    for batch in range(num_images // batch_size):
        # Randomly sample a batch ID
        batch_id = random.sample(batch_numbers, 1)[0]
        # Find those images/ labels/ params stipulated by the batch ID
        transferred_subject_ids = batch_reshuffle_csv.subject_id[batch_id:batch_id + batch_size]
        transferred_filenames = batch_reshuffle_csv.Filename[batch_id:batch_id + batch_size]
        transferred_label_filenames = batch_reshuffle_csv.Label_Filename[batch_id:batch_id + batch_size]
        transferred_physics = batch_reshuffle_csv.subject_physics[batch_id:batch_id + batch_size]
        transferred_folds = batch_reshuffle_csv.fold[batch_id:batch_id + batch_size]
        # Append these to respective lists
        new_subject_ids.extend(transferred_subject_ids)
        new_filenames.extend(transferred_filenames)
        new_label_filenames.extend(transferred_label_filenames)
        new_physics.extend(transferred_physics)
        new_folds.extend(transferred_folds)
        # Remove batch number used to reshuffle certain batches
        batch_numbers.remove(batch_id)

    altered_basic_csv = pd.DataFrame({
        'subject_id': new_subject_ids,
        'Filename': new_filenames,
        'subject_physics': new_physics,
        'fold': new_folds,
        'Label_Filename': new_label_filenames
    })
    return altered_basic_csv


def visualise_batch_patches(loader, bs, ps, comparisons=2):
    print('Calculating tester...')
    assert comparisons <= batch_size
    next_data = next(iter(loader))
    batch_samples = random.sample(list(range(bs)), comparisons)
    import matplotlib.pyplot as plt
    # Set up figure for ALL intra-batch comparisons
    f, axarr = plt.subplots(3, comparisons)
    for comparison in range(comparisons):
        # print(f'Label shape is {next_data["seg"]["data"].shape}')
        # print(f'Data shape is {next_data["mri"]["data"].shape}')
        example_batch_patch = np.squeeze(next_data['mri']['data'][batch_samples[comparison], ..., int(ps/2)])
        # For segmentation need to check that all classes (in 4D) have same patch that ALSO matches data
        example_batch_patch2 = np.squeeze(next_data['seg']['data'][batch_samples[comparison], 0, ..., int(ps/2)])
        example_batch_patch3 = np.squeeze(next_data['seg']['data'][batch_samples[comparison], 1, ..., int(ps/2)])
        axarr[0, comparison].imshow(example_batch_patch)
        axarr[0, comparison].axis('off')
        axarr[1, comparison].imshow(example_batch_patch2)
        axarr[1, comparison].axis('off')
        axarr[2, comparison].imshow(example_batch_patch3)
        axarr[2, comparison].axis('off')
    plt.show()


# Stratification specific functions
def feature_loss_func(volume1, volume2, tm):
    if tm == 'stratification':
        if type(volume2) == np.ndarray:
            return np.mean((volume1 - volume2) ** 2)
        else:
            return torch.mean((volume1 - volume2) ** 2).item()
    elif tm == 'kld':
        kld = torch.nn.KLDivLoss()
        if type(volume2) == np.ndarray:
            raise TypeError
        else:
            return kld(volume1.detach().cpu(), volume2.detach().cpu())


def stratification_checker(input_volume):
    # Will only work for batch size 4 for now, but that comprises most experiments
    return int(torch.sum(input_volume[0, ...] + input_volume[3, ...] - input_volume[1, ...] - input_volume[2, ...]))


def calc_feature_loss(input_volume, tm):
    feature_loss1 = feature_loss_func(
        volume1=input_volume[0, ...],
        volume2=input_volume[1, ...], tm=tm)
    feature_loss2 = feature_loss_func(
        volume1=input_volume[0, ...],
        volume2=input_volume[2, ...], tm=tm)
    feature_loss3 = feature_loss_func(
        volume1=input_volume[0, ...],
        volume2=input_volume[3, ...], tm=tm)
    feature_loss4 = feature_loss_func(
        volume1=input_volume[1, ...],
        volume2=input_volume[2, ...], tm=tm)
    feature_loss5 = feature_loss_func(
        volume1=input_volume[1, ...],
        volume2=input_volume[3, ...], tm=tm)
    feature_loss6 = feature_loss_func(
        volume1=input_volume[2, ...],
        volume2=input_volume[3, ...], tm=tm)

    total_feature_loss = np.mean([feature_loss1,
                                  feature_loss2,
                                  feature_loss3,
                                  feature_loss4,
                                  feature_loss5,
                                  feature_loss6])
    return total_feature_loss


def normalise_image(array):
    return (array - np.min(array)) / (np.max(array) - np.min(array))


# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
# torch.cuda.empty_cache()

# Writer will output to ./runs/ directory by default
log_dir = f'/nfs/home/pedro/PhysicsPyTorch/logger/logs/{arguments.job_name}'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
SAVE_PATH = os.path.join(f'/nfs/home/pedro/PhysicsPyTorch/logger/models/{arguments.job_name}')
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
SAVE = True
LOAD = True
patch_test = False
val_test = False

# Physics specific parameters
physics_flag = arguments.physics_flag
physics_experiment_type = 'MPRAGE'
physics_input_size = {'MPRAGE': 2,
                      'SPGR': 6}


def physics_preprocessing(physics_input, experiment_type):
    if experiment_type == 'MPRAGE':
        expo_physics = torch.exp(-physics_input)
        overall_physics = torch.stack((physics, expo_physics), dim=1)
    elif experiment_type == 'SPGR':
        TR_expo_params = torch.unsqueeze(torch.exp(-physics_input[:, 0]), dim=1)
        TE_expo_params = torch.unsqueeze(torch.exp(-physics_input[:, 1]), dim=1)
        FA_sin_params = torch.unsqueeze(torch.sin(physics_input[:, 2] * 3.14159265 / 180), dim=1)
        overall_physics = torch.stack((physics, TR_expo_params, TE_expo_params, FA_sin_params), dim=1)
    return overall_physics


# Uncertainty: Pure noise
def corrected_paper_stochastic_loss(logits, sigma, labels, num_passes):
    total_loss = 0
    # sigma_epsilon = float(self.segmentation_param.uncertainty_epsilon)
    # noise_array = tf.random_normal(shape=[num_passes])
    logits_shape = logits.get_shape().as_list()
    logits_shape.append(num_passes)
    noise_array = torch.normal(mean=0.0, std=1.0, size=logits_shape)
    # SCALE the uncertainty
    sigma = sigma  # + sigma_epsilon
    for fpass in range(num_passes):
        stochastic_output = logits + sigma * noise_array[..., fpass]
        exponent_B = torch.log(torch.sum(torch.exp(stochastic_output), dim=-1, keepdim=True))
        inner_logits = exponent_B - stochastic_output
        soft_inner_logits = labels * inner_logits
        total_loss += torch.exp(soft_inner_logits)
    mean_loss = total_loss / num_passes
    actual_loss = torch.sum(torch.log(mean_loss))
    return actual_loss


# Check if SAVE_PATH is empty
file_list = os.listdir(path=SAVE_PATH)
num_files = len(file_list)


# Hyper-parameter loading: General parameters so doesn't matter which model file is loaded exactly
if LOAD and num_files > 0:
    model_files = glob.glob(os.path.join(SAVE_PATH, '*.pth'))
    latest_model_file = max(model_files, key=os.path.getctime)
    checkpoint = torch.load(latest_model_file, map_location=torch.device('cuda:0'))
    print(f'Loading {latest_model_file}!')
    loaded_epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    running_iter = checkpoint['running_iter']
    EPOCHS = 100

    # Memory related variables
    batch_size = checkpoint['batch_size']
    queue_length = batch_size
    patch_size = checkpoint['patch_size']
    samples_per_volume = 1
else:
    running_iter = 0
    loaded_epoch = -1
    EPOCHS = 100

    # Memory related variables
    patch_size = arguments.patch_size
    batch_size = 4
    queue_length = batch_size
    samples_per_volume = 1

# Stratification
training_modes = ['standard', 'stratification', 'kld']
training_mode = arguments.experiment_mode
stratification_epsilon = 0.2

# Some necessary variables
dataset_csv = arguments.csv_label
img_dir = arguments.images_dir  # '/nfs/home/pedro/COVID/Data/KCH_CXR_JPG'
label_dir = arguments.labels_dir  # '/nfs/home/pedro/COVID/Labels/KCH_CXR_JPG.csv'
print(img_dir)
print(label_dir)
val_batch_size = 4


# Read csv + add directory to filenames
df = pd.read_csv(dataset_csv)
df['Label_Filename'] = df['Filename']
df['Filename'] = img_dir + '/' + df['Filename'].astype(str)
df['Label_Filename'] = label_dir + '/' + 'Label_' + df['Label_Filename'].astype(str)
num_folds = df.fold.nunique()

# Image generation code is hiding under this comment
# On demand image generation
# def mprage(T1, PD, TI, TD, tau, Gs=1):
#     mprage_img = Gs * PD * (1 - 2 * np.exp(-TI / T1) / (1 + np.exp(-(TI + TD + tau) / T1)))
#     return mprage_img
#
#
# class RandomMPRAGE(RandomTransform):
#     def __init__(
#             self,
#             TI: Union[float, Tuple[float, float]] = (600, 1200),
#             p: float = 1,
#             seed: Optional[int] = None,
#             keys: Optional[List[str]] = None,
#             ):
#         super().__init__(p=p, seed=seed, keys=keys)
#         self.coefficients_range = self.parse_range(
#             TI, 'TI_range')
#         self.order = self.parse_order(order)
#
#     def apply_transform(self, sample: Subject) -> dict:
#         random_parameters_images_dict = {}
#         TI_list = []
#         for image_name, image_dict in sample.get_images_dict().items():
#             TI = torch.FloatTensor(1).uniform_(*self.TI)
#             random_parameters_dict = {'TI': TI}
#             random_parameters_images_dict[image_name] = random_parameters_dict
#
#             generated_mprage = self.generated_mprage(T1=image_dict['data'][..., 0],
#                                                      TD=image_dict['data'][..., 2],
#                                                      TI=TI)
#             image_dict['data'] = generated_mprage
#             TI_list.append(TI)
#         sample['physics'] = torch.Tensor(TI_list)
#         sample.add_transform(self, random_parameters_images_dict)
#         return sample

    # @staticmethod
    # def generate_MPRAGE(T1, PD, TI, TD=10e-3, tau=10e-3, Gs=1):
    #     mprage_img = Gs * PD * (1 - 2 * np.exp(-TI / T1) / (1 + np.exp(-(TI + TD + tau) / T1)))
    #     return mprage_img


# Transforms

training_transform = porchio.Compose([
    # porchio.RescaleIntensity((0, 1)),  # so that there are no negative values for RandomMotion
    # porchio.RandomMotion(),
    # porchio.HistogramStandardization({MRI: landmarks}),
    porchio.RandomBiasField(coefficients=0.2),  # Bias field coeffs: Default 0.5 may be a bit too high!
    porchio.ZNormalization(masking_method=None),  # This is whitening
    porchio.RandomNoise(std=(0, 0.1)),
    # porchio.ToCanonical(),
    # porchio.Resample((4, 4, 4)),
    # porchio.CropOrPad((48, 60, 48)),
    # porchio.RandomFlip(axes=(0,)),
    # porchio.OneOf({
    #     porchio.RandomAffine(): 0.8,
    #     porchio.RandomElasticDeformation(): 0.2,}),
])

validation_transform = porchio.Compose([
    # porchio.HistogramStandardization({MRI: landmarks}),
    porchio.ZNormalization(masking_method=None),
    # porchio.ToCanonical(),
    # porchio.Resample((4, 4, 4)),
    # porchio.CropOrPad((48, 60, 48)),
])

# CUDA variables
use_cuda = torch.cuda.is_available()
print('Using cuda', use_cuda)

if use_cuda and torch.cuda.device_count() > 1:
    print('Using', torch.cuda.device_count(), 'GPUs!')

stacked_cv = False
if not stacked_cv:
    inf_fold = 5
    inf_df = df[df.fold == inf_fold]
    inf_df.reset_index(drop=True, inplace=True)

# For aggregation
overall_val_names = []
overall_val_metric = []
overall_gm_volumes = []


print('\nStarting training!')
for fold in range(num_folds):
    print('\nFOLD', fold)
    # Pre-loading sequence
    model = nnUNet(1, 2, physics_flag=physics_flag, physics_input=physics_input_size[physics_experiment_type],
                   physics_output=40)
    model = nn.DataParallel(model)
    # optimizer = RangerLars(model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9)

    # Early Stopping
    early_stopping = pytorchtools.EarlyStopping(patience=5, verbose=True)

    # Running lists
    running_val_names = []
    running_val_metric = []
    running_gm_volumes = []

    # Specific fold writer
    writer = SummaryWriter(log_dir=os.path.join(log_dir, f'fold_{fold}'))

    if LOAD and num_files > 0:
        # Get model file specific to fold
        loaded_model_file = f'model_epoch_{loaded_epoch}_fold_{fold}.pth'
        checkpoint = torch.load(os.path.join(SAVE_PATH, loaded_model_file), map_location=torch.device('cuda:0'))
        # Main model variables
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # Get the validation entries from previous folds!
        running_val_names = checkpoint['running_val_names']
        running_val_metric = checkpoint['running_val_metric']
        running_gm_volumes = checkpoint['running_gm_volumes']
        overall_val_names = checkpoint['overall_val_names']
        overall_val_metric = checkpoint['overall_val_metric']
        overall_gm_volumes = checkpoint['overall_gm_volumes']
        # Ensure that no more loading is done for future folds
        LOAD = False

    if stacked_cv:  # Pretty much never use this one
        # Train / Val/ Inf split
        val_fold = fold
        inf_fold = num_folds - fold - 1
        excluded_folds = [val_fold, inf_fold]
        train_df = df[~df.fold.isin(excluded_folds)]
        val_df = df[df.fold == val_fold]
        inf_df = df[df.fold == inf_fold]
        train_df.reset_index(drop=True, inplace=True)
        val_df.reset_index(drop=True, inplace=True)
        inf_df.reset_index(drop=True, inplace=True)
    else:
        # Train / Val split
        val_fold = fold
        excluded_folds = [val_fold]
        train_df = df[~df.fold.isin(excluded_folds)]
        val_df = df[df.fold == val_fold]
        train_df.reset_index(drop=True, inplace=True)
        val_df.reset_index(drop=True, inplace=True)

    print(f'The length of the training is {len(train_df)}')
    print(f'The length of the validation is {len(val_df)}')
    print(f'The length of the validation is {len(inf_df)}')

    model.cuda()
    print('\nStarting training!')
    for epoch in range(0, EPOCHS):
        print('Training Epoch')
        running_loss = 0.0
        model.train()
        train_acc = 0
        total_dice = 0
        new_seed = np.random.randint(10000)

        # Shuffle training and validation:
        new_train_df = reshuffle_csv(og_csv=train_df, batch_size=batch_size)
        new_val_df = reshuffle_csv(og_csv=val_df, batch_size=batch_size)

        # Val test
        if val_test:
            new_train_df = new_train_df[:20]

        # And generate new loaders
        patches_training_set = BespokeDataset(new_train_df, training_transform, patch_size, batch_seed=new_seed)
        train_loader = DataLoader(patches_training_set, batch_size=batch_size, shuffle=False)
        patches_validation_set = BespokeDataset(new_val_df, validation_transform, patch_size, batch_seed=new_seed)
        val_loader = DataLoader(patches_validation_set, batch_size=val_batch_size)

        # Early stopping
        best_val_dice = 0.0
        best_counter = 0

        # Patch test
        if patch_test and epoch == 0 and fold == 0:
            visualise_batch_patches(loader=train_loader, bs=batch_size, ps=patch_size, comparisons=4)
        for i, sample in enumerate(train_loader):
            start = time.time()
            images = sample['mri']['data'].cuda()
            labels = sample['seg']['data'].cuda()
            physics = sample['physics'].cuda().float()
            names = sample['mri']['path']
            names = [os.path.basename(name) for name in names]

            # Pass images to the model
            if physics_flag:
                # Calculate physics extensions
                processed_physics = physics_preprocessing(physics, physics_experiment_type)
                # print(f'Processed physics shape is {processed_physics.shape}')
                out, features_out = model(images, processed_physics)
            # print(f'Images shape is {images.shape}')
            else:
                out, features_out = model(images)

            # Need loss here
            eps = 1e-10
            data_loss = F.binary_cross_entropy_with_logits(out+eps, labels, reduction='mean')

            if training_mode == 'standard':
                loss = data_loss
                # print(f"iter: {running_iter}, Loss: {loss.item():.4f},"
                #       f"                                           ({(time.time() - start):.3f}s)")
                # Tracking feature loss for comparison purposes
                total_feature_loss = 0.1 * np.abs(calc_feature_loss(
                    features_out, tm='stratification'))  # NOTE: This needs to be the feature tensor!
                writer.add_scalar('Loss/Feature_loss', total_feature_loss, running_iter)

            elif training_mode == 'stratification' or training_mode == 'kld':
                total_feature_loss = 0.1 * np.abs(calc_feature_loss(
                    features_out, tm=training_mode))  # NOTE: This needs to be the feature tensor!
                regulatory_ratio = data_loss / total_feature_loss

                # print(f'The stratification check value is {stratification_checker(labels)}')
                loss = data_loss + stratification_epsilon * total_feature_loss / (
                        1 + stratification_checker(labels) * float(1e9)) ** 2
                # print(f"iter: {running_iter}, Loss: {loss.item():.4f}, strat: {stratification_checker(labels):.3f}"
                #       f"                                    ({(time.time() - start):.3f} s)")
                writer.add_scalar('Loss/Feature_loss', total_feature_loss, running_iter)

            # Softmax to convert to probabilities
            out = torch.softmax(out, dim=1)

            # pGM = PairwiseMeasures(labels[:, 0, ...].detach().cpu().numpy(), out[:, 0, ...].detach().cpu().numpy())
            # print(pGM.dice_score())
            pGM_dice = soft_dice_score(labels.cpu().detach().numpy(), out.cpu().detach().numpy())
            print(pGM_dice)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Name check: Shuffling sanity check
            if i == 0:
                print(f'The test names are: {names[0]}, {names[-2]}')

            # Terminal logging
            print(f"iter: {running_iter}, Loss: {loss.item():.4f}, strat: {stratification_checker(labels):.3f}"
                  f"                                    ({(time.time() - start):.3f} s)")

            # Writing to tensorboard
            if running_iter % 50 == 0:
                # Normalise images
                images = images.cpu().detach().numpy()
                out = out.cpu().detach().numpy()
                images = normalise_image(images)
                out = normalise_image(out)
                labels = labels.cpu().detach().numpy()

                writer.add_scalar('Loss/train', loss.item(), running_iter)
                img2tensorboard.add_animated_gif(writer=writer, image_tensor=images[0, ...],
                                                 tag=f'Visuals/Images_Fold_{fold}', max_out=patch_size//2,
                                                 scale_factor=255, global_step=running_iter)
                img2tensorboard.add_animated_gif(writer=writer, image_tensor=labels[0, 0, ...][None, ...],
                                                 tag=f'Visuals/Labels_Fold_{fold}', max_out=patch_size//2,
                                                 scale_factor=255, global_step=running_iter)
                img2tensorboard.add_animated_gif(writer=writer, image_tensor=out[0, 0, ...][None, ...],
                                                 tag=f'Visuals/Output_Fold_{fold}', max_out=patch_size//2,
                                                 scale_factor=255, global_step=running_iter)

            running_iter += 1

        print("Epoch: {}, Loss: {},\n Train Dice: Not implemented".format(epoch, running_loss))

        print('Validation step')
        model.eval()
        val_metric = DiceLoss(include_background=True, to_onehot_y=False, sigmoid=False, softmax=True)
        val_running_loss = 0
        # correct = 0
        val_counter = 0
        names_collector = []
        metric_collector = []
        gm_volumes_collector = []

        with torch.no_grad():
            for val_sample in val_loader:
                val_images = val_sample['mri']['data'].squeeze().cuda()
                val_names = val_sample['mri']['path']
                # Readjust dimensions to match expected shape for network
                # if len(val_images.shape) == 3:
                #     val_images = torch.unsqueeze(torch.unsqueeze(val_images, 0), 0)
                # elif len(val_images.shape) == 4:
                #     val_images = torch.unsqueeze(val_images, 0)
                val_labels = val_sample['seg']['data'].squeeze().cuda()
                # print(f'val_images shape is {val_images.shape}')
                # print(f'val_labels shape is {val_labels.shape}')
                # Readjust dimensions to match expected shape
                if len(val_labels.shape) == 4:
                    val_labels = torch.unsqueeze(val_labels, 1)
                if len(val_images.shape) == 4:
                    val_images = torch.unsqueeze(val_images, 1)
                val_physics = val_sample['physics'].squeeze().cuda().float()
                val_names = val_sample['mri']['path']
                val_names = [os.path.basename(val_name) for val_name in val_names]

                # Small name check
                print(val_names)

                # Pass images to the model
                if physics_flag:
                    # Calculate physics extensions
                    val_processed_physics = physics_preprocessing(val_physics, physics_experiment_type)
                    out, features_out = model(val_images, val_processed_physics)
                # print(f'Images shape is {images.shape}')
                else:
                    out, features_out = model(val_images)

                val_data_loss = F.binary_cross_entropy_with_logits(out, val_labels, reduction="mean")

                # Loss depends on training mode
                if training_mode == 'standard':
                    val_loss = val_data_loss
                elif training_mode == 'stratification' or training_mode == 'kld':
                    val_total_feature_loss = 0.1 * calc_feature_loss(
                        features_out, tm=training_mode)  # NOTE: This needs to be the feature tensor!
                    regulatory_ratio = val_data_loss / val_total_feature_loss
                    val_loss = val_data_loss + stratification_epsilon * total_feature_loss / (
                                1 + stratification_checker(val_labels) * float(1e9)) ** 2
                    # writer.add_scalar('Loss/Val_Feature_loss', val_total_feature_loss, running_iter)

                # print(f"out val shape is {out.shape}")  # Checking for batch dimension inclusion or not
                out = torch.softmax(out, dim=1)
                gm_out = out[:, 0, ...]

                val_running_loss += val_loss.item()

                # Metric calculation
                # pGM = PairwiseMeasures(val_labels[:, 0, ...].detach().cpu().numpy(), gm_out.detach().cpu().numpy())
                pGM_dice = soft_dice_score(val_labels.cpu().detach().numpy(), out.cpu().detach().numpy())
                print(pGM_dice)
                # dice_performance = val_metric.forward(out, val_labels)
                gm_volume = gm_out.view(4, -1).sum(1)
                metric_collector += [pGM_dice.tolist()]
                names_collector += val_names
                gm_volumes_collector += gm_volume

                # Convert to numpy arrays
                val_images = val_images.cpu().detach().numpy()
                val_labels = val_labels.cpu().detach().numpy()
                val_images = normalise_image(val_images)
                out = out.cpu().detach().numpy()
                out = normalise_image(out)

                val_counter += val_batch_size

        # Write to tensorboard
        writer.add_scalar('Loss/val', val_running_loss / val_counter, running_iter)
        writer.add_scalar('Loss/dice_val', np.mean(metric_collector)
                          , running_iter)
        img2tensorboard.add_animated_gif(writer=writer, image_tensor=val_images[0, ...],
                                         tag=f'Validation/Images_Fold_{fold}', max_out=patch_size // 4,
                                         scale_factor=255, global_step=running_iter)
        img2tensorboard.add_animated_gif(writer=writer, image_tensor=val_labels[0, 0, ...][None, ...],
                                         tag=f'Validation/Labels_Fold_{fold}', max_out=patch_size // 4,
                                         scale_factor=255, global_step=running_iter)
        img2tensorboard.add_animated_gif(writer=writer, image_tensor=out[0, 0, ...][None, ...],
                                         tag=f'Validation/Output_Fold_{fold}', max_out=patch_size // 4,
                                         scale_factor=255, global_step=running_iter)

        # Check if current val dice is better than previous best
        # true_dice = np.mean(metric_collector)
        # true_val = val_running_loss / val_counter  # alternative
        # if true_dice > best_val_dice:
        #     best_val_dice = true_dice
        #     append_string = 'not_best'
        #     best_counter = 0
        # else:
        #     append_string = 'nb'
        #     best_counter += 1

        # Aggregation
        running_val_metric.append(true_val)
        running_val_names.append(names_collector)
        running_gm_volumes.append(gm_volumes_collector)

        # # Save model
        # if SAVE and append_string == 'best':
        #     MODEL_PATH = os.path.join(SAVE_PATH, f'model_epoch_{epoch}_fold_{fold}.pth')
        #     print(MODEL_PATH)
        #     torch.save({'model_state_dict': model.state_dict(),
        #                 'optimizer_state_dict': optimizer.state_dict(),
        #                 'epoch': epoch,
        #                 'loss': loss,
        #                 'running_iter': running_iter,
        #                 'batch_size': batch_size,
        #                 'patch_size': patch_size,
        #                 'running_val_names': running_val_names,
        #                 'running_val_metric': running_val_metric,
        #                 'running_gm_volumes': running_gm_volumes,
        #                 'overall_gm_volumes': overall_gm_volumes,
        #                 'overall_val_names': overall_val_names,
        #                 'overall_val_metric': overall_val_metric}, MODEL_PATH)

        # Early stopping
        early_stopping(val_running_loss, model)

        if early_stopping.early_stop:
            # Save model
            if SAVE:
                MODEL_PATH = os.path.join(SAVE_PATH, f'model_epoch_{epoch}_fold_{fold}.pth')
                print(MODEL_PATH)
                torch.save({'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'epoch': epoch,
                            'loss': loss,
                            'running_iter': running_iter,
                            'batch_size': batch_size,
                            'patch_size': patch_size,
                            'running_val_names': running_val_names,
                            'running_val_metric': running_val_metric,
                            'running_gm_volumes': running_gm_volumes,
                            'overall_gm_volumes': overall_gm_volumes,
                            'overall_val_names': overall_val_names,
                            'overall_val_metric': overall_val_metric}, MODEL_PATH)

            # Set overalls to best epoch
            best_epoch = int(np.argmin(running_val_metric))
            print(f'The best epoch is Epoch {best_epoch}')
            overall_val_metric.append(running_val_metric[best_epoch])
            overall_val_names.extend(running_val_names[best_epoch])
            overall_gm_volumes.extend(running_gm_volumes[best_epoch])
            print('Early stopping!')
            break

        # if best_counter >= 5 and epoch > 10:
        #     # Set overalls to best epoch
        #     best_epoch = int(np.argmin(running_val_metric))
        #     print(f'The best epoch is Epoch {best_epoch}')
        #     overall_val_metric.append(running_val_metric[best_epoch])
        #     overall_val_names.extend(running_val_names[best_epoch])
        #     overall_gm_volumes.extend(running_gm_volumes[best_epoch])
        #     break

    # Now that this fold's training has ended, want starting points of next fold to reset
    latest_epoch = -1
    latest_fold = 0
    running_iter = 0

## Totals: What to collect after training has finished
# Dice for all validation? Volumes and COVs?

overall_val_metric = np.array(overall_val_metric)
overall_gm_volumes = np.array(overall_gm_volumes)
print(overall_val_names)
overall_subject_ids = [int(vn[0].split('_')[2]) for vn in overall_val_names]

# Folds analysis
print('Names', len(overall_val_names), 'Dice', len(overall_val_metric), 'GM volumes', len(overall_gm_volumes))

# Folds Dice
print('Overall Dice:', np.mean(overall_val_metric), 'std:', np.std(overall_val_metric))

sub = pd.DataFrame({"Filename": overall_val_names,
                    "subject_id": overall_subject_ids,
                    "Dice": overall_val_metric.tolist(),
                    "GM_volumes": overall_gm_volumes})

sub.to_csv(os.path.join(SAVE_PATH, 'dice_gm_volumes.csv'), index=False)

# CoVs
subject_CoVs = []
subject_dice = []
for ID in range(sub.subject_id.nunique()):
    # CoV, see: https://en.wikipedia.org/wiki/Coefficient_of_variation
    subject_CoV = np.std(sub[sub.subject_id == ID].GM_volumes) / np.mean(sub[sub.subject_id == ID].GM_volumes)
    subject_CoVs.append(subject_CoV)
    subject_dice.append(np.mean(sub[sub.subject_id == ID].Dice))
    print(f'The CoV for subject {ID} is {subject_CoV}')
cov_sub = pd.DataFrame({"subject_id": list(range(sub.subject_id.nunique())),
                        "subject_dice": subject_dice,
                        "subject_CoVs": subject_CoVs})
print(f"The mean and std of all subject CoVs is: {np.mean(subject_CoVs)}, {np.std(subject_CoVs)}")
print('Finished!')
