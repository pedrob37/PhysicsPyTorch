import numpy as np
import matplotlib
# matplotlib.use('TkAgg')  # TkAgg
matplotlib.get_backend()
print(matplotlib.get_backend())
import matplotlib.pyplot as plt
import glob
import os
import re
import nibabel as nib


plt.close('all')


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


# Functions
def sort_dir(dir_name, sub_split=-3, param_split=-1):
    """
    :param dir_name: Full directory name to be sorted
    :param sub_split: Location of subID to split in sorting
    :param param_split: Location of parameter to split in sorting
    :return: Sorted directory according to first (subID) and third value (TI) following underscores
    """
    print(dir_name)
    # float(xx.split('_')[param_split].lstrip('0'))])
    return sorted(os.listdir(dir_name), key=lambda xx: [float(xx.replace('.nii.gz', '').split('_')[sub_split]),
                                                        float(xx.replace('.nii.gz', '').split('_')[param_split])])


def read_file(filename):
    img = nib.load(filename)
    data = img.get_fdata()
    aff = img.affine
    return data, aff


def save_img(image, affine, filename):
    nifti_img = nib.Nifti1Image(image, affine)
    nib.save(nifti_img, filename)


def calc_prob_tiss_vols(volume_name):
    if type(volume_name) == str:
        volume_name, _ = read_file(volume_name)
    GM_prob_vol = np.sum(volume_name[..., 0])
    return GM_prob_vol


def standardise_volumes(volumes):
    return volumes - volumes[int(len(volumes)/2)]


def standardise_gt_volumes(volumes, sub_gt_seg):
    return volumes - sub_gt_seg


import argparse
parser = argparse.ArgumentParser(description="General processing: Volumes + uncertainty")
parser.add_argument("--inference_directories", type=str, nargs='+',
                    help="List of inference directories")
# parser.add_argument("--inference_tags", type=str, nargs='+',
#                     help="List of tags to assign each inference directory")
arguments = parser.parse_args()
all_directories = arguments.inference_directories
# Main directories to analyse
inference_directories = {}
# Some candidates:
# ood-base-mprage-generation-test-stand, ood-base-stratification-test-adam-144-50-v2, ood-base-standard-test-adam-144-v2, ood-mprage-generation-test, ood-mprage-generation-test-1, ood-phys-strat-144-es-covdice-5000-v2, ood-inf/new_maybe
for entry in all_directories:
    inference_directories[os.path.basename(entry)] = entry

# To keep track of what experiments are being tracked
combined_folder = '_'.join(list(inference_directories.keys()))

if not os.path.exists(f'/nfs/home/pedro/PhysicsPyTorch/Analyses/BlandAltman/{combined_folder}'):
    os.makedirs(f'/nfs/home/pedro/PhysicsPyTorch/Analyses/BlandAltman/{combined_folder}')
if not os.path.exists(f'/nfs/home/pedro/PhysicsPyTorch/Analyses/Volumes/{combined_folder}'):
    os.makedirs(f'/nfs/home/pedro/PhysicsPyTorch/Analyses/Volumes/{combined_folder}')

# Important reference directories
gt_seg_dir = 'unlisted'
prob_gt_seg_dir = '/nfs/project/pborges/AnalysisReferences/ProbSegs'

# Important params
TI_min = 100  # 600
TI_max = 2000  # 3000  # 1200
TI_increment = 100
# Number of samples = number of realisation per subject
num_reals = len(list(range(TI_min, TI_max+TI_increment, TI_increment)))
TIs = np.linspace(TI_min, TI_max, num_reals)
num_tissues = 1
num_experiments = len(inference_directories)
num_inference_subjects = 5

# Set up regex to extract subject ID from directory name
regex = re.compile(r'\d+')

# Create a placeholder array for the total tissue vols, for all experiments, for all inference subjects + dir tracking
all_tot_tiss_vols = np.zeros((num_experiments, num_inference_subjects, num_reals, num_tissues))
all_tot_tiss_dice = np.zeros((num_experiments, num_inference_subjects, num_reals, num_tissues))
dirs = []

# Histograms
uncertainty_dict = {'Dropout': 0,
                    'Hetero': 1,
                    'Total': 2}

num_hist_bins = 20
all_tot_histograms = np.zeros((len(inference_directories), num_inference_subjects, num_reals,
                               num_hist_bins, num_hist_bins, 3))  # 3 for: Dropout, Hetero, Total
x_bin_edges = np.zeros((len(inference_directories), num_inference_subjects, num_reals,
                        num_hist_bins + 1, 3))
y_bin_edges = np.zeros((len(inference_directories), num_inference_subjects, num_reals,
                        num_hist_bins + 1, 3))
# Argmax
argmax_all_tot_histograms = np.zeros((len(inference_directories), num_inference_subjects, num_reals,
                               num_hist_bins, num_hist_bins, 3))  # 3 for: Dropout, Hetero, Total
argmax_x_bin_edges = np.zeros((len(inference_directories), num_inference_subjects, num_reals,
                        num_hist_bins + 1, 3))
argmax_y_bin_edges = np.zeros((len(inference_directories), num_inference_subjects, num_reals,
                        num_hist_bins + 1, 3))
# Argmax
partial_argmax_all_tot_histograms = np.zeros((len(inference_directories), num_inference_subjects, num_reals,
                               num_hist_bins, num_hist_bins, 3))  # 3 for: Dropout, Hetero, Total
partial_argmax_x_bin_edges = np.zeros((len(inference_directories), num_inference_subjects, num_reals,
                        num_hist_bins + 1, 3))
partial_argmax_y_bin_edges = np.zeros((len(inference_directories), num_inference_subjects, num_reals,
                        num_hist_bins + 1, 3))
# Upper and lower bounds
lower_bounds = np.zeros((len(inference_directories), num_inference_subjects, num_reals))
upper_bounds = np.zeros((len(inference_directories), num_inference_subjects, num_reals))

lower_dice_bounds = np.zeros((len(inference_directories), num_inference_subjects, num_reals))
upper_dice_bounds = np.zeros((len(inference_directories), num_inference_subjects, num_reals))
hist_counter = 0
histogram_directories = {}
for num, directory in enumerate(inference_directories.items()):
    os.chdir(directory[1])
    print(directory[1])
    for inf_ID, dirname in enumerate(glob.glob('inf_*')):
        if inf_ID is not None:
            # Loop through the individual subject directories.
            # Append to parent directory to get full dir. to cd to
            current_dir = os.path.join(directory[1], dirname)
            os.chdir(current_dir)

            # Identify subject number via regular expressions
            sub_ID = np.int(regex.findall(dirname)[0])
            dirs.append(sub_ID) if sub_ID not in dirs else dirs
            # if 'overlap' not in directory[1]:
            #     seg_files = sort_dir(current_dir, sub_split=4, param_split=6)
            # else:
            #     seg_files = sort_dir(current_dir, sub_split=5, param_split=7)
            seg_files = sort_dir(current_dir)
            seg_files = [x for x in seg_files if "3.0000" not in x]
            for exc in range(1, 10):
                seg_files = [x for x in seg_files if f"2.{exc}" not in x]

            # Load in ground truth image (Obsolete for current Unc work)
            gt_seg_file = os.path.join(prob_gt_seg_dir, f'Dil_fin_merged_sub_{sub_ID}.nii.gz')
            gt_seg, _ = read_file(gt_seg_file)
            print(gt_seg_file)
            # Create a placeholder array for the individual tissue vols for each simulation volume
            tot_tiss_vols = np.zeros([len(seg_files), num_tissues])
            tot_tiss_dice = np.zeros([len(seg_files), num_tissues])

            # Calculate the total volume, for each tissue type (CSF, WM, GM) across subjects into N x 3 array
            dice_counter = 0
            for ID, seg_file in enumerate(seg_files):
                print(seg_file)
                # Dropout histogram
                try:
                    image_globber = f'*{seg_file.split("MPRAGE")[1]}'
                    print(f'The image globber is {image_globber}')
                    # print(f'Made it as far as loading! For experiment: {image_globber}')
                    qualitative_directory = os.path.join(directory[1], 'Qualitative_uncertainty')
                    qualitative_volume, _ = read_file(glob.glob(os.path.join(qualitative_directory, image_globber))[0])
                    print(f'Loaded volume: {os.path.basename(glob.glob(os.path.join(qualitative_directory, image_globber))[0])}')
                    if ID == 0:
                        print(f'Made it as far as loading! For experiment: {directory[0]}')
                        hist_counter += 1
                        histogram_directories[directory[0]] = os.path.basename(directory[1])
                    all_tot_histograms[num, inf_ID, ID, :, :, uncertainty_dict['Dropout']], \
                    x_bin_edges[num, inf_ID, ID, :, uncertainty_dict['Dropout']], \
                    y_bin_edges[num, inf_ID, ID, :, uncertainty_dict['Dropout']] = \
                        np.histogram2d(qualitative_volume[..., 1].flatten(),
                                       qualitative_volume[..., 0].flatten(),
                                       bins=20)
                    # Argmax
                    argmax_all_tot_histograms[num, inf_ID, ID, :, :, uncertainty_dict['Dropout']], \
                    argmax_x_bin_edges[num, inf_ID, ID, :, uncertainty_dict['Dropout']], \
                    argmax_y_bin_edges[num, inf_ID, ID, :, uncertainty_dict['Dropout']] = \
                        np.histogram2d(qualitative_volume[..., 3].flatten(),
                                       qualitative_volume[..., 2].flatten(),
                                       bins=20)
                    # Partial-Argmax
                    partial_argmax_all_tot_histograms[num, inf_ID, ID, :, :, uncertainty_dict['Dropout']], \
                    partial_argmax_x_bin_edges[num, inf_ID, ID, :, uncertainty_dict['Dropout']], \
                    partial_argmax_y_bin_edges[num, inf_ID, ID, :, uncertainty_dict['Dropout']] = \
                        np.histogram2d(qualitative_volume[..., 4].flatten(),
                                       qualitative_volume[..., 2].flatten(),
                                       bins=20)
                    # Try to find quantitative directory to extract minimum and maximum volumes
                    quantitative_directory = os.path.join(directory[1], 'Combined_seg_uncertainty')
                    # Ordering in last axis: Segmentation, min_volume, max_volume
                    print(f'Loaded quant. volume: '
                          f'{os.path.basename(glob.glob(os.path.join(quantitative_directory, image_globber))[0])}')
                    quantitative_volume, _ = read_file(
                        glob.glob(os.path.join(quantitative_directory, image_globber))[0])
                    pred_seg = quantitative_volume[..., 0]
                    min_vol = quantitative_volume[..., 1]
                    max_vol = quantitative_volume[..., 2]

                    # Subtract upper and lower bounds from the mean
                    lower_bounds[num, inf_ID, ID] = np.abs(np.sum(min_vol) - np.sum(pred_seg))
                    upper_bounds[num, inf_ID, ID] = np.abs(np.sum(max_vol) - np.sum(pred_seg))

                    # Dice score
                    pGM = soft_dice_score(gt_seg[..., 0], pred_seg)

                    # Also calculate dice scores
                    lower_dice_bounds[num, inf_ID, ID] = np.abs(soft_dice_score(gt_seg[..., 0], min_vol) - pGM)
                    upper_dice_bounds[num, inf_ID, ID] = np.abs(soft_dice_score(gt_seg[..., 0], max_vol) - pGM)

                    print(f'Errorbar example values are {lower_bounds[num, inf_ID, ID]},'
                          f'{upper_bounds[num, inf_ID, ID]}')
                except Exception as e:
                    print(e)
                    # Ordering in last axis: Segmentation, min_volume, max_volume
                    pred_seg, _ = read_file(seg_file)
                    pred_seg[~np.isfinite(pred_seg)] = 0
                    # Want histogram directories to be same length as number of experiments
                    if ID == 0:
                        histogram_directories[directory[0]] = 'No histogram data'
                    if len(pred_seg.shape) == 4:
                        pred_seg = pred_seg[..., 0]
                    pGM = soft_dice_score(gt_seg[..., 0], pred_seg)
                tot_tiss_vols[ID, 0] = np.sum(pred_seg)
                tot_tiss_dice[ID, 0] = pGM

            all_tot_tiss_vols[num, inf_ID, ...] = tot_tiss_vols
            all_tot_tiss_dice[num, inf_ID, ...] = tot_tiss_dice

# Plotting
# Individual plots
plot_skip = 1
standardised_tiss_volumes = np.zeros_like(all_tot_tiss_vols)
gt_standardised_tiss_volumes = np.zeros_like(all_tot_tiss_vols)
gt_vols = np.zeros((len(dirs), 1))
for counter, sub_ID in enumerate(dirs):
    # Calculate ranges for GM
    plt.interactive(False)
    plot_gt_seg_file = os.path.join(prob_gt_seg_dir, f'Dil_fin_merged_sub_{sub_ID}.nii.gz')
    gt_GM = calc_prob_tiss_vols(plot_gt_seg_file)
    gt_vols[counter] = gt_GM
    fig, axes = plt.subplots(1, num_tissues+2, sharex='all', figsize=(18.0, 10.0))
    np.random.seed(44)
    seeded_color = np.random.rand(len(inference_directories.items()), 3)
    # seeded_color[1, :] = [1, 0, 0]
    # seeded_color[2, :] = [0, 1, 0]
    axes[0].axvline(x=(TI_max + TI_min) / 2)
    axes[1].axvline(x=(TI_max + TI_min) / 2)
    for exp_num, directory in enumerate(inference_directories.items()):
        # Identify qualitative array
        standardised_tiss_volumes[exp_num,
                                  counter,
                                  ::plot_skip,
                                  0] = standardise_volumes(all_tot_tiss_vols[exp_num,
                                                                             counter,
                                                                             ::plot_skip,
                                                                             0])
        gt_standardised_tiss_volumes[exp_num,
                                     counter,
                                     ::plot_skip,
                                     0] = standardise_gt_volumes(all_tot_tiss_vols[exp_num,
                                                                                   counter,
                                                                                   ::plot_skip,
                                                                                   0], gt_GM)
        # tr = matplotlib.transforms.offset_copy(axes[0].transData, fig=fig, x=1.5 * (exp_num+0.1), y=0,
        #                                        units='points')
        # axes[0].scatter(TIs[::plot_skip],
        #                       # Decide whether to standardise with respect to Mid point or wrt PGS segmentation
        #                       standardised_tiss_volumes[exp_num, counter, ::plot_skip, 0],
        #                       label=f'GM: {directory[0]}',
        #                       c=seeded_color[exp_num])
        # axes[1].scatter(TIs[::plot_skip],
        #                       # Decide whether to standardise with respect to Mid point or wrt PGS segmentation
        #                       gt_standardised_tiss_volumes[exp_num, counter, ::plot_skip, 0],
        #                       label=f'GM: {directory[0]}',
        #                       c=seeded_color[exp_num])
        # axes[2].scatter(TIs[::plot_skip],
        #                       # Decide whether to standardise with respect to Mid point or wrt PGS segmentation
        #                       all_tot_tiss_dice[exp_num, counter, ::plot_skip, 0],
        #                       label=f'GM: {directory[0]}',
        #                       c=seeded_color[exp_num])
        # Errorbar plots
        axes[0].errorbar(TIs[::plot_skip],
                         # Decide whether to standardise with respect to Mid point or wrt PGS segmentation
                         standardised_tiss_volumes[exp_num, counter, ::plot_skip, 0],
                         yerr=[lower_bounds[exp_num, counter, ::plot_skip],
                               upper_bounds[exp_num, counter, ::plot_skip]],
                         label=f'GM: {directory[0]}',
                         c=seeded_color[exp_num], fmt='o')
        axes[1].errorbar(TIs[::plot_skip],
                         # Decide whether to standardise with respect to Mid point or wrt PGS segmentation
                         gt_standardised_tiss_volumes[exp_num, counter, ::plot_skip, 0],
                         yerr=[lower_bounds[exp_num, counter, ::plot_skip],
                               upper_bounds[exp_num, counter, ::plot_skip]],
                         label=f'GM: {directory[0]}',
                         c=seeded_color[exp_num], fmt='o')
        axes[2].errorbar(TIs[::plot_skip],
                         # Decide whether to standardise with respect to Mid point or wrt PGS segmentation
                         all_tot_tiss_dice[exp_num, counter, ::plot_skip, 0],
                         yerr=[lower_dice_bounds[exp_num, counter, ::plot_skip],
                               upper_dice_bounds[exp_num, counter, ::plot_skip]],
                         label=f'GM: {directory[0]}',
                         c=seeded_color[exp_num], fmt='o')
        axes[0].set_title(f'GM volumes sub {sub_ID}', fontsize=17)  # Turned off sub
        axes[0].legend(loc='upper left', fontsize=14)
        axes[0].grid(which='minor')
        axes[0].set_ylabel('Volume deviation from reference', fontsize=17)
        axes[0].set_xlabel('Inversion time (ms)', fontsize=17)
        axes[0].axvspan(600, 1200, color='cyan', alpha=0.25)
        axes[0].set_xlim([TI_min, TI_max])
        axes[0].tick_params(labelsize=14)

        axes[1].set_title(f'GM volumes sub {sub_ID} (GT)', fontsize=17)  # Turned off sub
        axes[1].legend(loc='upper left', fontsize=14)
        axes[1].grid(which='minor')
        axes[1].set_ylabel('Volume deviation from GT reference', fontsize=17)
        axes[1].set_xlabel('Inversion time (ms)', fontsize=17)
        axes[1].axvspan(600, 1200, color='cyan', alpha=0.25)
        axes[1].set_xlim([TI_min, TI_max])
        axes[1].tick_params(labelsize=14)

        axes[2].set_title(f'GM volumes Dice {sub_ID}', fontsize=17)  # Turned off sub
        axes[2].legend(loc='upper left', fontsize=14)
        axes[2].grid(which='minor')
        axes[2].set_ylabel('Dice score', fontsize=17)
        axes[2].set_xlabel('Inversion time (ms)', fontsize=17)
        axes[2].axvspan(600, 1200, color='cyan', alpha=0.25)
        axes[2].set_xlim([TI_min, TI_max])
        axes[2].tick_params(labelsize=14)
    # manager = plt.get_current_fig_# manager()
    # manager.resize(*# manager.window.maxsize())
    fig.savefig(os.path.join(f'/nfs/home/pedro/PhysicsPyTorch/Analyses/Volumes/{combined_folder}', f'Volume_consistency_sub_{sub_ID}.png'),
                figsize=(32.0, 20.0), dpi=100)

# Non OOD
ood_standardised_tiss_volumes = np.zeros_like(all_tot_tiss_vols)
gt_ood_standardised_tiss_volumes = np.zeros_like(all_tot_tiss_vols)

train_TI_min = 600.
train_TI_max = 1200.
ood_min = np.where(TIs == train_TI_min)[0][0]
ood_max = np.where(TIs == train_TI_max)[0][0] + 1
for counter, sub_ID in enumerate(dirs):
    # Calculate ranges for GM
    plt.interactive(False)
    plot_gt_seg_file = os.path.join(prob_gt_seg_dir, f'Dil_fin_merged_sub_{sub_ID}.nii.gz')
    gt_GM = calc_prob_tiss_vols(plot_gt_seg_file)
    gt_vols[counter] = gt_GM
    fig, axes = plt.subplots(1, num_tissues+2, sharex='all', figsize=(18.0, 10.0))
    np.random.seed(44)
    seeded_color = np.random.rand(len(inference_directories.items()), 3)
    # seeded_color[1, :] = [1, 0, 0]
    # seeded_color[2, :] = [0, 1, 0]
    axes[0].axvline(x=(TI_max + TI_min) / 2)
    axes[1].axvline(x=(TI_max + TI_min) / 2)
    for exp_num, directory in enumerate(inference_directories.items()):
        # Identify qualitative array
        ood_standardised_tiss_volumes[exp_num,
                                  counter,
                                  ood_min:ood_max,
                                  0] = standardise_volumes(all_tot_tiss_vols[exp_num,
                                                                             counter,
                                                                             ood_min:ood_max,
                                                                             0])
        gt_ood_standardised_tiss_volumes[exp_num,
                                     counter,
                                     ood_min:ood_max,
                                     0] = standardise_gt_volumes(all_tot_tiss_vols[exp_num,
                                                                                   counter,
                                                                                   ood_min:ood_max,
                                                                                   0], gt_GM)
        # tr = matplotlib.transforms.offset_copy(axes[0].transData, fig=fig, x=1.5 * (exp_num+0.1), y=0,
        #                                        units='points')
        # axes[0].scatter(TIs[ood_min:ood_max],
        #                       # Decide whether to standardise with respect to Mid point or wrt PGS segmentation
        #                       ood_standardised_tiss_volumes[exp_num, counter, ood_min:ood_max, 0],
        #                       label=f'GM: {directory[0]}',
        #                       c=seeded_color[exp_num])
        # axes[1].scatter(TIs[ood_min:ood_max],
        #                       # Decide whether to standardise with respect to Mid point or wrt PGS segmentation
        #                       gt_ood_standardised_tiss_volumes[exp_num, counter, ood_min:ood_max, 0],
        #                       label=f'GM: {directory[0]}',
        #                       c=seeded_color[exp_num])
        # axes[2].scatter(TIs[ood_min:ood_max],
        #                       # Decide whether to standardise with respect to Mid point or wrt PGS segmentation
        #                       all_tot_tiss_dice[exp_num, counter, ood_min:ood_max, 0],
        #                       label=f'GM: {directory[0]}',
        #                       c=seeded_color[exp_num])
        # Errorbar plots
        axes[0].errorbar(TIs[ood_min:ood_max],
                         # Decide whether to standardise with respect to Mid point or wrt PGS segmentation
                         ood_standardised_tiss_volumes[exp_num, counter, ood_min:ood_max, 0],
                         yerr=[lower_bounds[exp_num, counter, ood_min:ood_max],
                               upper_bounds[exp_num, counter, ood_min:ood_max]],
                         label=f'GM: {directory[0]}',
                         c=seeded_color[exp_num], fmt='o')
        axes[1].errorbar(TIs[ood_min:ood_max],
                         # Decide whether to standardise with respect to Mid point or wrt PGS segmentation
                         gt_ood_standardised_tiss_volumes[exp_num, counter, ood_min:ood_max, 0],
                         yerr=[lower_bounds[exp_num, counter, ood_min:ood_max],
                               upper_bounds[exp_num, counter, ood_min:ood_max]],
                         label=f'GM: {directory[0]}',
                         c=seeded_color[exp_num], fmt='o')
        axes[2].errorbar(TIs[ood_min:ood_max],
                         # Decide whether to standardise with respect to Mid point or wrt PGS segmentation
                         all_tot_tiss_dice[exp_num, counter, ood_min:ood_max, 0],
                         yerr=[lower_dice_bounds[exp_num, counter, ood_min:ood_max],
                               upper_dice_bounds[exp_num, counter, ood_min:ood_max]],
                         label=f'GM: {directory[0]}',
                         c=seeded_color[exp_num], fmt='o')
        axes[0].set_title(f'GM volumes sub {sub_ID}', fontsize=17)  # Turned off sub
        axes[0].legend(loc='upper left', fontsize=14)
        axes[0].grid(which='minor')
        axes[0].set_ylabel('Volume deviation from reference', fontsize=17)
        axes[0].set_xlabel('Inversion time (ms)', fontsize=17)
        # axes[0].axvspan(600, 1200, color='cyan', alpha=0.25)
        axes[0].set_xlim([600, 1200])
        axes[0].tick_params(labelsize=14)

        axes[1].set_title(f'GM volumes sub {sub_ID} (GT)', fontsize=17)  # Turned off sub
        axes[1].legend(loc='upper left', fontsize=14)
        axes[1].grid(which='minor')
        axes[1].set_ylabel('Volume deviation from GT reference', fontsize=17)
        axes[1].set_xlabel('Inversion time (ms)', fontsize=17)
        # axes[1].axvspan(600, 1200, color='cyan', alpha=0.25)
        axes[1].set_xlim([600, 1200])
        axes[1].tick_params(labelsize=14)

        axes[2].set_title(f'GM volumes Dice {sub_ID}', fontsize=17)  # Turned off sub
        axes[2].legend(loc='upper left', fontsize=14)
        axes[2].grid(which='minor')
        axes[2].set_ylabel('Dice score', fontsize=17)
        axes[2].set_xlabel('Inversion time (ms)', fontsize=17)
        # axes[2].axvspan(600, 1200, color='cyan', alpha=0.25)
        axes[2].set_xlim([600, 1200])
        axes[2].tick_params(labelsize=14)
    # manager = plt.get_current_fig_# manager()
    # manager.resize(*# manager.window.maxsize())
    fig.savefig(os.path.join(f'/nfs/home/pedro/PhysicsPyTorch/Analyses/Volumes/{combined_folder}', f'Volume_consistency_sub_{sub_ID}_non_ood.png'),
                figsize=(32.0, 20.0), dpi=100)

# Averaged plot
# Calculate ranges for GM
fig, axes = plt.subplots(1, num_tissues+2, sharex='all', figsize=(18.0, 10.0))
mean_GT = np.mean(gt_vols)
seeded_color = ['r', 'g', 'b', 'cyan', 'black', 'purple']
for exp_num, directory in enumerate(inference_directories.items()):
    axes[0].axvline(x=(TI_max + TI_min) / 2)
    axes[1].axvline(x=(TI_max + TI_min) / 2)
    axes[0].scatter(TIs[::plot_skip],
                    # Decide whether to standardise with respect to Mid point or wrt PGS segmentation
                    np.mean(np.abs(standardised_tiss_volumes[exp_num, :, ::plot_skip, 0]), axis=0),
                    label=f'GM: {directory[0]}',
                    c=seeded_color[exp_num])
    # print(f'Shape is {np.abs(gt_standardised_tiss_volumes[exp_num, :, ::plot_skip, 0]).shape}')
    axes[1].scatter(TIs[::plot_skip],
                    # Decide whether to standardise with respect to Mid point or wrt PGS segmentation
                    np.mean(np.abs(gt_standardised_tiss_volumes[exp_num, :, ::plot_skip, 0]), axis=0),
                    label=f'GM: {directory[0]}',
                    c=seeded_color[exp_num])
    axes[2].scatter(TIs[::plot_skip],
                    # Decide whether to standardise with respect to Mid point or wrt PGS segmentation
                    np.mean(np.abs(all_tot_tiss_dice[exp_num, :, ::plot_skip, 0]), axis=0),
                    label=f'GM: {directory[0]}',
                    c=seeded_color[exp_num])
    axes[0].set_title(f'GM volumes sub ALL', fontsize=17)  # Turned off sub
    axes[0].legend(loc='upper left', fontsize=14)
    axes[0].grid(which='minor')
    axes[0].set_ylabel('Volume deviation from reference', fontsize=17)
    axes[0].set_xlabel('Inversion time (ms)', fontsize=17)
    axes[0].axvspan(600, 1200, color='cyan', alpha=0.25)
    axes[0].set_xlim([TI_min, TI_max])
    axes[0].tick_params(labelsize=14)

    axes[1].set_title(f'GM volumes sub ALL (GT)', fontsize=17)  # Turned off sub
    axes[1].legend(loc='upper left', fontsize=14)
    axes[1].grid(which='minor')
    axes[1].set_ylabel('Volume deviation from GT reference', fontsize=17)
    axes[1].set_xlabel('Inversion time (ms)', fontsize=17)
    axes[1].axvspan(600, 1200, color='cyan', alpha=0.25)
    axes[1].set_xlim([TI_min, TI_max])
    axes[1].tick_params(labelsize=14)

    axes[2].set_title(f'GM volumes Dice ALL', fontsize=17)  # Turned off sub
    axes[2].legend(loc='upper left', fontsize=14)
    axes[2].grid(which='minor')
    axes[2].set_ylabel('Dice score', fontsize=17)
    axes[2].set_xlabel('Inversion time (ms)', fontsize=17)
    axes[2].axvspan(600, 1200, color='cyan', alpha=0.25)
    axes[2].set_xlim([TI_min, TI_max])
    axes[2].tick_params(labelsize=14)

# manager = plt.get_current_fig_# manager()
# manager.resize(*# manager.window.maxsize())
fig.savefig(os.path.join(f'/nfs/home/pedro/PhysicsPyTorch/Analyses/Volumes/{combined_folder}', 'Volume_consistency_average.png'),
            figsize=(32.0, 20.0), dpi=100)# Calculate ranges for GM

# Non ood
fig, axes = plt.subplots(1, num_tissues+2, sharex='all', figsize=(18.0, 10.0))
mean_GT = np.mean(gt_vols)
seeded_color = ['r', 'g', 'b', 'cyan', 'black', 'purple']
for exp_num, directory in enumerate(inference_directories.items()):
    axes[0].axvline(x=(TI_max + TI_min) / 2)
    axes[1].axvline(x=(TI_max + TI_min) / 2)
    axes[0].scatter(TIs[ood_min:ood_max],
                    # Decide whether to standardise with respect to Mid point or wrt PGS segmentation
                    np.mean(np.abs(ood_standardised_tiss_volumes[exp_num, :, ood_min:ood_max, 0]), axis=0),
                    label=f'GM: {directory[0]}',
                    c=seeded_color[exp_num])
    # print(f'Shape is {np.abs(gt_ood_standardised_tiss_volumes[exp_num, :, ood_min:ood_max, 0]).shape}')
    axes[1].scatter(TIs[ood_min:ood_max],
                    # Decide whether to standardise with respect to Mid point or wrt PGS segmentation
                    np.mean(np.abs(gt_ood_standardised_tiss_volumes[exp_num, :, ood_min:ood_max, 0]), axis=0),
                    label=f'GM: {directory[0]}',
                    c=seeded_color[exp_num])
    axes[2].scatter(TIs[ood_min:ood_max],
                    # Decide whether to standardise with respect to Mid point or wrt PGS segmentation
                    np.mean(np.abs(all_tot_tiss_dice[exp_num, :, ood_min:ood_max, 0]), axis=0),
                    label=f'GM: {directory[0]}',
                    c=seeded_color[exp_num])
    axes[0].set_title(f'GM volumes sub ALL', fontsize=17)  # Turned off sub
    axes[0].legend(loc='upper left', fontsize=14)
    axes[0].grid(which='minor')
    axes[0].set_ylabel('Volume deviation from reference', fontsize=17)
    axes[0].set_xlabel('Inversion time (ms)', fontsize=17)
    # axes[0].axvspan(600, 1200, color='cyan', alpha=0.25)
    axes[0].set_xlim([TI_min, TI_max])
    axes[0].tick_params(labelsize=14)

    axes[1].set_title(f'GM volumes sub ALL (GT)', fontsize=17)  # Turned off sub
    axes[1].legend(loc='upper left', fontsize=14)
    axes[1].grid(which='minor')
    axes[1].set_ylabel('Volume deviation from GT reference', fontsize=17)
    axes[1].set_xlabel('Inversion time (ms)', fontsize=17)
    # axes[1].axvspan(600, 1200, color='cyan', alpha=0.25)
    axes[1].set_xlim([TI_min, TI_max])
    axes[1].tick_params(labelsize=14)

    axes[2].set_title(f'GM volumes Dice ALL', fontsize=17)  # Turned off sub
    axes[2].legend(loc='upper left', fontsize=14)
    axes[2].grid(which='minor')
    axes[2].set_ylabel('Dice score', fontsize=17)
    axes[2].set_xlabel('Inversion time (ms)', fontsize=17)
    # axes[2].axvspan(600, 1200, color='cyan', alpha=0.25)
    axes[2].set_xlim([600, 1200])
    axes[2].tick_params(labelsize=14)

# manager = plt.get_current_fig_# manager()
# manager.resize(*# manager.window.maxsize())
fig.savefig(os.path.join(f'/nfs/home/pedro/PhysicsPyTorch/Analyses/Volumes/{combined_folder}', 'Volume_consistency_average_non_ood.png'),
            figsize=(32.0, 20.0), dpi=100)


# Joint histograms plotting
# Get rid of zeros for log plotting
minval = np.min(all_tot_histograms[np.nonzero(all_tot_histograms)])
all_tot_histograms[all_tot_histograms == 0.0] = minval
argmax_minval = np.min(argmax_all_tot_histograms[np.nonzero(argmax_all_tot_histograms)])
argmax_all_tot_histograms[argmax_all_tot_histograms == 0.0] = argmax_minval
partial_argmax_minval = np.min(partial_argmax_all_tot_histograms[np.nonzero(partial_argmax_all_tot_histograms)])
partial_argmax_all_tot_histograms[partial_argmax_all_tot_histograms == 0.0] = partial_argmax_minval

num_subplots = 1
ylabels = 'Dropout uncertainty'
subject_skip = 1
print(f'The hist counter is {hist_counter}')
for i, directory in enumerate(inference_directories.items()):  # range(num_reals // subject_skip):
    for TI_sample in range(num_reals):
        fig, axes = plt.subplots(1, num_subplots, figsize=(18.0, 10.0))  # Baseline
        for axis_ID in range(num_subplots):
            # Averaging over subjects
            # Experiment, subjects, TIs, num_hist_bins, num_hist_bins, 3
            current_hist = np.mean(all_tot_histograms[i, :, TI_sample, :, :, axis_ID], axis=0)
            print(f'The mean of the current hist is {np.mean(current_hist)}, {TI_sample}')
            x_edges = np.mean(x_bin_edges[i, :, TI_sample, :, axis_ID], axis=0)
            y_edges = np.mean(y_bin_edges[i, :, TI_sample, :, axis_ID], axis=0)
            try:
                axis_aspect = np.max(x_edges)/np.max(y_edges)
            except:
                axis_aspect = None
            im = axes.imshow(np.log10(current_hist),
                             cmap='viridis', origin='lower', vmin=-1, vmax=7,
                                  extent=[x_edges[0], x_edges[-1],
                                          y_edges[0], y_edges[-1]], aspect=axis_aspect)
            axes.plot(x_edges[:-1],
                      np.sum(current_hist * y_edges[:-1, np.newaxis],
                             axis=0) / np.sum(current_hist, axis=0), color='red')
            # axes.set_xlim([x_edges[0], x_edges[-1]])
            axes.set_ylabel('Error rate')
            axes.set_xlabel(ylabels)
            # axes.set_aspect('equal', adjustable='box')
            # axes.axis('equal')
            c = plt.colorbar(im, cax=fig.add_axes([0.915, 0.282, 0.02, 0.425]))
        np.save(os.path.join(f'/nfs/home/pedro/PhysicsPyTorch/Analyses/BlandAltman/{combined_folder}',
                             '_'.join(['hist', histogram_directories[directory[0]], f'TI_{TIs[TI_sample]}_BA.npy'])),
                current_hist)
        np.save(os.path.join(f'/nfs/home/pedro/PhysicsPyTorch/Analyses/BlandAltman/{combined_folder}',
                             '_'.join(['x_edges', histogram_directories[directory[0]], f'TI_{TIs[TI_sample]}_BA.npy'])),
                x_edges)
        np.save(os.path.join(f'/nfs/home/pedro/PhysicsPyTorch/Analyses/BlandAltman/{combined_folder}',
                             '_'.join(['y_edges', histogram_directories[directory[0]], f'TI_{TIs[TI_sample]}_BA.npy'])),
                y_edges)
        # manager = plt.get_current_fig_# manager()
        # manager.resize(*# manager.window.maxsize())
        fig.savefig(os.path.join(f'/nfs/home/pedro/PhysicsPyTorch/Analyses/BlandAltman/{combined_folder}',
                                 histogram_directories[directory[0]] + f'_TI_{TIs[TI_sample]}_BA.png'),
                    figsize=(32.0, 32.0), dpi=100)
        # fig, axes = plt.subplots(1, num_subplots)  # Physics
        # for axis_ID in range(num_subplots):
        #     # Averaging over subjects
        #     current_hist = np.mean(all_tot_histograms[1, :, i * 3, :, :, axis_ID], axis=0)
        #     x_edges = np.mean(x_bin_edges[1, :, i*subject_skip, :, axis_ID], axis=0)
        #     y_edges = np.mean(x_bin_edges[1, :, i*subject_skip, :, axis_ID], axis=0)
        #     im = axes.imshow(np.log10(current_hist),  # Transpose is on!
        #                      cmap='viridis', origin='lower', vmin=-1, vmax=7,
        #                           extent=[x_edges[0], x_edges[-1],
        #                                   y_edges[0], y_edges[-1]])
        #     axes.plot(x_edges[:-1],
        #               np.sum(current_hist * y_edges[:-1, np.newaxis],
        #                      axis=0) / np.sum(current_hist, axis=0), color='red')
        #     print(current_hist)
        #     print(y_edges[:-1])
        #     print(np.sum(current_hist * y_edges[None, :-1], axis=0))
        #     # axes.set_xlim([x_edges[0], x_edges[-1]])
        #     axes.set_ylabel('Error rate')
        #     axes.set_xlabel(ylabels[0])
        # c = plt.colorbar(im, cax=fig.add_axes([0.915, 0.278, 0.02, 0.425]))
for i, directory in enumerate(inference_directories.items()):  # range(num_reals // subject_skip):
    for TI_sample in range(num_reals):
        fig, axes = plt.subplots(1, num_subplots, figsize=(18.0, 10.0))  # Baseline
        for axis_ID in range(num_subplots):
            # Averaging over subjects
            # Experiment, subjects, TIs, num_hist_bins, num_hist_bins, 3
            argmax_current_hist = np.mean(argmax_all_tot_histograms[i, :, TI_sample, :, :, axis_ID], axis=0)
            print(f'The mean of the current hist is {np.mean(argmax_current_hist)}, {TI_sample}')
            argmax_x_edges = np.mean(argmax_x_bin_edges[i, :, TI_sample, :, axis_ID], axis=0)
            argmax_y_edges = np.mean(argmax_y_bin_edges[i, :, TI_sample, :, axis_ID], axis=0)
            try:
                axis_aspect = np.max(argmax_x_edges)/np.max(argmax_y_edges)
            except:
                axis_aspect = None
            im = axes.imshow(np.log10(argmax_current_hist),
                             cmap='viridis', origin='lower', vmin=-1, vmax=7,
                                  extent=[argmax_x_edges[0], argmax_x_edges[-1],
                                          argmax_y_edges[0], argmax_y_edges[-1]], aspect=axis_aspect)
            axes.plot(argmax_x_edges[:-1],
                      np.sum(argmax_current_hist * argmax_y_edges[:-1, np.newaxis],
                             axis=0) / np.sum(argmax_current_hist, axis=0), color='red')
            # axes.set_xlim([x_edges[0], x_edges[-1]])
            axes.set_ylabel('Error rate')
            axes.set_xlabel(ylabels)
            # axes.set_aspect('equal', adjustable='box')
            # axes.axis('equal')
            c = plt.colorbar(im, cax=fig.add_axes([0.915, 0.282, 0.02, 0.425]))
        np.save(os.path.join(f'/nfs/home/pedro/PhysicsPyTorch/Analyses/BlandAltman/{combined_folder}',
                             '_'.join(['Argmax_hist', histogram_directories[directory[0]], f'TI_{TIs[TI_sample]}_BA.npy'])),
                argmax_current_hist)
        np.save(os.path.join(f'/nfs/home/pedro/PhysicsPyTorch/Analyses/BlandAltman/{combined_folder}',
                             '_'.join(['Argmax_x_edges', histogram_directories[directory[0]], f'TI_{TIs[TI_sample]}_BA.npy'])),
                argmax_x_edges)
        np.save(os.path.join(f'/nfs/home/pedro/PhysicsPyTorch/Analyses/BlandAltman/{combined_folder}',
                             '_'.join(['Argmax_y_edges', histogram_directories[directory[0]], f'TI_{TIs[TI_sample]}_BA.npy'])),
                argmax_y_edges)
        # manager = plt.get_current_fig_# manager()
        # manager.resize(*# manager.window.maxsize())
        fig.savefig(os.path.join(f'/nfs/home/pedro/PhysicsPyTorch/Analyses/BlandAltman/{combined_folder}',
                                 '_'.join(['Argmax', histogram_directories[directory[0]], f'TI_{TIs[TI_sample]}_BA.png'])),
                    figsize=(32.0, 32.0), dpi=100)
        # fig, axes = plt.subplots(1, num_subplots)  # Physics
        # for axis_ID in range(num_subplots):
        #     # Averaging over subjects
        #     current_hist = np.mean(all_tot_histograms[1, :, i * 3, :, :, axis_ID], axis=0)
        #     x_edges = np.mean(x_bin_edges[1, :, i*subject_skip, :, axis_ID], axis=0)
        #     y_edges = np.mean(x_bin_edges[1, :, i*subject_skip, :, axis_ID], axis=0)
        #     im = axes.imshow(np.log10(current_hist),  # Transpose is on!
        #                      cmap='viridis', origin='lower', vmin=-1, vmax=7,
        #                           extent=[x_edges[0], x_edges[-1],
        #                                   y_edges[0], y_edges[-1]])
        #     axes.plot(x_edges[:-1],
        #               np.sum(current_hist * y_edges[:-1, np.newaxis],
        #                      axis=0) / np.sum(current_hist, axis=0), color='red')
        #     print(current_hist)
        #     print(y_edges[:-1])
        #     print(np.sum(current_hist * y_edges[None, :-1], axis=0))
        #     # axes.set_xlim([x_edges[0], x_edges[-1]])
        #     axes.set_ylabel('Error rate')
        #     axes.set_xlabel(ylabels[0])
        # c = plt.colorbar(im, cax=fig.add_axes([0.915, 0.278, 0.02, 0.425]))

for i, directory in enumerate(inference_directories.items()):  # range(num_reals // subject_skip):
    for TI_sample in range(num_reals):
        fig, axes = plt.subplots(1, num_subplots, figsize=(18.0, 10.0))  # Baseline
        for axis_ID in range(num_subplots):
            # Averaging over subjects
            # Experiment, subjects, TIs, num_hist_bins, num_hist_bins, 3
            partial_argmax_current_hist = np.mean(partial_argmax_all_tot_histograms[i, :, TI_sample, :, :, axis_ID], axis=0)
            print(f'The mean of the current hist is {np.mean(partial_argmax_current_hist)}, {TI_sample}')
            partial_argmax_x_edges = np.mean(partial_argmax_x_bin_edges[i, :, TI_sample, :, axis_ID], axis=0)
            partial_argmax_y_edges = np.mean(partial_argmax_y_bin_edges[i, :, TI_sample, :, axis_ID], axis=0)
            try:
                axis_aspect = np.max(partial_argmax_x_edges)/np.max(partial_argmax_y_edges)
            except:
                axis_aspect = None
            im = axes.imshow(np.log10(partial_argmax_current_hist),
                             cmap='viridis', origin='lower', vmin=-1, vmax=7,
                                  extent=[partial_argmax_x_edges[0], partial_argmax_x_edges[-1],
                                          partial_argmax_y_edges[0], partial_argmax_y_edges[-1]], aspect=axis_aspect)
            axes.plot(partial_argmax_x_edges[:-1],
                      np.sum(partial_argmax_current_hist * partial_argmax_y_edges[:-1, np.newaxis],
                             axis=0) / np.sum(partial_argmax_current_hist, axis=0), color='red')
            # axes.set_xlim([x_edges[0], x_edges[-1]])
            axes.set_ylabel('Error rate')
            axes.set_xlabel(ylabels)
            # axes.set_aspect('equal', adjustable='box')
            # axes.axis('equal')
            c = plt.colorbar(im, cax=fig.add_axes([0.915, 0.282, 0.02, 0.425]))
        np.save(os.path.join(f'/nfs/home/pedro/PhysicsPyTorch/Analyses/BlandAltman/{combined_folder}',
                             '_'.join(['partial_argmax_hist', histogram_directories[directory[0]], f'TI_{TIs[TI_sample]}_BA.npy'])),
                partial_argmax_current_hist)
        np.save(os.path.join(f'/nfs/home/pedro/PhysicsPyTorch/Analyses/BlandAltman/{combined_folder}',
                             '_'.join(['partial_argmax_x_edges', histogram_directories[directory[0]], f'TI_{TIs[TI_sample]}_BA.npy'])),
                partial_argmax_x_edges)
        np.save(os.path.join(f'/nfs/home/pedro/PhysicsPyTorch/Analyses/BlandAltman/{combined_folder}',
                             '_'.join(['partial_argmax_y_edges', histogram_directories[directory[0]], f'TI_{TIs[TI_sample]}_BA.npy'])),
                partial_argmax_y_edges)
        # manager = plt.get_current_fig_# manager()
        # manager.resize(*# manager.window.maxsize())
        fig.savefig(os.path.join(f'/nfs/home/pedro/PhysicsPyTorch/Analyses/BlandAltman/{combined_folder}',
                                 '_'.join(['Partial_Argmax', histogram_directories[directory[0]], f'TI_{TIs[TI_sample]}_BA.png'])),
                    figsize=(32.0, 32.0), dpi=100)
        # fig, axes = plt.subplots(1, num_subplots)  # Physics
        # for axis_ID in range(num_subplots):
        #     # Averaging over subjects
        #     current_hist = np.mean(all_tot_histograms[1, :, i * 3, :, :, axis_ID], axis=0)
        #     x_edges = np.mean(x_bin_edges[1, :, i*subject_skip, :, axis_ID], axis=0)
        #     y_edges = np.mean(x_bin_edges[1, :, i*subject_skip, :, axis_ID], axis=0)
        #     im = axes.imshow(np.log10(current_hist),  # Transpose is on!
        #                      cmap='viridis', origin='lower', vmin=-1, vmax=7,
        #                           extent=[x_edges[0], x_edges[-1],
        #                                   y_edges[0], y_edges[-1]])
        #     axes.plot(x_edges[:-1],
        #               np.sum(current_hist * y_edges[:-1, np.newaxis],
        #                      axis=0) / np.sum(current_hist, axis=0), color='red')
        #     print(current_hist)
        #     print(y_edges[:-1])
        #     print(np.sum(current_hist * y_edges[None, :-1], axis=0))
        #     # axes.set_xlim([x_edges[0], x_edges[-1]])
        #     axes.set_ylabel('Error rate')
        #     axes.set_xlabel(ylabels[0])
        # c = plt.colorbar(im, cax=fig.add_axes([0.915, 0.278, 0.02, 0.425]))
