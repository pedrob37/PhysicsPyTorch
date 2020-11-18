import numpy as np
import os
import nibabel as nib
import glob
import re
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('TKAgg')  # Use this for console plotting
import argparse


def read_file(filename):
    img = nib.load(filename)
    data = img.get_fdata()
    aff = img.affine
    return data, aff


def save_img(image, affine, filename):
    nifti_img = nib.Nifti1Image(image, affine)
    nib.save(nifti_img, filename)


def sort_dir(dir_name):
    """
    :param dir_name: Full directory name to be sorted
    :return: Sorted directory according to first (subID) and third value (TI) following underscores
    """
    print(dir_name)
    listed_niis = [os.path.basename(x) for x in glob.glob(os.path.join(dir_name, '*nii.gz'))]
    return sorted(listed_niis, key=lambda xx: float(xx.rsplit('.nii.gz')[0].split('_')[11]))


def sort_dir_plurality(dir_name):
    """
    :param dir_name: Full directory name to be sorted
    :return: Sorted directory according to first (subID) and third value (TI) following underscores
    """
    print(dir_name)
    listed_niis = [os.path.basename(x) for x in glob.glob(os.path.join(dir_name, '*nii.gz'))]
    return sorted(listed_niis, key=lambda xx: [float(xx.rsplit('.nii.gz')[0].split('_')[9]),
                                               float(xx.rsplit('.nii.gz')[0].split('_')[11])])


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


def entropy_to_sigma(entropy, distribution):
    if distribution == 'logistic':
        return (np.pi/np.sqrt(3))*np.exp(entropy-2)
    elif distribution == 'gibbs':
        gamma = np.euler_gamma
        return np.sqrt(3/2-4/np.pi)*np.exp(entropy - gamma - 0.5)/np.sqrt(np.pi)


def calc_bernoulli_std(prob):
    return np.sqrt(prob*(1-prob))


def compute_softmax(input):
    exponent = np.exp(input)
    return exponent, np.sum(exponent), exponent / np.sum(exponent)


parser = argparse.ArgumentParser(description="Uncertainty processing")
parser.add_argument("--input_directory", type=str,
                    help="Input directory containing organised MC inferences")
parser.add_argument("--num_dropout_samples", type=int,
                    help="Number of dropout samples")
parser.add_argument("--processing_flag", type=str2bool, nargs='?', const=True, default=True,
                    help="Whether to process all images again or not")
arguments = parser.parse_args()
input_directory = arguments.input_directory
directory_name = os.path.basename(input_directory)

# Calculate variance of dropout samples + combine with heteroscedastic uncertainty
inference_directories = {directory_name: input_directory}

example_inference_directory = os.path.join(input_directory, 'Inference0')  #  '/storage/DGX2_backups/DGX2_experiments/Uncertainty/60k_OOD_Fuller/Physics/No_FCL_Dropout/60k_Inference2'
gt_seg_dir = '/nfs/project/pborges/Resampled_Data/MPM_GM_Labels/'

# Regex
regex = re.compile(r'\d+')

dropout_samples = arguments.num_dropout_samples
processing_flag = arguments.processing_flag
argmax_flag = False
image_names = sort_dir(example_inference_directory)
image_names = [x for x in image_names if "3.0000" not in x]
for exc in range(1, 10):
    image_names = [x for x in image_names if f"2.{exc}" not in x]

# Don't want to necessarily process data every time if directories are the same
if processing_flag:
    for directory in inference_directories.items():
        os.chdir(directory[1])
        output_quantitative_directory = os.path.join(directory[1], 'Combined_seg_uncertainty')
        print(f'The output quantitative directory is {output_quantitative_directory}')
        if argmax_flag:
            output_qualitative_directory = os.path.join(directory[1], 'Argmax_qualitative_uncertainty')
        else:
            output_qualitative_directory = os.path.join(directory[1], 'Qualitative_uncertainty')
        print(output_quantitative_directory)
        print(output_qualitative_directory)
        if not os.path.exists(output_quantitative_directory):
            os.makedirs(output_quantitative_directory)
        if not os.path.exists(output_qualitative_directory):
            os.makedirs(output_qualitative_directory)
        # Sanity check
        # print(image_names[::10])
        for image in image_names:
            # Isolate relevant parts of image string: Don't want to include the DO identifier!
            image_globber = f'*{image.split("MPRAGE")[1]}'
            # Isolate subject ID in order to determine GT segmentation
            sub_ID = np.int(regex.findall(image)[2])
            gt_seg, gt_aff = read_file(glob.glob(os.path.join(gt_seg_dir, f'*sub_{sub_ID}.*'))[0])
            print(f'The image name is: {image}\n The gt seg is: {glob.glob(os.path.join(gt_seg_dir, f"*sub_{sub_ID}.*"))[0]}')
            # print(glob.glob(os.path.join(gt_seg_dir, f'*sub_{sub_ID}*')))
            # Placeholder volumes for argmax uncertainties + final output volume
            dropout_volumes = np.zeros([181, 217, 181, dropout_samples])
            argmax_dropout_volumes = np.zeros([181, 217, 181, dropout_samples])
            # hetero_std_volumes = np.zeros([181, 217, 181, dropout_samples])
            argmax_residual_volume = np.zeros([181, 217, 181])
            partial_argmax_residual_volume = np.zeros([181, 217, 181])
            residual_volume = np.zeros([181, 217, 181])
            seg_uncertainty_volume = np.zeros([181, 217, 181, 3])
            # Two if no aleatoric uncertainty, four if argmaxed volumes are included
            qualitative_uncertainty_volume = np.zeros([181, 217, 181, 5])
            for MCID, mc_directory in enumerate(glob.glob('*Inference*')):
                # 4D file: Scaled segmentation (Prob), GM logits, BG logits, Uncertainty
                volume, aff = read_file(glob.glob(os.path.join(mc_directory, image_globber))[0])
                # print(f'Image globber is {image_globber}')
                # print(f'The MC sample is {glob.glob(os.path.join(mc_directory, image_globber))[0]}')
                seg_volume = np.nan_to_num(volume)
                dropout_volumes[..., MCID] = seg_volume
                argmax_dropout_volumes[..., MCID] = np.round(seg_volume)
                # hetero_std_volumes[..., MCID] = calc_bernoulli_std(np.nan_to_num(volume))
                # Argmax
                argmax_residual_volume += np.abs(np.round(gt_seg[..., 0]) - np.round(seg_volume))
                # Non-argmax
                residual_volume += np.abs(gt_seg[..., 0] - seg_volume)
                # Partial argmax
                partial_argmax_residual_volume += np.abs(np.round(gt_seg[..., 0]) - seg_volume)
            # MC Argmax Standard Deviation
            argmax_MC_std = np.std(argmax_dropout_volumes, axis=-1)
            # MC Standard Deviation
            MC_std = np.std(dropout_volumes, axis=-1)
            # Mean volume
            MC_mean = np.mean(dropout_volumes, axis=-1)
            # Mean hetero std
            # hetero_std_mean = np.mean(hetero_std_volumes, axis=-1)
            # Error rate
            error_rate = residual_volume / dropout_samples
            argmax_error_rate = argmax_residual_volume / dropout_samples
            partial_argmax_error_rate = partial_argmax_residual_volume / dropout_samples

            # Min and maximum volumes
            # Sum each 3D volume to obtain N numbers (N = dropout samples)
            MC_sum = np.sum(dropout_volumes, axis=tuple(range(dropout_volumes.ndim - 1)))
            min_volume_ID = np.argmin(MC_sum)
            max_volume_ID = np.argmax(MC_sum)
            min_volume = dropout_volumes[..., min_volume_ID]
            max_volume = dropout_volumes[..., max_volume_ID]

            # Quantitative: Concatenate MC mean, Max and min volumes
            seg_uncertainty_volume[..., 0] = MC_mean
            seg_uncertainty_volume[..., 1] = min_volume
            seg_uncertainty_volume[..., 2] = max_volume

            # Qualitative: Concatenate MC std (Epistemic), Uncertainty (Aleatoric)
            qualitative_uncertainty_volume[..., 0] = MC_std
            # qualitative_uncertainty_volume[..., 1] = hetero_std_mean
            qualitative_uncertainty_volume[..., 1] = error_rate
            qualitative_uncertainty_volume[..., 2] = argmax_MC_std
            qualitative_uncertainty_volume[..., 3] = argmax_error_rate
            qualitative_uncertainty_volume[..., 4] = partial_argmax_error_rate

            # Save volume
            mean_save_path = os.path.join(directory[1], f'inf_{sub_ID}')
            if not os.path.exists(mean_save_path):
                os.makedirs(mean_save_path)
            # Comprised of Mean, Min vol, Max vol
            save_img(seg_uncertainty_volume, aff, os.path.join(output_quantitative_directory, image))
            # Save a mean copy only as well to speed up analyses (No need to load larder nii): MC_mean here!
            save_img(MC_mean, aff, os.path.join(mean_save_path, image))
            if argmax_flag:
                save_img(qualitative_uncertainty_volume, aff,
                         os.path.join(output_qualitative_directory, 'argmax_' + image))
            else:
                save_img(qualitative_uncertainty_volume, aff, os.path.join(output_qualitative_directory, image))


# Uncertainty analysis
uncertainty_flag = True
if uncertainty_flag:
    for directory in inference_directories.items():
        output_quantitative_directory = os.path.join(directory[1], 'Combined_seg_uncertainty')
        physics_uncertainty_directory = output_quantitative_directory
        # baseline_uncertainty_directory = '/data/DGX2_experiments/Uncertainty/Baseline/Dropout/Combined_uncertainty'
        os.chdir(physics_uncertainty_directory)
        uncertainty_images = sort_dir_plurality(physics_uncertainty_directory)
        # NOT dropout samples, how many actual realisation are there per subject
        imgs_per_sub = 20
        num_subjects = 5

        epistemic_uncertainty_array = np.zeros((2, num_subjects, imgs_per_sub))
        aleatoric_uncertainty_array = np.zeros((2, num_subjects, imgs_per_sub))
        volume_consistency_array = np.zeros((2, num_subjects, imgs_per_sub))
        for uncID, uncertainty_image in enumerate(uncertainty_images):
            print(uncertainty_image)
            physics_uncertainty_volume, _ = read_file(os.path.join(physics_uncertainty_directory, uncertainty_image))
            # baseline_uncertainty_volume, _ = read_file(baseline_uncertainty_directory+'/'+uncertainty_image)

            # First dimension dictates Physics or Baseline
            # Second dimension dictates the subject you are on: Ever imgs_per_sub iterate: Therefore use //
            # Third dimension dictates individual image per subject up to imgs_per_sub: Therefore use %
            epistemic_uncertainty_array[0,
                                        uncID//imgs_per_sub,
                                        uncID % imgs_per_sub] = np.sum(physics_uncertainty_volume[..., 0])
            # aleatoric_uncertainty_array[0,
            #                             uncID//imgs_per_sub,
            #                             uncID % imgs_per_sub] = np.sum(physics_uncertainty_volume[..., 1])

            # epistemic_uncertainty_array[1,
            #                             uncID//imgs_per_sub,
            #                             uncID % imgs_per_sub] = np.sum(baseline_uncertainty_volume[..., 0])
            # aleatoric_uncertainty_array[1,
            #                             uncID//imgs_per_sub,
            #                             uncID % imgs_per_sub] = np.sum(baseline_uncertainty_volume[..., 1])

        save_on = True
        if save_on:
            np.save(os.path.join(output_quantitative_directory, 'epistemic.npy'), epistemic_uncertainty_array)
            # np.save(os.path.join(output_quantitative_directory, 'aleatoric.npy'), aleatoric_uncertainty_array)
        else:
            epistemic_uncertainty_array = np.load(os.path.join(output_quantitative_directory, 'epistemic.npy'))
            # aleatoric_uncertainty_array = np.load(os.path.join(output_quantitative_directory, 'aleatoric.npy'))

# Plotting
plot_names = {'Epistemic': epistemic_uncertainty_array}
              # 'Aleatoric': aleatoric_uncertainty_array}

# TI range
TI_min = 100  # 600
TI_max = 2000  # 3000  # 1200
TI_increment = 100
num_samples = len(list(range(TI_min, TI_max+TI_increment, TI_increment)))
TIs = np.linspace(TI_min, TI_max, num_samples)
num_subjects = 5

# Histograms
num_hist_bins = 20
all_tot_histograms = np.zeros((len(plot_names), num_subjects, num_samples,
                               num_hist_bins, num_hist_bins, len(plot_names)))  # 3 for: Dropout, Hetero, Total
x_bin_edges = np.zeros((len(plot_names), num_subjects, num_samples,
                        num_hist_bins + 1, 3))
y_bin_edges = np.zeros((len(plot_names), num_subjects, num_samples,
                        num_hist_bins + 1, 3))

# Epistemic uncertainty plotting?
fig, axes = plt.subplots(1, len(plot_names.items()), sharex='all')
for subject in range(num_subjects):
    for axis_ID, plot_name in enumerate(plot_names.items()):
        print(axis_ID)
        print(len(TIs), len(plot_name[1][0, subject, :]))
        axes.scatter(TIs, plot_name[1][0, subject, :], label=f'Physics: Subject {subject}')
        # axes.scatter(TIs, plot_name[1][1, subject, :], label='Baseline', c='r')
        axes.set_title('{} uncertainty for subject {}'.format(plot_name[0], subject), fontsize=17)
        axes.legend(loc='best', fontsize=12)
        axes.grid(which='minor')
        # axes[axis_ID].set_ylabel('Volume deviation from reference', fontsize=17)
        axes.set_xlabel('Inversion time (ms)', fontsize=17)
        axes.set_xlim([TI_min - 5, TI_max + 5])
        axes.tick_params(labelsize=14)
fig.savefig(os.path.join(input_directory, f'Physics_figure_all_subs.png'))

        # # Dropout histogram
        # all_tot_histograms[num, inf_ID, ID, :, :, uncertainty_dict['Dropout']], \
        # x_bin_edges[num, inf_ID, ID, :, uncertainty_dict['Dropout']], \
        # y_bin_edges[num, inf_ID, ID, :, uncertainty_dict['Dropout']] = \
        #     np.histogram2d(qualitative_volume[..., 2].flatten(),
        #                    qualitative_volume[..., 1].flatten(),
        #                    bins=20)
# plt.show()

