import numpy as np
import monai
import torchio
from torchio import Queue
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
from ignite.handlers import EarlyStopping
from model.metric import DiceLoss
import glob

import monai.visualize.img2tensorboard as img2tensorboard
import sys
sys.path.append('/home/pedro/over9000')
from over9000 import RangerLars


class BadDataset:
    def __init__(self, df, transform):
        self.df = df
        self.loader = torchio.ImagesDataset
        self.transform = transform
        self.sampler = torchio.data.UniformSampler(patch_size=80)

    def __getitem__(self, index):
        # These names are arbitrary
        MRI = 'mri'
        SEG = 'seg'
        PHYSICS = 'physics'

        subjects = []
        for (image_path, label_path, subject_physics) in zip(self.df.Filename, self.df.Label_Filename,
                                                             self.df.subject_physics):
            subject_dict = {
                MRI: torchio.ScalarImage(image_path),
                SEG: torchio.LabelMap(label_path),
                PHYSICS: subject_physics
            }
            subject = torchio.Subject(subject_dict)
            subjects.append(subject)
        this_dataset = self.loader(subjects, self.transform)

        patches_dataset = torchio.Queue(
            subjects_dataset=this_dataset,
            max_length=queue_length,
            samples_per_volume=samples_per_volume,
            sampler=torchio.sampler.UniformSampler(patch_size),
            shuffle_subjects=False,
            shuffle_patches=False,
        )

        return patches_dataset

    def __len__(self):
        return self.df.shape[0]


def BespokeDataset(df, transform, patch_size, batch_seed):
    loader = torchio.ImagesDataset
    sampler = torchio.data.UniformSampler(patch_size=patch_size, batch_seed=batch_seed)

    # These names are arbitrary
    MRI = 'mri'
    SEG = 'seg'
    PHYSICS = 'physics'

    subjects = []
    for (image_path, label_path, subject_physics) in zip(df.Filename, df.Label_Filename, df.subject_physics):
        subject_dict = {
            MRI: torchio.ScalarImage(image_path),
            SEG: torchio.LabelMap(label_path),
            PHYSICS: subject_physics
        }
        subject = torchio.Subject(subject_dict)
        subjects.append(subject)
    this_dataset = loader(subjects, transform)

    patches_dataset = torchio.Queue(
        subjects_dataset=this_dataset,
        max_length=queue_length,
        samples_per_volume=samples_per_volume,
        sampler=sampler,
        shuffle_subjects=False,
        shuffle_patches=False,
    )

    return patches_dataset


# Not enough to shuffle batches, shuffle WITHIN batches!
# Take original csv, shuffle between subjects!
def reshuffle_csv(og_csv, batch_size):
    # Calculate some necessary variables
    batch_reshuffle_csv = pd.DataFrame({})
    num_images = len(og_csv)
    print(num_images)
    batch_numbers = list(np.array(range(num_images // batch_size)) * batch_size)
    num_unique_subjects = og_csv.subject_id.nunique()
    unique_subject_ids = og_csv.subject_id.unique()
    print(num_unique_subjects)

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


def visualise_batch_patches(loader, bs, comparisons=2):
    print('Calculating tester...')
    assert comparisons <= batch_size
    next_data = next(iter(loader))
    batch_samples = random.sample(list(range(bs)), comparisons)
    import matplotlib.pyplot as plt
    for comparison in range(comparisons):
        example_batch_patch = np.squeeze(next_data['mri']['data'][batch_samples[comparison], :, :, 5])
        plt.figure(comparison)
        plt.imshow(example_batch_patch)
    plt.show()


# Stratification specific functions
def feature_loss_func(volume1, volume2):
    if type(volume2) == np.ndarray:
        return np.mean((volume1 - volume2) ** 2)
    else:
        return torch.mean((volume1 - volume2) ** 2).item()


def stratification_checker(input_volume):
    # Will only work for batch size 4 for now, but that comprises most experiments
    return int((input_volume[0, ...] + input_volume[3, ...] - input_volume[1, ...], input_volume[2, ...]).sum())


def calc_feature_loss(input_volume):
    feature_loss1 = feature_loss_func(
        volume1=input_volume[0, ...],
        volume2=input_volume[1, ...])
    feature_loss2 = feature_loss_func(
        volume1=input_volume[0, ...],
        volume2=input_volume[2, ...])
    feature_loss3 = feature_loss_func(
        volume1=input_volume[0, ...],
        volume2=input_volume[3, ...])
    feature_loss4 = feature_loss_func(
        volume1=input_volume[1, ...],
        volume2=input_volume[2, ...])
    feature_loss5 = feature_loss_func(
        volume1=input_volume[1, ...],
        volume2=input_volume[3, ...])
    feature_loss6 = feature_loss_func(
        volume1=input_volume[2, ...],
        volume2=input_volume[3, ...])

    total_feature_loss = np.mean([feature_loss1,
                                 feature_loss2,
                                 feature_loss3,
                                 feature_loss4,
                                 feature_loss5,
                                 feature_loss6])
    return total_feature_loss


os.environ['CUDA_VISIBLE_DEVICES'] = "0"
torch.cuda.empty_cache()

# Writer will output to ./runs/ directory by default
log_dir = f'/home/pedro/PhysicsPyTorch/logger/preliminary_tests'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
SAVE_PATH = os.path.join(f'/home/pedro/PhysicsPyTorch/logger/preliminary_tests/models')
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
SAVE = True
LOAD = True
patch_test = False

# Check if SAVE_PATH is empty
file_list = os.listdir(path=SAVE_PATH)
num_files = len(file_list)


# Hyper-parameter loading: General parameters so doesn't matter which model file is loaded exactly
if LOAD and num_files > 0:
    model_files = glob.glob(os.path.join(SAVE_PATH, '*.pth'))
    latest_model_file = max(model_files, key=os.path.getctime)
    checkpoint = torch.load(latest_model_file, map_location=torch.device('cuda:0'))
    print(f'Loading {latest_model_file}')
    loaded_epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    running_iter = checkpoint['running_iter']
    EPOCHS = 1000

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
    patch_size = 80
    batch_size = 4
    queue_length = batch_size
    samples_per_volume = 1

# Stratification
stratification_epsilon = 0.05

# Some necessary variables
dataset_csv = '/home/pedro/PhysicsPyTorch/local_physics_csv.csv'
# img_dir = '/data/MPRAGE_subjects_121T/Train_121T'  # '/nfs/home/pedro/COVID/Data/KCH_CXR_JPG'
# label_dir = '/data/Segmentation_MPRAGE_121T/All_labels'  # '/nfs/home/pedro/COVID/Labels/KCH_CXR_JPG.csv'
img_dir = '/data/Resampled_Data/Images/SS_GM_Images'  # '/nfs/home/pedro/COVID/Data/KCH_CXR_JPG'
label_dir = '/data/Resampled_Data/Labels/GM_Labels'  # '/nfs/home/pedro/COVID/Labels/KCH_CXR_JPG.csv'
print(img_dir)
print(label_dir)


# Read csv + add directory to filenames
df = pd.read_csv(dataset_csv)
df['Label_Filename'] = df['Filename']
df['Filename'] = img_dir + '/' + df['Filename'].astype(str)
df['Label_Filename'] = label_dir + '/' + 'Label_' + df['Label_Filename'].astype(str)
num_folds = df.fold.nunique()

# Transforms
training_transform = torchio.Compose([
    # torchio.RescaleIntensity((0, 1)),  # so that there are no negative values for RandomMotion
    # torchio.RandomMotion(),
    # torchio.HistogramStandardization({MRI: landmarks}),
    torchio.RandomBiasField(),
    torchio.ZNormalization(masking_method=None),
    torchio.RandomNoise(),
    # torchio.ToCanonical(),
    # torchio.Resample((4, 4, 4)),
    # torchio.CropOrPad((48, 60, 48)),
    # torchio.RandomFlip(axes=(0,)),
    # torchio.OneOf({
    #     torchio.RandomAffine(): 0.8,
    #     torchio.RandomElasticDeformation(): 0.2,}),
])

validation_transform = torchio.Compose([
    # torchio.HistogramStandardization({MRI: landmarks}),
    torchio.ZNormalization(masking_method=None),
    # torchio.ToCanonical(),
    # torchio.Resample((4, 4, 4)),
    # torchio.CropOrPad((48, 60, 48)),
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

# Early stopping
best_val_dice = 0.0
best_counter = 0

print('\nStarting training!')
for fold in range(num_folds):
    print('\nFOLD', fold)
    # Pre-loading sequence
    model = nnUNet(1, 1)
    model = nn.DataParallel(model)
    optimizer = RangerLars(model.parameters())

    # Running lists
    running_val_names = []
    running_val_metric = []

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
        overall_val_names = checkpoint['overall_val_names']
        overall_val_metric = checkpoint['overall_val_metric']
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
        print('Training step')
        running_loss = 0.0
        model.train()
        train_acc = 0
        total = 0
        new_seed = np.random.randint(10000)

        # Shuffle training and validation:
        new_train_df = reshuffle_csv(og_csv=train_df, batch_size=batch_size)
        new_val_df = reshuffle_csv(og_csv=val_df, batch_size=batch_size)

        # And generate new loaders
        patches_training_set = BespokeDataset(new_train_df, training_transform, patch_size, batch_seed=new_seed)
        train_loader = DataLoader(patches_training_set, batch_size=batch_size, num_workers=8, shuffle=False)
        patches_validation_set = BespokeDataset(new_val_df, validation_transform, patch_size, batch_seed=new_seed)
        val_loader = DataLoader(patches_validation_set, batch_size=int(batch_size / 4), num_workers=8)

        # Patch test
        if patch_test and epoch == 0 and fold == 0:
            visualise_batch_patches(train_loader, 4, 2)

        for i, sample in enumerate(train_loader):
            images = sample['mri']['data'].cuda()
            labels = sample['seg']['data'].cuda()
            physics = sample['physics'].cuda()
            names = sample['mri']['path']
            names = [os.path.basename(name) for name in names]

            out, features_out = model(images)

            # Need loss here
            eps = 1e-10
            data_loss = F.binary_cross_entropy_with_logits(out+eps, labels, reduction='mean')
            total_feature_loss = 0.1 * calc_feature_loss(features_out)  # NOTE: This needs to be the feature tensor!
            regulatory_ratio = loss / total_feature_loss
            loss = data_loss + stratification_epsilon * total_feature_loss / (1 + stratification_checker(labels) * float(1e9)) ** 2

            out = torch.softmax(out, dim=1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Name check: Shuffling sanity check
            if i == 0:
                print(f'The test names are: {names[0]}, {names[-2]}')

            # Writing to tensorboard
            # Normalise images
            images = images.cpu().detach().numpy()
            out = out.cpu().detach().numpy()
            images = (images - np.min(images)) / (np.max(images) - np.min(images))
            out = (out - np.min(out)) / (np.max(out) - np.min(out))
            if running_iter % 50 == 0:
                writer.add_scalar('Loss/train', loss.item(), running_iter)
                img2tensorboard.add_animated_gif_no_channels(writer=writer, image_tensor=images[0, ...],
                                                             tag='Visuals/Images', max_out=patch_size//4, scale_factor=255)
                img2tensorboard.add_animated_gif_no_channels(writer=writer, image_tensor=labels[0, ...].cpu().detach().numpy(),
                                                             tag='Visuals/Labels', max_out=patch_size//4, scale_factor=255)
                img2tensorboard.add_animated_gif_no_channels(writer=writer, image_tensor=out[0, ...],
                                                             tag='Visuals/Output', max_out=patch_size//4, scale_factor=255)

            print("iter: {}, Loss: {}".format(running_iter, loss.item()))
            running_iter += 1

        print("Epoch: {}, Loss: {},\n Train Accuracy: {}".format(epoch, running_loss, train_acc / total))
        running_iter = 0

        print('Validation step')
        model.eval()
        val_metric = DiceLoss(include_background=True, to_onehot_y=False, sigmoid=False, softmax=True)
        running_loss = 0
        # correct = 0
        val_counter = 0
        total = 0
        names_collector = []
        metric_collector = []

        with torch.no_grad():
            for val_sample in val_loader:
                val_images = val_sample['mri']['data'].squeeze().cuda()
                val_labels = val_sample['seg']['data'].squeeze().cuda()
                val_physics = val_sample['physics'].squeeze().cuda()
                val_names = val_sample['mri']['path']
                val_names = [os.path.basename(val_name) for val_name in val_names]

                out = model(val_images)

                val_loss = F.binary_cross_entropy_with_logits(out, val_labels, reduction="mean")
                out = torch.softmax(out, dim=1)

                running_loss += val_loss.item()

                # Metric calculation
                dice_performance = val_metric.forward(out, labels)
                metric_collector += dice_performance
                names_collector += val_names

                val_counter += 1

        # Write to tensorboard
        writer.add_scalar('Loss/val', running_loss / val_counter, running_iter)
        img2tensorboard.add_animated_gif_no_channels(writer=writer, image_tensor=val_images[0, ...],
                                                     tag='Validation/Images', max_out=patch_size // 4, scale_factor=255)
        img2tensorboard.add_animated_gif_no_channels(writer=writer, image_tensor=val_labels[0, ...],
                                                     tag='Validation/Labels', max_out=patch_size // 4, scale_factor=255)
        img2tensorboard.add_animated_gif_no_channels(writer=writer, image_tensor=out[0, ...],
                                                     tag='Validation/Output', max_out=patch_size // 4, scale_factor=255)

        # Check if current val dice is better than previous best
        true_dice = np.mean(metric_collector)
        if true_dice > best_val_dice:
            best_val_dice = true_dice
            append_string = 'best'
            best_counter = 0
        else:
            append_string = 'nb'
            best_counter += 1

        # Aggregation
        running_val_metric.append(true_dice)
        running_val_names.append(val_names)

        # Save model
        if SAVE and append_string == 'best':
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
                        'overall_val_names': overall_val_names,
                        'overall_val_metric': overall_val_metric}, MODEL_PATH)

        if best_counter >= 5:
            # Set overalls to best epoch
            best_epoch = int(np.argmin(running_val_metric))
            print(f'The best epoch is Epoch {best_epoch}')
            overall_val_metric.append(running_val_metric[best_epoch])
            overall_val_names.extend(running_val_names[best_epoch])
            break