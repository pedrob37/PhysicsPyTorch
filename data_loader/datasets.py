from torch.utils.data import Dataset
import torch
from pathlib import Path
import nibabel as nib
import numpy as np
from monai import transforms


class NiftiDataset(Dataset):
    """
    Class for loading .nii/.nii.gz images and paired segmentation masks.
    """
    def __init__(self, image_dir, label_dir, label_mapper, patch_size, transformations=None):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.image_list = list(self.image_dir.glob('*.nii*'))
        self.label_list = list(self.label_dir.glob('*.nii*'))
        self.transformations = transformations
        self.label_mapper = label_mapper
        self.patch_size = patch_size

        # some sanity checks on the data
        assert len(self.image_list) == len(self.label_list), 'Number of images and labels found are not equal.'
        assert all([(i.name == l.name) for i, l in zip(self.image_list, self.label_list)]), 'Image and label names do not correspond'

        # Data transforms + patch selection
        self.train_transforms = transforms.Compose(
            [
                transforms.LoadNiftid(keys=["img", "seg"], as_closest_canonical=True),
                # transforms.adaptor(label_mapper_transform, 'seg'),
                # transforms.AddChannelD(keys=["img", "seg"]),
                transforms.Whiteningd(keys=["img"]),
                transforms.RandSpatialCropD(keys=["img", "seg"], roi_size=self.patch_size, random_center=True,
                                            random_size=False),
                transforms.CastToTypeD(keys=["seg"], dtype=np.long)
            ]
        )
        self.val_transforms = transforms.Compose(
            [
                transforms.LoadNiftid(keys=["img", "seg"], as_closest_canonical=True),
                # transforms.adaptor(label_mapper_transform, 'seg'),
                # transforms.AddChannelD(keys=["img", "seg"]),
                transforms.Whiteningd(keys=["img"]),
                transforms.CastToTypeD(keys=["seg"], dtype=np.long)
            ]
        )

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = str(self.image_list[idx])
        label_path = str(self.label_list[idx])
        image, label = nib.load(image_path).get_fdata(dtype=np.float32), nib.load(label_path).get_fdata(dtype=np.float32).astype(np.uint8)
        # map labels from raw to desired level
        label = self.label_mapper.convert_raw_to_level(label)
        # add a channel dimension in if there is just one channel
        if image.ndim == 3:
            image = image[None, ...]
        # convert to tensors
        sample = (torch.from_numpy(image), torch.from_numpy(label))
        if self.transformations:
            sample = (self.transformations(sample[0]), self.transformations(sample[1]).type(torch.uint8))
        return sample
