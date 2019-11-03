import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from albumentations import (
    Compose,
    Normalize,
    VerticalFlip,
    HorizontalFlip,
    GaussNoise,
    Rotate,
    RandomResizedCrop,
)
from albumentations.pytorch.transforms import ToTensorV2
from preprocess.preprocess import make_mask, make_mask_only3


def get_augmentation_list():
    """
    return data augumentation list by albumentations
    """
    transforms_list = []
    transforms_list.extend(
        [
            VerticalFlip(),
            HorizontalFlip(),
            GaussNoise(),
            Rotate(limit=10, value=(0, 0, 0)),
            RandomResizedCrop(256, 1600, ratio=(1.5, 5.5), p=0.5),
        ]
    )
    transforms_list = Compose(transforms_list)

    return transforms_list


def get_to_tensor_list():
    """
    return converting to pytorch tensor function list 
    by albumentations
    """
    transforms_list = []
    transforms_list.extend([Normalize(), ToTensorV2()])
    transforms_list = Compose(transforms_list)

    return transforms_list


class SegmentationDataset(Dataset):
    def __init__(self, df, image_foloder, train=True):
        """
        input 
        df           : After preprocessing dataframe which contains 
                       4-class defect information. 
        image_folder : a directory where the images exist.
        train        : train mode or valid mode.
                       If false, no data augmentation.
        output
        image        : torch tensor image
        mask         : torch tensor mask
        label        : if mask is large enough, return labels
        """
        self.df = df
        self.image_folder = image_folder
        self.augmentation = get_augmentation_list()
        self.totensor_list = get_to_tensor_list()
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, id):
        fname, mask = make_mask(id, self.df)
        image_path = os.path.join(self.image_folder, fname)
        image = Image.open(image_path)
        image = np.asarray(image)
        if self.train:
            aug_data = self.augmentation(image=image, mask=mask)
            image, mask = aug_data["image"], aug_data["mask"]

        # 4-class classification
        label = (mask.reshape(-1, 4).sum(0) > 8).astype(np.int32)
        label = torch.from_numpy(label).float()

        mask = mask.transpose(2, 0, 1)
        tensor_data = self.totensor_list(image=image, mask=mask)
        image, mask = tensor_data["image"], tensor_data["mask"]

        return image, mask, label


class SegmentationDatasetOnly3(Dataset):
    def __init__(self, df, input_filepath, train=True):
        self.df = df
        self.image_folder = input_filepath
        self.augmentation = get_augmentation_list()
        self.totensor_list = get_to_tensor_list()
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, id):
        fname, mask = make_mask_only3(id, self.df)
        image_path = os.path.join(self.image_folder, fname)
        image = Image.open(image_path)
        image = np.asarray(image)
        if self.train:
            aug_data = self.augmentation(image=image, mask=mask)
            image, mask = aug_data["image"], aug_data["mask"]
        mask = mask.transpose(2, 0, 1)
        tensor_data = self.totensor_list(image=image, mask=mask)
        image, mask = tensor_data["image"], tensor_data["mask"]
        return image, mask
