import numpy as np
import torch
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

import kornia.augmentation as K

##########################################################
# basic_transform: toTensor, /norm, Resize
##########################################################

def get_transforms(norm=False, img_size=256):
    basic_transform = []
    basic_transform.append(T.ToTensor())  # ndarray转为 torch.FloatTensor， 范围[0,1]
    if norm:
        basic_transform.append(T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    basic_transform.append(T.Resize(size=(img_size, img_size), interpolation=InterpolationMode.BILINEAR))
    return T.Compose(basic_transform)


def get_mask_transforms(img_size=256):
    basic_target_transform = T.Compose(
        [
            MaskToTensor(),
            T.Resize(size=(img_size, img_size), interpolation=InterpolationMode.NEAREST),
        ]
    )
    return basic_target_transform


class MaskToTensor:
    def __call__(self, mask):
        return torch.from_numpy(np.array(mask, np.uint8)).unsqueeze(dim=0).float()

    def __repr__(self):
        return self.__class__.__name__ + '()'


##########################################################
# augmentations:
##########################################################


def get_ssl_augs(img_size=256, data_keys=('input', 'mask')):

    default_ssl_augs = K.AugmentationSequential(
        K.RandomHorizontalFlip(p=0.5),
        K.RandomVerticalFlip(p=0.5),
        K.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),
        K.RandomGrayscale(p=0.2),
        K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=0.5),
        K.RandomResizedCrop(
            size=(img_size, img_size), scale=(0.8, 1.0), resample="bilinear", align_corners=False
            ),
        K.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        data_keys=data_keys,
    )
    return default_ssl_augs


def get_ssl_augs_geometry(img_size=256, data_keys=('input', 'mask')):
    default_ssl_augs = K.AugmentationSequential(
        K.RandomHorizontalFlip(p=0.5),
        K.RandomVerticalFlip(p=0.5),
        # K.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),
        # K.RandomGrayscale(p=0.2),
        K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=0.5),
        K.RandomResizedCrop(
            size=(img_size, img_size), scale=(0.8, 1.0), resample="bilinear", align_corners=False
        ),
        K.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        data_keys=data_keys,
    )
    return default_ssl_augs

