import os
from typing import Dict, Sequence, Tuple, Optional, List

from PIL import Image
import numpy as np

from torch.utils import data

from datasets.transforms import get_transforms, get_mask_transforms


"""
some basic data loader
for example:
bitemporal image loader, change detection folder

data root
├─A
├─B
├─label
└─list
"""


def load_img_name_list(dataset_path):
    img_name_list = np.loadtxt(dataset_path, dtype=str)
    if img_name_list.ndim == 2:
        return img_name_list[:, 0]
    return img_name_list


class BiImageDataset(data.Dataset):
    """VOCdataloder"""
    def __init__(self,
                 root_dir: str,
                 split: str = 'train',
                 img_size: int =256,
                 norm: bool = False,
                 img_folder_names: Tuple[str, str] = ('A', 'B'),
                 list_folder_name: str = 'list'):
        super(BiImageDataset, self).__init__()
        self.root_dir = root_dir
        self.split = split  # train | train_aug | val
        self.list_path = os.path.join(self.root_dir, list_folder_name, self.split+'.txt')
        self.img_name_list = load_img_name_list(self.list_path)
        self.A_size = len(self.img_name_list)  # get the size of dataset A
        self.img_folder_names = img_folder_names
        self.img_size = img_size
        assert len(img_folder_names) == 2
        self.basic_transforms = get_transforms(norm=norm, img_size=img_size)

    def _get_bi_images(self, name):
        imgs = []
        for img_folder_name in self.img_folder_names:
            A_path = os.path.join(self.root_dir, img_folder_name, name)
            img = np.asarray(Image.open(A_path).convert('RGB'))
            imgs.append(img)
        if self.basic_transforms is not None:
            imgs = [self.basic_transforms(img) for img in imgs]

        return imgs

    def __getitem__(self, index):
        name = self.img_name_list[index % self.A_size]
        imgs = self._get_bi_images(name)
        return {'A': imgs[0],  'B': imgs[1], 'name': name}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.A_size


class CDDataset(BiImageDataset):
    '''
    注意：这里仅应用基础的transforms，即tensor化，resize等
        其他transforms在外部的augs中应用
    '''
    def __init__(self,
                 root_dir: str,
                 split: str = 'train',
                 img_size: int = 256,
                 norm: bool = False,
                 img_folder_names: Tuple[str, str] = ('A', 'B'),
                 list_folder_name: str = 'list',
                 label_transform: str = 'norm',
                 label_folder_name: str = 'label',):
        super(CDDataset, self).__init__(root_dir, split=split,
                                        img_folder_names=img_folder_names,
                                        list_folder_name=list_folder_name,
                                        img_size=img_size,
                                        norm=norm)
        self.basic_mask_transforms = get_mask_transforms(img_size=img_size)
        self.label_folder_name = label_folder_name
        self.label_transform = label_transform

    def _get_label(self, name):
        mask_path = os.path.join(self.root_dir, self.label_folder_name, name)
        mask = np.array(Image.open(mask_path), dtype=np.uint8)
        #  二分类中，前景标注为255
        if self.label_transform == 'norm':
            mask = mask // 255
        elif self.label_transform == 'ignore0_sub1':
            mask = mask - 1
            # 原来label==0的部分变为255，自动被ignore
        if self.basic_mask_transforms is not None:
            mask = self.basic_mask_transforms(mask)
        return mask

    def __getitem__(self, index):
        name = self.img_name_list[index]
        imgs = self._get_bi_images(name)
        mask = self._get_label(name)
        return {'A': imgs[0], 'B': imgs[1], 'mask': mask, 'name': name}

