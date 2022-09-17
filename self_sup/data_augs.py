import torch
import random
import torch.nn as nn
import kornia.augmentation as augs
from kornia import filters
import kornia
from models.ops_color import ColorAug


def virtual_swap_bg_in_batch(x1, m1, x2=None, m2=None):
    b = x1.shape[0]
    if x2 is None:
        #  batch内倒序
        index = list(range(b - 1, -1, -1))
        x2 = x1[index]
        m2 = m1[index]
    x1_, _ = merge_bg_ref_mask(x1, x2, m1, m2)
    return x1_, _


def merge_bg_ref_mask(x1, x2, m1, m2, color=True, blur=True):
    # x1: b*n*h*w
    # 将x2的背景部分，融入到x1的背景中；
    if color:
        colorex = ColorAug(norm_type='in', p=1)
        x2, _ = colorex(x2, x1)
    # else:
    #     print('no color transfer...................................')
    if blur:
        kernel_size = 7
        kernel = torch.ones(kernel_size + 2, kernel_size + 2).to(x1.device)
        m1 = kornia.morphology.dilation(m1, kernel=kernel)
        m2 = kornia.morphology.dilation(m2, kernel=kernel)
        blur = kornia.filters.GaussianBlur2d((kernel_size, kernel_size), (0.1, 2.0))
        m1 = blur(m1)
        m2 = blur(m2)
    mask = (1 - m1) * (1 - m2)
    x1_ = x1 * (1 - mask) + x2 * (mask)
    return x1_, mask


class RandomApply(nn.Module):
    def __init__(self, fn, p, _params=None):
        super().__init__()
        self.fn = fn
        self.p = p
        self._params = _params

    def forward(self, x, _params=None):
        if _params is None:
            self._params = torch.rand(1).item()
        else:
            self._params = _params
        if self._params > self.p:
            return x
        return self.fn(x)


class GaussianBlur_tensor(object):
    """
    对tenosr做处理，
    Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709
    """

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = filters.GaussianBlur2d(kernel_size=(3, 3), sigma=(sigma, sigma))(x)
        return x


# main class
class MyAugmentation(nn.Module):
    """原文链接： https://blog.csdn.net/ouening/article/details/119493807
    支持list/tuple的imgs，以及masks输入，用于扩充；masks可以为None
    """
    def __init__(self, image_size, with_random_resize_crop=False, weak_color_aug=False,
                 norm=True):
        super(MyAugmentation, self).__init__()
        # we define and cache our operators as class members
        # default SimCLR augmentation
        if weak_color_aug:
            self.k1 = RandomApply(augs.ColorJitter(0.2, 0.4, 0.4, 0.1), p=1)
            p_gray = 0
        else:
            self.k1 = RandomApply(augs.ColorJitter(0.4, 0.4, 0.4, 0.1), p=0.8)
            p_gray = 0.2
        self.k2 = augs.RandomGrayscale(p=0)
        self.k3 = augs.RandomHorizontalFlip()
        self.k4 = augs.RandomVerticalFlip()
        self.k5 = RandomApply(GaussianBlur_tensor([.1, 1.5]), p=0.5)
        self.with_random_resize_crop = with_random_resize_crop
        if with_random_resize_crop:
            self.k6 = augs.RandomResizedCrop((image_size, image_size), scale=(0.6, 1.))
            # NEAREST = 0
            # BILINEAR = 1
            # BICUBIC = 2
            self.k6_ = augs.RandomResizedCrop((image_size, image_size), scale=(0.6, 1.),
                                          # resample=0,
                                              )
        # 输入的dataset，已经做过归一化，不用再做了。修改之后，需要做了。
        self.norm = norm
        if self.norm:
            self.k7 = augs.Normalize(mean=torch.tensor([0.5, 0.5, 0.5]),
                                 std=torch.tensor([0.5, 0.5, 0.5]))

    def _forward_img(self, img: torch.Tensor) -> torch.Tensor:
        self.h, self.w = img.shape[-2], img.shape[-1]
        # 1. apply color only in image
        img_out = self.k5(self.k4(self.k3(self.k2(self.k1(img)))))
        # 2. apply geometric tranform
        if self.with_random_resize_crop:
            img_out = self.k6(img_out)
        if self.norm:
            img_out = self.k7(img_out)
        return img_out

    def _forward_imgs(self, imgs):
        """对imgs做相同的图像变换"""
        img = imgs[0]
        nums = len(imgs)
        self.h, self.w = img.shape[-2], img.shape[-1]
        imgs_out = []
        # 1. apply color only in image
        img_out = self.k5(self.k4(self.k3(self.k2(self.k1(img)))))
        # 2. apply geometric tranform
        if self.with_random_resize_crop:
            img_out = self.k6(img_out)
        if self.norm:
            img_out = self.k7(img_out)
        imgs_out.append(img_out)
        for i in range(1, nums):
            assert imgs[i].shape[-2] == self.h
            assert imgs[i].shape[-1] == self.w
            img_out = self.k5(self.k4(self.k3(self.k2(
                self.k1(imgs[i], self.k1._params), self.k2._params), self.k3._params,
            ), self.k4._params), self.k5._params)
            if self.with_random_resize_crop:
                img_out = self.k6(img_out, self.k6._params)
            if self.norm:
                img_out = self.k7(img_out)
            imgs_out.append(img_out)
        return imgs_out

    def _forward_mask(self, mask):
        assert mask.shape[-1] == self.w
        assert mask.shape[-2] == self.h
        if mask.dtype is not torch.float32:
            mask = mask.float()
        # 3. infer geometry params to mask
        # TODO: this will change in future so that no need to infer params
        mask_out = self.k4(self.k3(mask, self.k3._params), self.k4._params)
        if self.with_random_resize_crop:
            mask_out = self.k6_(mask_out, self.k6._params)
        return mask_out

    def forward(self, imgs, masks=None, consistency=True):
        if isinstance(imgs, list) or isinstance(imgs, tuple):
            if consistency:
                imgs_out = self._forward_imgs(imgs)
            else:
                # 如果后续，接着mask做扩充，那么mask扩充参数为最后一个img的扩充参数
                imgs_out = [self._forward_img(img) for img in imgs]
        else:
            imgs_out = self._forward_img(imgs)
        if masks is not None:
            if isinstance(masks, list) or isinstance(masks, tuple):
                masks_out = [self._forward_mask(mask) for mask in masks]
            else:
                masks_out = self._forward_mask(masks)
            return (imgs_out, masks_out)
        else:
            return imgs_out
