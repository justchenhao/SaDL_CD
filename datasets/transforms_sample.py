from typing import Dict, Optional, Union
import torch
from torch import nn
import kornia as K
from einops import rearrange
import copy


def sample_points_from_mask(mask: torch.Tensor,
                            target_id: int,
                            num_samples: int,
                            mask_valid: Optional[torch.Tensor] = None,
                            mask_importance: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    在mask上采样值为target_id的点， 返回num_samples个点的(x,y)坐标
    Args:
        mask: Bs * H * W
        target_id:
        num_samples:
        mask_valid:
        mask_importance: Bs * H * W,
    Returns: Bs * num_samples * 2

    """
    if mask.ndim == 4:
        assert mask.shape[1] == 1
        mask = mask.squeeze(dim=1)
    #  如果一个都采样不到怎么办？  TODO:
    h, w = mask.shape[-2], mask.shape[-1]
    sel_mask = (mask == target_id) + 1e-11  # 1e-11 / 1
    if mask_importance is not None:
        assert mask_importance.ndim == 3
        sel_mask = sel_mask * mask_importance
    if mask_valid is not None:
        assert mask_valid.ndim == 3
        sel_mask[mask_valid == 0] = 0  # 0 永远不会被抽到——》multinomial
    # from misc.torchutils import visualize_tensors
    # visualize_tensors(sel_mask[0])
    sel_mask = rearrange(sel_mask, 'b h w -> b (h w)')
    mask_id = torch.multinomial(sel_mask,
                                num_samples=num_samples,
                                replacement=True)  # Bs * num_samples
    sampled_h = mask_id // w
    sampled_w = mask_id % w
    sampled_wh = torch.stack([sampled_w, sampled_h], dim=-1)  # Bs * num_samples * 2
    return sampled_wh


class BiDataAugWithSample(nn.Module):
    """
    分别对前后时相图像做augs，并获取前后时相对应的geometry trans
    利用trans信息，反推原始空间的有效区域，在其中采样，获得points
    """
    def __init__(self, img_size: Union[int, tuple] = 256,
                 num_samples: int = 16,
                 num_classes: int = 2,
                 downsample: int = 32,
                 downsample_init: bool = False,  # 是否在downsample的mask上采样点（可能更分散一点）
                 ) -> None:
        super().__init__()
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        assert len(img_size) == 2
        # declare kornia components as class members
        aug_list = K.augmentation.AugmentationSequential(
            K.augmentation.RandomHorizontalFlip(p=0.5),
            K.augmentation.RandomVerticalFlip(p=0.5),
            K.augmentation.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=0.5),
            K.augmentation.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),
            K.augmentation.RandomResizedCrop(
                size=img_size, scale=(0.8, 1.0), resample="bilinear",
                align_corners=False, cropping_mode="resample",
            ),
            K.augmentation.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            data_keys=["input", "mask"],
            return_transform=True,
            same_on_batch=False,
        )
        self.augs1 = aug_list
        self.augs2 = copy.deepcopy(aug_list)
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.downsample = downsample
        self.downsample_init = downsample_init
        if self.downsample_init:
            self.pool = nn.AvgPool2d(kernel_size=self.downsample, stride=self.downsample)

    @torch.no_grad()
    def sample_points(self, mask, mask_valid=None, mask_importance=None) -> torch.Tensor:
        """
        根据mask采样num_samples个points
        Returns:

        """
        if self.downsample_init:
            mask = self.pool(mask)
            mask = (mask > 0.5).float()
            # print(np.unique(m.cpu().numpy()))
            mask_valid = self.pool(mask_valid.float()).long()
            if mask_importance is not None:
                mask_importance = self.pool(mask_importance.float())
        else:
            self.downsample = 1

        out_id = []
        for i in range(self.num_classes):
            sampled_id = sample_points_from_mask(mask, i, self.num_samples,
                                                 mask_valid, mask_importance)
            out_id.append(sampled_id)
        out_id = torch.concat(out_id, dim=1)  # Bs * (num_classes*num_samples) * 2

        out_id = out_id * self.downsample + self.downsample // 2  # 20220414修复

        return out_id

    @torch.no_grad()
    def forward_one(self, x1: torch.Tensor, m1: torch.Tensor):
        out_tensor = self.augs1(x1, m1)
        x1, trans = out_tensor[0]
        m1 = out_tensor[1]
        return x1, m1

    @torch.no_grad()
    def get_augs_from_predefined(self, x, m):
        """
        根据已有的augs参数，对输入的x做图像变换（几何/颜色）
        @param x: b 3 h w
        @param m: b 1 h w
        @return:
        """
        # https://kornia.readthedocs.io/en/latest/augmentation.container.html
        (x1, trans1), m1 = self.augs1(x, m, params=self.augs1._params, data_keys=["input", "mask"])
        (x2, trans2), m2 = self.augs2(x, m, params=self.augs2._params, data_keys=["input", "mask"])

        return x1, x2, m1, m2

    @torch.no_grad()
    def forward(self, x: torch.Tensor,
                m: torch.Tensor,
                pts: Optional[torch.Tensor] = None,
                x_v2: Optional[torch.Tensor] = None,
                importance_mask: Optional[torch.Tensor] = None):
        """

        Args:
            x: b 3 h w
            m: b 1 h w
            pts: None /
            x_v2: None /
        Returns:

        """
        if x_v2 is None:
            #  适配多个x输入的情况
            x_v2 = x
        h, w = x.shape[-2], x.shape[-1]
        (x1, trans1), m1 = self.augs1(x, m)
        (x2, trans2), m2 = self.augs2(x_v2, m)
        if pts is None:
            x1_, _ = self.augs1.inverse(x1, m1)
            x2_, _ = self.augs2.inverse(x2, m2)
            mask_valid = (x1_.sum(1) != 0) * (x2_.sum(1) != 0)
            pts = self.sample_points(m, mask_valid, importance_mask)
            self.pts = pts

        pts1 = K.geometry.transform_points(trans1, points_1=pts.float()).long()
        pts2 = K.geometry.transform_points(trans2, points_1=pts.float()).long()
        #  边界条件
        pts1 = check_points_within_boundary(pts1, h, w, message='pts1')
        pts2 = check_points_within_boundary(pts2, h, w, message='pts2')

        return x1, x2, m1, m2, pts1, pts2


def check_torch_greater_than_val(tensor, val):
    import numpy as np
    data = tensor.cpu().numpy()
    index = np.argwhere(data > val)
    if len(index) > 0:
        print(f'data: {tensor[index[0][0],index[0][1]]}, '
              f'index:{index[0][0]},{index[0][1]}')
    return index


def check_torch_greater_equal_than_val(tensor, val):
    import numpy as np
    data = tensor.cpu().numpy()
    index = np.argwhere(data >= val)
    if len(index) > 0:
        print(f'data: {tensor[index[0][0],index[0][1]]}, '
              f'index:{index[0][0]},{index[0][1]}')
    return index


def check_points_within_boundary(pts, h, w, message=''):
    import warnings
    if pts[..., 0].greater_equal(w).sum() > 0:
        warnings.warn(f'{message}: points greater than width {w}')
        # check_torch_greater_equal_than_val(pts[..., 0], w)
        pts[..., 0][pts[..., 0].greater_equal(w)] = w - 1
    if pts[..., 1].greater_equal(h).sum() > 0:
        warnings.warn(f'{message}: points greater than height {h}')
        # check_torch_greater_equal_than_val(pts[..., 0], h)
        pts[..., 1][pts[..., 1].greater_equal(h)] = h - 1
    if (-pts[..., 0]).greater(0).sum() > 0:
        warnings.warn(f'{message}: points smaller than 0')
        # check_torch_greater_than_val(-pts[..., 0], 0)
        pts[..., 0][(-pts[..., 0]).greater(0)] = 0
    if (-pts[..., 1]).greater(0).sum() > 0:
        warnings.warn(f'{message}: points smaller than 0')
        # check_torch_greater_than_val(-pts[..., 0], 0)
        pts[..., 1][(-pts[..., 1]).greater(0)] = 0
    return pts
