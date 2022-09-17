import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat


##############################useful tools#########################################
# exponential moving average
class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


def detach_fn(data):
    if isinstance(data, torch.Tensor):
        data = data.detach()
    elif isinstance(data, list) or isinstance(data, tuple):
        if isinstance(data[0], torch.Tensor):
            data = [data_.detach() for data_ in data]
    else:
        raise NotImplementedError(data.dtype)
    return data


def gather_x_non_zeros(x, mask_id):
    # input: x  # c*n
    # mask_id: n
    c, n = x.shape
    index_id = torch.nonzero(mask_id, as_tuple=False)  # k*1
    if index_id.size(0) == 0:
        #  如果mask均为0，导致index_id为空，提取不出来，那么就置为零，防止出现nan
        x = torch.zeros(c, dtype=x.dtype, device=x.device, requires_grad=x.requires_grad)
    else:
        index_id = repeat(index_id, 'k 1 -> (1 c) k', c=c)  # c*k
        x = torch.gather(x, dim=1, index=index_id.long())  # c*k
        x = x.mean(-1)  # c
    return x


############################## networks #########################################


class MaskPooling(nn.Module):
    def __init__(self,
                 num_classes: int = 2,
                 downsample: int = 32) -> None:
        super().__init__()
        assert num_classes == 2 or num_classes == 1 #TODO
        self.num_classes = num_classes
        self.downsample = downsample
        self.mask_ids = torch.arange(num_classes)
        self.pool = nn.AvgPool2d(kernel_size=downsample, stride=downsample)
        print(f'downsample:{downsample}~~~~~~~~~~~~~~~~~~')

    def pool_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """Create binary masks and performs mask pooling
        Args:
            masks: (b, 1, h, w)
        Returns:
            masks: (b, num_classes, h*w)
        """
        if masks.ndim < 4:
            masks = masks.unsqueeze(dim=1)

        masks = masks == self.mask_ids[None, :, None, None].to(masks.device)
        masks = self.pool(masks.to(torch.float))
        masks = rearrange(masks, "b c h w -> b c (h w)")
        masks = torch.argmax(masks, dim=1)
        masks = torch.eye(self.num_classes).to(masks.device)[masks]
        masks = rearrange(masks, "b d c -> b c d")
        return masks

    def forward(self, x: torch.Tensor, masks: torch.Tensor):
        bs = x.shape[0]
        h, w = masks.shape[-2] // self.downsample, masks.shape[-1] // self.downsample
        if x.shape[-1] != w:
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
        x = rearrange(x, "b c h w -> b c 1 (h w)")
        binary_masks = self.pool_masks(masks)   # (b, num_classes, h*w)
        assert binary_masks.max() == 1
        area = binary_masks.sum(dim=-1, keepdim=True)
        weight_masks = binary_masks / torch.maximum(area, torch.tensor(1.0))
        weight_masks = rearrange(weight_masks, "b n d -> b 1 n d")
        x = (x * weight_masks).sum(-1)  # b, c, n

        return x


class MaskSampling(MaskPooling):
    """
    根据pts信息，对特征张量做采样，获得采样的特征点；
    """
    def __init__(self, num_classes: int = 2,
                 downsample: int = 32):
        super(MaskSampling, self).__init__(num_classes=num_classes,
                                           downsample=downsample)

    def forward(self, x: torch.Tensor, pts: torch.Tensor):
        c, h, w = x.shape[-3], x.shape[-2], x.shape[-1]
        pts = (pts / self.downsample).long()  # Bs * (num_classes*num_samples) * 2--[w,h]
        pts = pts[..., 0] + pts[..., 1] * w  # Bs * (num_classes*num_samples)
        # pts = rearrange(pts, 'b n -> b 1 n')
        pts = repeat(pts, 'b n -> b c n', c=c)
        x = rearrange(x, 'b c h w -> b c (h w)')
        x_sampled = x.gather(dim=-1, index=pts)   # Bs * c * (num_samples*num_classes)
        x_sampled = rearrange(x_sampled, 'b c (nc ns) -> b c ns nc', nc=self.num_classes)

        return x_sampled


# MLP class for projector and predictor
class MLP(nn.Module):
    def __init__(self, dim, projection_size, hidden_size=4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, projection_size)
        )

    def forward(self, x):
        return self.net(x)


class MLP_MS(nn.Module):
    """
    允许输入多组x特征（list of feats），分别经过2层MLP，得到多组投影；
    可以定义各个mlp是否共享参数
    """
    def __init__(self, dim_in=512, dim_out=512,
                       predictor_hidden_dim=128, multi_level_nums=4, is_shared=False):
        super().__init__()
        #  对于list的特征输入，是否对其共享参数？
        self.is_shared = is_shared
        if isinstance(dim_in, list) or isinstance(dim_in, tuple):
            assert len(dim_in) == multi_level_nums
        else:
            assert isinstance(dim_in, int)
            dim_in = [dim_in, ] * multi_level_nums
        if not is_shared:
            self.multi_level_nums = multi_level_nums
            self.online_predictors = nn.ModuleList()
            for i in range(self.multi_level_nums):
                online_predictor = MLP(dim_in[i], dim_out, predictor_hidden_dim)
                self.online_predictors.append(online_predictor)
        else:
            self.online_predictor = MLP(dim_in[0], dim_out, predictor_hidden_dim)

    def forward(self, x):
        """输入feats: f5,f4,f3,f2"""
        if self.is_shared:
            x = [self.online_predictor(x_) for x_ in x]
        else:
            assert len(x) == self.multi_level_nums
            x = [self.online_predictors[i](x[i]) for i in range(self.multi_level_nums)]
        # outs = []
        # for i in range(self.multi_level_nums):
        #     out = self.online_predictors[i](x[i])
        #     outs.append(out)
        return x


