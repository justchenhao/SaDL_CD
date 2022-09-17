import copy
from functools import wraps

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from self_sup.data_augs import MyAugmentation, merge_bg_ref_mask
from self_sup.losses import loss_fn_ms, loss_fn
from self_sup.networks import update_moving_average, EMA, MLP, detach_fn
from models.networks import Backbone
from self_sup.networks import MaskPooling

from misc.torchutils import save_visuals


DEBUG = False
debug_dir = 'vis'
# helper functions
"""
simsiam baseline
输出单个时相图像，
也可以结合mask信息，

"""


def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance
        return wrapper
    return inner_fn


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


class BaseNetWrapper(nn.Module):
    def __init__(self, backbone_name: str = 'resnet18',
                 structure: str = 'simple5',
                 pretrained: str = 'imagenet',
                 dim_in=512, #TODO
                 dim_out=1024,
                 dim_hidden=2048,
                 **kwargs):
        super().__init__()
        self.net = Backbone(backbone_name=backbone_name, input_nc=3, output_nc=512,
                          structure=structure, pretrained=pretrained,
                          head_pretrained=False, with_out_conv=False)
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        # self.dim_in = dim_in
        self.dim_in = self.net.out_layers
        # global projector
        self.projector = MLP(self.dim_in, self.dim_out, self.dim_hidden)

    def forward(self, x1, m=None, suffix_name=''):
        x1 = self.net(x1)  # b*c*h*w
        representation_t1 = self._forward_global_pool(x1)
        projection_t1 = self._forward_projector(representation_t1)
        out_dict = {}
        out_dict['representation_'+suffix_name] = representation_t1
        out_dict['projection_'+suffix_name] = projection_t1
        return out_dict

    def _forward_global_pool(self, x):
        # input x: b*c*h*w
        out_global = F.adaptive_avg_pool2d(x, [1, 1]).squeeze(-1).squeeze(-1)  # b*c
        return out_global

    def _forward_projector(self, x):
        # x = x.reshape([-1, self.dim_in])
        x = self.projector(x)  # N*projection_hidden_size
        return x


class NetWrapper(BaseNetWrapper):
    def __init__(self,
                 class_num=2, select_id=1, with_mask=True, masked_feats_n=2,
                 pool_mode=0, interpolate_mode=0,
                 **kwargs):
        """
        proj shared 参数
        """
        super().__init__(**kwargs)
        self.interpolate_mode = interpolate_mode
        self.pool_mode = pool_mode
        self.with_mask = with_mask
        self.masked_feats_n = masked_feats_n
        self.class_num = class_num
        self.select_id = select_id
        self.downsample = kwargs.get('downsample', 32)

        self.maskpooling = MaskPooling(num_classes=2, downsample=self.downsample)

    def _forward_masked_pool(self, x, m):
        # input x: b*c*h*w
        out = self.maskpooling(x, m)
        out_ms = [out[:, :, 0], out[:, :, 1]]
        return out_ms

    def _forward_projector_m(self, xs):
        # 输入xs为多个vector，每个代表对应区域的pooled features
        assert len(xs) == self.masked_feats_n
        xs = [self.projector(x) for x in xs]
        return xs

    def forward(self, x1, m1=None, suffix_name=''):
        assert m1 is not None
        x1 = self.net(x1)  # b*c*h*w
        representation_t1 = self._forward_global_pool(x1)
        projection_t1 = self._forward_projector(representation_t1)
        out_dict = {}
        out_dict['representation_'+suffix_name] = representation_t1
        out_dict['projection_'+suffix_name] = projection_t1

        representations_m_t1 = self._forward_masked_pool(x1, m1)
        projection_m_t1 = self._forward_projector_m(representations_m_t1)
        out_dict['representation_m_'+suffix_name] = representations_m_t1
        out_dict['projection_m_'+suffix_name] = projection_m_t1
        return out_dict

    def forward_with_mask(self, x1, m1):
        x1 = self.net(x1)  # b*c*h*w
        representations_m = self._forward_masked_pool(x1, m1)
        projection_m = self._forward_projector_m(representations_m)
        # out_dict = {}
        # out_dict['projection_m']=projection_m
        return projection_m


class NetWrapperWithSample(BaseNetWrapper):
    """
    编码器，对图像编码，并提取输入图像特定位置points处的特征；
    """
    def __init__(self, num_classes: int = 2,
                 num_samples: int = 16,
                 downsample: int = 32,
                 **kwargs):
        super(NetWrapperWithSample, self).__init__(**kwargs)
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.downsample = downsample
        from self_sup.networks import MaskSampling
        self.masksampling = MaskSampling(num_classes=num_classes,
                                         downsample=downsample)

    def _forward_global(self, x: torch.Tensor, suffix_name=''):
        representation_t1 = self._forward_global_pool(x)
        projection_t1 = self._forward_projector(representation_t1)
        out_dict = {}
        out_dict['representation_' + suffix_name] = representation_t1
        out_dict['projection_' + suffix_name] = projection_t1
        return out_dict

    def _forward_mask_sampling(self, x, pts, suffix_name=''):
        out_dict = {}
        repr_sampled = self.masksampling(x, pts)  # b c ns nc
        from einops import rearrange
        repr_sampled = rearrange(repr_sampled, 'b c ns nc -> (b ns nc) c')
        proj_sampled = self.projector(repr_sampled)

        repr_sampled = rearrange(repr_sampled, '(b ns nc) c -> (b ns) c  nc',
                                 nc=self.num_classes, ns=self.num_samples)
        proj_sampled = rearrange(proj_sampled, '(b ns nc) c -> (b ns) c  nc',
                                 nc=self.num_classes, ns=self.num_samples)
        # TODO: 暂时仅实现nc==2
        if self.num_classes == 2:
            out_dict['representation_m_' + suffix_name] = \
                [repr_sampled[..., 0], repr_sampled[..., 1]]
            out_dict['projection_m_' + suffix_name] = \
                [proj_sampled[..., 0], proj_sampled[..., 1]]
        elif self.num_classes == 1:
            out_dict['representation_m_' + suffix_name] = \
                [repr_sampled[..., 0]]
            out_dict['projection_m_' + suffix_name] = \
                [proj_sampled[..., 0]]
        else:
            raise NotImplementedError
        return out_dict

    def forward(self, x1, p1=None,
                suffix_name: str = ''):
        x1 = self.net(x1)  # b*c*h*w
        out_dict = self._forward_global(x1, suffix_name)
        out_dict_m = self._forward_mask_sampling(x1, p1, suffix_name)
        out_dict.update(out_dict_m)
        return out_dict


@torch.no_grad()
def gather_elements_by_indexes(simmap, locs):
    """
    获取simmap中locs位置处的得分值
    @param simmap:  h * w
    @param locs: [[h,w], ...], n * 2
    @return:
    """
    h, w = simmap.shape[0], simmap.shape[1]
    # simmap = rearrange(simmap, 'h w 1 -> (h w)')
    if not isinstance(locs, torch.Tensor):
        locs = torch.LongTensor(locs).to(simmap.device)  # n 2
    indexes = locs[..., 0] * w + locs[..., 1]  # n
    socres = torch.gather(simmap, index=indexes, dim=-1)
    return socres


@torch.no_grad()
def get_selected_cor_points(sim_value_cor, indexes_cor, threhold=0, max_num=None):
    """
    TODO： threshold的逻辑
    从密集对应关系图中选择值最大的max_num个点，对应的位置[[w, h], ]. 以及对应的匹配得分
    @param sim_value_cor: b h w
    @param indexes_cor: b h w
    @param threhold: float
    @param max_num: int
    @return:  torch.Tensor, [[w, h], ], b n 2
    """
    h, w = sim_value_cor.shape[-2], sim_value_cor.shape[-1]
    sim_value_cor = rearrange(sim_value_cor, 'b h w -> b (h w)')
    indexes_cor = rearrange(indexes_cor, 'b h w -> b (h w)')

    #  取得分较高的几个
    indexes_sorted = torch.argsort(sim_value_cor, dim=-1, descending=True)  # b hw
    if max_num is None:
        max_num = sim_value_cor.shape[-1]
    indexes_selected = indexes_sorted[..., :max_num]  # b max_num
    indexes_cor_selected = torch.gather(indexes_cor, dim=-1, index=indexes_selected)
    sim_val_cor_selected = torch.gather(sim_value_cor, dim=-1, index=indexes_selected)

    #  把index转化为[w, h]形式
    h_indexes_cor_selected = indexes_cor_selected // w
    w_indexes_cor_selected = indexes_cor_selected % w
    indexes_cor_selected = torch.stack([w_indexes_cor_selected,
                                        h_indexes_cor_selected], dim=-1)  # b n 2
    h_indexes_selected = indexes_selected // w
    w_indexes_selected = indexes_selected % w
    indexes_selected = torch.stack([w_indexes_selected,
                                    h_indexes_selected], dim=-1)  # b n 2
    return indexes_selected, indexes_cor_selected, sim_val_cor_selected


@torch.no_grad()
def get_simmap(x, x2 = None, norm = None):
    """
    计算特征图x1,x2之间的任两像素之间的相关性图；
    若x2==None，则计算自相关性图；
    @param x:
    @return:
    """
    if x2 is None:
        x2 = x
    c, h, w = x.shape[-3], x.shape[-2], x.shape[-1]
    simmap = torch.einsum('bchw,bcuv->bhwuv', x, x2)
    if norm == 'softmax':
        simmap = rearrange(simmap, 'b h w u v -> b h w (u v)')
        simmap = torch.softmax(simmap, dim=-1)
        simmap = rearrange(simmap, 'b h w (u v) -> b h w u v', u=h)
    return simmap


@torch.no_grad()
def get_corresponding_map(x, x2, norm=None, sorted=False, ret_simmap=False):
    """计算特征图x1与x2之间的对应关系
    """
    c, h, w = x.shape[-3], x.shape[-2], x.shape[-1]
    simmap = get_simmap(x, x2, norm=norm)  # bhwuv
    #  计算xi 与 {xj}最大相似度的j
    simmap = rearrange(simmap, 'b h w u v -> b h w (u v)')
    indexes_cor = torch.argmax(simmap, dim=-1, keepdim=True)  # b h w 1
    sim_value_cor = torch.gather(simmap, index=indexes_cor, dim=-1)  # b h w 1
    # TODO  舍去sim_val过小的元素
    # simmap = rearrange(simmap, 'b h w (u v) -> b h w u v', u=h)
    sim_value_cor = sim_value_cor.squeeze(-1)  # b h w
    indexes_cor = indexes_cor.squeeze(-1)
    if sorted:
        sim_value_cor = rearrange(sim_value_cor, 'b h w -> b (h w)')
        sorted_index = torch.argsort(sim_value_cor, dim=-1, descending=True)  # b hw
        sim_value_cor = rearrange(sim_value_cor, 'b (h w) -> b h w', h=h)
        sorted_h_index = sorted_index // h
        sorted_w_index = sorted_index % h
        sorted_index = torch.stack([sorted_h_index, sorted_w_index], dim=-1)  # b hw 2
        return sim_value_cor, indexes_cor, sorted_index
    if ret_simmap:
        simmap = rearrange(simmap, 'b h w (u v) -> b h w u v', u=h)
        return sim_value_cor, indexes_cor, simmap
    return sim_value_cor, indexes_cor


class BaseSimSiam(nn.Module):
    """
    仅支持单时相输入
    1、virtual color consistent
    """
    def __init__(self,
                 backbone_name: str = 'resnet18',
                 structure: str = 'simple5',
                 pretrained: str = 'imagenet',
                 image_size: int = 256,
                 dim_in: int = 512,
                 dim_out: int = 1024,
                 dim_hidden: int = 2048,
                 moving_average_decay: float = 0.99,
                 use_momentum: bool = False,
                 with_random_resize_crop: bool = True,
                 attenpool = None,
                 **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.backbone_name = backbone_name
        self.structure = structure
        self.pretrained = pretrained
        # default SimCLR augmentation
        self.image_size = image_size
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.with_random_resize_crop = with_random_resize_crop
        self.attenpool = attenpool

        self.use_momentum = use_momentum
        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average_decay)

        predictor_hidden_dim = dim_out // 4
        self.predictor_hidden_dim = predictor_hidden_dim

        self._get_augs()
        self.get_online_projectors()
        self.get_online_predictors()

    @torch.no_grad()
    def get_vis_feat(self, batch: dict, which_head='x_proj', downsample=None):
        """
        用于提取给定batch的特征图，用于可视化用
        @param batch:
        @return:
        """
        batch = self.on_after_batch_transfer(batch)
        A = batch['A']
        x1 = batch['x1_t1']
        x2 = batch['x2_t1']
        f_x1 = self.get_encoder_featmap(x1, downsample=downsample)[which_head]
        f_x2 = self.get_encoder_featmap(x2, downsample=downsample)[which_head]

        return x1, x2, f_x1, f_x2

    @torch.no_grad()
    def get_vis_feat_bidata(self, x1, x2, which_head='x_proj', downsample=None):
        """
        用于提取给定batch的特征图，用于可视化用
        @param batch:
        @return:
        """
        x1 = (x1 - 0.5) / 0.5
        x2 = (x2 - 0.5) / 0.5
        f_x1 = self.get_encoder_featmap(x1, downsample=downsample)[which_head]
        f_x2 = self.get_encoder_featmap(x2, downsample=downsample)[which_head]

        return x1, x2, f_x1, f_x2

    @torch.no_grad()
    def get_encoder_featmap(self, x: torch.Tensor, downsample=None):
        """
        获得图像x的特征图（encode 编码）以及dense projection，以及dense prediction，
        @param x:
        @param m:
        @return: dict
        """
        x_pres = self.online_encoder.net(x)  # b c h w
        c, h, w = x_pres.shape[-3], x_pres.shape[-2], x_pres.shape[-1]
        x_pres = rearrange(x_pres, 'b c h w -> (b h w) c')
        x_proj = self.online_encoder.projector(x_pres)
        x_pred = self.online_predictor(x_proj)
        x_pres = rearrange(x_pres, '(b h w) c -> b c h w', c=c, h=h, w=w)
        x_proj = rearrange(x_proj, '(b h w) c -> b c h w', c=self.dim_out, h=h, w=w)
        x_pred = rearrange(x_pred, '(b h w) c -> b c h w', c=self.dim_out, h=h, w=w)
        if downsample:
            out_h = x.shape[-2] // downsample
            out_w = x.shape[-1] // downsample
            x_pres = F.interpolate(x_pres, [out_h, out_w], mode='bilinear')
            x_proj = F.interpolate(x_proj, [out_h, out_w], mode='bilinear')
            x_pred = F.interpolate(x_pred, [out_h, out_w], mode='bilinear')

        out_dict = {}
        out_dict['x_pres'] = x_pres
        out_dict['x_proj'] = x_proj
        out_dict['x_pred'] = x_pred

        return out_dict

    def _get_augs(self):
        DEFAULT_AUG = MyAugmentation(image_size=self.image_size,
                                     with_random_resize_crop=self.with_random_resize_crop)
        self.augment1 = DEFAULT_AUG
        self.augment2 = self.augment1

    def get_online_projectors(self):
        self.online_encoder = BaseNetWrapper(backbone_name=self.backbone_name,
                                     pretrained=self.pretrained,
                                     structure=self.structure,
                                     dim_in=self.dim_in,
                                     dim_out=self.dim_out,
                                     dim_hidden=self.dim_hidden,)

    def get_online_predictors(self):
        # online_predictor color
        self.online_predictor = MLP(self.dim_out, self.dim_out, self.predictor_hidden_dim)

    def on_after_batch_transfer(self, batch: dict):
        x_t1 = batch['A']
        x1_t1 = self.augment1(x_t1)
        x2_t1 = self.augment2(x_t1)
        if DEBUG:
            save_visuals({'x_t1': x_t1}, img_dir=debug_dir, name=['test_ori'])
            save_visuals({'x1_t1':x1_t1, 'x2_t1':x2_t1,},img_dir=debug_dir, name=['test_aug'])
        dict_data = {'x1_t1': x1_t1, 'x2_t1': x2_t1}
        return dict_data

    def training_step(self, batch, batch_id):
        return self.forward(batch)

    def forward(self, batch: dict):
        return self.forward_no_mask(batch)

    def forward_no_mask(self, dict_data: dict):
        x1_t1, x2_t1 = \
            dict_data['x1_t1'], dict_data['x2_t1']
        #  step1: 分别对两个时相做，其中单个时相图，做颜色变换得到两个view，计算两个view之间的一致性
        #  约束1：全局global映射，跨颜色变换一致性，
        dict_online_one, dict_target_one = self._forward_encoder(x1_t1, suffix_name='t1')

        dict_online_two, dict_target_two = self._forward_encoder(x2_t1, suffix_name='t1')

        #  获得online prediction
        dict_pred_one = {}
        dict_pred_two = {}
        for k in ['projection_t1']:
            out_key = k.replace('projection', 'prediction')
            dict_pred_one[out_key] = self._forward_color_predictor_ms(dict_online_one[k])
            dict_pred_two[out_key] = self._forward_color_predictor_ms(dict_online_two[k])

        #  颜色变换一致性
        loss_color_g = 1 / 2 * (loss_fn_ms(dict_pred_one['prediction_t1'], dict_online_two['projection_t1'])
                                + loss_fn_ms(dict_pred_two['prediction_t1'], dict_online_one['projection_t1']))
        loss_color = loss_color_g

        loss_color = loss_color.mean()
        loss = loss_color
        loss_dict = {'loss': loss, 'loss_color': loss_color}

        return loss_dict

    def _forward_color_predictor_ms(self, online_proj_one):
        # 输入list的projs
        if isinstance(online_proj_one, tuple) or isinstance(online_proj_one, list):
            online_pred_one = [self.online_predictor(online_proj_one_) for online_proj_one_ in online_proj_one]
        else:
            online_pred_one = self.online_predictor(online_proj_one)
        return online_pred_one

    def _forward_encoder(self, image_one, m_one=None, suffix_name=''):
        #  修改为字典形式的projections
        dict_online = self.online_encoder(image_one, m_one, suffix_name=suffix_name)

        with torch.no_grad():
            if not self.use_momentum:  # fix 20220731, 减少重复计算target的值
                # dict_target = copy.deepcopy(dict_online)  # 无法深拷贝
                dict_target = dict_online.copy()
            else:
                target_encoder = self._get_target_encoder() if self.use_momentum else self.online_encoder
                dict_target = target_encoder(image_one, m_one, suffix_name=suffix_name)
            for k, v in dict_target.items():
                dict_target[k] = detach_fn(v)

        return dict_online, dict_target

    def get_rm_pre(self):
        return 'online_encoder.net.'

    def on_before_zero_grad(self):
        """在优化之前，更新动量编码器的参数"""
        if self.use_momentum:
            self.update_moving_average()

    @singleton('target_encoder')
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.online_encoder)
        set_requires_grad(target_encoder, False)
        return target_encoder

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        assert self.use_momentum, 'you do not need to update the moving average, since you have turned off momentum for the target encoder'
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)


class SimSiamM(BaseSimSiam):
    """
    仅支持单时相输入
    1、virtual color consistent
    2、masked consistent
    """
    def __init__(
        self,
        with_mask=2,  # False, 1：用mask，且用global, 2: 有mask，但不用global
        with_masked_push=True,  # 把不同语义块的聚合特征，拉开，增加一个不相似损失；
        push_head='representation',  # representation | projection
        push_loss_mode='push_loss4',  # push_loss1 | push_loss2 | batch_push_loss | batch_push_pull_loss
        with_syn_data=False,
        **kwargs,
    ):
        self.push_head = push_head
        self.push_loss_mode = push_loss_mode
        self.with_masked_push = with_masked_push
        self.with_mask = with_mask
        self.with_syn_data = with_syn_data
        self.context_blend = kwargs.get('context_blend', True)
        super().__init__(**kwargs)

    def get_online_projectors(self):
        self.class_num = self.kwargs.get('class_num', 2)
        self.pool_mode = self.kwargs.get('pool_mode', 0)
        self.select_id = self.kwargs.get('select_id', 1)
        self.interpolate_mode = self.kwargs.get('interpolate_mode', 0)
        self.masked_feats_n = self.kwargs.get('masked_feats_n', 2)
        self.downsample = self.kwargs.get('downsample', 32)
        self.online_encoder = NetWrapper(backbone_name=self.backbone_name,
                                     pretrained=self.pretrained,
                                     structure=self.structure,
                                     dim_in=self.dim_in,
                                     dim_out=self.dim_out,
                                     dim_hidden=self.dim_hidden,
                                     class_num=self.class_num,
                                     select_id=self.select_id,
                                     with_mask=self.with_mask,
                                     pool_mode=self.pool_mode,
                                     interpolate_mode=self.interpolate_mode,
                                     masked_feats_n=2,
                                     attenpool=self.attenpool,
                                     downsample=self.downsample)

    def _get_augs(self):
        from datasets.transforms import get_ssl_augs
        DEFAULT_AUG = get_ssl_augs(data_keys=['input', 'mask'])
        self.augment1 = DEFAULT_AUG
        self.augment2 = self.augment1

    def on_after_batch_transfer(self, batch):
        x_t1 = batch['A']
        m_t1 = batch['mask'].float()
        x1_t1, m1_t1 = self.augment1(x_t1, m_t1)
        x2_t1, m2_t1 = self.augment2(x_t1, m_t1)
        # x_t1_c2, x_t2_c1 = self.ex_cd_color_func(x_t1, x_t2)
        if DEBUG:
            save_visuals({'x_t1':x_t1},img_dir=debug_dir,name=['test_ori'])
            save_visuals({'x1_t1':x1_t1,
                          'x2_t1':x2_t1,},img_dir=debug_dir, name=['test_aug'])
        dict_data = {'x1_t1': x1_t1, 'x2_t1': x2_t1,
                     'm1_t1': m1_t1, 'm2_t1': m2_t1}
        if self.with_syn_data:
            dict_data.update(self._get_syn_data(dict_data))
        batch.update(dict_data)
        return batch

    def training_step(self, batch, batch_id):
        return self.forward(batch)

    def forward(self, batch: dict):
        return self.forward_with_mask(batch)

    def _get_syn_data_one(self, x1, m1, x2=None, m2=None):
        b = x1.shape[0]
        if x2 is None:
            #  batch内倒序
            index = list(range(b-1, -1, -1))
            x2 = x1[index]
            m2 = m1[index]
        x1_, _ = merge_bg_ref_mask(x1, x2, m1, m2,
                                   color=self.context_blend, blur=self.context_blend)
        return x1_

    def _get_syn_data(self, dict_data):
        dict_data['x1_f2'] = self._get_syn_data_one(dict_data['x1_t1'], dict_data['m1_t1'])
        dict_data['x2_f1'] = self._get_syn_data_one(dict_data['x2_t1'], dict_data['m2_t1'])
        return dict_data

    def _get_syn_data_bi(self, dict_data):
        dict_data['x1_f2'] = self._get_syn_data_one(dict_data['x1_t1'], dict_data['m1_t1'],
                                                    dict_data['x1_t2'], dict_data['m1_t2'])
        dict_data['x2_f1'] = self._get_syn_data_one(dict_data['x2_t1'], dict_data['m2_t1'],
                                                    dict_data['x2_t2'], dict_data['m2_t2'])
        return dict_data

    def forward_with_mask(self, dict_data):
        x1_t1, x2_t1, m1_t1, m2_t1 = \
            dict_data['x1_t1'], dict_data['x2_t1'],\
            dict_data['m1_t1'], dict_data['m2_t1']
        #  step1: 单个时相图，做颜色变换得到两个view，计算两个view之间的一致性
        #  约束1：全局global映射，跨颜色变换一致性，
        #  约束2：加入map指导下的跨颜色变换一致性；

        # 获得online 和 target projections
        # 'projection_t1',
        # 'projection_m_t1',
        dict_online_one, dict_target_one = self._forward_encoder(x1_t1, m1_t1, suffix_name='t1')

        dict_online_two, dict_target_two = self._forward_encoder(x2_t1, m2_t1, suffix_name='t1')

        #  获得online prediction
        dict_pred_one = {}
        dict_pred_two = {}
        for k in ['projection_t1', 'projection_m_t1']:
            out_key = k.replace('projection', 'prediction')
            dict_pred_one[out_key] = self._forward_color_predictor_ms(dict_online_one[k])
            dict_pred_two[out_key] = self._forward_color_predictor_ms(dict_online_two[k])

        #  颜色变换一致性
        loss_color_m = 1 / 2 * (loss_fn_ms(dict_pred_one['prediction_m_t1'], dict_target_two['projection_m_t1'])
                                + loss_fn_ms(dict_pred_two['prediction_m_t1'], dict_target_one['projection_m_t1']))
        if self.with_mask != 2:
            # print('cal ***********global feat********')
            loss_color_g = 1/2 * (loss_fn_ms(dict_pred_one['prediction_t1'], dict_target_two['projection_t1'])
                              + loss_fn_ms(dict_pred_two['prediction_t1'], dict_target_one['projection_t1']))
            loss_color = 1/2 * (loss_color_g + loss_color_m)
        else:
            loss_color = loss_color_m
        loss_color = loss_color.mean()
        loss = loss_color
        loss_dict = {'loss': loss, 'loss_color': loss_color}

        if self.with_masked_push:
            #  暂时对representation_m_t1, representation_m_t2，操作
            if self.push_head == 'representation':
                loss_m_push = 1 / 2 * (self._get_m_push_loss(dict_online_one['representation_m_t1']) +
                                       self._get_m_push_loss(dict_online_two['representation_m_t1']))
            elif self.push_head == 'projection':
                loss_m_push = 1 / 2 * (self._get_m_push_loss_proj(dict_pred_one, dict_target_one) +
                                       self._get_m_push_loss_proj(dict_pred_two, dict_target_two))
            else:
                raise NotImplementedError(self.push_head)

            loss_m_push = loss_m_push.mean()
            loss = loss + loss_m_push
            loss_dict['loss'] = loss
            loss_dict['loss_push'] = loss_m_push

        if self.with_syn_data:
            dict_online_syn, dict_target_syn = self._forward_encoder(
                dict_data['x1_f2'], m1_t1, suffix_name='x1_f2')
            dict_online_syn_, dict_target_syn_ = self._forward_encoder(
                dict_data['x2_f1'], m2_t1, suffix_name='x2_f1')
            dict_online_syn.update(dict_online_syn_)
            dict_target_syn.update(dict_target_syn_)
            #  获得online prediction
            dict_pred_syn = {}
            for k in ['projection_m_x1_f2', 'projection_m_x2_f1']:
                out_key = k.replace('projection', 'prediction')
                dict_pred_syn[out_key] = self._forward_color_predictor_ms(dict_online_syn[k])
            loss_syn_m = 1 * (loss_fn_ms(dict_pred_syn['prediction_m_x1_f2'][1::2],
                                             dict_target_one['projection_m_t1'][1::2])
                              # +loss_fn_ms(dict_pred_one['prediction_m_t1'][::2],
                              #                dict_target_syn['projection_m_x1_f2'][::2])
            #TODO: 改成对称形状吧，还能少用一个输入
                             + loss_fn_ms(dict_pred_syn['prediction_m_x2_f1'][1::2],
                                         dict_target_two['projection_m_t1'][1::2])
                             # + loss_fn_ms(dict_pred_two['prediction_m_t1'][::2],
                             #             dict_target_syn['projection_m_x2_f1'][::2])
                              )
            loss_syn_m = loss_syn_m.mean()
            # print(loss_syn_m)
            loss = loss + 1 / 2 * loss_syn_m
            loss_dict['loss'] = loss
            loss_dict['loss_syn_m'] = loss_syn_m
        return loss_dict

    def _get_m_push_loss(self, representation_m_t1):
        # 计算同一样本内，不同masked feats之间互斥性
        loss_dis = (loss_fn_ms(representation_m_t1[::2], representation_m_t1[1::2],
                                         mode=self.push_loss_mode, margin=1))
        return loss_dis

    def _get_m_push_loss_proj(self, dict_pred_one, dict_target_one):
        prediction_0 = dict_pred_one['prediction_m_t1'][::2]
        target_projection_1 = dict_target_one['projection_m_t1'][1::2]
        prediction_1 =dict_pred_one['prediction_m_t1'][1::2]
        target_projection_0 = dict_target_one['projection_m_t1'][::2]
        loss_dis = 1/2 * (loss_fn_ms(prediction_0, target_projection_1,
                               mode=self.push_loss_mode, margin=1)+
                    loss_fn_ms(prediction_1, target_projection_0,
                               mode=self.push_loss_mode, margin=1))

        return loss_dis


class SimSiamSample(BaseSimSiam):
    """
    全图采样
    """
    def __init__(self, num_samples=16, num_classes=1,
                 **kwargs):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.downsample = kwargs.get('downsample', 32)
        super(SimSiamSample, self).__init__(**kwargs)

    def get_online_projectors(self):
        self.online_encoder = NetWrapperWithSample(backbone_name=self.backbone_name,
                                                   pretrained=self.pretrained,
                                                   structure=self.structure,
                                                   dim_in=self.dim_in,
                                                   dim_out=self.dim_out,
                                                   dim_hidden=self.dim_hidden,
                                                   num_classes=self.num_classes,
                                                   num_samples=self.num_samples,
                                                   downsample=self.downsample)

    def _get_augs(self):
        from datasets.transforms_sample import BiDataAugWithSample
        self.aug = BiDataAugWithSample(img_size=self.image_size,
                                          num_classes=self.num_classes,
                                          num_samples=self.num_samples,
                                       downsample_init=False,
                                       downsample=self.downsample,
                                       )

    @torch.no_grad()
    def on_after_batch_transfer(self, batch):
        x_t1 = batch['A']
        b, c, h, w = x_t1.shape
        # m_t1 = batch['mask'].float()
        #  虚拟的mask，用于采样
        m_t1 = torch.zeros([b, 1, h, w], dtype=torch.float32).to(x_t1.device)
        x1_t1, x2_t1, m1_t1, m2_t1, pts1, pts2 = self.aug(x_t1, m_t1)

        if DEBUG:
            save_visuals({'x_t1':x_t1},img_dir=debug_dir,name=['test_ori'])
            save_visuals({'x1_t1':x1_t1,
                          'x2_t1':x2_t1,},img_dir=debug_dir, name=['test_aug'])
        dict_data = {'x1_t1': x1_t1, 'x2_t1': x2_t1,
                     'm1_t1': m1_t1, 'm2_t1': m2_t1,
                     'pts1_t1':pts1, 'pts2_t1':pts2}
        batch.update(dict_data)
        return batch

    def forward_no_mask(self, dict_data):
        x1_t1, x2_t1, m1_t1, m2_t1 = \
            dict_data['x1_t1'], dict_data['x2_t1'],\
            dict_data['pts1_t1'], dict_data['pts2_t1']
        dict_online_one, dict_target_one = self._forward_encoder(x1_t1, m1_t1, suffix_name='t1')
        dict_online_two, dict_target_two = self._forward_encoder(x2_t1, m2_t1, suffix_name='t1')

        #  获得online prediction
        dict_pred_one = {}
        dict_pred_two = {}
        for k in ['projection_t1', 'projection_m_t1']:
            out_key = k.replace('projection', 'prediction')
            dict_pred_one[out_key] = self._forward_color_predictor_ms(dict_online_one[k])
            dict_pred_two[out_key] = self._forward_color_predictor_ms(dict_online_two[k])

        #  颜色变换一致性
        loss_color_m = 1 / 2 * (loss_fn_ms(dict_pred_one['prediction_m_t1'], dict_target_two['projection_m_t1'])
                                + loss_fn_ms(dict_pred_two['prediction_m_t1'], dict_target_one['projection_m_t1']))

        loss_color = loss_color_m.mean()

        # loss_color = loss_color.mean()
        loss = loss_color
        loss_dict = {'loss': loss, 'loss_color': loss_color}

        return loss_dict


class SimSiamMSample(SimSiamM):
    """
    采用采样方式，从masked 区域中采样若干点，构造正样本对；
    """
    def __init__(self, num_samples=16, num_classes=2,
                 with_cl=False,  # contrastive loss
                 with_syn_cl=False,  # syn与原始图像做对比学习（仅正例之间执行）
                 downsample_init=False,  # 采样时，是否对mask做下采样
                 syn_bitime=0,  # 使用多时相图像数据来增广syn样本，实现前景一致性；
                 **kwargs):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.downsample_init = downsample_init
        self.downsample = kwargs.get('downsample', 32)
        super(SimSiamMSample, self).__init__(**kwargs)
        self.syn_bitime = syn_bitime
        self.with_syn_cl = with_syn_cl
        self.with_cl = with_cl

    def get_online_projectors(self):
        self.online_encoder = NetWrapperWithSample(backbone_name=self.backbone_name,
                                                   pretrained=self.pretrained,
                                                   structure=self.structure,
                                                   dim_in=self.dim_in,
                                                   dim_out=self.dim_out,
                                                   dim_hidden=self.dim_hidden,
                                                   num_classes=self.num_classes,
                                                   num_samples=self.num_samples,
                                                   downsample=self.downsample)

    def _get_augs(self):
        from datasets.transforms_sample import BiDataAugWithSample
        self.aug = BiDataAugWithSample(img_size=self.image_size,
                                          num_classes=self.num_classes,
                                          num_samples=self.num_samples,
                                       downsample_init=self.downsample_init,
                                       downsample=self.downsample,
                                       )

    @torch.no_grad()
    def on_after_batch_transfer(self, batch):
        x_t1 = batch['A']
        m_t1 = batch['mask'].float()

        x1_t1, x2_t1, m1_t1, m2_t1, pts1, pts2 = self.aug(x_t1, m_t1)

        if DEBUG:
            save_visuals({'x_t1':x_t1},img_dir=debug_dir,name=['test_ori'])
            save_visuals({'x1_t1':x1_t1,
                          'x2_t1':x2_t1,},img_dir=debug_dir, name=['test_aug'])
        dict_data = {'x1_t1': x1_t1, 'x2_t1': x2_t1,
                     'm1_t1': m1_t1, 'm2_t1': m2_t1,
                     'pts1_t1':pts1, 'pts2_t1':pts2}
        if self.with_syn_data:
            if self.syn_bitime == 1:
                # 使用另一时相的图像中的背景来做背景替换
                x_t2 = batch['B']
                x1_t2, x2_t2, m1_t2, m2_t2 = self.aug.get_augs_from_predefined(x_t2, m_t1)
                dict_data['x1_t2'] = x1_t2
                dict_data['x2_t2'] = x2_t2
                dict_data['m1_t2'] = m1_t2
                dict_data['m2_t2'] = m2_t2
                syn_data = self._get_syn_data_bi(dict_data)
            elif self.syn_bitime == 2:
                # 直接用另一时相的图作为第三个view，也是做前景的一致性（真正的多时相前景的一致性）
                x_t2 = batch['B']
                x1_t2, x2_t2, m1_t2, m2_t2 = self.aug.get_augs_from_predefined(x_t2, m_t1)
                syn_data = {}
                syn_data['x1_f2'] = x1_t2
            else:
                syn_data = self._get_syn_data(dict_data)
            dict_data.update(syn_data)
        batch.update(dict_data)
        return batch

    def forward_with_mask(self, dict_data):
        x1_t1, x2_t1, m1_t1, m2_t1 = \
            dict_data['x1_t1'], dict_data['x2_t1'],\
            dict_data['pts1_t1'], dict_data['pts2_t1']
        dict_online_one, dict_target_one = self._forward_encoder(x1_t1, m1_t1, suffix_name='t1')
        dict_online_two, dict_target_two = self._forward_encoder(x2_t1, m2_t1, suffix_name='t1')

        #  获得online prediction
        dict_pred_one = {}
        dict_pred_two = {}
        for k in ['projection_t1', 'projection_m_t1']:
            out_key = k.replace('projection', 'prediction')
            dict_pred_one[out_key] = self._forward_color_predictor_ms(dict_online_one[k])
            dict_pred_two[out_key] = self._forward_color_predictor_ms(dict_online_two[k])

        #  颜色变换一致性
        loss_color_m = 1 / 2 * (loss_fn_ms(dict_pred_one['prediction_m_t1'], dict_target_two['projection_m_t1'])
                                + loss_fn_ms(dict_pred_two['prediction_m_t1'], dict_target_one['projection_m_t1']))
        if self.with_mask != 2:
            # print('cal ***********global feat********')
            loss_color_g = 1/2 * (loss_fn_ms(dict_pred_one['prediction_t1'], dict_target_two['projection_t1'])
                              + loss_fn_ms(dict_pred_two['prediction_t1'], dict_target_one['projection_t1']))
            if self.with_mask == 3:
                #  仅计算全局的一致性；
                # print(f'全局的一致性')
                loss_color = loss_color_g.mean()
            else:
                loss_color = 1 / 2 * (loss_color_g.mean() + loss_color_m.mean())
        else:
            loss_color = loss_color_m.mean()

        # loss_color = loss_color.mean()
        loss = loss_color
        loss_dict = {'loss': loss, 'loss_color': loss_color}

        if self.with_masked_push:
            #  暂时对representation_m_t1, representation_m_t2，操作
            if self.push_head == 'representation':
                loss_m_push = 1 / 2 * (self._get_m_push_loss(dict_online_one['representation_m_t1']) +
                                       self._get_m_push_loss(dict_online_two['representation_m_t1']))
            elif self.push_head == 'projection':
                loss_m_push = 1 / 2 * (self._get_m_push_loss_proj(dict_pred_one, dict_target_two) +
                                       self._get_m_push_loss_proj(dict_pred_two, dict_target_one))
            else:
                raise NotImplementedError(self.push_head)

            loss_m_push = loss_m_push.mean()
            loss = loss + loss_m_push
            loss_dict['loss'] = loss
            loss_dict['loss_push'] = loss_m_push

        if self.with_syn_data:
            items = ['x1_f2']
            dict_online_syn, dict_target_syn = self._forward_encoder(
                dict_data['x1_f2'], m1_t1, suffix_name='x1_f2')
            if self.with_syn_data >= 2 and self.with_syn_data != 3:
                items = ['x1_f2', 'x2_f1']
                dict_online_syn_, dict_target_syn_ = self._forward_encoder(
                dict_data['x2_f1'], m2_t1, suffix_name='x2_f1')
                dict_online_syn.update(dict_online_syn_)
                dict_target_syn.update(dict_target_syn_)
            #  获得online prediction
            dict_pred_syn = {}
            loss_syn_m = 0
            for k in items:
                k = 'projection_m_' + k  # ['projection_m_x1_f2', 'projection_m_x2_f1']
                out_key = k.replace('projection', 'prediction')
                dict_pred_syn[out_key] = self._forward_color_predictor_ms(dict_online_syn[k])
                if 'x1_f2' in k:
                    # self.with_syn_cl = True  # syn与原始图像做对比学习（仅正例之间执行）
                    if self.with_syn_cl:
                        loss_syn_m_ = self.pointsContrastiveLoss([dict_pred_syn[out_key][1]],
                                              [dict_pred_one['prediction_m_t1'][1]],
                                              [dict_target_syn['projection_m_x1_f2'][1]],
                                              [dict_target_one['projection_m_t1'][1]])
                    else:
                        if self.with_syn_data >= 3:  # 增加对称性
                            loss_syn_m_ = 1 / 2 * (loss_fn_ms(dict_pred_syn[out_key][1],
                                                     dict_target_one['projection_m_t1'][1])
                                               + loss_fn_ms(dict_pred_one['prediction_m_t1'][1],
                                                     dict_target_syn['projection_m_x1_f2'][1])
                                                   )
                        else:
                            loss_syn_m_ = 1 * (loss_fn_ms(dict_pred_syn[out_key][1],
                                                      dict_target_one['projection_m_t1'][1])
                                        )

                elif 'x2_f1' in k:
                    if self.with_syn_data == 4:
                        loss_syn_m_ = 1 / 2 * (loss_fn_ms(dict_pred_syn[out_key][1],
                                             dict_target_two['projection_m_t1'][1])
                                       + loss_fn_ms(dict_pred_two['prediction_m_t1'][1],
                                             dict_target_syn['projection_m_x2_f1'][1])
                                       )
                    else:
                        loss_syn_m_ = loss_fn_ms(dict_pred_syn[out_key][1],
                                                 dict_target_two['projection_m_t1'][1])
                else:
                    raise NotImplementedError
                loss_syn_m += loss_syn_m_ / len(items)
            loss_syn_m = loss_syn_m.mean()
            # print(loss_syn_m)
            loss = loss + 1 / 2 * loss_syn_m
            loss_dict['loss'] = loss
            loss_dict['loss_syn_m'] = loss_syn_m

        return loss_dict

