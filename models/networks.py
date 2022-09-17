import torch
import torch.nn as nn
from torch.nn import init
import functools
import os
from models import resnet


###############################################################################
# Helper Functions
###############################################################################


def init_weights(net, init_type='normal', init_gain=0.02,
                 ignore_prefix='backbone', Info=True):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
    if Info:
        print('initialize network with %s' % init_type)
    # for name, model in net.named_modules(): # 这个有多层结构的，不行。。。
    for name, model in net.named_children():
        if ignore_prefix not in name:
            if Info:
                print('model init: %s' % name)
            # print(model)
            model.apply(init_func)
    # net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        if len(gpu_ids) > 1:
            net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs

    init_weights(net, init_type, init_gain=init_gain)
    return net


##################################################################################
def get_backbone(backbone_name, pretrained=True, backbone_stages_num=5,
                 structure='simple3'):
    if structure == 'simple3':
        #  表示backbone输出下采样2^3倍
        replace_stride_with_dilation = [False, True, True]
    elif structure == 'simple4':
        #  表示backbone输出下采样2^4倍
        replace_stride_with_dilation = [False, False, True]
    else:
        #  simple5
        #  表示backbone输出下采样2^5倍
        replace_stride_with_dilation = [False, False, False]
    out_layer_n = 1
    if 'resnet' in backbone_name:
        expand = 1
        if backbone_name == 'resnet18':
            backbone = resnet.resnet18(pretrained=pretrained,
                                          replace_stride_with_dilation=replace_stride_with_dilation)
        elif backbone_name == 'resnet34':
            backbone = resnet.resnet34(pretrained=pretrained,
                                          replace_stride_with_dilation=replace_stride_with_dilation)
        elif backbone_name == 'resnet50':
            backbone = resnet.resnet50(pretrained=pretrained,
                                          replace_stride_with_dilation=replace_stride_with_dilation)
            expand = 4
        else:
            raise NotImplementedError
        if backbone_stages_num == 5:
            out_layer_n = 512 * expand
        elif backbone_stages_num == 4:
            out_layer_n = 256 * expand
        elif backbone_stages_num == 3:
            out_layer_n = 128 * expand
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()
    return backbone, out_layer_n


class Backbone(torch.nn.Module):
    """
    # 当with_out_conv==True时，output_nc无效，与encoder输出的维度有关
    """
    def __init__(self, input_nc, output_nc,
                 backbone_stages_num=5, backbone_name='resnet18',
                 pretrained=True,
                 head_pretrained=False,
                 frozen_backbone_weights=False,
                 structure='simple3', backbone_fuse_mode='add',
                 with_out_conv=True, out_upsample2x=False,
                 backbone_out_feats=False):
        super(Backbone, self).__init__()
        self.backbone_stages_num = backbone_stages_num
        self.backbone_name = backbone_name
        self.resnet, layers = get_backbone(backbone_name, pretrained=pretrained,
                                           backbone_stages_num=backbone_stages_num,
                                           structure=structure)
        self.structure = structure
        self.head_pretrained = head_pretrained
        self.with_out_conv = with_out_conv
        self.out_upsample2x = out_upsample2x
        if self.structure == 'fcn':
            assert 'resnet' in self.backbone_name
            self.expand = 1
            if self.backbone_name == 'resnet50':
                self.expand = 4
            if self.backbone_stages_num == 5:
                self.conv_fpn1 = nn.Conv2d(512 * self.expand,
                                           256 * self.expand, kernel_size=3, padding=1)
            if self.backbone_stages_num >= 4:
                self.conv_fpn2 = nn.Conv2d(256 * self.expand, 128 * self.expand, kernel_size=3, padding=1)

            self.conv_fpn3 = nn.Conv2d(128 * self.expand, 64 * self.expand, kernel_size=3, padding=1)
            self.relu = nn.ReLU()
            self.upsamplex2 = nn.Upsample(scale_factor=2)
            layers = 64 * self.expand

        elif self.structure == 'fpn':
            from models.FPN import FPNDecoder
            # only support resnet18 currently
            assert 'resnet18' in backbone_name
            encoder_channels = [64, 128, 256, 512][::-1]
            pyramid_channels = 256
            self.neck = FPNDecoder(encoder_channels=encoder_channels, encoder_depth=5,
                            pyramid_channels=pyramid_channels, with_segmentation_head=False)
            layers = pyramid_channels
        elif 'simple' not in self.structure:
            raise NotImplementedError
        self.out_layers = layers
        if self.with_out_conv:
            self.conv_pred = nn.Conv2d(layers, output_nc, kernel_size=3, padding=1)
            self.out_layers = output_nc
        if self.out_upsample2x:
            self.upsamplex2 = nn.Upsample(scale_factor=2)
        self.backbone_out_feats = backbone_out_feats
        if self.backbone_out_feats:
            assert 'simple' in structure
            assert out_upsample2x is False
            assert with_out_conv is False
        init_weights(self, ignore_prefix='resnet')
        self._load_backbone_pretrain(pretrained)
        if frozen_backbone_weights:
            self._frozen_backbone()

    def _load_backbone_pretrain(self, pretrained):
        """用于加载预训练模型，用于下游迁移训练,
        以及是否迁移head 参数"""

        if pretrained is not None and os.path.isfile(pretrained):
            assert os.path.exists(pretrained)
            state_dict = torch.load(pretrained)
            loaded_state_dict = {}
            for k, v in state_dict.items():
                if 'resnet.' in k:
                    loaded_state_dict[k] = state_dict[k]
                else:
                    if self.head_pretrained:
                        loaded_state_dict[k] = state_dict[k]
            if not self.head_pretrained:
                print('do not loaded head parameter from %s' % pretrained)
            try:
                print(f'loading pretrained with items: {loaded_state_dict.keys()}')
                self.load_state_dict(loaded_state_dict, strict=False)
            except RuntimeError as e:
                print(e)
                # params_model = list(self.resnet.conv1.named_parameters())
                # params = state_dict['resnet.conv1.weight']
                # print('%s, shape: %s;[0][0][0]: %s' % (params_model[0][0],
                #                                        params_model[0][1].shape,
                #                                        params_model[0][1][0][0][0]))
                # print('%s shape: %s;[0][0][0]: %s' % ('resnet.conv1.weight',
                #                                       params.shape, params[0][0][0]))

            print('Backbone --> load from pretrain: %s' % pretrained)
        else:
            print('Backbone init: %s' % pretrained)

    def _frozen_backbone(self, frozen_layers='resnet'):
        if frozen_layers == 'resnet':
            m = self.resnet
            [x.requires_grad_(False) for x in m.parameters()]
            print(f'frozen resnet pretrained weights')

    def _forward_backbone(self, x):
        if 'resnet' in self.backbone_name:
            x = self.forward_resnet(x)
        else:
            x = self.resnet(x)
        return x

    def forward(self, x):
        if 'simple' in self.structure:
            if self.backbone_out_feats:
                x = self._forward_resnet_with_feats(x)[::-1][:4]  # p5,p4,p3,p2
            else:
                x = self._forward_backbone(x)
        elif self.structure == 'fcn':
            x = self.forward_resnet_fcn(x)
        elif self.structure == 'fpn':
            feats = self._forward_resnet_with_feats(x)[::-1]  # p5,p4,p3,p2
            x = self.neck(feats)
        if self.with_out_conv:
            x = self.conv_pred(x)
        if self.out_upsample2x:
            x = self.upsamplex2(x)
        return x

    def forward_resnet(self, x):
        # resnet layers
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x_4 = self.resnet.layer1(x)  # 1/4, in=64, out=64
        x_8 = self.resnet.layer2(x_4)  # 1/8, in=64, out=128

        if self.backbone_stages_num >= 4:
            x_8 = self.resnet.layer3(x_8)  # 1/8, in=128, out=256
        if self.backbone_stages_num == 5:
            x_8 = self.resnet.layer4(x_8)  # 1/32, in=256, out=512
        elif self.backbone_stages_num > 5:
            raise NotImplementedError
        return x_8

    def _forward_resnet_with_feats(self, x):
        # resnet layers
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x_2 = self.resnet.relu(x)
        x = self.resnet.maxpool(x_2)
        x_4 = self.resnet.layer1(x)  # 1/4, in=64, out=64
        x_8 = self.resnet.layer2(x_4)  # 1/8, in=64, out=128
        x_16 = self.resnet.layer3(x_8)  # 1/16, in=128, out=256
        if self.backbone_stages_num == 5:
            x_32 = self.resnet.layer4(x_16)  # 1/32, in=256, out=512
            return x_2, x_4, x_8, x_16, x_32
        elif self.backbone_stages_num == 4:
            return x_2, x_4, x_8, x_16
        else:
            raise NotImplementedError

    def forward_resnet_fcn(self, x):
        if self.backbone_stages_num == 5:
            x_4, x_8, x_16, x_32 = self._forward_resnet_with_feats(x)[1:]
            # FPN layers
            x = self.upsamplex2(self.relu(self.conv_fpn1(x_32)))
            x = self.upsamplex2(self.relu(self.conv_fpn2(x + x_16)))
            x = self.upsamplex2(self.relu(self.conv_fpn3(x + x_8)))
        elif self.backbone_stages_num == 4:
            x_4, x_8, x_16 = self._forward_resnet_with_feats(x)[1:]
            # FPN layers
            x = self.upsamplex2(self.relu(self.conv_fpn2(x_16)))
            x = self.upsamplex2(self.relu(self.conv_fpn3(x + x_8)))
        else:
            raise NotImplementedError(self.resnet_stages_num)
        return x

