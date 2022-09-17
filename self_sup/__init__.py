from self_sup.masksiamsim import SimSiamMSample

import torch
from models.resnet import resnet18


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
    # init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(args):
    gpu_ids = args.gpu_ids
    image_size = args.img_size
    pretrained = args.pretrained

    if args.net_G == 'sadl_fpn_m2_resnet18_sample16_syn1':
        net = SimSiamMSample(backbone_name='resnet18', pretrained=pretrained,
                        structure='fpn', downsample=4, with_syn_data=1,
                        image_size=image_size, with_mask=2, num_samples=16)
    elif args.net_G == 'sadl_fpn_m2_resnet18_sample16_syn2':
        net = SimSiamMSample(backbone_name='resnet18', pretrained=pretrained,
                        structure='fpn', downsample=4, with_syn_data=2,
                        image_size=image_size, with_mask=2, num_samples=16)

############################some other methods ###################################
    elif args.net_G == 'moco_resnet18':
        from self_sup.moco import MoCoV2
        net = MoCoV2(base_encoder=resnet18, dim=128)
    elif args.net_G == 'simsiamOri_resnet18':
        from self_sup.moco import SimSiamV1
        net = SimSiamV1(base_encoder=resnet18, dim=512)
    elif args.net_G == 'seco_resnet18':
        from self_sup.moco import SeCoMain
        net = SeCoMain(base_encoder=resnet18, emb_dim=128)
    elif args.net_G == 'densecl_resnet18':
        from self_sup.moco import DenseCLMain
        net = DenseCLMain(backbone_name='resnet18',)
    else:
        raise NotImplementedError(args.net_G)

    return init_net(net, init_type='normal', init_gain=0.02, gpu_ids=gpu_ids)

