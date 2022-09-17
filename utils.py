import numpy as np
import torch
from torchvision import utils


def modify_state_dicts(state_dict, key_pre='', rm_pre='', rm_key=''):
    """
    提取关键字key开头的keys，并移除其中开头的rm_pre的字符
    Args:
        state_dict:
        key_pre: 提取关键字为key_pre的键，
        rm_pre:  提取的键中rm_pre开头的字符删除，留下后续字符作为新的key
        rm_key: 去除含有rm_key关键字的key
    Returns: out_state_dict

    """
    out_state_dict = {}
    keys = list(state_dict.keys())
    values = list(state_dict.values())
    for key, value in zip(keys, values):
        if rm_key in key and rm_key:
            print('remove key: %s' % key)
            continue
        if key_pre in key:
            out_key = key[key.find(rm_pre)+len(rm_pre):]
            out_state_dict[out_key] = value
            print('set key: %s --> out_key: %s' % (key, out_key))
    return out_state_dict


def convert_ckpt2pretrained(args, rm_pre='online_encoder.net.', ckpt_name='best_ckpt'):
    import os
    ckpt_path = os.path.join(args.checkpoint_dir, ckpt_name+'.pt')
    state_dict = torch.load(ckpt_path, map_location='cpu')['model_state_dict']
    # print('ssl mocel: ',state_dict.keys())
    out_path = os.path.join(args.checkpoint_dir, 'pretrained.pth')
    # rm_pre = net_G.get_rm_pre()
    if rm_pre is not None:
        rm_dict={}
        for k,v in state_dict.items():
            if 'fc.' in k:
                continue
            if rm_pre in k:
                rm_dict[k.replace(rm_pre,'')]=v
        state_dict = rm_dict
    print('backbone pretrained model: ', state_dict.keys())
    torch.save(state_dict, out_path)
    print('save backbone pretrained model at %s' % out_path)


def make_numpy_grid(tensor_data, pad_value=0, padding=0):
    tensor_data = tensor_data.detach()
    vis = utils.make_grid(tensor_data, pad_value=pad_value, padding=padding)
    vis = np.array(vis.cpu()).transpose((1, 2, 0))
    if vis.shape[2] == 1:
        vis = np.stack([vis, vis, vis], axis=-1)
    return vis


def make_numpy_grid_singledim(tensor_data, padding=2, pad_value=0):
    tensor_data = tensor_data.detach()
    b, c, h, w = tensor_data.shape
    tensor_data = tensor_data.view([b*c, 1, h, w])
    vis = utils.make_grid(tensor_data, padding=padding, pad_value=pad_value)
    vis = np.array(vis.cpu()).transpose((1, 2, 0))
    return vis[:, :, 0]


def make_numpy_grid_image_and_feature(tensor_images, tensor_features, padding=2,pad_value=0):
    tensor_images = tensor_images.detach().cpu()
    b1, c, h, w = tensor_images.shape
    assert c == 3
    tensor_feature = tensor_features.detach().cpu()
    b2,c,h,w = tensor_feature.shape
    assert b1 == b2
    tensor_feature = tensor_feature.view([b2*c,1,h,w])
    tensor_feature = torch.cat([tensor_feature,]*3,dim=1)
    tensor_data = torch.cat([tensor_images, tensor_feature],dim=0)
    vis = utils.make_grid(tensor_data, padding=padding, pad_value=pad_value)
    vis = np.array(vis.cpu()).transpose((1,2,0))
    return vis


def de_norm(tensor_data):
    return tensor_data * 0.5 + 0.5


def get_device(gpu_ids: str):
    # set gpu ids
    str_ids = gpu_ids.split(',')
    gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            gpu_ids.append(id)
    if len(gpu_ids) > 0:
        torch.cuda.set_device(gpu_ids[0])
    return gpu_ids

