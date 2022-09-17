import torch.optim as optim


def get_params_groups(net, lr, lr_multi=10):
    if hasattr(net, 'trainable_parameters'):
        param_groups = net.trainable_parameters()
    else:
        parameters = filter(lambda p: p.requires_grad, net.parameters())   # 适配frozen layers的情况
        param_groups = list(parameters)
        # print(f'layers that requires grad')
        # for k, v in net.named_parameters():
        #     if v.requires_grad:
        #         print(k)
    if isinstance(param_groups, list):
        params_list = [{'params': param_groups, 'lr': lr_multi * lr}]
    elif isinstance(param_groups, tuple):
        params_list = [{'params': param_groups[0]},
                       {'params': param_groups[1], 'lr': lr_multi * lr}]
    else:
        raise NotImplementedError
    return params_list


def get_optimizer(model_params, lr, optim_mode='sgd', lr_policy='linear', init_step=0, max_step=None):
    # if lr_policy != 'poly':
    if optim_mode == 'sgd':
        optimizer_G = optim.SGD(model_params, lr=lr, momentum=0.9, weight_decay=5e-4)
    elif optim_mode == 'adam':
        optimizer_G = optim.Adam(model_params, lr=lr, betas=(0.9, 0.999))
    elif optim_mode == 'adamw':
        optimizer_G = optim.AdamW(model_params, lr=lr, betas=(0.9, 0.999))
    else:
        raise NotImplementedError()
    return optimizer_G
