import torch
import torch.nn.functional as F

from torch import Tensor
from typing import Iterable, Set, Tuple

from misc.torchutils import class2one_hot


# loss fn
def push_loss_fn(x, y, margin=1, mode='push_loss1'):
    x = F.normalize(x, dim=-1, p=2)  # b*c
    y = F.normalize(y, dim=-1, p=2)  # b*c
    #  这里强制搞一个正交约束
    b = x.size(0)  # batch size
    if mode == 'push_loss1':
        #     .. math::
        #         l_n = \begin{cases}
        #             x_n, & \text{if}\; y_n = 1,\\
        #             \max \{0, \Delta - x_n\}, & \text{if}\; y_n = -1,
        #         \end{cases}
        dist_val = 1 - (x * y).sum(dim=-1)  # [0, 2]
        target = -torch.ones(b, device=x.device).long()
        loss = torch.nn.functional.hinge_embedding_loss(dist_val, target=target, margin=margin,
                                                        reduction='none')
    elif mode == 'push_loss2':
        #  这里增加正交约束，使得x与y正交，无margin
        ##  loss = |1-d|
        sim_val = (x * y).sum(dim=-1)  # [-1, 1]
        loss = torch.abs(sim_val)
    elif mode == 'push_loss3':
        #  这里增加正交约束，使得x与y正交，有margin
        ##  loss = |1-d|
        sim_val = (x * y).sum(dim=-1)  # [-1, 1]
        dist_val = 1 - torch.abs(sim_val)    # [0, 1]
        target = -torch.ones(b, device=x.device).long()
        loss = torch.nn.functional.hinge_embedding_loss(dist_val, target=target, margin=0.999,
                                                        reduction='none')
    elif mode == 'push_loss4':
        #  使得的相似度越小越好，无margin
        ##  loss = |1-d|
        sim_val = (x * y).sum(dim=-1)  # [-1, 1]
        loss = sim_val + 1  # [0, 2]
    elif mode == 'batch_push_loss':
        #  考虑一个batch，xs,ys正交，xs类内一致
        ##1  正负样本之间，不相似
        dot_prod_neg = torch.matmul(x, y.t())
        loss_neg = torch.abs(dot_prod_neg).mean()
        loss = loss_neg
    elif mode == 'batch_push_pull_loss':
        #  考虑一个batch，xs,ys正交，xs类内一致
        ##1  batch正负样本之间，不相似
        dot_prod_neg = torch.matmul(x, y.t())
        loss_neg = torch.abs(dot_prod_neg).mean()
        ##2  batch正样本之间，自相似
        dot_prod_pos = torch.matmul(x, x.t())
        loss_pos = 1 - dot_prod_pos.mean()
        loss = loss_pos + loss_neg
    else:
        raise NotImplementedError(mode)
    return loss


def loss_fn(x, y, mode='cos_loss',margin=1):
    if mode == 'cos_loss':
        x = F.normalize(x, dim=-1, p=2)
        y = F.normalize(y, dim=-1, p=2)
        loss = 2 - 2 * (x * y).sum(dim=-1)
    elif mode == 'push_loss1' or mode == 'push_loss2' or mode == 'batch_push_pull_loss'\
            or mode == 'batch_push_loss' or mode == 'push_loss3' or mode == 'push_loss4':
        loss = push_loss_fn(x, y, mode=mode, margin=margin)
    elif mode == 'abs':
        loss = torch.abs(x - y)
    else:
        raise NotImplementedError(mode)
    return loss


def loss_fn_ms(xs, ys, mode='cos_loss', margin=1):
    if isinstance(xs, list) or isinstance(xs, tuple):
        assert len(xs) == len(ys)
        loss = 0
        nums = len(xs)
        for x, y in zip(xs, ys):
            loss += loss_fn(x, y, mode, margin)
        loss = loss/nums
    elif isinstance(xs, torch.Tensor):
        loss = loss_fn(xs, ys)
    else:
        raise NotImplementedError
    return loss


def dist_func(x, y, mode='dif', norm=True):
    if norm:
        x = F.normalize(x, dim=-1, p=2)
        y = F.normalize(y, dim=-1, p=2)
    if mode == 'dif':
        dist = x - y
    elif mode == 'cos':
        dist = (x * y).sum(dim=-1)
    elif mode == 'abs':
        dist = torch.abs(x - y)
    elif mode == 'add/2':
        dist = (x + y) / 2
    else:
        raise NotImplementedError(mode)
    return dist


def dist_func_ms(xs, ys, mode='dif', norm=True):
    if isinstance(xs, list) or isinstance(xs, tuple):
        assert len(xs) == len(ys)
        dist = []
        for x, y in zip(xs, ys):
            dist.append(dist_func(x, y, mode, norm))
    elif isinstance(xs, torch.Tensor):
        dist = dist_func(xs, ys, mode, norm)
    else:
        raise NotImplementedError
    return dist


def sim_func(pred, target, pmask, tmask, pos_weight=None, norm_mode='after'):
    """
    计算pred与target在指定mask之间的相似性损失，
    pred, target: N*C*H*W
    pmask, tmask: N*H*W
    """
    N, C, H, W = pred.shape
    assert pred.shape == target.shape
    pmask = pmask.unsqueeze(dim=1)
    pmask = pmask.view(N, 1, -1)
    tmask = tmask.unsqueeze(dim=1)
    tmask = tmask.view(N, 1, -1)
    pred = pred.view(N, C, -1)
    target = target.view(N, C, -1)

    # Weighting of the loss
    if pos_weight is None:
        # 统计一个batch中的正负类样本，计算weight
        #  label：正负两类
        num_labels_pos = torch.sum(pmask) + torch.sum(tmask)
        num_labels_neg = torch.sum(1.0 - pmask) + torch.sum(1.0 - tmask)
        num_total = num_labels_pos + num_labels_neg
        w = num_labels_neg / num_total

    else:
        w = pos_weight
    # TODO: batch之间是否算
    #  如果存在mask==1的区域，需要计算相似性loss
    pmask_pos_nums = (pmask == 1).sum(-1) + 0.01
    tmask_pos_nums = (tmask == 1).sum(-1) + 0.01
    pred = (w * pred * pmask).sum(dim=-1) / pmask_pos_nums  # out: N * C
    target = (w * target * tmask).sum(dim=-1) / tmask_pos_nums
    # TODO: 确认pool之后的特征还是否需要再做归一化
    if norm_mode == 'after':
        pred = F.normalize(pred, dim=1, p=2)
        target = F.normalize(target, dim=1, p=2)

    loss = 1 - 1 * (pred * target).sum() / (N)

    return loss


def sim_loss_with_mask_one(pred1, target2, pmask1, tmask2,
                       class_num=2, weight=None, ignore_id=[0],
                        norm_mode='after'):
    """
    ref: https://github.com/deepmind/detcon/blob/main/utils/losses.py
    Args:
    pred1 (tensor): the prediction from first view.
    target2 (tensor): the projection from second view.
    pmask1 (tensor): mask indices for first view's prediction.
    tmask2 (tensor): mask indices for second view's projection.
    Returns:
    相似性损失，带mask的；
    """
    B, C, H, W = pred1.shape
    assert pred1.shape == target2.shape
    assert pmask1.shape == tmask2.shape

    pmask1_onehot = class2one_hot(pmask1, C=class_num)
    tmask2_onehot = class2one_hot(tmask2, C=class_num)

    if norm_mode == 'before':
        pred1 = F.normalize(pred1, dim=1, p=2)
        target2 = F.normalize(target2, dim=1, p=2)

    loss = 0
    for i in range(class_num):
        if i in ignore_id:
            #  对于某些class id 需要ignore
            continue
        # TODO: 完善weight机制
        #  现在weight=None时，表达出来的意思是，
        #  对于每个mask==c, 负类为其他不为c的样本，这样权重是否合理，发现出来的结果数值更小一点。相比于weight==1
        loss += sim_func(pred1, target2, pmask=pmask1_onehot[:, i],
                         tmask=tmask2_onehot[:, i],
                         pos_weight=weight)

    return loss


def sim_loss_with_mask(pred1, pred2, target1, target2, pmask1, pmask2, tmask1, tmask2,
                       class_num=2, weight=None):
    sim_loss1 = sim_loss_with_mask_one(pred1, target2, pmask1, tmask2,
                                       weight=weight, class_num=class_num)
    sim_loss2 = sim_loss_with_mask_one(pred2, target1, pmask2, tmask1,
                                       weight=weight, class_num=class_num)
    sim_loss = sim_loss1 + sim_loss2

    return sim_loss


def define_loss(loss_name):
    if loss_name == 'sim_loss':
        loss = sim_loss_with_mask_one

    else:
        return NotImplementedError(loss_name)
    return loss

