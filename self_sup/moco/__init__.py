# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from torch import nn
import torch
import torchvision.transforms as transforms


from self_sup.moco.builder import MoCo
from self_sup.moco.loader import GaussianBlur_tensor, TwoCropsTransform

from kornia import augmentation as augs

normalize = augs.Normalize(mean=torch.tensor([0.5, 0.5, 0.5]),
                                 std=torch.tensor([0.5, 0.5, 0.5]))


def get_moco_aug(image_size=256, aug_plus=True):
    if aug_plus:
        augmentation = [
            augs.RandomResizedCrop((image_size, image_size), scale=(0.2, 1.)),
            transforms.RandomApply([
                augs.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            augs.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur_tensor([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            # transforms.ToTensor(),
            normalize
        ]
    else:
        # MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978
        augmentation = [
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]

    aug_func = TwoCropsTransform(transforms.Compose(augmentation))
    return aug_func


class MoCoV2(nn.Module):
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False,
                 aug_plus=True, image_size=256):
        super(MoCoV2, self).__init__()
        self.moco = MoCo(base_encoder, dim=dim, K=K, m=m, T=T, mlp=mlp)

        self.aug_func = get_moco_aug(image_size=image_size, aug_plus=aug_plus)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x_one, x_two = self.aug_func(x)
        logits, labels = self.moco(x_one, x_two)

        loss = self.criterion(logits, labels)
        return loss

    def on_before_zero_grad(self):
        pass

    def trainable_parameters(self):
        pretrain_parameters = list(self.moco.parameters())
        return pretrain_parameters

    def on_after_batch_transfer(self, batch):
        return batch

    def training_step(self, batch, batch_id):
        dict_loss = {}
        dict_loss['loss'] = self.forward(batch['A'])
        return dict_loss


from self_sup.moco.builder import SimSiam


class SimSiamV1(nn.Module):
    """
    """
    def __init__(self, base_encoder, dim=2048, pred_dim=512, image_size=256, aug_plus=True):
        super(SimSiamV1, self).__init__()

        self.simsiam = SimSiam(base_encoder, dim=dim, pred_dim=pred_dim)
        self.aug_func = get_moco_aug(image_size=image_size, aug_plus=aug_plus)
        self.criterion = nn.CosineSimilarity(dim=1)

    def forward(self, x):
        x_one, x_two = self.aug_func(x)
        p1, p2, z1, z2 = self.simsiam(x_one, x_two)
        loss = 1/2 * (self.criterion(p1, z2).mean()+self.criterion(p2, z1).mean())

        return loss

    def on_after_batch_transfer(self, batch):
        return batch

    def training_step(self, batch, batch_id):
        dict_loss = {}
        dict_loss['loss'] = self.forward(batch['A'])
        return dict_loss

    def on_before_zero_grad(self):
        pass

    def trainable_parameters(self):
        """
        以元组形式返回可训练的参数,包括两部分，encoder参数（预训练参数），以及其他参数（随机初始化）
        :return: tuple (0,1)
        """
        pretrain_parameters = list(self.simsiam.encoder.parameters())
        others_parameters = list(self.simsiam.predictor.parameters())
        return (pretrain_parameters, others_parameters)


from self_sup.seco.seco_main import SeCo
import torch.nn.functional as F


class SeCoMain(nn.Module):
    """
    """
    def __init__(self, base_encoder, emb_dim=128, image_size=256, aug_plus=True):
        super(SeCoMain, self).__init__()

        self.seco = SeCo(base_encoder, emb_dim=emb_dim)
        self.augment = transforms.Compose([
            transforms.RandomApply([
                augs.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur_tensor([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
        ])
        self.preprocess = transforms.Compose([
            augs.RandomResizedCrop((image_size, image_size), scale=(0.2, 1.)),
            # transforms.ToTensor(),
            normalize
        ])

    def forward(self, t0, t1=None):
        # TODO: tmp for 计算复杂度
        if t1 is None:
            t1 = 1 - t0
        q = t0
        k0 = self.augment(t1)
        k1 = t1
        k2 = self.augment(t0)

        q = self.preprocess(q)
        k0 = self.preprocess(k0)
        k1 = self.preprocess(k1)
        k2 = self.preprocess(k2)

        output, target = self.seco(q, [k0, k1, k2])
        losses = []
        for out in output:
            losses.append(F.cross_entropy(out.float(), target.long()))
        loss = torch.sum(torch.stack(losses))
        loss = loss / 3
        return loss

    def on_after_batch_transfer(self, batch):
        return batch

    def training_step(self, batch, batch_id):
        dict_loss = {}
        dict_loss['loss'] = self.forward(batch['A'], batch['B'])
        return dict_loss

    def on_before_zero_grad(self):
        pass

    def trainable_parameters(self):
        """
        以元组形式返回可训练的参数,包括两部分，encoder参数（预训练参数），以及其他参数（随机初始化）
        :return: tuple (0,1)
        """
        pretrain_parameters = list(self.seco.parameters())
        return pretrain_parameters


from self_sup.densecl.densecl import DenseCL


class DenseCLMain(nn.Module):
    def __init__(self, backbone_name='resnet18', pretrained='imagenet', image_size=256):
        super(DenseCLMain, self).__init__()
        self.densecl = DenseCL(pretrained='imagenet', backbone_name='resnet18')
        self.aug_func = get_moco_aug(image_size=image_size, aug_plus=True)

    def forward(self, x):
        x_one, x_two = self.aug_func(x)
        img = torch.stack([x_one, x_two], dim=1)
        loss_dict = self.densecl(img)
        return loss_dict

    def on_after_batch_transfer(self, batch):
        return batch

    def training_step(self, batch, batch_id):
        dict_loss = self.forward(batch['A'])
        return dict_loss

    def on_before_zero_grad(self):
        pass

    def trainable_parameters(self):
        """
        以元组形式返回可训练的参数,包括两部分，encoder参数（预训练参数），以及其他参数（随机初始化）
        :return: tuple (0,1)
        """
        pretrain_parameters = list(self.densecl.parameters())
        return pretrain_parameters
