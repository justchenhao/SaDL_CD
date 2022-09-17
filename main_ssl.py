from argparse import ArgumentParser
import torch
import os

import utils
from datasets import get_loaders
print(torch.cuda.is_available())
from self_sup.ssl_trainer import SSLTrainer
from misc.torchutils import seed_torch

"""
自监督预训练的入口函数
"""


def train(args):
    kwargs = args.__dict__
    dataloaders = get_loaders(**kwargs)
    model = SSLTrainer(args=args, dataloaders=dataloaders)
    model.train_models()


seed_torch(seed=2021)


def update_num_workers(args):
    args.num_workers = args.num_workers if args.num_workers < args.batch_size \
        else args.batch_size
    print('num_workers: %d' % args.num_workers)


if __name__ == '__main__':
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--project_name', default='test', type=str)
    parser.add_argument('--checkpoint_root', default='checkpoints', type=str)

    # logger
    parser.add_argument('--with_wandb',  type=bool, default=False,
                        help="use wandb or not")
    parser.add_argument('--project_task',  type=str, default='pretrain',
                        help="which task is it?: cd? seg? ssl?, pretrain")

    # data
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--persistent_workers', default=False, type=bool)

    parser.add_argument('--dataset_type', default='SegDataset', type=str,
                        help='SegDataset, CDDataset, BiImageDataset, ImageDataset')
    parser.add_argument('--data_name', default='get_start', type=str,
                        help='inria256 | get_start')
    parser.add_argument('--n_class', default=2, type=int)
    parser.add_argument('--val_data_norm', default=False, type=bool,
                        help='ssl任务中，train和val dataset出来的data都不需要norm')

    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--split', default="get_start", type=str,
                        help='get_start | pos0.1_train')
    parser.add_argument('--split_val', default="get_start", type=str,
                        help='get_start | pos0.1_val')

    parser.add_argument('--img_size', default=256, type=int)

    # model
    parser.add_argument('--net_G', default='sadl_fpn_m2_resnet18_sample16_syn1', type=str,
                        help="sadl_fpn_m2_resnet18_sample16_syn1 | sadl_fpn_m2_resnet18_sample16_syn2")

    # init backbone parameter from pretrained
    parser.add_argument('--pretrained', default='imagenet', type=str,
                        help="imagenet | None | pretrain_path")
    parser.add_argument('--num_epoch_to_save_best', default=50, type=int,
                        help='save best checkpoint every n epoch')

    # optimizer
    parser.add_argument('--optim_mode', default='sgd', type=str)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--max_epochs', default=200, type=int)
    parser.add_argument('--lr_policy', default='linear', type=str,
                        help='linear | step')
    parser.add_argument('--lr_decay_iters', default=100, type=int)

    args = parser.parse_args()
    args.gpu_ids = utils.get_device(args.gpu_ids)
    print(args.gpu_ids)

    #  update pretrained path
    import data_config
    args.pretrained = data_config.get_pretrained_path(args.pretrained)
    update_num_workers(args)

    #  checkpoints dir
    args.checkpoint_dir = os.path.join(args.checkpoint_root, args.project_name)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    train(args)




