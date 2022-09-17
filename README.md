# SaDL_CD
Here, we provide the pytorch implementation of the paper: Semantic-Aware Dense Representation Learning for Remote Sensing Image Change Detection.

For more ore information, please see our published paper at [IEEE TGRS](https://ieeexplore.ieee.org/document/9874899/) or [arxiv](https://arxiv.org/abs/2205.13769). 

![overview](images/overview.png)

## Requirements

```
Python 3.7
pytorch 1.10.1
torchvision 0.11.2
einops  0.3.0
kornia 0.6.3
```

## Installation

Clone this repo:

```shell
git clone https://github.com/justchenhao/SaDL_CD.git
cd SaDL_CD
```

## Quick Start

You can simply run `python main_ssl.py` to train our model (``sadl_fpn_m2_resnet18_sample16_syn1`) on the given small samples (in the folder `samples`).

## Training

You can find the training script `train_ssl.sh` in the folder `scripts`. You can run the script file by `sh scripts/train_ssl.sh` in the command environment.

The detailed script file `train_ssl.sh` is as follows:

```shell
#!/usr/bin/env bash

gpus=0

checkpoint_root=checkpoints
dataset_type=SegDataset
img_size=256
batch_size=64
optim_mode=sgd
lr_policy=poly

lr=0.01
max_epochs=200
net_G=sadl_fpn_m2_resnet18_sample16_syn1


data_name=inria256

split=pos0.1_train
split_val=pos0.1_val

project_name=SSLM_${net_G}_${data_name}_b${batch_size}_lr${lr}_${split}_${split_val}_${max_epochs}_${lr_policy}_${optim_mode}

python main_ssl.py --dataset_type ${dataset_type} --img_size ${img_size} --optim_mode ${optim_mode}  --checkpoint_root ${checkpoint_root} --lr_policy ${lr_policy} --split ${split} --split_val ${split_val} --net_G ${net_G} --gpu_ids ${gpus} --max_epochs ${max_epochs} --project_name ${project_name} --batch_size ${batch_size} --data_name ${data_name}  --lr ${lr}
```

## Dataset Preparation

### Pretraining dataset

We leverage image-mask pairs from the existing Inria building segmentation dataset. We cut the original samples into small patches of size 256 × 256.  We additionally obtain the coregistered image patch of the corresponding geospatial region.

The original Inria building segmentation dataset can be found at: https://project.inria.fr/aerialimagelabeling/

Our processed pretraining dataset can be accessed by Baidu yun (code: vldo): [link](https://pan.baidu.com/s/1mEy5RfomClXWdi9I28Rp8A)

Note that you need only the image-mask pairs in `A` and `label` to train our model. We also provide the spatially registered image of another temporal in `B` for possible usage. 

#### Data structure

```
"""
The pretraining data set with bitemporal images and building mask for one temporal. Note that the masks in the folder 'label' are aligned with the corresponding images in folder 'A'；
├─A
├─B
├─label
└─list
"""
```

#### Data Download 

### Downstream Datasets

We test our pretrained model at three downstream change detection datasets.

#### Data structure

```
"""
Change detection data set with pixel-level binary labels；
├─A
├─B
├─label
└─list
"""
```

#### Data Download 

LEVIR-CD: https://justchenhao.github.io/LEVIR/

WHU-CD: https://study.rsgis.whu.edu.cn/pages/download/building_dataset.html

GZ-CD: https://github.com/daifeng2016/Change-Detection-Dataset-for-High-Resolution-Satellite-Imagery

## License

Code is released for non-commercial and research purposes **only**. For commercial purposes, please contact the authors.

## Citation

If you use this code for your research, please cite our paper:

```
@Article{chen2022,
    title={Semantic-Aware Dense Representation Learning for Remote Sensing Image Change Detection},
    author={Hao Chen, Wenyuan Li, Song Chen and Zhenwei Shi},
    year={2022},
    journal={IEEE Transactions on Geoscience and Remote Sensing},
    volume={},
    number={},
    pages={1-18},
    doi={10.1109/TGRS.2022.3203769}
}
```

