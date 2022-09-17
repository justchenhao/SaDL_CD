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

