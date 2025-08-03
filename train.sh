#!/usr/bin/env bash
# use multiple GPUs to train the vertebrae detection model 

# Specify which GPUs to use (modify as needed). And this number should match the number of GPUs(nproc_per_node) you have available.
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5

PYTHONPATH="$(dirname $0)":$PYTHONPATH \
python -m torch.distributed.launch \
    --nproc_per_node=6 \
    --master_port=29500 \
    /train/train_mmdet.py \
    /train/det_cascade-mask-rcnn.py \
    --work-dir work-dir/ \
    --launcher pytorch
    