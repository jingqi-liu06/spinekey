#!/usr/bin/env bash

PYTHONPATH="$(dirname $0)":$PYTHONPATH \
python -m torch.distributed.launch \
    --nproc_per_node=6 \
    --master_port=29500 \
    /train/train_mmdet.py \
    /train/det_cascade-mask-rcnn.py \
    --work-dir work-dir/ \
    --launcher pytorch
    