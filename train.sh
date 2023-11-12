#! bin/bash

#   --test-dataset2 TSUNAMI_total \
CUDA_VISIBLE_DEVICES=0 python train.py \
    --seed 0 \
    --model resnet18_mtf_msf_fcn \
    --train-dataset VL_CMU_CD \
    --test-dataset VL_CMU_CD \
    --test-dataset3 PSCD_total \
    --input-size 512 \
    --batch-size 4 \
    --lr 0.0001 \
    --lr-scheduler cosine \
    --data-cv 0 \
    --mtf id \
    --cycle \
    --msf 4 \
    --warmup \
    --loss-weight 