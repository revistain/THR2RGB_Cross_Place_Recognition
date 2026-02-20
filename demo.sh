#!/bin/bash

CUDA_VISIBLE_DEVICES=3 \
NCCL_P2P_DISABLE=1 \
OMP_NUM_THREADS=8 \
python create_demo_video.py \
    --resume "/home/jwkim/workspace/THR2RGB_Cross_Place_Recognition/logs/twoStage-CroCo-ViTs-numBlock0-distanceDecoder-1e-4(neg,tau10)/260219_113028/best_model.pth" \
    --foundation_model_path "/home/jwkim/workspace/THR2RGB_Cross_Place_Recognition/backbone/dinov2/pretrained/dinov2_vits14_pretrain.pth" \
    --datasets_folder "/home/jwkim/workspace/THR2RGB_Cross_Place_Recognition/Dataset/save_mat" \
    --sequences Urban \
    --img_time daytime \
    --use_reranking distance \
    --output_path demo_urban.mp4
