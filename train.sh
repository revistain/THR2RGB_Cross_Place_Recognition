OMP_NUM_THREADS=6 CUDA_VISIBLE_DEVICES=7 python3 train_wandb.py \
    --save_dir './logs' \
    --features_dim 768 \
    --sequences KAIST \
    --foundation_model_path /home/sjkwon/workspace/VPR/THR2RGB_Cross_Place_Recognition/backbone/dinov2/pretrained/dinov2_vitb14_pretrain.pth \
    --queries_per_epoch 2000 \
    --num_trainable_blocks_RGB 6 \
    --num_trainable_blocks_THERMAL 6 \
    --comment "croco_mask-ratio-0.01_decoder-depth-4_recon-loss-weight-0.1-SmoothL1" \
    --use_rgb_adapter \
    --use_thermal_adapter \
    --croco_mask_ratio 0.75


######### 설명 #########
# comment: wandb 기록명
# use_alignment_loss: alignment loss 사용 여부
# use_sepearte_backbone_lr/backbone_lr: backbone과 나머지 모듈의 learning rate를 다르게 설정
# use_GeMAdditionalLayer: GeM pooling 후에 Linear-ReLU-Linear 추가
# use_rgb_adapter     : adapter 사용여부(없으면 미사용)
# use_thermal_adapter : adapter 사용여부(없으면 미사용)