OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 python3 train_wandb.py \
    --save_dir './logs' \
    --features_dim 768 \
    --sequences KAIST \
    --foundation_model_path /home/sjkwon/workspace/VPR/THR2RGB_Cross_Place_Recognition/backbone/dinov2/pretrained/dinov2_vitb14_pretrain.pth \
    --queries_per_epoch 2000 \
    --num_trainable_blocks_RGB 6 \
    --num_trainable_blocks_THERMAL 6 \
    --comment "baseline_test" \
    --use_rgb_adapter \
    --use_thermal_adapter \
    --use_alignment_loss

#    --use_sepearte_backbone_lr
#    --backbone_lr 1e-5
#    --lr 1e-5
#    --use_alignment_loss
#    --use_GeMAdditionalLayer
#    --comment "baseline_clip-cls-loss_al0.5_GeMLinear[784,ReLU,784]_seperateLR[1e-5,1e-4]" \
#    --comment "baseline_alignment-attn[dot_product,rgb-attn-map]-loss_al1.0_num-train-block6" \
#    --negs_num_per_query 10
#    --margin 0.1

######### 설명 #########
# comment: wandb 기록명
# use_alignment_loss: alignment loss 사용 여부
# use_sepearte_backbone_lr/backbone_lr: backbone과 나머지 모듈의 learning rate를 다르게 설정
# use_GeMAdditionalLayer: GeM pooling 후에 Linear-ReLU-Linear 추가
    # --use_rgb_adapter
    # --use_thermal_adapter