OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=2 python3 train_wandb.py \
    --save_dir './logs' \
    --features_dim 768 \
    --sequences KAIST \
    --foundation_model_path /home/sjkwon/workspace/VPR/THR2RGB_Cross_Place_Recognition/backbone/dinov2/pretrained/dinov2_vitb14_pretrain.pth \
    --queries_per_epoch 2000 \
    --num_trainable_blocks 4 \
    --lr 1e-4 \
    --backbone_lr 1e-5 \
    --comment "baseline_clip_cls_al0.5_GeMLinear[784,ReLU,784]_seperateLR[1e-5,1e-4]" \
    --use_alignment_loss \
    --use_sepearte_backbone_lr \
    --use_GeMAdditionalLayer

#    --use_sepearte_backbone_lr
#    --use_alignment_loss
#    --backbone_lr 1e-5
#    --use_GeMAdditionalLayer