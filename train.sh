OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 python3 train_wandb.py \
    --save_dir './logs' \
    --features_dim 768 \
    --sequences KAIST \
    --foundation_model_path /home/sjkwon/workspace/VPR/THR2RGB_Cross_Place_Recognition/backbone/dinov2/pretrained/dinov2_vitb14_pretrain.pth \
    --queries_per_epoch 2000 \
    --num_trainable_blocks 4 \
    --comment baseline_clip_cls_al0.5 \
    --use_alignment_loss

#    --use_alignment_loss
#    --use_GeMAdditionalLayer