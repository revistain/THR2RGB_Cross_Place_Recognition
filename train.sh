OMP_NUM_THREADS=6 CUDA_VISIBLE_DEVICES=4 python3 train_wandb.py \
    --save_dir './logs' \
    --features_dim 768 \
    --sequences KAIST \
    --foundation_model_path /home/jwkim/workspace/THR2RGB_Cross_Place_Recognition/backbone/dinov2/pretrained/dinov2_vitb14_pretrain.pth \
    --queries_per_epoch 2000 \
    --num_trainable_blocks_RGB 4 \
    --num_trainable_blocks_THERMAL 4 \
    --use_reduced_thermal_patch \
    --comment "croco-0.5-alpha10.0-decDepth2" \
    --margin 0.1 \
    --epochs_num 100 \
    --negs_num_per_query 10 \
    --croco_mask_ratio 0.5 \
    --num_decoder_depth 2 \
    --recon_weight 10 \
    --use_reranking
    # --rerank_weight 1 \
    # --use_rerank_loss
    # --debug_subset 1000

    # --use_single_pass \

#    --use_sepearte_backbone_lr
#    --backbone_lr 1e-5
#    --lr 1e-5
#    --use_alignment_loss
#    --use_GeMAdditionalLayer
#    --comment "baseline_clip-cls-loss_al0.5_GeMLinear[784,ReLU,784]_seperateLR[1e-5,1e-4]" \
#    --comment "baseline_alignment-attn[dot_product,rgb-attn-map]-loss_al1.0_num-train-block6" \

######### 설명 #########
# comment: wandb 기록명
# use_alignment_loss: alignment loss 사용 여부
# use_sepearte_backbone_lr/backbone_lr: backbone과 나머지 모듈의 learning rate를 다르게 설정
# use_GeMAdditionalLayer: GeM pooling 후에 Linear-ReLU-Linear 추가