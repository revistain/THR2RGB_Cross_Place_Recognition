# branch: BASELINE
NCCL_P2P_DISABLE=1 \
OMP_NUM_THREADS=8 \
CUDA_VISIBLE_DEVICES=3 \
python3 train_wandb.py \
    --save_dir './logs' \
    --features_dim 768 \
    --sequences KAIST \
    --foundation_model_path /home/jwkim/workspace/THR2RGB_Cross_Place_Recognition/backbone/dinov2/pretrained/dinov2_vitb14_pretrain.pth \
    --queries_per_epoch 2000 \
    --num_trainable_blocks_RGB 4 \
    --num_trainable_blocks_THERMAL 4 \
    --comment "CroCo-RGBrecon-maskRatio0.9-reconWeight10-decDepth8" \
    --margin 0.1 \
    --epochs_num 100 \
    --negs_num_per_query 10 \
    --croco_mask_ratio 0.9 \
    --num_decoder_depth 8 \
    --recon_weight 10 \
    --attention_mask_type 'none' \
    --use_pos_as_paired_rgb
    # --use_reranking \
    # --use_constrastive_recon_loss \
    # --use_confidence_map \
    # --use_reranking \
    # --recon_loss_fn_type "MAE"
    # --use_ssim_recon_loss
    # --use_only_cross_decoder
    # --use_decode_mask \
    # --num_workers 0 \
    # --use_feature_level_recon_loss \
    # --use_bireconstruction \


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