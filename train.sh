NCCL_P2P_DISABLE=1 \
OMP_NUM_THREADS=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 train_wandb.py \
    --save_dir './logs' \
    --features_dim 384 \
    --sequences KAIST \
    --foundation_model_path /home/jwkim/workspace/THR2RGB_Cross_Place_Recognition/backbone/dinov2/pretrained/dinov2_vits14_pretrain.pth \
    --queries_per_epoch 2000 \
    --num_trainable_blocks_RGB 4 \
    --num_trainable_blocks_THERMAL 4 \
    --comment "476x644-CrocoFixedR2former-0.8-alpha10-decDepth8-reranking" \
    --margin 0.1 \
    --epochs_num 100 \
    --negs_num_per_query 10 \
    --croco_mask_ratio 0.8 \
    --num_decoder_depth 8 \
    --recon_weight 1.6 \
    --recon_loss_type 'mse+ssim' \
    --r2_penultimate_layer \
    --r2_global_local_score \
    --use_reranking \
    --use_recon_loss
    # --use_fast_track 
    
# NPY 안겹치게 잘하자.... 까먹지말고 아님 방지하던가...

    # --use_fast_track \
    # --use_cls_for_vpr \
    # --r2_add_random_patch \
    # --use_fast_track \
    # --rerank_with_RGB \
    # --use_fast_track \
    # --use_only_cross_decoder \
    # --use_contrastive_recon_loss \
    # --ssim-contrastLoss
    # --use_confidence_map
    # --use_ssim_recon_loss
    # --use_only_cross_decoder
    # --use_pos_as_aligned_rgb \
    # --use_decode_mask \
    # --num_workers 0 \
    # --use_feature_level_recon_loss \
    # --use_bireconstruction \

    # confidence_map은 decodeMask에서는 구현안되어 있음 (주의!!)



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

# reranking이 안되는 이유??
# - reconstruction 결과가 다 뭉개짐(thermal이라서?)
# - thermal의 normalization 문제? (히스토그램 shift)
# - reconstruction이 다 뭉개지니까 pixel level에서 명확한 비교가 안됨
# - decoder가 가벼우니까, encoder에서 최대한 압축된 정보를 만들어내려고함
# - 그럼 pixel 레벨 다 버려짐 