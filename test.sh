CUDA_VISIBLE_DEVICES=4

NCCL_P2P_DISABLE=1 \
OMP_NUM_THREADS=8 \
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
python3 fast_inference.py \
    --save_dir './logs' \
    --test_seq Urban Residential \
    --cuda_device $CUDA_VISIBLE_DEVICES \
    --foundation_model_path /home/jwkim/workspace/THR2RGB_Cross_Place_Recognition/backbone/dinov2/pretrained/dinov2_vits14_pretrain.pth \
    --resume "/home/jwkim/workspace/THR2RGB_Cross_Place_Recognition/logs/biRecon(w10,l1,dep8,r0.6)-attnMask(stage1, 1e-4)/260217_062526/last_model.pth" \
    --queries_per_epoch 2000 \
    --num_trainable_blocks_RGB 0 \
    --num_trainable_blocks_THERMAL 0 \
    --soft_positives_dist_threshold 10 \
    --hard_positives_dist_threshold 10 \
    --comment "224x224-Croco-ViTs-numBlock0-inference-pairVPR" \
    --margin 0.1 \
    --epochs_num 100 \
    --negs_num_per_query 10 \
    --croco_mask_ratio 0.8 \
    --recon_weight 10 \
    --use_recon_loss \
    --use_reranking 'recon' \
    --recon_loss_type 'l1' \
    --masking_method 'random' \
    --num_decoder_depth 8 \
    --visualize_attention

    # --selaVPR_rerank_score_type 'none' \
    # --use_reranking 'reconSelaVPR' \

    #### Recon Reranking (bidirectional reconstruction loss)
    # --use_reranking 'recon' \
    # --use_gem_recon_weight \  # Enable GeM-weighted loss (default: off)

    #### Croco
    # --croco_mask_ratio 0.8 \
    # --recon_weight 10 \
    # --use_recon_loss \
    # --recon_loss_type 'mse+ssim' \
    # --num_decoder_depth 8

    #### r2former
    # --r2_penultimate_layer \
    # --r2_add_random_patch \
    # --use_reranking 'r2former' \
    # --use_cls_for_vpr \

    #### selaVPR
    # --use_reranking 'selaVPR' \
    # --match_conf_weight \
    # --match_conf_top_k \
    # --match_conf_embed_dim \
    # --use_selaVPR_attn_score \
    ###
    # --use_reranking 'match_conf' \


    # --use_fast_track \
    # --use_selaLocalFeature \
    # --use_recon_loss
    # --r2_add_random_patch \
    # --rerank_with_RGB \
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


## ~/.local/share/claude/versions/2.1.19
