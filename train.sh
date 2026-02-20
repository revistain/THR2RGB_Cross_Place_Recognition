CUDA_VISIBLE_DEVICES=7

NCCL_P2P_DISABLE=1 \
OMP_NUM_THREADS=8 \
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
python3 train_wandb.py \
    --train_batch_size 4 \
    --save_dir './logs' \
    --train_seq Campus \
    --test_seq Urban Residential \
    --lr 1e-4 \
    --cuda_device $CUDA_VISIBLE_DEVICES \
    --foundation_model_path /home/jwkim/workspace/THR2RGB_Cross_Place_Recognition/backbone/dinov2/pretrained/dinov2_vits14_pretrain.pth \
    --queries_per_epoch 2000 \
    --soft_positives_dist_threshold 10 \
    --hard_positives_dist_threshold 10 \
    --num_trainable_blocks_RGB 0 \
    --num_trainable_blocks_THERMAL 0 \
    --comment "biRecon(w10,l1,dep4,r0.8)-(stage1, 1e-4)" \
    --margin 0.1 \
    --epochs_num 100 \
    --negs_num_per_query 10 \
    --croco_mask_ratio 0.8 \
    --recon_weight 10 \
    --use_recon_loss \
    --num_decoder_depth 4 \
    --recon_loss_type 'l1' \
    --use_warmup \
    --masking_method 'random'

    # --masking_method 'GeM'
    # --use_mlp_dim_before_decoder 768
    # --unfreeze_dino_decoder \
    # --use_diff_loss \
    # --recon_loss_type 'mse+ssim' \
    # --brightness 0.5 \
    # --contrast 0.5 \
    # --saturation 0.5 \
    # --hue 0.1 \

    # --use_pos_as_aligned_rgb \
    # --num_decoder_depth 8 \
    # --rand_perspective 0.1

    # --use_pos_as_aligned_rgb \
    # --img_time 'latetime'
    # --selaVPR_rerank_score_type 'none' \
    # --use_reranking 'selaVPR'

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

    #### DINO decoder
    # --use_dino_decoder \
    # --dino_decoder_layer_start 6 \
    # --dino_decoder_layer_end 12


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
