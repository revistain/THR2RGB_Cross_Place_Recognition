NCCL_P2P_DISABLE=1 \
OMP_NUM_THREADS=8 \
CUDA_VISIBLE_DEVICES=3 \
python3 attn_map_match.py \
    --save_dir './logs' \
    --features_dim 768 \
    --sequences KAIST \
    --foundation_model_path /home/jwkim/workspace/THR2RGB_Cross_Place_Recognition/backbone/dinov2/pretrained/dinov2_vitb14_pretrain.pth \
    --queries_per_epoch 2000 \
    --num_trainable_blocks_RGB 4 \
    --num_trainable_blocks_THERMAL 4 \
    --margin 0.1 \
    --epochs_num 100 \
    --negs_num_per_query 10 \
    --croco_mask_ratio 0.8 \
    --num_decoder_depth 8 \
    --recon_weight 10 \
    --recon_loss_type 'mse' \
    --resume /home/jwkim/workspace/THR2RGB_Cross_Place_Recognition/logs/test-croco-0.8-alpha10.0-decDepth8-bidirectional-mse-reranking/260114_040035/last_model.pth \
    --use_reranking