CUDA_VISIBLE_DEVICES=0

NCCL_P2P_DISABLE=1 \
OMP_NUM_THREADS=4 \
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
python3 train_wandb_2stage.py \
    --cuda_device $CUDA_VISIBLE_DEVICES \
    --foundation_model_path /home/jwkim/workspace/THR2RGB_Cross_Place_Recognition/backbone/dinov2/pretrained/dinov2_vits14_pretrain.pth \
    --resume '/home/jwkim/workspace/THR2RGB_Cross_Place_Recognition/logs/CroCo-beforeA6000(Sthereo,lr3e-5)-posAsPaired/260228_190611/best_model.pth' \
    --save_dir './logs' \
    --lr 1e-4 \
    --train_seq Campus \
    --test_seq Urban Residential \
    --comment "2stage-gmrw-warp" \
    --epochs_num 50 \
    --use_recon_loss \
    --recon_loss_type 'l1' \
    --num_decoder_depth 8 \
    --use_pos_as_aligned_rgb \
    --affinity_dim 384 \
    --use_2stage \
    --gmrw_weight 1.0 \
    --use_label_warp \
    --score_method trace \
    --freeze_encoder

# Optional: train with negatives for margin loss
# --train_with_negatives \
# --neg_margin 0.1

# Optional: smoothness loss
# --use_smoothness_loss \
# --smoothness_weight 0.1
