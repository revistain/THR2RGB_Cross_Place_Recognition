CUDA_VISIBLE_DEVICES=4

NCCL_P2P_DISABLE=1 \
OMP_NUM_THREADS=4 \
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
python3 fast_inference.py \
    --save_dir './logs' \
    --cuda_device $CUDA_VISIBLE_DEVICES \
    --foundation_model_path /home/jwkim/workspace/THR2RGB_Cross_Place_Recognition/backbone/dinov2/pretrained/dinov2_vits14_pretrain.pth \
    --resume "/home/jwkim/workspace/THR2RGB_Cross_Place_Recognition/logs/biRecon(w10,l1, dep4)-warmup(3e-5)/260213_034204/best_model.pth" \
    --test_seq KAIST \
    --comment "224x224" \
    --croco_mask_ratio 0.8 \
    --recon_weight 10 \
    --use_recon_loss \
    --recon_loss_type 'mse+ssim' \
    --num_decoder_depth 8 