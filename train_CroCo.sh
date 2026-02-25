CUDA_VISIBLE_DEVICES=3

NCCL_P2P_DISABLE=1 \
OMP_NUM_THREADS=4 \
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
python3 train_wandb.py \
    --cuda_device $CUDA_VISIBLE_DEVICES \
    --foundation_model_path /home/jwkim/workspace/THR2RGB_Cross_Place_Recognition/backbone/dinov2/pretrained/dinov2_vits14_pretrain.pth \
    --save_dir './logs' \
    --lr 3e-5 \
    --train_seq Campus \
    --test_seq Urban Residential \
    --comment "CroCo-beforeA6000(Sthereo,lr3e-5)-interintraRecon(r0.9, fixed, CLSmask)" \
    --croco_mask_ratio 0.9 \
    --recon_weight 10 \
    --use_recon_loss \
    --recon_loss_type 'l1' \
    --num_decoder_depth 8 \
    --masking_method CLS


    # --use_dist_cls_while_Recon \
    # --distance_weight 1.0

# Scene 종류 : ['Campus', 'Residential', 'Urban', 'KAIST', 'SNU', 'Valley'] 
# recon_loss_type: ['mse', 'l1', 'ssim', 'mse+ssim']

# claude-monitor --plan pro --timezone Asia/Seoul