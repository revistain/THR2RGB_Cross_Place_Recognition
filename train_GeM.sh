CUDA_VISIBLE_DEVICES=0

NCCL_P2P_DISABLE=1 \
OMP_NUM_THREADS=4 \
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
python3 train_wandb.py \
    --cuda_device $CUDA_VISIBLE_DEVICES \
    --foundation_model_path /DATA2/datasets/PR/STheReO/dinov2_vits14_pretrain.pth \
    --save_dir './logs' \
    --lr 3e-5 \
    --train_seq KAIST \
    --test_seq SNU Valley \
    --comment "GeM-beforeA6000(Sthereo,lr3e-5)"

# Scene 종류 : ['Campus', 'Residential', 'Urban', 'KAIST', 'SNU', 'Valley'] 
# recon_loss_type: ['mse', 'l1', 'ssim', 'mse+ssim']