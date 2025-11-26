OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=7 python3 train.py --save_dir './logs/baseline_freeze(-4)' \
                    --features_dim 768 --sequences KAIST --foundation_model_path ./backbone/dinov2/pretrained/dinov2_vitb14_pretrain.pth \
                    --queries_per_epoch 2000 --num_trainable_blocks 4