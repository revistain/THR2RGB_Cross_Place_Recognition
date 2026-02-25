# Parser.py
# Argument parser for cross-modal VPR
import os
import argparse
import yaml


class Parser():
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description=None,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )

        # Training settings
        self.parser.add_argument("--optim", type=str, default='adam', choices=['adam', 'sgd'])
        self.parser.add_argument("--margin", type=float, default=0.1)
        self.parser.add_argument("--lr", type=float, default=0.00001)
        self.parser.add_argument("--epochs_num", type=int, default=50)
        self.parser.add_argument("--train_batch_size", type=int, default=4)
        self.parser.add_argument("--patience", type=int, default=5)

        self.parser.add_argument("--soft_positives_dist_threshold", type=int, default=10)
        self.parser.add_argument("--hard_positives_dist_threshold", type=int, default=10)

        self.parser.add_argument("--mining", type=str, default="partial", choices=["partial", "full", "random"])
        self.parser.add_argument("--cache_refresh_rate", type=int, default=1000)
        self.parser.add_argument("--queries_per_epoch", type=int, default=2000)
        self.parser.add_argument("--negs_num_per_query", type=int, default=10)
        self.parser.add_argument("--neg_samples_num", type=int, default=1000)
        self.parser.add_argument("--num_trainable_blocks", type=int, default=0, help="number of trainable blocks")

        # Inference settings
        self.parser.add_argument("--infer_batch_size", type=int, default=64)
        self.parser.add_argument('--test_method', type=str, default="hard_resize",
                            choices=["hard_resize", "single_query", "central_crop", "five_crops", "nearest_crop", "maj_voting"])

        # Evaluation settings
        self.parser.add_argument("--majority_weight", type=float, default=0.01)
        self.parser.add_argument('--recall_values', type=int, default=[1, 5, 10, 20], nargs="+")

        # Model settings
        self.parser.add_argument('--resize', type=int, default=[224, 224], nargs=2)
        self.parser.add_argument("--foundation_model_path", type=str, default=None)
        self.parser.add_argument("--features_dim", type=int, default=768)

        # Dataset parameters
        self.parser.add_argument("--img_time", type=str, default="allday",
                            choices=["allday", "daytime", "nighttime", "latetime"])
        self.parser.add_argument("--train_seq", type=str, default=None, nargs='+')
        self.parser.add_argument("--test_seq", type=str, default=None, nargs='+')

        # Data augmentation
        self.parser.add_argument("--brightness", type=float, default=None)
        self.parser.add_argument("--contrast", type=float, default=None)
        self.parser.add_argument("--saturation", type=float, default=None)
        self.parser.add_argument("--hue", type=float, default=None)
        self.parser.add_argument("--rand_perspective", type=float, default=None)
        self.parser.add_argument("--random_resized_crop", type=float, default=None)
        self.parser.add_argument("--random_rotation", type=float, default=None)
        self.parser.add_argument("--horizontal_flip", action='store_true')

        # Record settings
        self.parser.add_argument("--config", type=str, default=None)
        self.parser.add_argument("--save_dir", type=str, default="default")

        # Common settings
        self.parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
        self.parser.add_argument("--num_workers", type=int, default=8)
        self.parser.add_argument("--seed", type=int, default=42)
        self.parser.add_argument("--resume", type=str, default=None, nargs='*')
        self.parser.add_argument("--save_all", type=bool, default=False)

        # Custom settings
        self.parser.add_argument('--comment', type=str)
        self.parser.add_argument('--cuda_device', type=str)

        # Reconstruction loss settings (CroCo decoder)
        self.parser.add_argument("--use_recon_loss", action='store_true', default=False) # CroCo 사용여부
        self.parser.add_argument("--recon_loss_type", type=str, default="mse", # reconstruction loss 계산방식
                            choices=['mse', 'l1', 'ssim', 'mse+ssim'])
        self.parser.add_argument("--recon_weight", type=float, default=1) # loss = (triplet_loss + recon_weight * reconstruction_loss)
        self.parser.add_argument("--croco_mask_ratio", type=float, default=0.75) # CroCo masking Ratio
        self.parser.add_argument("--num_decoder_depth", type=int, default=8) # decoder의 layer 개수
        self.parser.add_argument("--masking_method", type=str, default="random", choices=['random', 'CLS'])
        self.parser.add_argument("--use_dist_cls_while_Recon", action='store_true', default=False)

        # Training options
        self.parser.add_argument("--use_pos_as_aligned_rgb", action='store_true', default=False) # CroCo는 동시에 찍은 사진은 reference로 사용하는데, 이게 키면 pos로 수집한걸 사용
        self.parser.add_argument("--use_fast_track", action='store_true', default=False) # training 빠르게 건너뛰어서 실험할때 사용
        self.parser.add_argument("--isRGBGreyscale", action='store_true', default=False)
        
        # Test options
        self.parser.add_argument("--use_reranking", type=str, default="none", choices=['recon']) # reranking 사용여부 및 선택
        
    def parse_arguments(self):
        args = self.parser.parse_args()
        if args.config is not None:
            with open(args.config, 'r') as file:
                config = yaml.safe_load(file)

            self.parser.set_defaults(**config)
            args = self.parser.parse_args()
            args.save_dir = os.path.basename(os.path.dirname(args.save_dir))

        return args
