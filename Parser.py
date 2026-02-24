import os
import argparse
import yaml

'''
TODO
--model
get different models
'''

class Parser():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description=None,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        ### Training settings
        self.parser.add_argument("--optim", type=str, default='adam', help="_", choices=['adam', 'sgd'])
        self.parser.add_argument("--margin", type=float, default=0.1, help="_")
        self.parser.add_argument("--lr", type=float, default=0.00001, help="_")
        self.parser.add_argument("--epochs_num", type=int, default=100, help="number of epochs to train for")
        self.parser.add_argument("--train_batch_size", type=int, default=4,
                            help="Batch size for train")
        self.parser.add_argument("--patience", type=int, default=5)

        # Warmup scheduler settings
        self.parser.add_argument("--use_warmup", action="store_true", default=True,
                            help="Use warmup scheduler with cosine annealing")
        self.parser.add_argument("--no_warmup", action="store_false", dest="use_warmup",
                            help="Disable warmup scheduler")
        self.parser.add_argument("--warmup_epochs", type=int, default=5,
                            help="Number of warmup epochs")

        self.parser.add_argument("--soft_positives_dist_threshold", type=int, default=10, help="_")
        self.parser.add_argument("--hard_positives_dist_threshold", type=int, default=10, help="_")

        self.parser.add_argument("--mining", type=str, default="partial", choices=["partial", "full", "random"])
        self.parser.add_argument("--cache_refresh_rate", type=int, default=1000,
                        help="How often to refresh cache, in number of queries")
        self.parser.add_argument("--queries_per_epoch", type=int, default=5000,
                        help="How many queries to consider for one epoch. Must be multiple of cache_refresh_rate")
        self.parser.add_argument("--negs_num_per_query", type=int, default=10, help="_")
        self.parser.add_argument("--neg_samples_num", type=int, default=1000, help="How many negatives to use to compute the hardest ones")

        ### Inference settings
        self.parser.add_argument("--infer_batch_size", type=int, default=64,
                            help="Batch size for inference (caching and testing)")
        self.parser.add_argument("--reranking_batch_size", type=int, default=32,
                            help="Batch size for reranking (number of queries per batch)")
        self.parser.add_argument("--reranking_num_workers", type=int, default=8,
                            help="Number of workers for reranking DataLoader")
        self.parser.add_argument('--test_method', type=str, default="hard_resize",
                            choices=["hard_resize", "single_query", "central_crop", "five_crops", "nearest_crop", "maj_voting"],
                            help="This includes pre/post-processing methods and prediction refinement")

        ### Evaluation settings
        self.parser.add_argument("--majority_weight", type=float, default=0.01, 
                            help="only for majority voting, scale factor, the higher it is the more importance is given to agreement")
        self.parser.add_argument('--recall_values', type=int, default=[1, 5, 10, 20], nargs="+",
                            help="Recalls to be computed, such as R@5.")
        self.parser.add_argument('--fuse', type=str, default=None, choices=[None, 'cat', 'add'])
        
        ### Model settings
        self.parser.add_argument('--resize', type=int, default=[224, 224], nargs=2, help="Resizing shape for images (HxW) to be fed into the network.")
        self.parser.add_argument("--foundation_model_path", type=str, default=None, help="_")

        ### Dataset parameters
        self.parser.add_argument("--img_time", type=str, default="allday", choices=["allday", "daytime", "nighttime", "latetime"])
        # self.parser.add_argument("--sequences", type=str, default=['KAIST', 'SNU', 'Valley'], nargs="+",
        #                          help="List of sequences to load from the dataset. Default: ['KAIST', 'SNU', 'Valley']")
        self.parser.add_argument("--train_seq", type=str, default=None, nargs='+', help="_")
        self.parser.add_argument("--test_seq", type=str, default=None, nargs='+', help="_")
        
        # Data augmentation parameters, # applyed to the training set
        self.parser.add_argument("--brightness", type=float, default=None, help="_")
        self.parser.add_argument("--contrast", type=float, default=None, help="_")
        self.parser.add_argument("--saturation", type=float, default=None, help="_")
        self.parser.add_argument("--hue", type=float, default=None, help="_")
        self.parser.add_argument("--rand_perspective", type=float, default=None, help="_")
        self.parser.add_argument("--random_resized_crop", type=float, default=None, help="_")
        self.parser.add_argument("--random_rotation", type=float, default=None, help="_")
        self.parser.add_argument("--horizontal_flip", action='store_true', help="_")

        ### Record settings
        self.parser.add_argument("--config", type=str, default=None, help="Path to args config file")
        self.parser.add_argument("--save_dir", type=str, default="default", help="_")

        ### Common settings
        self.parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
        self.parser.add_argument("--num_workers", type=int, default=8, help="num_workers for all dataloaders")
        self.parser.add_argument("--efficient_ram_testing", action='store_true', help="_")
        self.parser.add_argument("--seed", type=int, default=42)
        self.parser.add_argument("--resume", type=str, default=None, nargs='*',
                            help="Path to load checkpoint from, for resuming training or testing.")
        self.parser.add_argument("--save_all", type=bool, default=False)
        
        self.parser.add_argument("--num_trainable_blocks_RGB", type=int, default=0, help="number of trainable blocks")
        self.parser.add_argument("--num_trainable_blocks_THERMAL", type=int, default=0, help="number of trainable blocks")
        
        # custom settings
        self.parser.add_argument('--comment', type=str)
        self.parser.add_argument('--cuda_device', type=str)
        self.parser.add_argument("--use_alignment_loss", action='store_true', default=False)
        self.parser.add_argument("--use_sepearte_backbone_lr", action='store_true', default=False)
        self.parser.add_argument("--backbone_lr", type=float, default=0.00001, help="_")
        self.parser.add_argument("--croco_mask_ratio", type=float, default=0.75, help="_")
        self.parser.add_argument("--recon_weight", type=float, default=1, help="_")
        self.parser.add_argument("--use_rerank_loss", action='store_true', default=False)
        self.parser.add_argument("--num_decoder_depth", type=int, default=8, help="_")
        self.parser.add_argument("--use_only_cross_decoder", action='store_true', default=False)
        self.parser.add_argument("--use_pos_as_aligned_rgb", action='store_true', default=False)
        self.parser.add_argument("--switch_to_pos_rgb_epoch", type=int, default=-1, help="Epoch to switch from aligned_rgb to pos_rgb for CroCo. -1 means no switch.")
        self.parser.add_argument("--recon_loss_type", type=str, default="none", choices=['mse', 'l1', 'ssim', 'mse+ssim', 'GV', 'GV+l1', 'l1+ssim'])
        self.parser.add_argument("--use_fast_track", action='store_true', default=False)
        self.parser.add_argument("--rerank_with_RGB", action='store_true', default=False)
        self.parser.add_argument("--r2_penultimate_layer", action='store_true', default=False)
        self.parser.add_argument("--r2_add_random_patch", action='store_true', default=False)
        self.parser.add_argument("--use_cls_for_vpr", action='store_true', default=False)
        self.parser.add_argument("--use_recon_loss", action='store_true', default=False)
        self.parser.add_argument("--use_reranking", type=str, default="none",
                                 choices=['none', 'recon', 'r2former', 'selaVPR', 'reconSelaVPR', 'GeM_KL', 'reconDiffVPR', 'diffGeM', 'reconPairVPR', 'reconAttn', 'distance'])
        self.parser.add_argument("--features_dim", type=int, default=768)
        self.parser.add_argument("--selaVPR_rerank_score_type", type=str, default="none", choices=['none', 'quantile_attn', 'mul_cossim'])
        self.parser.add_argument("--visualize_attention", action='store_true', default=False, help="Visualize CLS and GeM attention maps during inference")
        self.parser.add_argument("--visualize_sinkhorn", action='store_true', default=False, help="Visualize Sinkhorn assignment matrix during inference")
        self.parser.add_argument("--sinkhorn_vis_samples", type=int, default=20, help="Number of samples for Sinkhorn visualization")
        self.parser.add_argument("--use_sela_local_loss", action='store_true', default=False)
        self.parser.add_argument("--use_diff_loss", action='store_true', default=False)
        self.parser.add_argument("--use_mlp_dim_before_decoder", type=int, default=0)
        self.parser.add_argument("--masking_method", type=str, default="random", choices=['random', 'CLS', 'GeM', '22222'])
        self.parser.add_argument("--use_gem_recon_weight", action='store_true', default=False, help="Use GeM attention scores as weights for reconstruction loss")
        
        # Decoder DINO setting
        self.parser.add_argument('--use_dino_decoder', action='store_true', default=False)
        self.parser.add_argument('--dino_decoder_layer_start', type=int, default=6)
        self.parser.add_argument('--dino_decoder_layer_end', type=int, default=12)
        self.parser.add_argument('--unfreeze_dino_decoder', action='store_true', default=False,
                                 help="Unfreeze DINO blocks in decoder (train self-attn + MLP, not just cross-attn)")   
        self.parser.add_argument('--is_dino_dec_stage2', action='store_true', default=False)
        self.parser.add_argument('--stage2_train_adapter', action='store_true', default=False,
                                 help="Train adapter layers in Stage2 (like Stage1)")

        # Swin Decoder settings
        self.parser.add_argument("--use_swin_decoder", action='store_true', default=False,
                                 help="Use Swin V2 decoder blocks instead of CroCo decoder blocks")
        self.parser.add_argument("--swin_window_size", type=int, default=4,
                                 help="Window size for Swin V2 windowed self-attention")
        self.parser.add_argument("--drop_path_rate", type=float, default=0.1,
                                 help="Stochastic depth rate for Swin decoder blocks")

        # Stage 2: Distance-based geometric matching
        self.parser.add_argument("--use_distance_loss", action='store_true', default=False,
                                 help="Use distance-based loss for Stage 2 geometric matching training")
        self.parser.add_argument("--distance_tau", type=float, default=10.0,
                                 help="Temperature for distance->score conversion (default: 10m)")
        self.parser.add_argument("--train_with_negatives", action='store_true', default=False,
                                 help="Include negative pairs in distance-based training")

        # Stage 2 V2: CLS + Sinkhorn Soft Aggregation (deprecated)
        self.parser.add_argument("--use_distance_loss_v2", action='store_true', default=False,
                                 help="Use CLS + Sinkhorn soft aggregation loss (V2) [DEPRECATED]")
        self.parser.add_argument("--aux_loss_weight", type=float, default=0.3,
                                 help="Weight for auxiliary loss (flow or sinkhorn)")
        self.parser.add_argument("--sinkhorn_iters", type=int, default=100,
                                 help="Number of Sinkhorn iterations")
        self.parser.add_argument("--inference_cls_only", action='store_true', default=False,
                                 help="Use only CLS score for inference (ignore aux score)")

        # Stage 2 Flow: CLS + Flow matching auxiliary
        self.parser.add_argument("--use_distance_loss_flow", action='store_true', default=False,
                                 help="Use CLS + Flow matching auxiliary loss")

        # Distance-Aware Margin Loss
        self.parser.add_argument("--use_margin_loss", action='store_true', default=False,
                                 help="Use Distance-Aware Margin Loss for relative ranking")
        self.parser.add_argument("--margin_loss_weight", type=float, default=0.5,
                                 help="Weight for margin loss (default: 0.5)")
        self.parser.add_argument("--min_margin", type=float, default=0.1,
                                 help="Minimum margin to prevent gradient vanishing (default: 0.1)")


    def parse_arguments(self):
        args = self.parser.parse_args()
        if args.config is not None:
            with open(args.config, 'r') as file:
                config = yaml.safe_load(file)
        
            self.parser.set_defaults(**config)
            args = self.parser.parse_args()
            args.save_dir = os.path.basename(os.path.dirname(args.save_dir))
        
        return args
