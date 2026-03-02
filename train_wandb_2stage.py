# train_wandb_2stage.py
# 2-Stage GMRW training script for cross-modal VPR
import math
import torch
import logging
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import multiprocessing
from datetime import datetime
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import os
import random
import wandb

from Parser import Parser
import commons
import utils
import datasets_T2R
import inference
import network
from pathlib import Path


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    '''Setup'''
    set_seed()
    parser = Parser()
    args = parser.parse_arguments()

    # Set features_dim based on model
    import backbone.dinov2.block as dinoblock
    model_path = Path(args.foundation_model_path)
    model_name = model_path.parts[-1].lower()
    args.features_dim = 768 if 'vitb' in model_name else 384
    dinoblock.adapter_dim = args.features_dim

    # wandb init
    wandb.init(project="cross-modal-vpr-2stage", name=args.comment, config=vars(args))

    args.save_dir = os.path.join(args.save_dir, args.comment, utils.get_timestamp())
    commons.setup_logging(args.save_dir)
    commons.seed_everything(args.seed)

    start_time = datetime.now()

    utils.save_to_yaml(args)
    logging.debug(f"The outputs are being saved in {args.save_dir}")
    logging.info(f"Use {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs")

    DATASET_FOLDER = "./Dataset/save_mat"

    '''Datasets'''
    args.sequences = args.train_seq
    triplets_ds = datasets_T2R.TripletsSTheReODual(args, DATASET_FOLDER, use_align_rgb=True)
    train_ds = datasets_T2R.BaseSTheReODual(args, DATASET_FOLDER, split='train')
    logging.info(f"[Train - {args.train_seq}] Database: {train_ds.database_num}, Queries: {train_ds.queries_num}, Total: {len(train_ds)}")

    test_sequences = args.test_seq
    test_ds_list = []
    for seq in test_sequences:
        args.sequences = [seq]
        test_ds = datasets_T2R.BaseSTheReODual(args, DATASET_FOLDER, split='test')
        test_ds_list.append(test_ds)
        logging.info(f"[Test - {seq}] Database: {test_ds.database_num}, Queries: {test_ds.queries_num}, Total: {len(test_ds)}")

    '''Model'''
    model = network.CrossModalVPR_Net(
        args,
        pretrained_foundation=True,
        foundation_model_path=args.foundation_model_path,
    )

    '''Resume from 1-stage checkpoint'''
    if args.resume:
        if isinstance(args.resume, list):
            resume_path = args.resume[0]
        else:
            resume_path = args.resume
        logging.info(f"Loading 1-stage checkpoint from {resume_path}")
        checkpoint = torch.load(resume_path, map_location='cpu')

        # Handle DataParallel state dict
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v

        model.load_state_dict(new_state_dict, strict=False)
        logging.info("1-stage checkpoint loaded successfully")
    else:
        logging.warning("No 1-stage checkpoint provided! Training from scratch.")

    model = model.to(args.device)
    model = torch.nn.DataParallel(model)

    '''Freeze Strategy for 2-Stage'''
    # First freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze decoder if not frozen
    if not args.freeze_decoder:
        for param in model.module.decoder_blocks.parameters():
            param.requires_grad = True
        for param in model.module.decoder_norm.parameters():
            param.requires_grad = True
        for param in model.module.decoder_pos_embed:
            param.requires_grad = True
        logging.info("Decoder blocks unfrozen for training")

    # Always train GMRW-related components
    for param in model.module.label_warping.parameters():
        param.requires_grad = True
    for param in model.module.gmrw_loss_fn.parameters():
        param.requires_grad = True

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Trainable params: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

    '''Optimizer'''
    train_params = [p for p in model.parameters() if p.requires_grad]

    print("=" * 30)
    print(f"- 2-Stage Learning Rate: \t{args.lr}")
    print(f"- GMRW Weight: \t{args.gmrw_weight}")
    print(f"- Label Warp: \t{args.use_label_warp}")
    print(f"- Score Method: \t{args.score_method}")
    print("=" * 30)

    if args.optim == "adam":
        optimizer = torch.optim.Adam(train_params, lr=args.lr)
    elif args.optim == "sgd":
        optimizer = torch.optim.SGD(train_params, lr=args.lr, momentum=0.9, weight_decay=0.001)

    # Cosine Annealing with Warmup Scheduler
    warmup_epochs = 5
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs_num - warmup_epochs, eta_min=1e-7)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

    # Flags for forward pass
    thermal_flag = torch.zeros(1, dtype=torch.long)
    rgb_flags = torch.ones(1 + args.negs_num_per_query, dtype=torch.long)
    bundle_flags = torch.cat([thermal_flag, rgb_flags])
    flags = bundle_flags.repeat(args.train_batch_size)

    '''Training'''
    global_step = 0
    best_r1 = 0
    not_improved_num = 0

    for epoch_num in range(args.epochs_num):
        logging.info(f"Start 2-stage training epoch: {epoch_num:02d}")

        epoch_start_time = datetime.now()
        epoch_losses = np.zeros((0, 1), dtype=np.float32)

        loops_num = math.ceil(args.queries_per_epoch / args.cache_refresh_rate)
        for loop_num in range(loops_num):
            logging.debug(f"Cache: {loop_num + 1} / {loops_num}")

            # Compute triplets
            triplets_ds.is_inference = True
            print("- Computing triplets...")
            triplets_ds.compute_triplets(args, model)
            torch.cuda.empty_cache()
            triplets_ds.is_inference = False

            # Create DataLoader
            triplets_dl = DataLoader(
                dataset=triplets_ds,
                num_workers=args.num_workers,
                batch_size=args.train_batch_size,
                collate_fn=datasets_T2R.collate_fn,
                pin_memory=(args.device == "cuda"),
                drop_last=True
            )

            model = model.train()

            print("- Training 2-stage...")
            for images, triplets_local_indexes, _, aligned_rgbs in tqdm(triplets_dl, ncols=100, desc=f"GPU{args.cuda_device}/Epoch {epoch_num:02d}"):
                # Use positive as aligned RGB if specified
                if args.use_pos_as_aligned_rgb:
                    assert images.size(0) % args.train_batch_size == 0
                    size_of_batch = int(images.size(0) / args.train_batch_size)
                    pos_rgbs = [images[idx] for idx in range(1, images.size(0), size_of_batch)]
                    pos_rgbs = torch.stack(pos_rgbs)
                    pos_rgbs = triplets_ds.transform(pos_rgbs)
                    aligned_rgbs = pos_rgbs

                # Extract thermal and positive RGB
                # triplets_local_indexes: (B, negs_num, 3) where 3 = (query, pos, neg)
                B = args.train_batch_size
                size_of_batch = int(images.size(0) / B)

                # Thermal images (queries)
                thermal_imgs = images[0::size_of_batch]  # (B, 3, H, W)

                # Positive RGB images
                pos_rgb_imgs = images[1::size_of_batch]  # (B, 3, H, W)

                thermal_imgs = thermal_imgs.to(args.device)
                pos_rgb_imgs = pos_rgb_imgs.to(args.device)

                # 2-Stage GMRW Forward (positive pairs)
                gmrw_loss, cycle, A_T2R, warped_label, cycle_loss, smooth_loss = \
                    model.module.stage2_forward_gmrw(
                        thermal_imgs,
                        pos_rgb_imgs,
                        use_warp=args.use_label_warp
                    )

                overall_loss = gmrw_loss * args.gmrw_weight

                # Negative pairs (optional)
                if args.train_with_negatives:
                    neg_losses = []
                    for neg_idx in range(2, size_of_batch):  # Skip query and positive
                        neg_rgb_imgs = images[neg_idx::size_of_batch].to(args.device)

                        neg_loss, neg_cycle, _, _, _, _ = model.module.stage2_forward_gmrw(
                            thermal_imgs,
                            neg_rgb_imgs,
                            use_warp=args.use_label_warp
                        )

                        # Compute scores
                        pos_score = model.module.stage2_compute_score(cycle, method=args.score_method)
                        neg_score = model.module.stage2_compute_score(neg_cycle, method=args.score_method)

                        # Margin loss: pos_score should be higher than neg_score by margin
                        margin_loss = torch.relu(neg_score - pos_score + args.neg_margin).mean()
                        neg_losses.append(margin_loss)

                    if neg_losses:
                        neg_loss_avg = sum(neg_losses) / len(neg_losses)
                        overall_loss = overall_loss + neg_loss_avg
                        wandb.log({"train/neg_margin_loss": neg_loss_avg.item()}, step=global_step)

                optimizer.zero_grad()
                overall_loss.backward()
                optimizer.step()

                batch_loss = overall_loss.item()
                epoch_losses = np.append(epoch_losses, batch_loss)

                # Compute matching score for logging
                with torch.no_grad():
                    pos_score = model.module.stage2_compute_score(cycle, method=args.score_method)

                # wandb logging
                wandb.log({
                    "train/overall_loss": overall_loss.item(),
                    "train/gmrw_loss": gmrw_loss.item(),
                    "train/cycle_loss": cycle_loss.item(),
                    "train/smooth_loss": smooth_loss.item() if isinstance(smooth_loss, torch.Tensor) else smooth_loss,
                    "train/pos_score": pos_score.mean().item(),
                }, step=global_step)

                global_step += 1

                if args.use_fast_track:
                    break

            logging.info(f"Epoch[{epoch_num:02d}]({loop_num + 1}/{loops_num}): " +
                        f"current batch loss = {batch_loss:.8f}, " +
                        f"average epoch loss = {epoch_losses.mean():.8f}")

        # wandb logging (epoch level)
        wandb.log({"train/epoch_avg_loss": epoch_losses.mean(), "epoch": epoch_num}, step=global_step)
        logging.info(f"epoch {epoch_num:02d} time: {str(datetime.now() - epoch_start_time)[:-7]}")

        # Update learning rate scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        wandb.log({"train/learning_rate": current_lr}, step=global_step)
        logging.info(f"Learning rate: {current_lr:.2e}")

        # Evaluation
        current_epoch_r1_list = []
        for seq, test_ds in zip(test_sequences, test_ds_list):
            logging.info(f"===== Evaluating Sequence: {seq} =====")
            args.current_epoch = epoch_num
            recalls, recalls_str = inference.inference(args, test_ds, model)
            logging.info(f"Recalls for {seq}: {recalls_str}")
            logging.info(f"================================================")
            current_epoch_r1_list.append(recalls[0])

            # wandb logging (per sequence recall)
            wandb.log({f"val/{seq}_R@1": recalls[0], f"val/{seq}_R@5": recalls[1]}, step=global_step)

        current_avg_r1 = np.mean(current_epoch_r1_list)
        is_best = current_avg_r1 > best_r1

        if is_best:
            best_r1 = current_avg_r1
            not_improved_num = 0
        else:
            not_improved_num += 1

        # wandb logging (average recall)
        wandb.log({"val/avg_R@1": current_avg_r1, "val/best_R@1": best_r1}, step=global_step)

        utils.save_checkpoint(
            args,
            {
                "epoch_num": epoch_num,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "recalls": recalls,
                "best_r1": best_r1,
                "not_improved_num": not_improved_num
            },
            is_best,
            filename="last_model.pth"
        )

        # Early stopping
        if not_improved_num >= args.patience:
            logging.info(f"Early stopping at epoch {epoch_num} (patience: {args.patience})")
            break

    logging.info(f"Training finished. Best R@1: {best_r1:.2f}%")
    logging.info(f"Total time: {str(datetime.now() - start_time)[:-7]}")
    wandb.finish()
