# train_wandb_2stage.py
# Stage2: Attention-based masking + Reconstruction loss training
import math
import torch
import logging
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import multiprocessing
from datetime import datetime
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import os
from sklearn.cluster import KMeans
import torchvision.models as models
import wandb

from Parser import Parser
import commons
import utils
import datasets_T2R
import inference
import random
from croco.models.criterion import MaskedMSE
from recon_vis import visualize_during_training
from visual import visualize_attention_maps_pca, visualize_mnn_matches
from info_nce import InfoNCE, info_nce

import network
import network_only_GeM
from local_matching import LocalFeatureLoss
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
    if True:
        import backbone.dinov2.block as dinoblock
        model_path = Path(args.foundation_model_path)
        model_name = model_path.parts[-1].lower()
        args.features_dim = 768 if 'vitb' in model_name else 384
        dinoblock.adapter_dim = args.features_dim

    # wandb 초기화
    wandb.init(project="cross-modal-vpr-stage2", name=args.comment, config=vars(args))

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

    '''Resume from checkpoint (Stage1 weights)'''
    if args.resume:
        model, _, best_r1, start_epoch_num, not_improved_num = utils.resume_train(args, model, strict=False)
        best_r1 = 0  # Reset for Stage2
        start_epoch_num = 0  # Start fresh for Stage2
        logging.info(f"Loaded Stage1 checkpoint, starting Stage2 training")
    else:
        best_r1 = start_epoch_num = not_improved_num = 0
        logging.warning("No checkpoint provided. Starting from scratch (not recommended for Stage2)")

    model = model.to(args.device)
    model = torch.nn.DataParallel(model)

    '''Stage2 Training Parameters'''
    # Freeze everything except decoder components
    for name, param in model.named_parameters():
        param.requires_grad = False

    # Unfreeze Stage2 relevant parameters (CroCo decoder)
    stage2_keys = [
        'decoder_blocks',       # CroCo decoder blocks
        'decoder_norm',         # Decoder layer norm
        'prediction_thermal_head',  # Prediction heads for reconstruction
        'prediction_rgb_head',
        'mask_token',           # Mask token
        'decoder_pos_embed',    # Decoder positional embedding
    ]

    # Add V2-specific keys if using distance loss V2
    if args.use_distance_loss_v2:
        stage2_keys.extend([
            'dustbin_score',        # Sinkhorn dustbin score
            'aux_distance_head',    # Auxiliary distance head
            'decoder_cls_token',    # CLS token for distance prediction
            'decoder_pos_embed_with_cls',  # Positional embedding with CLS
            'distance_head',        # Main distance head
        ])

    # Add Flow-specific keys if using distance loss flow
    if args.use_distance_loss_flow:
        stage2_keys.extend([
            'decoder_cls_token',    # CLS token for distance prediction
            'decoder_pos_embed_with_cls',  # Positional embedding with CLS
            'distance_head',        # Main distance head
        ])

    for name, param in model.named_parameters():
        if any(key in name for key in stage2_keys):
            param.requires_grad = True

    # Unfreeze adapter layers if specified (like Stage1)
    train_adapter = args.stage2_train_adapter
    if train_adapter:
        for name, param in model.module.shared_backbone.named_parameters():
            if "adapter" in name:
                param.requires_grad = True

    # Unfreeze encoder (backbone) blocks if specified (like Stage1)
    train_encoder = args.num_trainable_blocks_RGB > 0
    if train_encoder:
        num_blocks = len(model.module.shared_backbone.blocks)
        for i in range(args.num_trainable_blocks_RGB):
            model.module.shared_backbone.blocks[num_blocks - i - 1].requires_grad_(True)

    # Print training configuration
    print("=" * 30)
    print("Stage2 Training Config:")
    print(f"  - Decoder: ON (default)")
    print(f"  - Adapter: {'ON' if train_adapter else 'OFF'}")
    print(f"  - Backbone blocks: {args.num_trainable_blocks_RGB}")
    print("=" * 30)

    # Collect trainable params
    train_params = [p for p in model.parameters() if p.requires_grad]

    # Log trainable parameters
    trainable_count = sum(p.numel() for p in train_params)
    total_count = sum(p.numel() for p in model.parameters())
    logging.info(f"Stage2 Trainable params: {trainable_count:,} / {total_count:,} ({100*trainable_count/total_count:.2f}%)")

    if args.optim == "adam":
        optimizer = torch.optim.Adam([
            {'params': train_params, 'lr': args.lr},
        ])
    elif args.optim == "sgd":
        optimizer = torch.optim.SGD([
            {'params': train_params, 'lr': args.lr, 'momentum': 0.9, 'weight_decay': 0.001},
        ])

    # Cosine Annealing with Warmup Scheduler
    if args.use_warmup:
        warmup_epochs = args.warmup_epochs
        warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs_num - warmup_epochs, eta_min=1e-7)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
        logging.info(f"Using warmup scheduler: {warmup_epochs} warmup epochs + cosine annealing")
    else:
        scheduler = None
        logging.info("Warmup scheduler disabled")

    '''Training Loop'''
    global_step = 0
    for epoch_num in range(start_epoch_num, args.epochs_num):
        logging.info(f"Start Stage2 training epoch: {epoch_num:02d}")

        epoch_start_time = datetime.now()
        epoch_losses = np.zeros((0, 1), dtype=np.float32)

        loops_num = math.ceil(args.queries_per_epoch / args.cache_refresh_rate)
        for loop_num in range(loops_num):
            logging.debug(f"Cache: {loop_num + 1} / {loops_num}")

            # Compute triplets for sampling positive pairs
            triplets_ds.is_inference = True
            print("- Computing triplets...")
            triplets_ds.compute_triplets(args, model)
            torch.cuda.empty_cache()
            triplets_ds.is_inference = False

            logging.debug("Finish computing triplets")

            triplets_dl = DataLoader(
                dataset=triplets_ds,
                num_workers=args.num_workers,
                batch_size=args.train_batch_size,
                collate_fn=datasets_T2R.collate_fn,
                pin_memory=(args.device == "cuda"),
                drop_last=True
            )

            model = model.train()
            logging.debug(f"Start loading {len(triplets_ds)} triplets as {len(triplets_dl)} batches")

            if args.use_distance_loss_flow:
                print("- Stage2 Training (Distance Flow: CLS + Flow Matching)...")
            elif args.use_distance_loss_v2:
                print("- Stage2 Training (Distance V2: CLS + Sinkhorn Soft Aggregation)...")
            elif args.use_distance_loss:
                print("- Stage2 Training (Distance-based Geometric Matching)...")
            else:
                print("- Stage2 Training (Attention-based Recon)...")

            for batch_data in tqdm(triplets_dl, ncols=100, desc=f"GPU{args.cuda_device}/Epoch {epoch_num:02d}"):
                # Unpack batch data (with or without distances)
                if len(batch_data) == 5:
                    images, triplets_local_indexes, _, aligned_rgbs, distances = batch_data
                else:
                    images, triplets_local_indexes, _, aligned_rgbs = batch_data
                    distances = None

                # Extract thermal queries and positive RGBs from batch
                # images layout: [query, pos, neg1, neg2, ...] * batch_size
                batch_size = args.train_batch_size
                size_of_bundle = 2 + args.negs_num_per_query  # query + pos + negs

                # Extract query (thermal) indices: 0, size_of_bundle, 2*size_of_bundle, ...
                query_indices = [i * size_of_bundle for i in range(batch_size)]
                # Extract positive (RGB) indices: 1, size_of_bundle+1, 2*size_of_bundle+1, ...
                pos_indices = [i * size_of_bundle + 1 for i in range(batch_size)]

                thermal_imgs = images[query_indices].to(args.device)  # [B, C, H, W]
                pos_rgb_imgs = images[pos_indices].to(args.device)    # [B, C, H, W]

                optimizer.zero_grad()

                if args.use_distance_loss or args.use_distance_loss_v2 or args.use_distance_loss_flow:
                    # Stage2 forward: Distance-based geometric matching
                    # Get encoder features (trainable if train_encoder or train_adapter)
                    with torch.set_grad_enabled(train_encoder or train_adapter):
                        thermal_feat = model.module.shared_backbone(thermal_imgs)["x_norm_patchtokens"]  # [B, 256, D]
                        pos_rgb_feat = model.module.shared_backbone(pos_rgb_imgs)["x_norm_patchtokens"]  # [B, 256, D]

                    # Positive pair distances
                    pos_distances = distances[:, 0].to(args.device) if distances is not None else torch.zeros(batch_size).to(args.device)

                    # Forward distance prediction for positive pairs
                    if args.use_distance_loss_flow:
                        # Flow: CLS + Flow matching auxiliary
                        distance_loss, main_score, flow_score = model.module.stage2_forward_distance_flow(
                            thermal_feat=thermal_feat,
                            rgb_feat=pos_rgb_feat,
                            distance_gt=pos_distances,
                            tau=args.distance_tau,
                            aux_weight=args.aux_loss_weight
                        )
                        pred_score = main_score
                        aux_score = flow_score
                    elif args.use_distance_loss_v2:
                        # V2: CLS + Sinkhorn Soft Aggregation
                        distance_loss, main_score, aux_score, P = model.module.stage2_forward_distance_v2(
                            thermal_feat=thermal_feat,
                            rgb_feat=pos_rgb_feat,
                            distance_gt=pos_distances,
                            tau=args.distance_tau,
                            aux_weight=args.aux_loss_weight,
                            sinkhorn_iters=args.sinkhorn_iters
                        )
                        pred_score = main_score
                    else:
                        # V1: CLS only
                        distance_loss, pred_score = model.module.stage2_forward_distance(
                            thermal_feat=thermal_feat,
                            rgb_feat=pos_rgb_feat,
                            distance_gt=pos_distances,
                            tau=args.distance_tau
                        )
                        aux_score = None

                    total_loss = distance_loss

                    # Optionally train with negative pairs
                    if args.train_with_negatives and distances is not None:
                        neg_loss_list = []
                        neg_aux_score_list = []
                        neg_scores_list = []  # For margin loss
                        neg_target_scores_list = []  # For margin loss

                        for neg_idx in range(args.negs_num_per_query):
                            neg_indices = [i * size_of_bundle + 2 + neg_idx for i in range(batch_size)]
                            neg_rgb_imgs = images[neg_indices].to(args.device)
                            neg_distances = distances[:, 1 + neg_idx].to(args.device)

                            with torch.set_grad_enabled(train_encoder or train_adapter):
                                neg_rgb_feat = model.module.shared_backbone(neg_rgb_imgs)["x_norm_patchtokens"]

                            if args.use_distance_loss_flow:
                                neg_loss, neg_main_score, neg_flow_score = model.module.stage2_forward_distance_flow(
                                    thermal_feat=thermal_feat,
                                    rgb_feat=neg_rgb_feat,
                                    distance_gt=neg_distances,
                                    tau=args.distance_tau,
                                    aux_weight=args.aux_loss_weight
                                )
                                neg_aux_score_list.append(neg_flow_score.mean().item())
                                neg_scores_list.append(neg_main_score)
                            elif args.use_distance_loss_v2:
                                neg_loss, neg_main_score, neg_aux_score, _ = model.module.stage2_forward_distance_v2(
                                    thermal_feat=thermal_feat,
                                    rgb_feat=neg_rgb_feat,
                                    distance_gt=neg_distances,
                                    tau=args.distance_tau,
                                    aux_weight=args.aux_loss_weight,
                                    sinkhorn_iters=args.sinkhorn_iters
                                )
                                neg_aux_score_list.append(neg_aux_score.mean().item())
                                neg_scores_list.append(neg_main_score)
                            else:
                                neg_loss, neg_pred_score = model.module.stage2_forward_distance(
                                    thermal_feat=thermal_feat,
                                    rgb_feat=neg_rgb_feat,
                                    distance_gt=neg_distances,
                                    tau=args.distance_tau
                                )
                                neg_scores_list.append(neg_pred_score)

                            neg_loss_list.append(neg_loss)
                            # Store target scores for margin loss
                            neg_target_scores_list.append(model.module.distance_to_score(neg_distances, tau=args.distance_tau))

                        # Average negative loss (regression)
                        neg_loss_avg = sum(neg_loss_list) / len(neg_loss_list)
                        total_loss = (distance_loss + neg_loss_avg) / 2.0

                        # Distance-Aware Margin Loss
                        margin_loss = torch.tensor(0.0, device=args.device)
                        if args.use_margin_loss:
                            pos_target_score = model.module.distance_to_score(pos_distances, tau=args.distance_tau)

                            margin_losses = []
                            for neg_score, neg_target_score in zip(neg_scores_list, neg_target_scores_list):
                                # Dynamic margin: difference between target scores
                                margin = (pos_target_score - neg_target_score).clamp(min=args.min_margin)
                                # Hinge loss: S_neg - S_pos + margin should be <= 0
                                m_loss = torch.relu(neg_score - pred_score + margin).mean()
                                margin_losses.append(m_loss)

                            margin_loss = sum(margin_losses) / len(margin_losses)
                            total_loss = total_loss + args.margin_loss_weight * margin_loss

                        if args.use_distance_loss_flow:
                            neg_aux_score_avg = sum(neg_aux_score_list) / len(neg_aux_score_list)
                            log_dict = {
                                "train/pos_distance_loss": distance_loss.item(),
                                "train/neg_distance_loss": neg_loss_avg.item(),
                                "train/total_distance_loss": total_loss.item(),
                                "train/pos_flow_score": aux_score.mean().item(),  # should → 1
                                "train/neg_flow_score": neg_aux_score_avg,         # should → 0
                            }
                        elif args.use_distance_loss_v2:
                            neg_aux_score_avg = sum(neg_aux_score_list) / len(neg_aux_score_list)
                            log_dict = {
                                "train/pos_distance_loss": distance_loss.item(),
                                "train/neg_distance_loss": neg_loss_avg.item(),
                                "train/total_distance_loss": total_loss.item(),
                                "train/pos_aux_score": aux_score.mean().item(),  # should → 1
                                "train/neg_aux_score": neg_aux_score_avg,         # should → 0
                            }
                        else:
                            log_dict = {
                                "train/pos_distance_loss": distance_loss.item(),
                                "train/neg_distance_loss": neg_loss_avg.item(),
                                "train/total_distance_loss": total_loss.item(),
                            }

                        # Log margin loss if enabled
                        if args.use_margin_loss:
                            log_dict["train/margin_loss"] = margin_loss.item()
                            log_dict["train/pos_score_mean"] = pred_score.mean().item()
                            log_dict["train/neg_score_mean"] = torch.stack(neg_scores_list).mean().item()

                        wandb.log(log_dict, step=global_step)
                    else:
                        log_dict = {"train/distance_loss": distance_loss.item()}
                        if args.use_distance_loss_flow:
                            log_dict["train/main_score_mean"] = pred_score.mean().item()
                            log_dict["train/flow_score_mean"] = aux_score.mean().item()
                        elif args.use_distance_loss_v2:
                            log_dict["train/main_score_mean"] = main_score.mean().item()
                            log_dict["train/aux_score_mean"] = aux_score.mean().item()
                        wandb.log(log_dict, step=global_step)

                    # Backward
                    total_loss.backward()
                    optimizer.step()

                    batch_loss = total_loss.item()
                    epoch_losses = np.append(epoch_losses, batch_loss)

                else:
                    # Stage2 forward: attention-based masking + reconstruction loss
                    total_recon_loss, recon_loss_thermal, recon_loss_rgb = model.module.stage2_forward_recon(
                        thermal_imgs=thermal_imgs,
                        rgb_imgs=pos_rgb_imgs,
                    )

                    # Backward
                    total_recon_loss.backward()
                    optimizer.step()

                    batch_loss = total_recon_loss.item()
                    epoch_losses = np.append(epoch_losses, batch_loss)

                    # wandb logging
                    wandb.log({
                        "train/total_recon_loss": total_recon_loss.item(),
                        "train/recon_loss_thermal": recon_loss_thermal.item(),
                        "train/recon_loss_rgb": recon_loss_rgb.item(),
                    }, step=global_step)

                global_step += 1

                if args.use_fast_track:
                    break

            logging.info(f"Epoch[{epoch_num:02d}]({loop_num + 1}/{loops_num}): " +
                        f"current batch loss = {batch_loss:.8f}, " +
                        f"average epoch loss = {epoch_losses.mean():.8f}")

        # Visualize reconstructions
        if args.use_recon_loss:
            visualize_during_training(
                args,
                model,
                triplets_dl,
                args.device,
                epoch_num,
                save_dir=os.path.join(args.save_dir, 'reconstructions'),
                comment=args.comment
            )

        # wandb logging (epoch)
        wandb.log({"train/epoch_avg_loss": epoch_losses.mean(), "epoch": epoch_num}, step=global_step)
        logging.info(f"epoch {epoch_num:02d} time: {str(datetime.now() - epoch_start_time)[:-7]}")

        # Update learning rate scheduler
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            wandb.log({"train/learning_rate": current_lr}, step=global_step)

        # Evaluation
        current_epoch_r1_list = []
        for seq, test_ds in zip(test_sequences, test_ds_list):
            logging.info(f"===== Evaluating Sequence: {seq} =====")
            args.current_epoch = epoch_num
            recalls, recalls_str = inference.inference(args, test_ds, model, seq_name=seq)
            logging.info(f"Recalls for {seq}: {recalls_str}")
            logging.info(f"================================================")
            current_epoch_r1_list.append(recalls[0])

            # wandb logging (per sequence)
            wandb.log({f"val/{seq}_R@1": recalls[0], f"val/{seq}_R@5": recalls[1]}, step=global_step)

        current_avg_r1 = np.mean(current_epoch_r1_list)
        is_best = current_avg_r1 > best_r1

        # wandb logging (average recall)
        wandb.log({"val/avg_R@1": current_avg_r1, "val/best_R@1": best_r1}, step=global_step)

        utils.save_checkpoint(args, {
            "epoch_num": epoch_num,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "recalls": recalls,
            "best_r1": best_r1,
            "not_improved_num": not_improved_num
        }, is_best, filename="last_model_stage2.pth")

        if is_best:
            logging.info(f"Improved: previous best R@1 = {best_r1:.1f}, current R@1 = {current_avg_r1:.1f}")
            best_r1 = current_avg_r1
            not_improved_num = 0
        else:
            not_improved_num += 1
            logging.info(f"Not improved: {not_improved_num} / {args.patience}: best R@1 = {best_r1:.1f}, current R@1 = {current_avg_r1:.1f}")
            if not_improved_num >= args.patience:
                print(f"Performance did not improve for {not_improved_num} epochs.")
                logging.info(f"Performance did not improve for {not_improved_num} epochs.")

        print(f"Comment: {args.comment} :: Stage2 Epoch {epoch_num:02d}")
        import gc
        del recalls, recalls_str
        if 'current_epoch_r1_list' in locals():
            del current_epoch_r1_list

        gc.collect()
        torch.cuda.empty_cache()

    logging.info(f"Best R@1: {best_r1:.2f}")
    logging.info(f"Stage2 trained for {epoch_num + 1:02d} epochs, in total {str(datetime.now() - start_time)[:-7]}")

    for seq, test_ds in zip(test_sequences, test_ds_list):
        logging.info(f"===== Final Evaluation - {seq} =====")
        recalls, recalls_str = inference.inference(args, test_ds, model)
        logging.info(f"Recalls for {seq}: {recalls_str}")
        logging.info(f"================================================")

    wandb.finish()
