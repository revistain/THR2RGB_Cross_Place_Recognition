# train_wandb.py
# Training script for cross-modal VPR
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
import inference
import recon_vis
import datasets_T2R

import network
import network_only_GeM
from pathlib import Path
from pair_sampler import PairSampler


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
    wandb.init(project="cross-modal-vpr-ms2", name=args.comment, config=vars(args))

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
    if args.use_recon_loss:
        sampler = PairSampler(triplets_ds.database_utms, triplets_ds.queries_utms, distance_threshold=20.0)
        model = network.CrossModalVPR_Net(
            args,
            pretrained_foundation=True,
            foundation_model_path=args.foundation_model_path,
            pair_sampler=sampler,
        )
    else:
        model = network_only_GeM.CrossModalVPR_Net(
            args,
            pretrained_foundation=True,
            foundation_model_path=args.foundation_model_path,
        )
            
    '''Resume from checkpoint'''
    if args.resume:
        model, _, best_r1, start_epoch_num, not_improved_num = utils.resume_train(args, model, strict=False)
        best_r1 = 0
        logging.info(f"Resuming from epoch {start_epoch_num}")
    else:
        best_r1 = start_epoch_num = not_improved_num = 0

    model = model.to(args.device)
    model = torch.nn.DataParallel(model)

    # Freeze backbone except adapter layers
    print("="*30)
    print("- Tuning backbone layers: ", args.num_trainable_blocks)
    print("="*30)
    for name, param in model.module.shared_backbone.named_parameters():
        if "adapter" not in name:
            param.requires_grad = False
        for i in range(args.num_trainable_blocks):
            num_blocks = len(model.module.shared_backbone.blocks)
            model.module.shared_backbone.blocks[num_blocks - i - 1].requires_grad_(True)

    # Initialize adapter layers
    for n, m in model.named_modules():
        if 'adapter' in n:
            for n2, m2 in m.named_modules():
                if 'D_fc2' in n2:
                    if isinstance(m2, nn.Linear):
                        nn.init.constant_(m2.weight, 0.)
                        nn.init.constant_(m2.bias, 0.)
            for n2, m2 in m.named_modules():
                if 'conv' in n2:
                    if isinstance(m2, nn.Conv2d):
                        nn.init.constant_(m2.weight, 0.00001)
                        nn.init.constant_(m2.bias, 0.00001)

    '''Loss Function'''
    GlobalTriplet = nn.TripletMarginLoss(margin=args.margin, p=2, reduction="sum")

    train_params = []
    print("="*30)
    print(f"- Learning Rate: \t{args.lr}")
    print("="*30)

    for name, param in model.named_parameters():
        if param.requires_grad:
            train_params.append(param)

    if args.optim == "adam":
        optimizer = torch.optim.Adam([
            {'params': train_params, 'lr': args.lr},
        ])
    elif args.optim == "sgd":
        optimizer = torch.optim.SGD([
            {'params': train_params, 'lr': args.lr, 'momentum': 0.9, 'weight_decay': 0.001},
        ])

    # Cosine Annealing with Warmup Scheduler
    warmup_epochs = 5
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs_num - warmup_epochs, eta_min=1e-7)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

    thermal_flag = torch.zeros(1, dtype=torch.long)
    rgb_flags = torch.ones(1 + args.negs_num_per_query, dtype=torch.long)
    bundle_flags = torch.cat([thermal_flag, rgb_flags])
    flags = bundle_flags.repeat(args.train_batch_size)

    '''Training'''
    global_step = 0
    for epoch_num in range(start_epoch_num, args.epochs_num):
        logging.info(f"Start training epoch: {epoch_num:02d}")

        epoch_start_time = datetime.now()
        epoch_losses = np.zeros((0, 1), dtype=np.float32)

        loops_num = math.ceil(args.queries_per_epoch / args.cache_refresh_rate)
        for loop_num in range(loops_num):
            logging.debug(f"Cache: {loop_num + 1} / {loops_num}")

            # Compute triplets for contrastive learning
            triplets_ds.is_inference = True
            print("- Computing triplets...")
            triplets_ds.compute_triplets(args, model)
            torch.cuda.empty_cache()
            triplets_ds.is_inference = False

            logging.debug("Finish computing triplets")

            # Create DataLoader for triplets
            triplets_dl = DataLoader(dataset=triplets_ds, num_workers=args.num_workers,
                                    batch_size=args.train_batch_size,
                                    collate_fn=datasets_T2R.collate_fn,
                                    pin_memory=(args.device == "cuda"),
                                    drop_last=True)

            model = model.train()
            logging.debug(f"Start loading {len(triplets_ds)} triplets as {len(triplets_dl)} batches")

            print("- Training...")
            for batch_data in tqdm(triplets_dl, ncols=100, desc=f"GPU{args.cuda_device}/Epoch {epoch_num:02d}"):
                # Unpack batch data (5 elements with recon_images)
                images, triplets_local_indexes, _, aligned_rgbs, recon_images = batch_data
                # Use positive as aligned RGB if specified
                if args.use_pos_as_aligned_rgb:
                    assert images.size(0) % args.train_batch_size == 0
                    size_of_batch = int(images.size(0) / args.train_batch_size)
                    pos_rgbs = [images[idx] for idx in range(1, images.size(0), size_of_batch)]
                    pos_rgbs = torch.stack(pos_rgbs)
                    pos_rgbs = triplets_ds.transform(pos_rgbs)
                    aligned_rgbs = pos_rgbs

                # Forward pass
                # recon_images를 device로 이동
                if args.use_recon_loss:
                    recon_images_device = None
                    if recon_images is not None:
                        recon_images_device = {}
                        for key, val in recon_images.items():
                            if val is not None:
                                recon_images_device[key] = val.to(args.device)
                            else:
                                recon_images_device[key] = None

                    global_features, patch_embedding, recon_loss, masks = model(
                        images.to(args.device),
                        flags=flags,
                        paired_rgb=aligned_rgbs.to(args.device) if aligned_rgbs is not None else None,
                        recon_pairs=recon_images_device,
                        return_mask=True
                    )
                else:
                    global_features, patch_embedding, recon_loss, masks = model(
                        images.to(args.device),
                        flags=flags,
                        paired_rgb=aligned_rgbs.to(args.device),
                        return_mask=True
                    )

                # Process triplet indices
                triplets_local_indexes = torch.transpose(
                    triplets_local_indexes.view(args.train_batch_size, args.negs_num_per_query, 3), 1, 0)

                overall_loss = 0
                triplet_loss_sum = 0

                # Compute triplet loss for each triplet
                for triplets in triplets_local_indexes:
                    queries_indexes, positives_indexes, negatives_indexes = triplets.T

                    query_features = global_features[queries_indexes]
                    positive_features = global_features[positives_indexes]
                    negative_features = global_features[negatives_indexes]

                    triplet_loss = GlobalTriplet(query_features, positive_features, negative_features)
                    overall_loss += triplet_loss
                    triplet_loss_sum += triplet_loss

                # Add reconstruction loss if enabled
                # recon_loss는 dict: {intra_thermal, intra_rgb, inter_t2r, inter_r2t}
                if args.use_recon_loss and recon_loss is not None:
                    recon_weight = args.recon_weight

                    # 유효한 loss들만 수집
                    valid_losses = []
                    loss_log = {}
                    scale_factor = args.train_batch_size * args.negs_num_per_query

                    # Pair 1: Intra-modal Thermal (Query ← Pos Thermal)
                    if recon_loss.get('intra_thermal') is not None:
                        loss_1 = recon_loss['intra_thermal']
                        valid_losses.append(loss_1)
                        loss_log["train/recon_intra_thermal"] = loss_1.mean().item() * recon_weight / scale_factor

                    # Pair 2: Intra-modal RGB (Similar RGB ← RGB-similar RGB)
                    if recon_loss.get('intra_rgb') is not None:
                        loss_2 = recon_loss['intra_rgb']
                        valid_losses.append(loss_2)
                        loss_log["train/recon_intra_rgb"] = loss_2.mean().item() * recon_weight / scale_factor

                    # Pair 3: Inter-modal (Query Thermal ← Similar RGB)
                    if recon_loss.get('inter_t2r') is not None:
                        loss_3 = recon_loss['inter_t2r']
                        valid_losses.append(loss_3)
                        loss_log["train/recon_inter_t2r"] = loss_3.mean().item() * recon_weight / scale_factor

                    # Pair 4: Inter-modal (Similar RGB ← Query Thermal)
                    if recon_loss.get('inter_r2t') is not None:
                        loss_4 = recon_loss['inter_r2t']
                        valid_losses.append(loss_4)
                        loss_log["train/recon_inter_r2t"] = loss_4.mean().item() * recon_weight / scale_factor

                    if valid_losses:
                        combined_recon_loss = torch.stack(valid_losses).mean()
                        overall_loss += (combined_recon_loss * recon_weight)
                        loss_log["train/recon_loss_combined"] = combined_recon_loss.mean().item() * recon_weight / scale_factor
                        wandb.log(loss_log, step=global_step)

                overall_loss /= (args.train_batch_size * args.negs_num_per_query)

                del global_features, query_features, positive_features, negative_features

                optimizer.zero_grad()
                overall_loss.backward()
                optimizer.step()

                batch_loss = overall_loss.item()
                epoch_losses = np.append(epoch_losses, batch_loss)

                # wandb logging
                wandb.log({
                    "train/overall_loss": overall_loss.item(),
                    "train/triplet_loss(scaled)": triplet_loss_sum.item() / (args.train_batch_size * args.negs_num_per_query),
                }, step=global_step)

                global_step += 1

                if args.use_fast_track: break

            logging.info(f"Epoch[{epoch_num:02d}]({loop_num + 1}/{loops_num}): " +
                        f"current batch triplet loss = {batch_loss:.8f}, " +
                        f"average epoch triplet loss = {epoch_losses.mean():.8f}")

        # wandb logging (epoch level)
        wandb.log({"train/epoch_avg_loss": epoch_losses.mean(), "epoch": epoch_num}, step=global_step)
        logging.info(f"epoch {epoch_num:02d} time: {str(datetime.now() - epoch_start_time)[:-7]}, ")

        # Reconstruction visualization (every epoch)
        if args.use_recon_loss and epoch_num % 1 == 0:
            model.eval()
            with torch.no_grad():
                # Get a sample batch for visualization
                sample_batch = next(iter(triplets_dl))
                images, _, _, _, recon_images = sample_batch
                
                if recon_images is not None:
                    recon_images_device = {}
                    for key in recon_images:
                        if recon_images[key] is not None:
                            recon_images_device[key] = recon_images[key].to(args.device)
                        else:
                            recon_images_device[key] = None

                    # Visualization 데이터 가져오기
                    vis_data = model.module.visualize_reconstruction(
                        images.to(args.device),
                        recon_images_device,
                        batch_idx=0
                    )

                    vis_dir = Path(args.save_dir) / "recon_vis"
                    vis_dir.mkdir(parents=True, exist_ok=True)

                    # 1. recon_vis를 사용해 개별 이미지 쌍 저장
                    recon_vis.save_reconstruction_images(vis_data, vis_dir, epoch=epoch_num)

                    # 2. recon_vis를 사용해 통합 Grid Figure 생성 및 로컬 저장
                    grid_path = vis_dir / f"epoch{epoch_num:03d}_grid.png"
                    fig = recon_vis.visualize_reconstruction_grid(
                        vis_data,
                        save_path=grid_path,
                        title=f"Epoch {epoch_num} Reconstruction"
                    )

                    # 3. WandB에 Grid 이미지 직접 로깅 (대시보드에서 확인 가능)
                    if fig is not None:
                        wandb.log({"val/reconstruction_vis": wandb.Image(fig)}, step=global_step)

                    logging.info(f"Saved and logged reconstruction visualization to {vis_dir}")
            
            model.train()

        # Update learning rate scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        wandb.log({"train/learning_rate": current_lr}, step=global_step)
        logging.info(f"Learning rate: {current_lr:.2e}")

        # Compute recalls
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

        # wandb logging (average recall)
        wandb.log({"val/avg_R@1": current_avg_r1, "val/best_R@1": best_r1}, step=global_step)

        utils.save_checkpoint(args, {"epoch_num": epoch_num, "model_state_dict": model.state_dict(),
                                    "optimizer_state_dict": optimizer.state_dict(), "recalls": recalls, "best_r1": best_r1,
                                    "not_improved_num": not_improved_num
                                    }, is_best, filename="last_model.pth")

        if is_best:
            logging.info(f"Improved: previous best R@1 = {best_r1:.1f}, current R@1 = {(current_avg_r1):.1f}")
            best_r1 = current_avg_r1
            not_improved_num = 0
        else:
            not_improved_num += 1
            logging.info(
                f"Not improved: {not_improved_num} / {args.patience}: best R@1 = {best_r1:.1f}, current R@1 = {(current_avg_r1):.1f}")
            if not_improved_num >= args.patience:
                print(f"Performance did not improve for {not_improved_num} epochs.")
                logging.info(f"Performance did not improve for {not_improved_num} epochs. Stop training.")

        print(f"Comment: {args.comment} :: Epoch {epoch_num:02d}")
        import gc
        del recalls, recalls_str
        if 'current_epoch_r1_list' in locals(): del current_epoch_r1_list

        gc.collect()
        torch.cuda.empty_cache()

    logging.info(f"Best R@1: {best_r1:.2f}")
    logging.info(f"Trained for {epoch_num + 1:02d} epochs, in total in {str(datetime.now() - start_time)[:-7]}")

    for seq, test_ds in zip(test_sequences, test_ds_list):
        logging.info(f"===== Evaluating Sequence: {seq} =====")
        recalls, recalls_str = inference.inference(args, test_ds, model)
        logging.info(f"Recalls for {seq}: {recalls_str}")
        logging.info(f"================================================")

    wandb.finish()
