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
import os
from sklearn.cluster import KMeans
import torchvision.models as models

from Parser import Parser
import commons
import utils
# import datasets_dual
import datasets_T2R
import inference
import network
import random

def clip_patch_alignment_loss(thermal_patches, rgb_patches, temperature=0.07):
    """
    CLIP-style contrastive loss for patch embeddings.
    thermal_patches: (B, N, D) - batch의 thermal query patch embeddings
    rgb_patches: (B, N, D) - batch의 aligned RGB patch embeddings
    """
    B, N, D = thermal_patches.shape
    
    # Flatten patches: (B, N, D) -> (B, N*D) or use mean pooling
    thermal_feat = thermal_patches.mean(dim=1)  # (B, D)
    rgb_feat = rgb_patches.mean(dim=1)          # (B, D)
    
    # L2 normalize
    thermal_feat = F.normalize(thermal_feat, dim=-1)
    rgb_feat = F.normalize(rgb_feat, dim=-1)
    
    # Similarity matrix: (B, B)
    logits = torch.matmul(thermal_feat, rgb_feat.T) / temperature
    
    # Labels: diagonal은 positive pair
    labels = torch.arange(B, device=logits.device)
    
    # Symmetric loss
    loss_t2r = F.cross_entropy(logits, labels)
    loss_r2t = F.cross_entropy(logits.T, labels)
    
    return (loss_t2r + loss_r2t) / 2

if __name__ == "__main__":
    '''Setup'''
    parser = Parser()
    args = parser.parse_arguments()

    commons.setup_logging(args.save_dir)
    commons.seed_everything(args.seed)

    start_time = datetime.now()

    utils.save_to_yaml(args)
    logging.debug(f"The outputs are being saved in {args.save_dir}")

    logging.info(f"Use {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs")

    DATASET_FOLDER = "./Dataset/save_mat"

    '''Datasets'''
    args.sequences = ['KAIST']  # Use KAIST sequence for training
    triplets_ds = datasets_T2R.TripletsSTheReODual(args, DATASET_FOLDER, use_align_rgb=True)
    train_ds = datasets_T2R.BaseSTheReODual(args, DATASET_FOLDER, split='train')

    args.sequences = ['SNU', 'Valley']
    test_sequences = args.sequences
    # test_ds = datasets_T2R.BaseSTheReODual(args, DATASET_FOLDER, split='test')
    test_ds_list = []
    for seq in test_sequences:
        args.sequences = [seq]
        test_ds = datasets_T2R.BaseSTheReODual(args, DATASET_FOLDER, split='test')
        test_ds_list.append(test_ds)


    '''Model'''
    model = network.CrossModalVPR_Net(pretrained_foundation = True, foundation_model_path = args.foundation_model_path)
    model = model.to(args.device)
    model = torch.nn.DataParallel(model)

    ## Freeze parameters except adapter
    for name, param in model.module.rgb_backbone.named_parameters():
        if "adapter" not in name:
            param.requires_grad = False
        # channge last args.num_trainable_blocks trainable
        for i in range(args.num_trainable_blocks):
            num_blocks = len(model.module.rgb_backbone.blocks)
            model.module.rgb_backbone.blocks[num_blocks - i - 1].requires_grad_(True)

    for name, param in model.module.thermal_backbone.named_parameters():
        if "adapter" not in name:
            param.requires_grad = False
        for i in range(args.num_trainable_blocks):
            num_blocks = len(model.module.thermal_backbone.blocks)
            model.module.thermal_backbone.blocks[num_blocks - i - 1].requires_grad_(True)

    ## initialize Adapter
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

    '''Optimizer'''
    if args.optim == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    elif args.optim == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.001)

    '''Loss Function'''
    GlobalTriplet = nn.TripletMarginLoss(margin=args.margin, p=2, reduction="sum")

    '''Resume from checkpoint'''
    if args.resume:
        model, _, best_r1, start_epoch_num, not_improved_num = utils.resume_train(args, model, strict=False)
        logging.info(f"Resuming from epoch {start_epoch_num} with best recall@1 {best_r1:.1f}")
    else:
        best_r1 = start_epoch_num = not_improved_num = 0

    bundle_flags =  ['thermal'] + ['rgb'] * (1 + args.negs_num_per_query) # ['thermal', 'rgb', 'rgb', 'rgb', 'rgb', 'rgb', 'rgb', 'rgb', 'rgb', 'rgb', 'rgb']
    num_bundle_flags = len(bundle_flags)

    '''Training'''
    for epoch_num in range(start_epoch_num, args.epochs_num):
        logging.info(f"Start training epoch: {epoch_num:02d}")

        epoch_start_time = datetime.now()
        epoch_losses = np.zeros((0, 1), dtype=np.float32)

        # How many loops should an epoch last (default is 5000/1000=5)
        loops_num = math.ceil(args.queries_per_epoch / args.cache_refresh_rate)
        for loop_num in range(loops_num):
            logging.debug(f"Cache: {loop_num + 1} / {loops_num}")

            # Compute triplets to use in the triplet loss
            triplets_ds.is_inference = True
            triplets_ds.compute_triplets(args, model)
            triplets_ds.is_inference = False

            logging.debug("Finish computing triplets")

            triplets_dl = DataLoader(dataset=triplets_ds, num_workers=args.num_workers,
                                    batch_size=args.train_batch_size,
                                    collate_fn=datasets_T2R.collate_fn,
                                    pin_memory=(args.device == "cuda"),
                                    drop_last=True)
            model = model.train()

            logging.debug(f"Start loading {len(triplets_ds)} triplets as {len(triplets_dl)} batches")

            # get image and loop
            # looping triplet_dataloader
            for images, triplets_local_indexes, _, aligned_rgbs in tqdm(triplets_dl, ncols=100):
                curr_batch_len = len(images) // num_bundle_flags
                flags = bundle_flags * curr_batch_len # to check rgb or thermal
                
                # global_features = model(images.to(args.device), flags=flags, return_embedding=False)
                global_features, patch_embedding = model(images.to(args.device), flags=flags, return_embedding=True)

                # CLIP-style alignment loss
                overall_loss = 0
                RT_alignment_loss = 0
                if aligned_rgbs is not None:
                    aligned_rgbs = aligned_rgbs.to(args.device)
                    # model eval?
                    _, aligned_rgb_patches = model(aligned_rgbs, flags=['rgb'] * len(aligned_rgbs), return_embedding=True)
                    
                    thermal_indices = [i * num_bundle_flags for i in range(curr_batch_len)]
                    thermal_patches = patch_embedding[thermal_indices]
                    
                    # thermal query의 patch embedding 추출 (각 샘플의 첫 번째 이미지)
                    thermal_indices = [i * num_bundle_flags for i in range(curr_batch_len)]
                    thermal_patches = patch_embedding[thermal_indices]  # (B, N, D)
                    
                    RT_alignment_loss = clip_patch_alignment_loss(thermal_patches, aligned_rgb_patches, temperature=0.07)

                triplets_local_indexes = torch.transpose(
                    triplets_local_indexes.view(args.train_batch_size, args.negs_num_per_query, 3), 1, 0)
                
                for triplets in triplets_local_indexes:
                    queries_indexes, positives_indexes, negatives_indexes = triplets.T
                    
                    query_features = global_features[queries_indexes]      # (query)    thermal descriptor
                    positive_features = global_features[positives_indexes] # (positive) rgb descriptor
                    negative_features = global_features[negatives_indexes] # (negative) rgb descriptor

                    triplet_loss = GlobalTriplet(query_features,
                                                positive_features,
                                                negative_features)
                    
                    
                    overall_loss += (triplet_loss + RT_alignment_loss)

                overall_loss /= (args.train_batch_size * args.negs_num_per_query)

                del global_features, query_features, positive_features, negative_features

                optimizer.zero_grad()
                overall_loss.backward()
                optimizer.step()

                batch_loss = overall_loss.item()
                epoch_losses = np.append(epoch_losses, batch_loss)
                del overall_loss, triplet_loss, RT_alignment_loss

            logging.info(f"Epoch[{epoch_num:02d}]({loop_num + 1}/{loops_num}): " +
                        f"current batch triplet loss = {batch_loss:.8f}, " +
                        f"average epoch triplet loss = {epoch_losses.mean():.8f}")

        logging.info(f"epoch {epoch_num:02d} time: {str(datetime.now() - epoch_start_time)[:-7]}, ")

        # Compute recalls
        current_epoch_r1_list = []
        for seq, test_ds in zip(test_sequences, test_ds_list):
            logging.info(f"===== Evaluating Sequence: {seq} =====")
            recalls, recalls_str = inference.inference(args, test_ds, model)
            logging.info(f"Recalls for {seq}: {recalls_str}")
            logging.info(f"================================================")
            current_epoch_r1_list.append(recalls[0])

        current_avg_r1 = np.mean(current_epoch_r1_list)
        is_best = current_avg_r1 > best_r1

        # Save latest checkpoint, which contains all training parameters
        utils.save_checkpoint(args, {"epoch_num": epoch_num, "model_state_dict": model.state_dict(),
                                    "optimizer_state_dict": optimizer.state_dict(), "recalls": recalls, "best_r1": best_r1,
                                    "not_improved_num": not_improved_num
                                    }, is_best, filename="last_model.pth")
        # Save all
        # logging.info(f"Saved checkpoint for epoch {epoch_num:02d}")
        # utils.save_checkpoint(args, {"epoch_num": epoch_num, "model_state_dict": model.state_dict(),
        #                             "optimizer_state_dict": optimizer.state_dict(), "recalls": recalls, "best_r1": best_r1,
        #                             "not_improved_num": not_improved_num
        #                             }, False, filename=f"epoch_{epoch_num}_model.pth")

        # If recall@1 did not improve for "many" epochs, stop training
        if is_best:
            logging.info(f"Improved: previous best R@1 = {best_r1:.1f}, current R@1 = {(current_avg_r1):.1f}")
            best_r1 = current_avg_r1
            not_improved_num = 0
        else:
            not_improved_num += 1
            logging.info(
                f"Not improved: {not_improved_num} / {args.patience}: best R@1 = {best_r1:.1f}, current R@1 = {(current_avg_r1):.1f}")
            if not_improved_num >= args.patience:
                logging.info(f"Performance did not improve for {not_improved_num} epochs. Stop training.")
                break

    logging.info(f"Best R@1: {best_r1:.2f}")
    logging.info(f"Trained for {epoch_num + 1:02d} epochs, in total in {str(datetime.now() - start_time)[:-7]}")


    for seq, test_ds in zip(test_sequences, test_ds_list):
        logging.info(f"===== Evaluating Sequence: {seq} =====")
        recalls, recalls_str = inference.inference(args, test_ds, model)
        logging.info(f"Recalls for {seq}: {recalls_str}")
        logging.info(f"================================================")