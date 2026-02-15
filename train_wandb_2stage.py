# train_wandb.py
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
    wandb.init(project="cross-modal-vpr-4", name=args.comment, config=vars(args))

    args.save_dir = os.path.join(args.save_dir, args.comment, utils.get_timestamp())
    commons.setup_logging(args.save_dir)
    commons.seed_everything(args.seed)

    start_time = datetime.now()

    utils.save_to_yaml(args)
    logging.debug(f"The outputs are being saved in {args.save_dir}")
    logging.info(f"Use {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs")

    DATASET_FOLDER = "./Dataset/save_mat"

    '''Datasets'''
    args.sequences = ['KAIST']
    triplets_ds = datasets_T2R.TripletsSTheReODual(args, DATASET_FOLDER, use_align_rgb=True)
    train_ds = datasets_T2R.BaseSTheReODual(args, DATASET_FOLDER, split='train')
    logging.info(f"[Train - KAIST] Database: {train_ds.database_num}, Queries: {train_ds.queries_num}, Total: {len(train_ds)}")

    args.sequences = ['Valley', 'SNU']
    test_sequences = args.sequences
    test_ds_list = []
    for seq in test_sequences:
        args.sequences = [seq]
        test_ds = datasets_T2R.BaseSTheReODual(args, DATASET_FOLDER, split='test')
        test_ds_list.append(test_ds)
        logging.info(f"[Test - {seq}] Database: {test_ds.database_num}, Queries: {test_ds.queries_num}, Total: {len(test_ds)}")

    '''Model'''
    if args.use_recon_loss:
        model = network.CrossModalVPR_Net(
            args,
            pretrained_foundation = True,
            foundation_model_path = args.foundation_model_path,
        )
    else:
        model = network_only_GeM.CrossModalVPR_Net(
            args,
            pretrained_foundation = True,
            foundation_model_path = args.foundation_model_path,
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

    '''Loss Function - Initialize early for optimizer'''
    GlobalTriplet = nn.TripletMarginLoss(margin=args.margin, p=2, reduction="sum")

    train_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.requires_grad = False
            if 'similarity_score' in name or \
                'recursive_' in name or \
                'dino_dec_cls_token' in name or \
                'dino_decoder' in name:
                param.requires_grad = True
    
    if args.unfreeze_dino_decoder:
        model.module.dino_decoder.unfreeze_dino_blocks()
    else:
        model.module.dino_decoder.freeze_dino_blocks()
        
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

    thermal_flag = torch.zeros(1, dtype=torch.long)
    rgb_flags = torch.ones(1 + args.negs_num_per_query, dtype=torch.long)
    bundle_flags = torch.cat([thermal_flag, rgb_flags]) # [0, 1, 1, ..., 1]
    flags = bundle_flags.repeat(args.train_batch_size)

    '''Training'''
    global_step = 0
    for epoch_num in range(start_epoch_num, args.epochs_num):
        # Log training mode based on epoch
        rgb_mode = "paired RGB" if epoch_num < args.paired_rgb_epochs else "positive RGB"
        logging.info(f"Start training epoch: {epoch_num:02d} (reconstruction with {rgb_mode})")

        epoch_start_time = datetime.now()
        epoch_losses = np.zeros((0, 1), dtype=np.float32)

        loops_num = math.ceil(args.queries_per_epoch / args.cache_refresh_rate)
        for loop_num in range(loops_num):
            logging.debug(f"Cache: {loop_num + 1} / {loops_num}")

            ### contrastive learning을 위한 triplet 샘플링
            triplets_ds.is_inference = True
            print("- Computing triplets...")
            triplets_ds.compute_triplets(args, model)
            torch.cuda.empty_cache()
            triplets_ds.is_inference = False

            logging.debug("Finish computing triplets")

            ### triplet을 위한 DataLoader 생성
            # DataLoader는 (Batch, images, triplets_local_indexes, triplets_global_indexes[index], paired_rgb)를 getitem
            # images: stacked(query, pos, *negs)
            # triplets_local_indexes: triplet 내에서의 local index
            #   ex) (0, 1, 2), (0, 1, 3), ..., (0, 1, 11) # (query, pos, neg)의 pairs
            # triplets_global_indexes[index]: ?
            # paired_rgb: thermal query와 맞는 rgb query 이미지
            triplets_dl = DataLoader(dataset=triplets_ds, num_workers=args.num_workers,
                                    batch_size=args.train_batch_size,
                                    collate_fn=datasets_T2R.collate_fn,
                                    pin_memory=(args.device == "cuda"),
                                    drop_last=True)
            
            model = model.train()
            logging.debug(f"Start loading {len(triplets_ds)} triplets as {len(triplets_dl)} batches")

            # Determine whether to use paired RGB or positive RGB based on epoch
            use_paired_rgb = epoch_num < args.paired_rgb_epochs
            if epoch_num == args.paired_rgb_epochs:
                logging.info(f"[Epoch {epoch_num}] Switching from paired RGB to positive RGB for reconstruction")

            print("- Training...")
            for images, triplets_local_indexes, _, aligned_rgbs in tqdm(triplets_dl, ncols=100, desc=f"GPU{args.cuda_device}/Epoch {epoch_num:02d}"):
                ### model을 통해, triplet의 descriptor와 patch embedding 추출
                # Use positive RGB instead of aligned RGB after paired_rgb_epochs
                if args.use_pos_as_aligned_rgb or not use_paired_rgb:
                    assert images.size(0) % args.train_batch_size == 0
                    size_of_batch = int(images.size(0) / args.train_batch_size)
                    train_batch_size = args.train_batch_size

                    pos_rgbs = [images[idx] for idx in range(1, images.size(0), size_of_batch)]
                    pos_rgbs = torch.stack(pos_rgbs)
                    pos_rgbs = triplets_ds.transform(pos_rgbs)
                    aligned_rgbs = pos_rgbs

                recon_loss = None
                if args.use_recon_loss:
                    global_features, patch_embedding, \
                    recon_loss, masks, cls_attn_map, \
                    penultimate_patch_embedding, local_embedding, _, masked_patch_embedding = model(
                        images.to(args.device),
                        flags=flags,
                        paired_rgb=aligned_rgbs.to(args.device),
                        return_mask=True,
                        return_masked_patch=True
                    )
                else:
                    outputs = model(
                        images.to(args.device),
                        flags=flags,
                        paired_rgb=aligned_rgbs.to(args.device),
                        return_mask=True,
                        return_masked_patch=True
                    )
                    global_features = outputs[0]
                    patch_embedding = outputs[1]
                    cls_attn_map = outputs[4]
                    local_embedding = outputs[6]

                # triplets_local_indexes = (batch, 3, neg_num) => [[[0, 1, 2], [0, 1, 3] ... [0, 1, neg_num+2]] * batch]
                triplets_local_indexes = torch.transpose(
                    triplets_local_indexes.view(args.train_batch_size, args.negs_num_per_query, 3), 1, 0)
                
                dino_dec_loss_sum = 0
                triplet_loss_sum = 0
                rerank_loss_sum = 0
                local_loss_sum = 0
                diff_loss_sum = 0
                optimizer.zero_grad()
                num_steps = len(triplets_local_indexes)
                # 각 triplet에 대해 triplet loss 계산
                for triplets in triplets_local_indexes:
                    overall_triplet_loss = 0
                    queries_indexes, positives_indexes, negatives_indexes = triplets.T
                    
                    # 각각에 해당하는 descriptor 추출
                    query_features = global_features[queries_indexes]
                    positive_features = global_features[positives_indexes]
                    negative_features = global_features[negatives_indexes]
                    
                    ## 새로 추가한거
                    dino_dec_loss = model.module.stage2_forward(patch_embedding, queries_indexes, positives_indexes, negatives_indexes)
                    overall_triplet_loss += dino_dec_loss
                    dino_dec_loss_sum += dino_dec_loss

                    # Reranking loss
                    if args.use_recon_loss and args.r2_penultimate_layer:
                        rerank_patch_embedding = penultimate_patch_embedding
                    else:
                        rerank_patch_embedding = patch_embedding

                    if args.use_reranking == 'r2former':
                        reranker = model.module.reranker
                        rerank_loss = reranker(rerank_patch_embedding.detach(), cls_attn_map.detach(),
                                            queries_indexes, positives_indexes, negatives_indexes,
                                            query_features.detach(), positive_features.detach(), negative_features.detach())
                        overall_triplet_loss += rerank_loss
                        rerank_loss_sum += rerank_loss

                    if args.use_sela_local_loss:
                        local_loss = model.module.MNNLocalFeatureLoss([
                                    local_embedding[queries_indexes],
                                    local_embedding[positives_indexes],
                                    local_embedding[negatives_indexes]])
                        overall_triplet_loss += local_loss
                        local_loss_sum += local_loss
                        
                    # new reranking loss
                    if args.use_diff_loss:
                        diff_loss, lambda_ = model.module.DiffGeMLoss(
                            model.module, queries_indexes, positives_indexes, negatives_indexes,
                            global_features.detach(), patch_embedding.detach(),
                            use_train=True, return_lambda=True
                        )
                        overall_triplet_loss += diff_loss
                        diff_loss_sum += diff_loss
                
                    overall_triplet_loss /= (args.train_batch_size * args.negs_num_per_query)
                    overall_triplet_loss.backward()
                    
                del global_features, query_features, positive_features, negative_features
                optimizer.step()

                batch_loss = overall_triplet_loss.item()
                epoch_losses = np.append(epoch_losses, batch_loss)

                # wandb logging
                wandb.log({
                    "train/overall_loss": overall_triplet_loss.item(),
                    "train/triplet_loss(scaled)": triplet_loss_sum.item() / (args.train_batch_size * args.negs_num_per_query)
                        if isinstance(triplet_loss_sum, torch.Tensor) else 0,
                    "train/reranking_loss(scaled)": rerank_loss_sum.item() / (args.train_batch_size * args.negs_num_per_query)
                        if isinstance(rerank_loss_sum, torch.Tensor) else 0,
                    "train/local_loss(scaled)": local_loss_sum.item() / (args.train_batch_size * args.negs_num_per_query)
                        if isinstance(local_loss_sum, torch.Tensor) else 0,
                    "train/diff_loss(scaled)": diff_loss_sum.item() / (args.train_batch_size * args.negs_num_per_query)
                        if isinstance(diff_loss_sum, torch.Tensor) else 0,
                }, step=global_step)
                
                global_step += 1
                
                # del patch_embedding, masks, cls_attn_map, penultimate_patch_embedding, masked_patch_embedding
                # del overall_loss, triplet_loss, recon_loss
                if args.use_fast_track: break
                
            logging.info(f"Epoch[{epoch_num:02d}]({loop_num + 1}/{loops_num}): " +
                        f"current batch triplet loss = {batch_loss:.8f}, " +
                        f"average epoch triplet loss = {epoch_losses.mean():.8f}")
        
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

        # Visualize attention maps (penultimate and last layer) using PCA
        visualize_attention_maps_pca(
            args,
            model,
            triplets_dl,
            args.device,
            epoch_num,
            num_samples=4,
            save_dir=os.path.join(args.save_dir, 'attention_maps'),
            comment=args.comment
        )

        # wandb 로깅 (epoch 단위)
        # Log RGB mode: 1 = paired, 0 = positive
        rgb_mode_flag = 1 if epoch_num < args.paired_rgb_epochs else 0
        wandb.log({
            "train/epoch_avg_loss": epoch_losses.mean(),
            "train/rgb_mode": rgb_mode_flag,  # 1=paired, 0=positive
            "epoch": epoch_num
        }, step=global_step)
        logging.info(f"epoch {epoch_num:02d} time: {str(datetime.now() - epoch_start_time)[:-7]}, ")

        # Compute recalls
        current_epoch_r1_list = []
        for seq, test_ds in zip(test_sequences, test_ds_list):
            logging.info(f"===== Evaluating Sequence: {seq} =====")
            args.current_epoch = epoch_num # 시각화
            recalls, recalls_str = inference.inference(args, test_ds, model, seq_name=seq)
            logging.info(f"Recalls for {seq}: {recalls_str}")
            logging.info(f"================================================")
            current_epoch_r1_list.append(recalls[0])
            
            # wandb 로깅 (sequence별 recall)
            wandb.log({f"val/{seq}_R@1": recalls[0], f"val/{seq}_R@5": recalls[1]}, step=global_step)

        current_avg_r1 = np.mean(current_epoch_r1_list)
        is_best = current_avg_r1 > best_r1

        # wandb 로깅 (평균 recall)
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
                logging.info(f"Performance did not improve for {not_improved_num} epochs.")
        
        print(f"Comment: {args.comment} :: Epoch {epoch_num:02d}")
        import gc
        del recalls, recalls_str # 필요 없다면 삭제
        if 'current_epoch_r1_list' in locals(): del current_epoch_r1_list
        
        gc.collect()            # Python 가비지 컬렉션 (참조 잃은 변수 제거)
        torch.cuda.empty_cache() # PyTorch VRAM 캐시 비우기 (OS로 반환)
        
    logging.info(f"Best R@1: {best_r1:.2f}")
    logging.info(f"Trained for {epoch_num + 1:02d} epochs, in total in {str(datetime.now() - start_time)[:-7]}")

    for seq, test_ds in zip(test_sequences, test_ds_list):
        logging.info(f"===== Evaluating Sequence: {seq} =====")
        recalls, recalls_str = inference.inference(args, test_ds, model)
        logging.info(f"Recalls for {seq}: {recalls_str}")
        logging.info(f"================================================")

    wandb.finish()