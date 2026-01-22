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
import network
import random
from croco.models.criterion import MaskedMSE
from recon_vis import visualize_during_training
from info_nce import InfoNCE, info_nce

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

    # wandb 초기화
    wandb.init(project="cross-modal-vpr-2", name=args.comment, config=vars(args))

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

    args.sequences = ['SNU', 'Valley']
    test_sequences = args.sequences
    test_ds_list = []
    for seq in test_sequences:
        args.sequences = [seq]
        test_ds = datasets_T2R.BaseSTheReODual(args, DATASET_FOLDER, split='test')
        test_ds_list.append(test_ds)

    '''Model'''
    model = network.CrossModalVPR_Net(
        args,
        pretrained_foundation = True,
        foundation_model_path = args.foundation_model_path,
    )
    model = model.to(args.device)
    model = torch.nn.DataParallel(model)

    backbone_params = []
    other_params    = []
    print("="*30)
    print("- Tuning RGB backbone layers: ", args.num_trainable_blocks_RGB)
    print("- Tuning THERMAL backbone layers: ", args.num_trainable_blocks_THERMAL)
    print("="*30)
    for name, param in model.module.rgb_backbone.named_parameters():
        if "adapter" not in name:
            param.requires_grad = False
        for i in range(args.num_trainable_blocks_RGB):
            num_blocks = len(model.module.rgb_backbone.blocks)
            model.module.rgb_backbone.blocks[num_blocks - i - 1].requires_grad_(True)

    for name, param in model.module.thermal_backbone.named_parameters():
        if "adapter" not in name:
            param.requires_grad = False
        for i in range(args.num_trainable_blocks_THERMAL):
            num_blocks = len(model.module.thermal_backbone.blocks)
            model.module.thermal_backbone.blocks[num_blocks - i - 1].requires_grad_(True)

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
                        
    if args.use_sepearte_backbone_lr:
        backbone_params = []
        other_params = []
        print("="*30)
        print(f"Using seperate LR !!!")
        print(f"- backbone LR: \t{args.backbone_lr}")
        print(f"- other LR: \t{args.lr}")
        print("="*30)

        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'rgb_backbone' in name or 'thermal_backbone' in name:
                    backbone_params.append(param)
                else: other_params.append(param)

        '''Seperate Learning Rate'''
        if args.optim == "adam":
            optimizer = torch.optim.Adam([
                {'params': backbone_params, 'lr': args.backbone_lr},
                {'params': other_params, 'lr': args.lr}
            ])
        elif args.optim == "sgd":
            optimizer = torch.optim.SGD([
                {'params': backbone_params, 'lr': args.backbone_lr, 'momentum': 0.9, 'weight_decay': 0.001},
                {'params': other_params, 'lr': args.lr, 'momentum': 0.9, 'weight_decay': 0.001}
            ])
    else:
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

    bundle_flags =  ['thermal'] + ['rgb'] * (1 + args.negs_num_per_query)
    num_bundle_flags = len(bundle_flags)

    '''Training'''
    global_step = 0
    for epoch_num in range(start_epoch_num, args.epochs_num):
        logging.info(f"Start training epoch: {epoch_num:02d}")

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

            print("- Training...")
            for images, triplets_local_indexes, _, aligned_rgbs in tqdm(triplets_dl, ncols=100, desc=f"Epoch {epoch_num:02d}"):
                curr_batch_len = len(images) // num_bundle_flags
                flags = bundle_flags * curr_batch_len
                
                ### model을 통해, triplet의 descriptor와 patch embedding 추출
                if args.use_pos_as_aligned_rgb:
                    assert images.size(0) % args.train_batch_size == 0
                    size_of_batch = int(images.size(0) / args.train_batch_size)
                    train_batch_size = args.train_batch_size
                    
                    pos_rgbs = [images[idx] for idx in range(1, images.size(0), size_of_batch)]
                    pos_rgbs = torch.stack(pos_rgbs)
                    aligned_rgbs = pos_rgbs

                global_features, patch_embedding, \
                recon_loss, masks, cls_attn_map, \
                penultimate_patch_embedding, masked_patch_embedding, \
                thermal_cross_attn_map = model(
                    images.to(args.device),
                    flags=flags,
                    paired_rgb=aligned_rgbs.to(args.device),
                    return_mask=True,
                    return_masked_patch=True
                )

                # triplets_local_indexes = (batch, 3, neg_num) => [[[0, 1, 2], [0, 1, 3] ... [0, 1, neg_num+2]] * batch]
                triplets_local_indexes = torch.transpose(
                    triplets_local_indexes.view(args.train_batch_size, args.negs_num_per_query, 3), 1, 0)
                
                overall_loss = 0
                triplet_loss_sum = 0
                rerank_loss_sum = 0
                # 각 triplet에 대해 triplet loss 계산
                for triplets in triplets_local_indexes:
                    queries_indexes, positives_indexes, negatives_indexes = triplets.T
                    
                    # 각각에 해당하는 descriptor 추출
                    query_features = global_features[queries_indexes]
                    positive_features = global_features[positives_indexes]
                    negative_features = global_features[negatives_indexes]

                    # Triplet Loss
                    triplet_loss = GlobalTriplet(query_features, positive_features, negative_features)
                    triplet_loss_sum += triplet_loss
                    
                    # Reranking loss
                    reranker = model.module.reranker
                    if args.r2_penultimate_layer:
                        rerank_patch_embedding = penultimate_patch_embedding
                    else:
                        rerank_patch_embedding = patch_embedding

                    if args.r2loss_div < 0.001:
                        # Decoder까지 학습 (encoder는 차단)
                        rerank_loss = reranker(
                            rerank_patch_embedding.detach(),
                            cls_attn_map.detach(),
                            queries_indexes, positives_indexes, negatives_indexes,
                            query_features.detach(),  # triplet loss와 분리
                            positive_features.detach(),
                            negative_features.detach(),
                            cross_attn_matrix=thermal_cross_attn_map
                        )
                        overall_loss += (triplet_loss.detach() + rerank_loss)  # triplet loss는 별도 학습
                        rerank_loss_sum += rerank_loss
                    else:
                        # 동일하게 수정
                        rerank_loss = reranker(
                            rerank_patch_embedding.detach(),
                            cls_attn_map.detach(),
                            queries_indexes, positives_indexes, negatives_indexes,
                            query_features.detach(),
                            positive_features.detach(),
                            negative_features.detach(),
                            cross_attn_matrix=thermal_cross_attn_map
                        )
                        overall_loss += (triplet_loss.detach() + rerank_loss / args.r2loss_div)
                    
                    
                # train_batch_size: 4, arg.negs_num_per_query: 10
                recon_weight = args.recon_weight
                overall_loss += (recon_loss * recon_weight)
                overall_loss /= (args.train_batch_size * args.negs_num_per_query)

                del global_features, query_features, positive_features, negative_features

                optimizer.zero_grad()
                overall_loss.backward()
                optimizer.step()

                batch_loss = overall_loss.item()
                epoch_losses = np.append(epoch_losses, batch_loss)

                # wandb logging
                wandb.log({
                    "train/overall_loss": overall_loss,
                    "train/triplet_loss(scaled)": triplet_loss_sum.item() / (args.train_batch_size * args.negs_num_per_query),
                    "train/reranking_loss(scaled)": rerank_loss_sum.item() / (args.train_batch_size * args.negs_num_per_query),
                    "train/recon_loss(scaled)": (recon_loss * recon_weight).item() / (args.train_batch_size * args.negs_num_per_query) if isinstance(recon_loss, torch.Tensor) else 0,
                }, step=global_step)
                
                global_step += 1

                del patch_embedding, masks, cls_attn_map, penultimate_patch_embedding, masked_patch_embedding
                del overall_loss, triplet_loss, recon_loss
                if args.use_fast_track: break
            
            logging.info(f"Epoch[{epoch_num:02d}]({loop_num + 1}/{loops_num}): " +
                        f"current batch triplet loss = {batch_loss:.8f}, " +
                        f"average epoch triplet loss = {epoch_losses.mean():.8f}")
        
        visualize_during_training(
            args,
            model, 
            triplets_dl, 
            args.device, 
            epoch_num,
            save_dir=os.path.join(args.save_dir, 'reconstructions'),
            comment=args.comment
        )

        # wandb 로깅 (epoch 단위)
        wandb.log({"train/epoch_avg_loss": epoch_losses.mean(), "epoch": epoch_num}, step=global_step)
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
                logging.info(f"Performance did not improve for {not_improved_num} epochs. Stop training.")
                # break # 굳이 멈출 필요까지야
        
        print(f"Comment: {args.comment} :: Epoch {epoch_num:02d}")

    logging.info(f"Best R@1: {best_r1:.2f}")
    logging.info(f"Trained for {epoch_num + 1:02d} epochs, in total in {str(datetime.now() - start_time)[:-7]}")

    for seq, test_ds in zip(test_sequences, test_ds_list):
        logging.info(f"===== Evaluating Sequence: {seq} =====")
        recalls, recalls_str = inference.inference(args, test_ds, model)
        logging.info(f"Recalls for {seq}: {recalls_str}")
        logging.info(f"================================================")

    wandb.finish()
    
    
                    # 3. decoder 통과시키기
                    # 3-1. Thermal decoder 통과
                    # negs_num = args.negs_num_per_query
                    # decoded_embeddings = torch.zeros(
                    #     (args.train_batch_size * (negs_num+1), 256, args.features_dim), device=args.device)   
                    
                    # for query_idx in range(query_embedding.size(0)):
                    #     # positive-query decoder 통과
                    #     thermal_dec = query_embedding[query_idx,:,:]
                    #     positive_dec = positive_embedding[query_idx]
                    #     print(f"query: {query_idx} / positive: {query_idx*(negs_num+1)}")
                        
                    #     for blk in model.module.decoder_blocks:
                    #         decoded_embeddings[query_idx*(negs_num+1)] = blk(thermal_dec, positive_dec)
                    #     decoded_embeddings[query_idx*(negs_num+1)] = model.module.decoder_norm(decoded_embeddings[query_idx*negs_num])                        
                        
                    #     # negatives-query decoder 통과
                    #     for negative_idx in range(negs_num):
                    #         negative_dec = negative_embedding[query_idx*(negs_num+1)+negative_idx]
                        
                    #         print(f"query: {query_idx} / negative: {query_idx*(negs_num+1)+negative_idx}")
                    #         for blk in model.module.decoder_blocks:
                    #             decoded_embeddings[query_idx*(negs_num+1)+negative_idx+1] = blk(thermal_dec, negative_dec)
                    #         decoded_embeddings[query_idx*(negs_num+1)+negative_idx+1] = model.module.decoder_norm(decoded_embeddings[query_idx*(negs_num+1)+negative_idx+1]) 

                    