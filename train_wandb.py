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

def clip_patch_alignment_mean_loss(thermal_patches, rgb_patches, temperature=0.07):
    # https://taeyuplab.tistory.com/16
    B, N, D = thermal_patches.shape # (batch, 16x16, feature_dim)
    
    # 
    thermal_feat = thermal_patches.mean(dim=1)
    rgb_feat = rgb_patches.mean(dim=1)
    
    thermal_feat = F.normalize(thermal_feat, dim=-1)
    rgb_feat = F.normalize(rgb_feat, dim=-1)
    
    logits = torch.matmul(thermal_feat, rgb_feat.T) / temperature
    labels = torch.arange(B, device=logits.device)
    
    loss_t2r = F.cross_entropy(logits, labels)
    # loss_r2t = F.cross_entropy(logits.T, labels)
    
    # FIXME: loss_r2t는 필요없지 않나
    # return (loss_t2r + loss_r2t) / 2
    return loss_t2r

def clip_alignment_cls_loss(thermal_cls, rgb_cls, temperature=0.07):
    """
    thermal_cls: (B, D) - CLS token features
    rgb_cls: (B, D) - CLS token features
    """
    thermal_feat = F.normalize(thermal_cls, dim=-1)
    rgb_feat = F.normalize(rgb_cls, dim=-1)
    
    logits = torch.matmul(thermal_feat, rgb_feat.T) / temperature
    labels = torch.arange(thermal_feat.size(0), device=logits.device)
    
    loss_t2r = F.cross_entropy(logits, labels)
    
    # FIXME: 이거 loss_r2t도 넣어서 해보기, 필요한거 같음
    return loss_t2r

def patch_alignment_loss(thermal_patches, rgb_patches):
    """
    thermal_patches: (B, 256, 768)
    rgb_patches: (B, 256, 768)
    """
    # L2 normalize
    thermal_norm = F.normalize(thermal_patches, dim=-1)  # (B, 256, 768)
    rgb_norm = F.normalize(rgb_patches, dim=-1)          # (B, 256, 768)
    
    # 같은 위치 patch끼리 cosine similarity
    cos_sim = (thermal_norm * rgb_norm).sum(dim=-1)  # (B, 256)
    
    # similarity가 1에 가까울수록 좋음
    loss = 1 - cos_sim.mean()
    return loss

def attention_weighted_patch_alignment_loss(thermal_patches, rgb_patches, thermal_attn, rgb_attn, use_intranorm=False):
    """
    공통 중요도로 가중치를 준 patch alignment
    
    Args:
        thermal_patches: (B, num_patches, 768)
        rgb_patches: (B, num_patches, 768)
        thermal_attn: (B, num_heads, num_tokens)
        rgb_attn: (B, num_heads, num_tokens)
        use_intranorm: if True, L2-normalize patches before averaging (GeM-style)
    
    Returns:
        loss
    """
    # Multi-head averaging + CLS 토큰 제거
    thermal_attn_flat = thermal_attn.mean(dim=1)[:, 1:]  # (B, num_patches)
    rgb_attn_flat = rgb_attn.mean(dim=1)[:, 1:]  # (B, num_patches)
    
    # 공통 중요도
    # importance = thermal_attn_flat * rgb_attn_flat
    # importance = importance / (importance.sum(dim=1, keepdim=True) + 1e-8)
    importance = rgb_attn_flat
    importance = importance / (importance.sum(dim=1, keepdim=True) + 1e-8)
    
    if use_intranorm:
        thermal_patches = F.normalize(thermal_patches, dim=-1)
        rgb_patches = F.normalize(rgb_patches, dim=-1)
        
        thermal_agg = (thermal_patches * importance.unsqueeze(-1)).sum(dim=1)
        rgb_agg = (rgb_patches * importance.unsqueeze(-1)).sum(dim=1)
        
        # F.cosine_similarity 사용
        similarity = F.cosine_similarity(thermal_agg, rgb_agg, dim=-1)
        loss = 1 - similarity.mean()
    else:
        ...
        # # 원래 방식: Patch-wise cosine similarity
        # patch_sim = (thermal_patches * rgb_patches).sum(dim=-1)  # (B, num_patches)
        
        # # Importance-weighted similarity
        # weighted_sim = (patch_sim * importance).sum(dim=1)  # (B,)
        # loss = 1 - weighted_sim.mean()
    
    return loss

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
    wandb.init(project="cross-modal-vpr", name=args.comment, config=vars(args))

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
    print("use_alignment_loss: ", args.use_alignment_loss)
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
        pretrained_foundation = True,
        foundation_model_path = args.foundation_model_path, 
        use_GeMAdditionalLayer=args.use_GeMAdditionalLayer,
        mask_ratio=args.croco_mask_ratio,
        use_single_pass=args.use_single_pass, use_reduced_thermal_patch=args.use_reduced_thermal_patch,
        num_decoder_depth=args.num_decoder_depth
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
            triplets_ds.is_inference = False

            logging.debug("Finish computing triplets")

            ### triplet을 위한 DataLoader 생성
            # DataLoader는 (Batch, images, triplets_local_indexes, triplets_global_indexes[index], aligned_rgb)를 getitem
            # images: stacked(query, pos, *negs)
            # triplets_local_indexes: triplet 내에서의 local index
            #   ex) (0, 1, 2), (0, 1, 3), ..., (0, 1, 11) # (query, pos, neg)의 pairs
            # triplets_global_indexes[index]: ?
            # aligned_rgb: thermal query와 맞는 rgb query 이미지
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
                global_features, patch_embedding, recon_loss, masks, masked_patch_embedding = model(
                    images.to(args.device),
                    flags=flags,
                    aligned_rgb=aligned_rgbs.to(args.device),
                    return_mask=True,
                    return_masked_patch=True
                )

                # triplets_local_indexes = (batch, 3, neg_num) => [[[0, 1, 2], [0, 1, 3] ... [0, 1, neg_num+2]] * batch]
                triplets_local_indexes = torch.transpose(
                    triplets_local_indexes.view(args.train_batch_size, args.negs_num_per_query, 3), 1, 0)
                
                overall_loss = 0
                triplet_loss_sum = 0
                # 각 triplet에 대해 triplet loss 계산
                for triplets in triplets_local_indexes:
                    queries_indexes, positives_indexes, negatives_indexes = triplets.T
                    
                    # 각각에 해당하는 descriptor 추출
                    query_features = global_features[queries_indexes]
                    positive_features = global_features[positives_indexes]
                    negative_features = global_features[negatives_indexes]

                    triplet_loss = GlobalTriplet(query_features, positive_features, negative_features)
                    triplet_loss_sum += triplet_loss
                    
                    # triplet_loss
                    overall_loss += triplet_loss
                
                rerank_loss = 0
                if args.use_rerank_loss:
                    # TODO: augmentation 문제 없는지도 확인
                    # constrastive_reconstruction loss 구하기
                    query_lists = [i for i in range(0, args.train_batch_size)] 
                    positive_lists = [i for i in range(1, args.train_batch_size * (args.negs_num_per_query+2), (args.negs_num_per_query+2))] 
                    negative_lists = [i 
                        for batch_idx in range(args.train_batch_size)
                        for i in range(
                            batch_idx * (args.negs_num_per_query + 2) + 2,  # start+2부터
                            (batch_idx + 1) * (args.negs_num_per_query + 2)  # start+12까지
                        )
                    ]
                    
                    query_embedding = masked_patch_embedding[query_lists].detach()
                    positive_embedding = patch_embedding[positive_lists].detach()
                    negative_embedding = patch_embedding[negative_lists].detach()
                    
                    # 아침에 여기 고치기
                    # loss는 어떻게? 비율은?
                    
                    # 2. 전부 pos embedding 추가
                    decoder_pos_embed = model.module.decoder_pos_embed
                    query_embedding = query_embedding + decoder_pos_embed
                    positive_embedding = positive_embedding + decoder_pos_embed
                    negative_embedding = negative_embedding + decoder_pos_embed

                    decoded_embeddings = torch.zeros(
                        (args.train_batch_size * (args.negs_num_per_query+1), 256, args.features_dim), device=args.device)
                    
                    # c. query를 batch로 만들어주기
                    for batch_idx in range(args.train_batch_size):
                        query_embedding_batch = query_embedding[batch_idx,:,:].unsqueeze(0).expand((1+args.negs_num_per_query), -1, -1)  # [RERANKING_TOP_K, N_visible, 768]
                        
                        # d. pos/neg 순으로 쌓아주기 (pos, pos, pos, pos, neg, neg ... neg)
                        pos_neg_embedding_batch = torch.cat(
                            [positive_embedding[batch_idx,:,:].unsqueeze(0),
                             negative_embedding[batch_idx*args.negs_num_per_query:(batch_idx+1)*args.negs_num_per_query,:,:]], dim=0)
                        
                        # 3. decoder 통과시키기
                        # 3-1. Thermal decoder 통과
                        thermal_dec = query_embedding_batch
                        for blk in model.module.decoder_blocks:
                            thermal_dec = blk(thermal_dec, pos_neg_embedding_batch)
                        thermal_dec = model.module.decoder_norm(thermal_dec)
                        
                        decoded_embeddings[batch_idx*(args.negs_num_per_query+1):(batch_idx+1)*(args.negs_num_per_query+1)] = thermal_dec
                    
                    # 5. infoNCE loss 계산
                    infoNCE()
                    decoded_embeddings.shape # (B*12,256,768)

                    # 5. descriptor 기반의 infoNCE loss
                    
                    # GAP하고 infoNCE loss 계산(의미 있나?)
                    # 그리고 GAP를 한다는 것은 global한 레벨에서 본다는것
                    # 그럼 GeM pooling하는 방법과 다른게 무엇인가?
                    

                # train_batch_size: 4, arg.negs_num_per_query: 10
                recon_weight = args.recon_weight
                rerank_weight = args.rerank_weight
                overall_loss += (recon_loss * recon_weight)
                overall_loss += (rerank_loss * rerank_weight)
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
                    "train/recon_loss(scaled)": (recon_loss * recon_weight).item() / (args.train_batch_size * args.negs_num_per_query) if isinstance(recon_loss, torch.Tensor) else 0,
                    "train/rerank_loss(scaled)": (rerank_loss * rerank_weight).item() / (args.train_batch_size * args.negs_num_per_query) if isinstance(rerank_loss, torch.Tensor) else 0,
                }, step=global_step)
                
                global_step += 1

                del overall_loss, triplet_loss, recon_loss
            logging.info(f"Epoch[{epoch_num:02d}]({loop_num + 1}/{loops_num}): " +
                        f"current batch triplet loss = {batch_loss:.8f}, " +
                        f"average epoch triplet loss = {epoch_losses.mean():.8f}")
        
        visualize_during_training(
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
            recalls, recalls_str = inference.inference(args, test_ds, model)
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

                    