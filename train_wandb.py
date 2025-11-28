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

def clip_patch_alignment_cls_loss(thermal_cls, rgb_cls, temperature=0.07):
    """
    thermal_cls: (B, D) - CLS token features
    rgb_cls: (B, D) - CLS token features
    """
    thermal_feat = F.normalize(thermal_cls, dim=-1)
    rgb_feat = F.normalize(rgb_cls, dim=-1)
    
    logits = torch.matmul(thermal_feat, rgb_feat.T) / temperature
    labels = torch.arange(thermal_feat.size(0), device=logits.device)
    
    loss_t2r = F.cross_entropy(logits, labels)
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
    triplets_ds = datasets_T2R.TripletsSTheReODual(args, DATASET_FOLDER, use_align_rgb=args.use_alignment_loss)
    train_ds = datasets_T2R.BaseSTheReODual(args, DATASET_FOLDER, split='train')

    args.sequences = ['SNU', 'Valley']
    test_sequences = args.sequences
    test_ds_list = []
    for seq in test_sequences:
        args.sequences = [seq]
        test_ds = datasets_T2R.BaseSTheReODual(args, DATASET_FOLDER, split='test')
        test_ds_list.append(test_ds)

    '''Model'''
    model = network.CrossModalVPR_Net(pretrained_foundation = True, foundation_model_path = args.foundation_model_path, use_GeMAdditionalLayer=args.use_GeMAdditionalLayer)
    model = model.to(args.device)
    model = torch.nn.DataParallel(model)

    backbone_params = []
    other_params    = []
    for name, param in model.module.rgb_backbone.named_parameters():
        if "adapter" not in name:
            param.requires_grad = False
        for i in range(args.num_trainable_blocks):
            num_blocks = len(model.module.rgb_backbone.blocks)
            model.module.rgb_backbone.blocks[num_blocks - i - 1].requires_grad_(True)

    for name, param in model.module.thermal_backbone.named_parameters():
        if "adapter" not in name:
            param.requires_grad = False
        for i in range(args.num_trainable_blocks):
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

        '''Seperate Optimizer'''
        if args.optim == "adam":
            optimizer = torch.optim.Adam([
                {'params': backbone_params, 'lr': args.lr * 0.1},  # backbone은 10배 작은 lr
                {'params': other_params, 'lr': args.lr}
            ])
        elif args.optim == "sgd":
            optimizer = torch.optim.SGD([
                {'params': backbone_params, 'lr': args.lr * 0.1, 'momentum': 0.9, 'weight_decay': 0.001},
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
            for images, triplets_local_indexes, _, aligned_rgbs in tqdm(triplets_dl, ncols=100, desc=f"alignmentLoss:{args.use_alignment_loss}::Epoch {epoch_num:02d}"):
                curr_batch_len = len(images) // num_bundle_flags
                flags = bundle_flags * curr_batch_len
                
                ### model을 통해, triplet의 descriptor와 patch embedding 추출
                global_features, patch_embedding, cls_embedding = model(images.to(args.device), flags=flags, return_embedding=True)

                overall_loss = 0
                alignment_loss = 0
                if aligned_rgbs is not None:
                    # 만약 alignment_loss를 사용한다면 (def clip_patch_alignment_mean_loss 참고)
                    aligned_rgbs = aligned_rgbs.to(args.device)
                    
                    model.eval()
                    with torch.no_grad():
                        _, aligned_rgb_embedding, aligned_rgb_cls_embedding = model(aligned_rgbs, flags=['rgb'] * len(aligned_rgbs), return_embedding=True)
                    
                    # thermal query에 해당하는 patch_embedding만 추출(0, 12, 24, 36, maybe...)
                    thermal_indices = [i * num_bundle_flags for i in range(curr_batch_len)]
                    # thermal_patches = patch_embedding[thermal_indices]
                    
                    thermal_cls = cls_embedding[thermal_indices]
                    aligned_rgb_cls = aligned_rgb_cls_embedding 
                    
                    # alignment_loss 구하기
                    # alignment_loss = clip_patch_alignment_mean_loss(thermal_patches, aligned_rgb_patches, temperature=0.07)
                    # alignment_loss = patch_alignment_loss(thermal_patches, aligned_rgb_patches)
                    alignment_loss = clip_patch_alignment_cls_loss(thermal_cls, aligned_rgb_cls)

                # triplets_local_indexes = (batch, 3, neg_num) => [[[0, 1, 2], [0, 1, 3] ... [0, 1, neg_num+2]] * batch]
                triplets_local_indexes = torch.transpose(
                    triplets_local_indexes.view(args.train_batch_size, args.negs_num_per_query, 3), 1, 0)
                
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

                # train_batch_size: 4, arg.negs_num_per_query: 10
                al_weight = 1.5 # loss 가중치(al: alignment loss)
                overall_loss += (alignment_loss * al_weight)
                overall_loss /= (args.train_batch_size * args.negs_num_per_query)

                del global_features, query_features, positive_features, negative_features

                optimizer.zero_grad()
                overall_loss.backward()
                optimizer.step()

                batch_loss = overall_loss.item()
                epoch_losses = np.append(epoch_losses, batch_loss)

                # wandb 로깅 (batch 단위)
                wandb.log({
                    "train/overall_loss": overall_loss,
                    "train/triplet_loss(scaled)": triplet_loss_sum.item() / (args.train_batch_size * args.negs_num_per_query),
                    "train/alignment_loss(scaled)": (alignment_loss * al_weight).item() / (args.train_batch_size * args.negs_num_per_query) if isinstance(alignment_loss, torch.Tensor) else 0,
                }, step=global_step)
                global_step += 1

                del overall_loss, triplet_loss, alignment_loss

            logging.info(f"Epoch[{epoch_num:02d}]({loop_num + 1}/{loops_num}): " +
                        f"current batch triplet loss = {batch_loss:.8f}, " +
                        f"average epoch triplet loss = {epoch_losses.mean():.8f}")

        # wandb 로깅 (epoch 단위)
        wandb.log({"train/epoch_avg_loss": epoch_losses.mean(), "epoch": epoch_num}, step=global_step)
        logging.info(f"epoch {epoch_num:02d} time: {str(datetime.now() - epoch_start_time)[:-7]}, ")

        # Compute recalls
        current_epoch_r1_list = []
        for seq, test_ds in zip(test_sequences, test_ds_list):
            logging.info(f"===== Evaluating Sequence: {seq} =====")
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