import faiss
import torch
import logging
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
import time
import cv2
import os
import torch.nn.functional as F
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

from croco.models.masking import RandomMask, AttentionMask
from croco.models.criterion import MaskedMSE
from local_matching import *

from recon_vis import *
from reranking_dataset import RerankingDataset, reranking_collate_fn, compute_batch_recon_loss

def patchify(imgs):
    """
    imgs: (B, 3, H, W)
    x: (B, L, patch_size**2 *3)
    """
    p = 14
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

    h = w = imgs.shape[2] // p
    x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
    
    return x

START_TIME = get_timestamp()
NPY_ROOTPATH = None
def save_npy(data, path):
    np.save(os.path.join(NPY_ROOTPATH, path), data)
    
def load_npy(path):
    return np.load(os.path.join(NPY_ROOTPATH, path)+".npy")


def batched_recon_reranking(
    model,
    dataloader,
    args,
    H_feat,
    W_feat,
    vis_query_indices=None
):
    """
    DataLoader 기반 배치 recon reranking.

    여러 쿼리를 한 번에 처리하여 GPU 활용도를 높입니다.

    Args:
        model: CrossModalVPR_Net 모델
        dataloader: RerankingDataset을 위한 DataLoader
        args: 인자들
        H_feat, W_feat: feature map 크기
        vis_query_indices: visualization할 쿼리 인덱스 리스트 (optional)

    Returns:
        predictions: [num_queries, K] reranked predictions
        rerank_scores_dict: {query_idx: scores} dictionary
        vis_data: visualization 데이터 (vis_query_indices가 주어진 경우)
    """
    predictions_list = []
    rerank_scores_dict = {}

    # Visualization 데이터 수집
    vis_data = {
        'top_k_indices': [],
        'recon_losses': [],
        'thermal_recon': [],
        'rgb_recon': [],
        'thermal_masks': [],
        'rgb_masks': [],
        'thermal_gem': [],
        'rgb_gem': [],
    } if vis_query_indices else None

    vis_query_set = set(vis_query_indices) if vis_query_indices else set()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Batched Recon Reranking"):
            B = batch['batch_size']
            K = batch['top_k']
            BK = B * K

            # GPU로 전송
            thermal_feat = batch['thermal_feat'].float().cuda()      # [BK, N, D]
            thermal_target = batch['thermal_target'].float().cuda()  # [BK, N, P]
            rgb_feat = batch['rgb_feat'].float().cuda()              # [BK, N, D]
            rgb_target = batch['rgb_target'].float().cuda()          # [BK, N, P]

            N, D = thermal_feat.shape[1], thermal_feat.shape[2]

            # ===== GeM scores =====
            thermal_spatial = thermal_feat.permute(0, 2, 1).view(BK, D, H_feat, W_feat)
            thermal_gem = model.module.thermal_aggregation(thermal_spatial)
            thermal_gem_scores = torch.einsum('bd,bnd->bn', thermal_gem.detach(), thermal_feat.detach())
            thermal_gem_weight = F.softmax(thermal_gem_scores, dim=-1)

            rgb_spatial = rgb_feat.permute(0, 2, 1).view(BK, D, H_feat, W_feat)
            rgb_gem = model.module.rgb_aggregation(rgb_spatial)
            rgb_gem_scores = torch.einsum('bd,bnd->bn', rgb_gem.detach(), rgb_feat.detach())
            rgb_gem_weight = F.softmax(rgb_gem_scores, dim=-1)

            # ===== Masking =====
            mask_generator = model.module.mask_generator
            thermal_masks = mask_generator(thermal_feat, gem_score=thermal_gem_scores)  # [BK, N]
            rgb_masks = mask_generator(rgb_feat, gem_score=rgb_gem_scores)              # [BK, N]

            thermal_masked = model.module.mask_token.expand(BK, N, -1).clone()
            thermal_masked[~thermal_masks] = thermal_feat[~thermal_masks]

            rgb_masked = model.module.mask_token.expand(BK, N, -1).clone()
            rgb_masked[~rgb_masks] = rgb_feat[~rgb_masks]

            # ===== Positional embedding =====
            thermal_masked = thermal_masked + model.module.decoder_pos_embed
            rgb_masked = rgb_masked + model.module.decoder_pos_embed
            thermal_ref = thermal_feat + model.module.decoder_pos_embed
            rgb_ref = rgb_feat + model.module.decoder_pos_embed

            # ===== Decoder forward =====
            thermal_dec = thermal_masked
            for blk in model.module.decoder_blocks:
                thermal_dec = blk(thermal_dec, rgb_ref)
            thermal_dec = model.module.decoder_norm(thermal_dec)

            rgb_dec = rgb_masked
            for blk in model.module.decoder_blocks:
                rgb_dec = blk(rgb_dec, thermal_ref)
            rgb_dec = model.module.decoder_norm(rgb_dec)

            # ===== Prediction heads =====
            thermal_recon = model.module.prediction_thermal_head(thermal_dec)  # [BK, N, P]
            rgb_recon = model.module.prediction_rgb_head(rgb_dec)              # [BK, N, P]

            # ===== Reshape for batch loss: [BK, ...] -> [B, K, ...] =====
            thermal_recon_bk = thermal_recon.view(B, K, N, -1)
            rgb_recon_bk = rgb_recon.view(B, K, N, -1)
            thermal_masks_bk = thermal_masks.view(B, K, N)
            rgb_masks_bk = rgb_masks.view(B, K, N)
            thermal_target_bk = thermal_target.view(B, K, N, -1)
            rgb_target_bk = rgb_target.view(B, K, N, -1)
            thermal_gem_weight_bk = thermal_gem_weight.view(B, K, N)
            rgb_gem_weight_bk = rgb_gem_weight.view(B, K, N)
            thermal_gem_scores_bk = thermal_gem_scores.view(B, K, N)
            rgb_gem_scores_bk = rgb_gem_scores.view(B, K, N)

            # ===== Vectorized loss computation =====
            recon_losses = compute_batch_recon_loss(
                thermal_recon_bk, rgb_recon_bk,
                thermal_masks_bk, rgb_masks_bk,
                thermal_target_bk, rgb_target_bk,
                thermal_gem_weight_bk, rgb_gem_weight_bk,
                use_weight=args.use_gem_recon_weight
            )  # [B, K]

            # ===== Rerank =====
            rerank_indices = recon_losses.argsort(dim=1)  # [B, K]

            # 결과 저장
            top_k_indices = batch['top_k_indices']  # [B, K] numpy
            query_indices = batch['query_indices']

            for i, query_idx in enumerate(query_indices):
                reranked_indices = rerank_indices[i].cpu().numpy()
                reranked = top_k_indices[i][reranked_indices]
                predictions_list.append(reranked)
                rerank_scores_dict[query_idx] = recon_losses[i, reranked_indices].cpu().numpy().tolist()

                # Visualization 데이터 수집
                if query_idx in vis_query_set:
                    vis_data['top_k_indices'].append(top_k_indices[i])
                    vis_data['recon_losses'].append(recon_losses[i].cpu().numpy())
                    vis_data['thermal_recon'].append(thermal_recon_bk[i].cpu())
                    vis_data['rgb_recon'].append(rgb_recon_bk[i].cpu())
                    vis_data['thermal_masks'].append(thermal_masks_bk[i].cpu())
                    vis_data['rgb_masks'].append(rgb_masks_bk[i].cpu())
                    vis_data['thermal_gem'].append(thermal_gem_scores_bk[i].cpu())
                    vis_data['rgb_gem'].append(rgb_gem_scores_bk[i].cpu())

    predictions = np.array(predictions_list)
    return predictions, rerank_scores_dict, vis_data


def batched_distance_reranking(
    model,
    dataloader,
    prev_predictions,
    reranking_top_k=5
):
    """
    DataLoader 기반 배치 distance reranking.

    여러 쿼리를 한 번에 처리하여 GPU 활용도를 높입니다.

    Args:
        model: CrossModalVPR_Net 모델
        dataloader: RerankingDataset을 위한 DataLoader (load_targets=False)
        prev_predictions: 원본 predictions [num_queries, max_recall]
        reranking_top_k: reranking할 top-K 수

    Returns:
        predictions: [num_queries, max_recall] reranked predictions
        rerank_scores_dict: {query_idx: scores} dictionary
    """
    predictions_list = []
    rerank_scores_dict = {}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Batched Distance Reranking"):
            B = batch['batch_size']
            K = batch['top_k']
            BK = B * K

            # GPU로 전송: [BK, N, D]
            thermal_feat = batch['thermal_feat'].float().cuda()
            rgb_feat = batch['rgb_feat'].float().cuda()

            # ===== Bidirectional distance scoring (batched) =====
            # stage2_forward_distance는 이미 bidirectional + batched
            _, pred_scores = model.module.stage2_forward_distance(
                thermal_feat, rgb_feat, distance_gt=None
            )  # [BK]

            # Reshape to [B, K]
            pred_scores = pred_scores.view(B, K)

            # Rerank (higher score = better)
            rerank_indices = pred_scores.argsort(dim=1, descending=True)  # [B, K]

            # 결과 저장
            top_k_indices = batch['top_k_indices']  # [B, K] numpy
            query_indices = batch['query_indices']

            for i, query_idx in enumerate(query_indices):
                reranked_order = rerank_indices[i].cpu().numpy()
                reranked_top_k = top_k_indices[i][reranked_order]

                # 전체 predictions 업데이트 (top-K만 rerank, 나머지 유지)
                full_pred = prev_predictions[query_idx].copy()
                full_pred[:reranking_top_k] = reranked_top_k
                predictions_list.append(full_pred)

                rerank_scores_dict[query_idx] = pred_scores[i, reranked_order].cpu().numpy().tolist()

    predictions = np.array(predictions_list)
    return predictions, rerank_scores_dict


def visualize_distance_reranking(vis_data, save_dir, seq_name, num_samples=20):
    """
    Visualize distance reranking results to verify geometric matching quality.

    Creates:
    1. Scatter plot: Predicted Score vs GT Distance (all candidates)
    2. Per-query analysis: Top-K candidates with scores and distances
    3. Success/Failure case comparison

    Args:
        vis_data: list of dicts with 'pred_scores', 'gt_distances', 'is_positive'
        save_dir: directory to save visualizations
        seq_name: sequence name for labeling
        num_samples: number of sample queries to visualize in detail
    """
    import matplotlib.pyplot as plt
    os.makedirs(save_dir, exist_ok=True)

    # Collect all data points
    all_scores = []
    all_distances = []
    all_is_positive = []

    for item in vis_data:
        all_scores.extend(item['pred_scores'])
        all_distances.extend(item['gt_distances'])
        all_is_positive.extend(item['is_positive'])

    all_scores = np.array(all_scores)
    all_distances = np.array(all_distances)
    all_is_positive = np.array(all_is_positive)

    # 1. Scatter plot: Predicted Score vs GT Distance
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # All candidates
    ax = axes[0]
    ax.scatter(all_distances[~all_is_positive], all_scores[~all_is_positive],
               alpha=0.3, s=10, c='blue', label='Negative')
    ax.scatter(all_distances[all_is_positive], all_scores[all_is_positive],
               alpha=0.8, s=30, c='red', marker='*', label='Positive')
    ax.set_xlabel('GT Distance (m)')
    ax.set_ylabel('Predicted Score')
    ax.set_title(f'{seq_name}: Pred Score vs GT Distance')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Zoomed view (0-50m)
    ax = axes[1]
    mask = all_distances < 50
    ax.scatter(all_distances[mask & ~all_is_positive], all_scores[mask & ~all_is_positive],
               alpha=0.3, s=10, c='blue', label='Negative')
    ax.scatter(all_distances[mask & all_is_positive], all_scores[mask & all_is_positive],
               alpha=0.8, s=30, c='red', marker='*', label='Positive')
    ax.set_xlabel('GT Distance (m)')
    ax.set_ylabel('Predicted Score')
    ax.set_title(f'{seq_name}: Zoomed (0-50m)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{seq_name}_score_vs_distance.png'), dpi=150)
    plt.close()

    # 2. Correlation analysis
    correlation = np.corrcoef(all_scores, all_distances)[0, 1]
    logging.info(f"[{seq_name}] Score-Distance Correlation: {correlation:.4f} (should be negative)")

    # 3. Per-query detailed visualization (sample)
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    axes = axes.flatten()

    sample_indices = np.random.choice(len(vis_data), min(num_samples, len(vis_data)), replace=False)

    for i, idx in enumerate(sample_indices):
        item = vis_data[idx]
        ax = axes[i]

        scores = np.array(item['pred_scores'])
        distances = np.array(item['gt_distances'])
        is_pos = np.array(item['is_positive'])

        # Plot candidates
        ax.scatter(distances[~is_pos], scores[~is_pos], c='blue', s=20, alpha=0.6, label='Neg')
        ax.scatter(distances[is_pos], scores[is_pos], c='red', s=50, marker='*', label='Pos')

        # Mark top-1 prediction
        top1_idx = scores.argmax()
        ax.scatter(distances[top1_idx], scores[top1_idx], c='green', s=100, marker='o',
                   edgecolors='black', linewidths=2, label='Top1', zorder=5)

        ax.set_xlabel('Distance (m)')
        ax.set_ylabel('Score')
        ax.set_title(f'Query {item["query_idx"]}')
        ax.grid(True, alpha=0.3)

        # Check if top1 is positive
        if is_pos[top1_idx]:
            ax.set_facecolor('#e6ffe6')  # Light green for success

    axes[0].legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{seq_name}_per_query_samples.png'), dpi=150)
    plt.close()

    # 4. Success vs Failure analysis
    success_queries = [item for item in vis_data
                       if np.array(item['is_positive'])[np.array(item['pred_scores']).argmax()]]
    failure_queries = [item for item in vis_data
                       if not np.array(item['is_positive'])[np.array(item['pred_scores']).argmax()]]

    logging.info(f"[{seq_name}] R@1 Success: {len(success_queries)}/{len(vis_data)} = {len(success_queries)/len(vis_data)*100:.1f}%")

    # Save statistics
    stats = {
        'seq_name': seq_name,
        'correlation': correlation,
        'r1_success': len(success_queries),
        'r1_total': len(vis_data),
        'r1_rate': len(success_queries) / len(vis_data) * 100
    }
    np.save(os.path.join(save_dir, f'{seq_name}_stats.npy'), stats)

    logging.info(f"Distance reranking visualization saved to: {save_dir}")

def visualize_top5_predictions(args, eval_ds, predictions, distances, positives_per_query, num_samples=10):
    """
    Query와 top-5 retrieved 이미지를 시각화
    
    Args:
        eval_ds: evaluation dataset
        predictions: (num_queries, K) faiss predictions
        distances: (num_queries, K) L2 distances
        positives_per_query: list of positive indices per query
        num_samples: 시각화할 query 개수
    """
    # Random하게 query 샘플링
    query_indices = np.random.choice(eval_ds.queries_num, min(num_samples, eval_ds.queries_num), replace=False)
    
    visualizations = []
    
    for query_idx in query_indices:
        # Query 이미지 (thermal)
        query_img = eval_ds.get_thermal_img(eval_ds.t_queries_paths[query_idx])
        query_img = cv2.resize(query_img, (224, 224))
        
        # Top-5 predictions
        top5_preds = predictions[query_idx, :5]
        top5_dists = distances[query_idx, :5]
        positives = positives_per_query[query_idx]
        
        # Top-5 database 이미지들 (RGB)
        retrieved_imgs = []
        for rank, (pred_idx, dist) in enumerate(zip(top5_preds, top5_dists)):
            db_img = eval_ds.get_rgb_img(eval_ds.rgb_database_paths[pred_idx])
            db_img = cv2.resize(db_img, (224, 224))
            
            # GT인지 확인
            is_correct = pred_idx in positives
            color = (0, 255, 0) if is_correct else (255, 0, 0)  # Green if correct, Red otherwise
            
            # Border와 텍스트 추가
            db_img = cv2.copyMakeBorder(db_img, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=color)
            
            # Rank와 distance 표시
            text = f"R{rank+1}: {dist:.2f}"
            if is_correct:
                text += " ✓"
            cv2.putText(db_img, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(db_img, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            retrieved_imgs.append(db_img)
        
        # Query에 "Query" 텍스트 추가
        query_img = cv2.copyMakeBorder(query_img, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(query_img, "Query (T)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(query_img, "Query (T)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # Horizontal stack: [Query | Top1 | Top2 | Top3 | Top4 | Top5]
        grid = np.hstack([query_img] + retrieved_imgs)
        
        # BGR to RGB (wandb uses RGB)
        grid = cv2.cvtColor(grid, cv2.COLOR_BGR2RGB)
        visualizations.append(grid)
    
    return visualizations

def index_to_image_tensor(dataset, index):
    return dataset[index][0]

def inference(args, eval_ds, model, pca=None, k=1, use_cuda=True, verbose=True,seq_name=""):
    '''
    hard_resize: directly use the resized image
    single_query: use the resized image, and set query_infer_batchsize=1 (used when the query images have varying size)
    central_crop: Take the biggest central crop of size self.resize. Preserves ratio.
    five_crops: use five crops of the image, and take the average of the features
    nearest_crop: use five crops of the image, 
    maj_voting: calculate features of five crops of the image, then use the nearest features
    * hard_size method for all database images
    * selected test_method for all query images
    '''
    global NPY_ROOTPATH
    if NPY_ROOTPATH is None:
        NPY_ROOTPATH = f"/home/jwkim/workspace/THR2RGB_Cross_Place_Recognition/npys_agg/{START_TIME}_{args.comment}"
    if os.path.exists(NPY_ROOTPATH):
        import shutil
        shutil.rmtree(NPY_ROOTPATH)
    os.makedirs(NPY_ROOTPATH, exist_ok=True)
    logging.info(f"Cleaned and created NPY directory: {NPY_ROOTPATH}")

    orig_W = args.resize[0]
    orig_H = args.resize[1]
    patch_W = int(orig_W / 14)
    patch_H = int(orig_H / 14)
    patch_count = patch_W * patch_H
    try:
        test_method = args.test_method
        model = model.eval()
        with torch.no_grad():
            # 1. database feature들 추출
            start_time = time.time()
            database_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num)))
            database_dataloader = DataLoader(dataset=database_subset_ds, num_workers=args.num_workers,
                                            batch_size=args.infer_batch_size, pin_memory=(args.device=="cuda"))
        
            database_features = np.empty((eval_ds.database_num, args.features_dim), dtype="float32")
            database_attn_map = np.empty((eval_ds.database_num, patch_count), dtype="float32")
            
            use_selaVPR = args.use_reranking in ['selaVPR', 'reconSelaVPR']
            use_penultimate = args.r2_penultimate_layer
            use_recon = args.use_reranking == 'recon'
            print(f"Using SelaVPR features: {use_selaVPR}")
            print(f"Using penultimate: {use_penultimate}")
            print(f"Using recon (saving target_patches): {use_recon}")
            for inputs, indices, flags in tqdm(database_dataloader, ncols=100):
                flags_int = [1 if f == 'rgb' else 0 for f in flags]
                flags = torch.tensor(flags_int, dtype=torch.long, device=args.device)
                outputs = model(inputs.to(args.device), flags)
                features = outputs[0].view(-1, args.features_dim)
                patch_features = outputs[1].view(-1, patch_W*patch_H, args.features_dim)

                indices_npy = indices.numpy()
                database_features[indices_npy,:] = features.cpu().numpy() # [B, C] # 이건 저장 x
                if args.use_reranking != 'none':
                    database_attn_map[indices_npy,:] = outputs[4].cpu().numpy() # [B, N] # 이것도 저장 x
                    # target_patches for recon reranking
                    if use_recon:
                        target_patches = patchify(inputs.to(args.device))  # [B, N, 588]
                    for num, idx in enumerate(indices_npy):
                        if use_penultimate: save_npy(outputs[5][num].cpu().numpy(), f"Db_{seq_name}_penultimate_{idx}")
                        if use_selaVPR: save_npy(outputs[6][num].cpu().numpy(), f"Db_{seq_name}_sela_{idx}")
                        if use_recon: save_npy(target_patches[num].cpu().numpy(), f"Db_{seq_name}_target_{idx}")
                        save_npy(patch_features[num].cpu().numpy(), f"Db_{seq_name}_{idx}")
                # if args.use_fast_track: break
                
            logging.info(f"Finished extracting {eval_ds.database_num} database features in {time.time() - start_time:.2f} s")

            # 2. query 전부의 feature 추출
            start_time = time.time()
            queries_infer_batch_size = args.infer_batch_size
            queries_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num, len(eval_ds))))
            queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args.num_workers,
                                            batch_size=queries_infer_batch_size, pin_memory=(args.device=="cuda"))

            queries_features = np.empty((eval_ds.queries_num, args.features_dim), dtype="float32")
            queries_attn_map = np.empty((eval_ds.queries_num, patch_count), dtype="float32")
                
            for inputs, indices, flags in tqdm(queries_dataloader, ncols=100):
                flags_int = [1 if f == 'rgb' else 0 for f in flags]
                flags = torch.tensor(flags_int, dtype=torch.long, device=args.device)
                outputs = model(inputs.to(args.device), flags)
                features = outputs[0].view(-1, args.features_dim)
                patch_features = outputs[1].view(-1, patch_W*patch_H, args.features_dim)

                indices_npy = indices.numpy()-eval_ds.database_num
                queries_features[indices.numpy()-eval_ds.database_num,:] = features.cpu().numpy()
                if args.use_reranking != 'none':
                    queries_attn_map[indices.numpy()-eval_ds.database_num,:] = outputs[4].cpu().numpy()
                    # target_patches for recon reranking
                    if use_recon:
                        target_patches = patchify(inputs.to(args.device))  # [B, N, 588]
                    for num, idx in enumerate(indices_npy):
                        if use_penultimate: save_npy(outputs[5][num].cpu().numpy(), f"Query_{seq_name}_penultimate_{idx}")
                        if use_selaVPR: save_npy(outputs[6][num].cpu().numpy(), f"Query_{seq_name}_sela_{idx}")
                        if use_recon: save_npy(target_patches[num].cpu().numpy(), f"Query_{seq_name}_target_{idx}")
                        save_npy(patch_features[num].cpu().numpy(), f"Query_{seq_name}_{idx}")
                # if args.use_fast_track: break
                    
            logging.info(f"Finished extracting {eval_ds.queries_num} query features in {time.time() - start_time:.2f} s")

        # 3. faiss를 이용하여, L2 distance로 가까운 descriptor 찾기
        faiss_index = faiss.IndexFlatL2(args.features_dim)
        faiss_index.add(database_features)
        
        start_time = time.time()
        _, predictions = faiss_index.search(queries_features, max(args.recall_values))
        del faiss_index
        
        import gc; gc.collect()
        torch.cuda.empty_cache()
        
        if args.use_reranking != 'none':
            prev_predictions = predictions.copy()
            #####################################
            ############# RERANKING #############
            print("=" * 30)
            print("- USING RERANKING -")
            RERANKING_TOP_K = 5
            if args.use_reranking == 'r2former':
                # 파일 개수만 확인
                saved_files = os.listdir(NPY_ROOTPATH)
                prefix = "penultimate" if args.r2_penultimate_layer else ""
                
                db_files = [f for f in saved_files if f.startswith(f"Db_{seq_name}") and prefix in f]
                query_files = [f for f in saved_files if f.startswith(f"Query_{seq_name}") and prefix in f]
                
                assert len(db_files) == eval_ds.database_num, \
                    f"DB files mismatch: {len(db_files)} vs {eval_ds.database_num}"
                assert len(query_files) == eval_ds.queries_num, \
                    f"Query files mismatch: {len(query_files)} vs {eval_ds.queries_num}"
                
                logging.info(f"✓ File count verified: {len(db_files)} DB + {len(query_files)} Query")
                
                reconstruction_losses_dict = {}
                
                queries_indexes = torch.zeros(RERANKING_TOP_K, dtype=torch.long).cuda()
                positives_indexes = torch.arange(RERANKING_TOP_K + 1, dtype=torch.long).cuda()
                positives_indexes = positives_indexes[positives_indexes % (RERANKING_TOP_K + 1) != 0]

                with torch.no_grad():
                    for query_idx in tqdm(range(eval_ds.queries_num), desc="Reranking", ncols=100):
                        # ========== 1. Query Features 준비 ==========
                        if args.r2_penultimate_layer:
                            encoded_queries_features = torch.from_numpy(
                                load_npy(f"Query_{seq_name}_penultimate_{query_idx}")
                            ).float().cuda().unsqueeze(0)
                        else:
                            encoded_queries_features = torch.from_numpy(
                                load_npy(f"Query_{seq_name}_{query_idx}")
                            ).float().cuda().unsqueeze(0) 
                        
                        encoded_queries_attn_map = torch.from_numpy(
                            queries_attn_map[query_idx]
                        ).float().cuda().unsqueeze(0)
                        
                        encoded_queries_descriptor = torch.from_numpy(
                            queries_features[query_idx]
                        ).float().cuda().unsqueeze(0) 

                        # ========== 2. Top-K Database Features 준비 ==========
                        top_k_db_indices_batch = predictions[query_idx, :RERANKING_TOP_K]  # [K]
                        
                        # 1. 파일 하나씩 로드해서 리스트에 담기
                        if args.r2_penultimate_layer:
                            db_features_list = [
                                torch.from_numpy(
                                    load_npy(f"Db_{seq_name}_penultimate_{db_idx}")
                                ) 
                                for db_idx in top_k_db_indices_batch
                            ]
                        else:
                            db_features_list = [
                                torch.from_numpy(
                                    load_npy(f"Db_{seq_name}_{db_idx}") # 파일명 포맷에 맞게 수정
                                ) 
                                for db_idx in top_k_db_indices_batch
                            ]

                        # 2. 리스트를 하나의 텐서로 합치기 (Stack) -> [K, 256, 768]
                        encoded_dbs_features = torch.stack(db_features_list).float().cuda()
                        
                        # 3. 배치 차원 추가 (Batch=1 이므로) -> [1, K, 256, 768]
                        # 뒤쪽 코드(concatenation 등)와 호환되게 reshape
                        encoded_dbs_features = encoded_dbs_features.reshape(1, RERANKING_TOP_K, patch_count, -1)
                        
                        encoded_dbs_attn_map = torch.from_numpy(
                            database_attn_map[top_k_db_indices_batch]
                        ).float().cuda()
                        
                        encoded_dbs_descriptor = torch.from_numpy(
                            database_features[top_k_db_indices_batch]
                        ).float().cuda()
                        
                        # Reshape: [1, K, 256, 768] (Batch=1 명시)
                        encoded_dbs_features = encoded_dbs_features.reshape(1, RERANKING_TOP_K, patch_count, -1)
                        encoded_dbs_attn_map = encoded_dbs_attn_map.reshape(1, RERANKING_TOP_K, patch_count)
                        encoded_dbs_descriptor = encoded_dbs_descriptor.reshape(1, RERANKING_TOP_K, -1)
                        
                        # ========== 5. Reranker용 데이터 준비 ==========
                        concated_db_patches = torch.cat([
                            encoded_queries_features.unsqueeze(1),  # [1, 1, 256, 768]
                            encoded_dbs_features  # [1, K, 256, 768]
                        ], dim=1)
                        
                        concated_db_attn_map = torch.cat([
                            encoded_queries_attn_map.unsqueeze(1),  # [1, 1, 256]
                            encoded_dbs_attn_map  # [1, K, 256]
                        ], dim=1)
                        
                        concated_db_patches_flat = concated_db_patches.reshape(-1, patch_count, concated_db_patches.size(-1))
                        concated_db_attn_map_flat = concated_db_attn_map.reshape(-1, patch_count)
                        
                        # ========== 7. Global Descriptors ==========
                        # Query Descriptor Expand [1, 768] -> [1, K, 768] -> [K, 768]
                        global_query_exp = encoded_queries_descriptor.unsqueeze(1).expand(
                            -1, RERANKING_TOP_K, -1
                        ).reshape(-1, encoded_queries_descriptor.size(-1))
                        
                        global_pos_flat = encoded_dbs_descriptor.reshape(-1, encoded_dbs_descriptor.size(-1))
                        
                        # ========== 9. Reranking ==========
                        reranker = model.module.reranker
                        reranker.global_query_cache = encoded_queries_descriptor[0].unsqueeze(0)
                        rerank_scores = reranker(
                            concated_db_patches_flat,
                            concated_db_attn_map_flat,
                            queries_indexes,
                            positives_indexes,
                            None,
                            global_query=global_query_exp,
                            global_pos=global_pos_flat,
                            global_neg=None,
                            cross_attn_matrix=None
                        )
                        
                        # ========== 10. Reshape & Reorder ==========
                        rerank_scores = rerank_scores.reshape(1, RERANKING_TOP_K)  # [1, K]
                    
                        sorted_indices = torch.argsort(rerank_scores[0], descending=True)
                        predictions[query_idx, :RERANKING_TOP_K] = top_k_db_indices_batch[sorted_indices.cpu()]
                        
                        del encoded_queries_features, encoded_queries_attn_map, encoded_queries_descriptor
                        del encoded_dbs_features, encoded_dbs_attn_map, encoded_dbs_descriptor, concated_db_patches_flat, concated_db_attn_map_flat
                        # if args.use_fast_track: break
            elif args.use_reranking == 'recon':
                # Bidirectional reconstruction loss reranking (DataLoader-based, parallelized)
                logging.info("Using bidirectional reconstruction loss reranking (DataLoader-based)")

                H_feat, W_feat = int(args.resize[0]/14), int(args.resize[1]/14)

                # ===== Visualization setup =====
                import random
                num_vis_queries = 5
                vis_query_indices = random.sample(range(eval_ds.queries_num), min(num_vis_queries, eval_ds.queries_num))
                vis_save_dir = os.path.join(args.save_dir, f'recon_reranking_vis_{seq_name}')
                logging.info(f"Will visualize recon reranking for queries: {vis_query_indices}")

                # ===== DataLoader 설정 =====
                reranking_dataset = RerankingDataset(
                    predictions=prev_predictions,
                    seq_name=seq_name,
                    npy_root_path=NPY_ROOTPATH,
                    reranking_top_k=RERANKING_TOP_K,
                    load_targets=True
                )

                reranking_batch_size = getattr(args, 'reranking_batch_size', 32)
                reranking_num_workers = getattr(args, 'reranking_num_workers', 8)

                reranking_dataloader = DataLoader(
                    dataset=reranking_dataset,
                    batch_size=reranking_batch_size,
                    shuffle=False,
                    num_workers=reranking_num_workers,
                    collate_fn=reranking_collate_fn,
                    pin_memory=True,
                    prefetch_factor=2
                )

                logging.info(f"Reranking DataLoader: batch_size={reranking_batch_size}, num_workers={reranking_num_workers}")

                # ===== Batched Reranking 실행 =====
                predictions, rerank_scores_dict, vis_data = batched_recon_reranking(
                    model=model,
                    dataloader=reranking_dataloader,
                    args=args,
                    H_feat=H_feat,
                    W_feat=W_feat,
                    vis_query_indices=vis_query_indices
                )

                # ===== Visualization =====
                if vis_data:
                    positives_per_query_vis = eval_ds.get_positives()
                    visualize_recon_reranking(
                        args, eval_ds, model,
                        query_indices=vis_query_indices,
                        top_k_db_indices_list=vis_data['top_k_indices'],
                        recon_losses_list=vis_data['recon_losses'],
                        thermal_recon_list=vis_data['thermal_recon'],
                        rgb_recon_list=vis_data['rgb_recon'],
                        thermal_masks_list=vis_data['thermal_masks'],
                        rgb_masks_list=vis_data['rgb_masks'],
                        thermal_gem_attn_list=vis_data['thermal_gem'],
                        rgb_gem_attn_list=vis_data['rgb_gem'],
                        positives_per_query=positives_per_query_vis,
                        save_dir=vis_save_dir,
                        seq_name=seq_name
                    )
                    logging.info(f"Recon reranking visualization saved to: {vis_save_dir}")
            elif args.use_reranking == 'selaVPR':
                saved_files = os.listdir(NPY_ROOTPATH)
                prefix = "sela"
                db_files = [f for f in saved_files if f.startswith(f"Db_{seq_name}") and prefix in f]
                query_files = [f for f in saved_files if f.startswith(f"Query_{seq_name}") and prefix in f]

                assert len(db_files) == eval_ds.database_num, \
                    f"DB files mismatch: {len(db_files)} vs {eval_ds.database_num}"
                assert len(query_files) == eval_ds.queries_num, \
                    f"Query files mismatch: {len(query_files)} vs {eval_ds.queries_num}"

                logging.info(f"✓ File count verified: {len(db_files)} DB + {len(query_files)} Query")

                use_GeM_attn = True
                print(f"use_GEM_attn: {use_GeM_attn}")
                predictions = []
                rerank_scores_dict = {}  # For visualization
                candidates_local_features = torch.zeros(RERANKING_TOP_K, 61, 61, 128, device='cuda')
                candidates_db_attn_map = torch.zeros(RERANKING_TOP_K, 256, device='cuda')
                for query_index, pred in enumerate(tqdm(prev_predictions)):
                    # Load query local features: [61, 61, features_dim]
                    query_local_features = torch.from_numpy(
                        load_npy(f"Query_{seq_name}_sela_{query_index}")
                    ).float().cuda()

                    # Load candidate local features: [K, 61, 61, features_dim]
                    for cnt, candidates_index in enumerate(pred[:RERANKING_TOP_K]):
                        candidates_local_features[cnt] = torch.from_numpy(
                            load_npy(f"Db_{seq_name}_sela_{candidates_index}")
                        ).float().cuda()
                        
                        if use_GeM_attn:
                            cur_database_features = load_npy(f"Db_{seq_name}_{candidates_index}")
                            cur_database_attn_map = database_features[candidates_index] @ cur_database_features.T # softmax?
                            candidates_db_attn_map[cnt] = F.softmax(torch.from_numpy(cur_database_attn_map), dim=-1).float().cuda()
                        else:
                            candidates_db_attn_map[cnt] = torch.from_numpy(database_attn_map[candidates_index]).float().cuda()
                        
                    # local_sim expects: query [H, W, C], candidates [B, H, W, C]
                    if use_GeM_attn:
                        cur_queries_features = load_npy(f"Query_{seq_name}_{query_index}")
                        cur_queries_attn_map = queries_features[query_index] @ cur_queries_features.T # softmax?
                        cur_queries_attn_map = F.softmax(torch.from_numpy(cur_queries_attn_map), dim=-1).float().cuda()
                    else:
                        cur_queries_attn_map = torch.from_numpy(queries_attn_map[query_index]).float().cuda()
                    
                    rerank_scores = local_sim(query_local_features, candidates_local_features, trainflag=False,
                        query_attn_map=cur_queries_attn_map,
                        db_attn_map=candidates_db_attn_map,
                        method_type=args.selaVPR_rerank_score_type
                    )

                    rerank_scores_np = rerank_scores.cpu().numpy()
                    rerank_index = rerank_scores_np.argsort()[::-1]
                    rerank_scores_dict[query_index] = rerank_scores_np[rerank_index].tolist()
                    predictions.append(pred[rerank_index])

                predictions = np.array(predictions)

                # Visualization for selaVPR
                positives_per_query_vis = eval_ds.get_positives()
                vis_save_dir = f"./selaVPR_visualizations/{args.comment}_{seq_name}"
                # visualize_selaVPR_reranking(
                #     args, eval_ds,
                #     prev_predictions, predictions,
                #     rerank_scores_dict,
                #     positives_per_query_vis,
                #     epoch=0,
                #     distances=None,
                #     npy_root_path=NPY_ROOTPATH,
                #     seq_name=seq_name,
                #     save_dir=vis_save_dir,
                #     num_samples=4,
                #     reranking_method='selaVPR'
                # )
            elif args.use_reranking == 'reconSelaVPR':
                # ReconSelaVPR: CroCo bidirectional decoder + SelaVPR-style local matching
                logging.info("Using reconSelaVPR reranking (bidirectional decoder + local matching)")

                # Verify saved patch features exist
                saved_files = os.listdir(NPY_ROOTPATH)
                db_files = [f for f in saved_files if f.startswith(f"Db_{seq_name}_")
                            and "sela" not in f and "penultimate" not in f]
                query_files = [f for f in saved_files if f.startswith(f"Query_{seq_name}_")
                               and "sela" not in f and "penultimate" not in f]

                assert len(db_files) == eval_ds.database_num, \
                    f"DB files mismatch: {len(db_files)} vs {eval_ds.database_num}"
                assert len(query_files) == eval_ds.queries_num, \
                    f"Query files mismatch: {len(query_files)} vs {eval_ds.queries_num}"
                logging.info(f"✓ Verified: {len(db_files)} DB + {len(query_files)} Query patch files")

                predictions = []
                rerank_scores_dict = {}
                candidates_features = torch.zeros(RERANKING_TOP_K, patch_count, args.features_dim, device='cuda')

                RETURN_DECODER_LAYER = 1
                print("Return Layer: ", RETURN_DECODER_LAYER)
                with torch.no_grad():
                    for query_index, pred in enumerate(tqdm(prev_predictions, desc="ReconSelaVPR Reranking")):
                        # Load query encoder features: [256, 768]
                        query_enc_features = torch.from_numpy(
                            load_npy(f"Query_{seq_name}_{query_index}")
                        ).float().cuda()

                        # Load top-K candidate encoder features
                        for cnt, candidate_idx in enumerate(pred[:RERANKING_TOP_K]):
                            candidates_features[cnt] = torch.from_numpy(
                                load_npy(f"Db_{seq_name}_{candidate_idx}")
                            ).float().cuda()

                        # Expand query to batch: [K, 256, 768]
                        query_batch = query_enc_features.unsqueeze(0).expand(RERANKING_TOP_K, -1, -1)

                        # Bidirectional decoding
                        thermal_decoded_local, rgb_decoded_local = model.module.forward_recon_sela_decode(
                            query_batch,         # thermal [K, 256, 768]
                            candidates_features,  # RGB [K, 256, 768]
                            return_layer=RETURN_DECODER_LAYER
                        )
                        # thermal_decoded_local: [K, 61, 61, 768] - thermal refined with RGB context
                        # rgb_decoded_local: [K, 61, 61, 768] - RGB refined with thermal context
                        # MNN matching between decoded features
                        query_local = thermal_decoded_local  # [K, 61, 61, 768]
                        db_local = rgb_decoded_local            # [K, 61, 61, 768]

                        rerank_scores = local_sim_batch(
                            query_local, db_local,
                            trainflag=False,
                        )

                        # Sort and reorder
                        rerank_scores_np = rerank_scores.cpu().numpy()
                        rerank_index = rerank_scores_np.argsort()[::-1]
                        rerank_scores_dict[query_index] = rerank_scores_np[rerank_index].tolist()
                        predictions.append(pred[rerank_index])

                predictions = np.array(predictions)
            elif args.use_reranking == 'GeM_KL':
                logging.info("Using GeM-KLdivergence reranking")
                saved_files = os.listdir(NPY_ROOTPATH)
                db_files = [f for f in saved_files if f.startswith(f"Db_{seq_name}")]
                query_files = [f for f in saved_files if f.startswith(f"Query_{seq_name}")]
                
                assert len(db_files) == eval_ds.database_num, \
                    f"DB files mismatch: {len(db_files)} vs {eval_ds.database_num}"
                assert len(query_files) == eval_ds.queries_num, \
                    f"Query files mismatch: {len(query_files)} vs {eval_ds.queries_num}"
                logging.info(f"✓ Verified: {len(db_files)} DB + {len(query_files)} Query patch files")

                predictions = []
                rerank_scores_dict = {}
                with torch.no_grad():
                    for query_index, pred in enumerate(tqdm(prev_predictions, desc="GeM KL divergence Reranking")):
                        top_k_db_indices_batch = prev_predictions[query_index, :RERANKING_TOP_K]  # [K]
                        
                        encoded_queries_features = torch.from_numpy(
                            load_npy(f"Query_{seq_name}_{query_index}")
                        ).float().cuda().unsqueeze(0) 
                        
                        encoded_queries_descriptor = torch.from_numpy(
                            queries_features[query_index]
                        ).float().cuda().unsqueeze(0) 

                        top_k_db_indices_batch = prev_predictions[query_index, :RERANKING_TOP_K]  # [K]
                        db_features_list = [
                            torch.from_numpy(
                                load_npy(f"Db_{seq_name}_{db_index}") # 파일명 포맷에 맞게 수정
                            ) 
                            for db_index in top_k_db_indices_batch
                        ]

                        encoded_dbs_features = torch.stack(db_features_list).float().cuda()
                        encoded_dbs_descriptor = torch.from_numpy(
                            database_features[top_k_db_indices_batch]
                        ).float().cuda()
                        
                        # 1. GeM pooling된 결과물들 L2 normalize
                        USE_INTRANORM = True
                        if USE_INTRANORM: 
                            encoded_queries_descriptor = F.normalize(encoded_queries_descriptor, p=2, dim=-1)
                            encoded_dbs_descriptor = F.normalize(encoded_dbs_descriptor, p=2, dim=-1)
                            
                        encoded_queries_descriptor = encoded_queries_descriptor.squeeze(0)
                        encoded_dbs_features = encoded_dbs_features.squeeze(0)
                        encoded_queries_features = encoded_queries_features.squeeze(0)

                        # encoded_queries_features: [256, 384]
                        # encoded_queries_descriptor: [384]
                        candidate_attn_maps = torch.zeros(RERANKING_TOP_K, 256, device='cuda')
                        
                        query_attn_map = torch.zeros(RERANKING_TOP_K, 256, device='cuda')
                        rerank_scores_query = torch.zeros(RERANKING_TOP_K, device='cuda')
                        for candidate_index in range(len(encoded_dbs_descriptor)):
                            query_attn_map[candidate_index] = F.softmax(encoded_dbs_features[candidate_index] @ encoded_dbs_descriptor[candidate_index], dim=-1)
                            candidate_attn_maps[candidate_index] = F.softmax(encoded_dbs_features[candidate_index] @ encoded_queries_descriptor, dim=-1)
                            rerank_scores_query[candidate_index] = F.kl_div(query_attn_map[candidate_index].log(), candidate_attn_maps[candidate_index], reduction='sum')
                        
                        rerank_scores_db = torch.zeros(RERANKING_TOP_K)
                        query_attn_map = F.softmax(encoded_queries_features @ encoded_queries_descriptor, dim=-1)
                        for candidate_index in range(len(encoded_dbs_descriptor)):
                            candidate_attn_maps[candidate_index] = F.softmax(encoded_queries_features @ encoded_dbs_descriptor[candidate_index], dim=-1)
                            rerank_scores_db[candidate_index] = F.kl_div(query_attn_map.log(), candidate_attn_maps[candidate_index], reduction='sum')
                                                    
                        # import lovely_tensors as lt; lt.monkey_patch()
                        # print("query: ", query_attn_map)
                        # for _ in range(len(encoded_dbs_descriptor)): print(f"cand[{_}]: {candidate_attn_maps[_]}")
                        
                        # Sort and reorder
                        rerank_scores_query_np = rerank_scores_query.cpu().numpy()
                        rerank_scores_db_np = rerank_scores_db.cpu().numpy()
                        rerank_query_index = rerank_scores_query_np.argsort()
                        rerank_db_index = rerank_scores_db_np.argsort()
                        # rerank_scores_dict[query_index] = rerank_scores_query_np[rerank_query_index].tolist()
                        # rerank_scores_dict[query_index] = rerank_scores_db_np[rerank_db_index].tolist()
                        if rerank_query_index[0] == rerank_db_index[0]:
                            predictions.append(pred[rerank_query_index])
                        else:
                            predictions.append(pred[:RERANKING_TOP_K])

                predictions = np.array(predictions)
            elif args.use_reranking == 'diffGeM':
                # DiffGeM reranking using trained DiffLoss module
                logging.info("Using diffGeM reranking (differential attention with GeM)")

                # Verify saved patch features exist
                saved_files = os.listdir(NPY_ROOTPATH)
                db_files = [f for f in saved_files if f.startswith(f"Db_{seq_name}_")
                            and "sela" not in f and "penultimate" not in f]
                query_files = [f for f in saved_files if f.startswith(f"Query_{seq_name}_")
                               and "sela" not in f and "penultimate" not in f]

                assert len(db_files) == eval_ds.database_num, \
                    f"DB files mismatch: {len(db_files)} vs {eval_ds.database_num}"
                assert len(query_files) == eval_ds.queries_num, \
                    f"Query files mismatch: {len(query_files)} vs {eval_ds.queries_num}"
                logging.info(f"✓ Verified: {len(db_files)} DB + {len(query_files)} Query patch files")

                predictions = []
                rerank_scores_dict = {}
                candidates_features = torch.zeros(RERANKING_TOP_K, patch_count, args.features_dim, device='cuda')
                candidates_GeMs = torch.zeros(RERANKING_TOP_K, args.features_dim, device='cuda')

                # Select 5 random queries for visualization
                import random
                num_vis_queries = 2
                vis_query_indices = random.sample(range(eval_ds.queries_num), min(num_vis_queries, eval_ds.queries_num))
                vis_save_dir = os.path.join(args.save_dir, f'diffGeM_attn_vis_{seq_name}')
                logging.info(f"Will visualize attention maps for queries: {vis_query_indices}")

                with torch.no_grad():
                    for query_index, pred in enumerate(tqdm(prev_predictions, desc="DiffGeM Reranking")):
                        # Load query encoder features: [256, features_dim]
                        query_enc_features = torch.from_numpy(
                            load_npy(f"Query_{seq_name}_{query_index}")
                        ).float().cuda().unsqueeze(0)  # [1, 256, D]

                        # Load query GeM descriptor
                        query_GeM = torch.from_numpy(
                            queries_features[query_index]
                        ).float().cuda().unsqueeze(0)  # [1, D]

                        # Load top-K candidate encoder features and GeMs
                        top_k_indices = pred[:RERANKING_TOP_K]
                        for cnt, candidate_idx in enumerate(top_k_indices):
                            candidates_features[cnt] = torch.from_numpy(
                                load_npy(f"Db_{seq_name}_{candidate_idx}")
                            ).float().cuda()
                            candidates_GeMs[cnt] = torch.from_numpy(
                                database_features[candidate_idx]
                            ).float().cuda()

                        # Get reranking scores using DiffLoss inference
                        rerank_scores = model.module.DiffGeMLoss.inference(
                            model.module,
                            query_enc_features,      # [1, N, D]
                            query_GeM,               # [1, D]
                            candidates_features,     # [K, N, D]
                            candidates_GeMs          # [K, D]
                        )

                        # Visualize attention maps for selected queries
                        if query_index in vis_query_indices:
                            # Load images for visualization
                            try:
                                query_img = eval_ds.get_thermal_img(eval_ds.t_queries_paths[query_index])
                                query_img = cv2.cvtColor(cv2.resize(query_img, (224, 224)), cv2.COLOR_BGR2RGB)
                                candidate_imgs = []
                                for candidate_idx in top_k_indices:
                                    cand_img = eval_ds.get_rgb_img(eval_ds.rgb_database_paths[candidate_idx])
                                    cand_img = cv2.cvtColor(cv2.resize(cand_img, (224, 224)), cv2.COLOR_BGR2RGB)
                                    candidate_imgs.append(cand_img)
                            except Exception as e:
                                logging.warning(f"Could not load images for visualization: {e}")
                                query_img = None
                                candidate_imgs = None

                            model.module.DiffGeMLoss.visualize_attention_maps(
                                model.module,
                                query_enc_features,
                                query_GeM,
                                candidates_features,
                                candidates_GeMs,
                                query_idx=query_index,
                                save_dir=vis_save_dir,
                                query_image=query_img,
                                candidate_images=candidate_imgs
                            )
                            logging.info(f"Saved attention visualization for query {query_index}")

                        # Sort and reorder (higher score = better match)
                        rerank_scores_np = rerank_scores.cpu().numpy()
                        rerank_index = rerank_scores_np.argsort()[::-1]
                        rerank_scores_dict[query_index] = rerank_scores_np[rerank_index].tolist()
                        predictions.append(pred[rerank_index])

                predictions = np.array(predictions)
                logging.info(f"Attention visualizations saved to: {vis_save_dir}")
            elif args.use_reranking == 'reconDiffVPR':
                # ReconSelaVPR: CroCo bidirectional decoder + SelaVPR-style local matching
                logging.info("Using reconDiffVPR reranking (bidirectional decoder + local matching)")

                # Verify saved patch features exist
                saved_files = os.listdir(NPY_ROOTPATH)
                db_files = [f for f in saved_files if f.startswith(f"Db_{seq_name}_")
                            and "sela" not in f and "penultimate" not in f]
                query_files = [f for f in saved_files if f.startswith(f"Query_{seq_name}_")
                               and "sela" not in f and "penultimate" not in f]

                assert len(db_files) == eval_ds.database_num, \
                    f"DB files mismatch: {len(db_files)} vs {eval_ds.database_num}"
                assert len(query_files) == eval_ds.queries_num, \
                    f"Query files mismatch: {len(query_files)} vs {eval_ds.queries_num}"
                logging.info(f"✓ Verified: {len(db_files)} DB + {len(query_files)} Query patch files")

                predictions = []
                rerank_scores_dict = {}
                candidates_features = torch.zeros(RERANKING_TOP_K, patch_count, args.features_dim, device='cuda')

                RETURN_DECODER_LAYER = 5
                print("Return Layer: ", RETURN_DECODER_LAYER)
                with torch.no_grad():
                    for query_index, pred in enumerate(tqdm(prev_predictions, desc="ReconDiffVPR Reranking")):
                        # Load query encoder features: [256, 768]
                        query_enc_features = torch.from_numpy(
                            load_npy(f"Query_{seq_name}_{query_index}")
                        ).float().cuda()

                        query_GeM = torch.from_numpy(queries_features[query_index]).float().cuda()
                        database_GeMs = torch.zeros(RERANKING_TOP_K, args.features_dim, device='cuda')
                        # Load top-K candidate encoder features
                        for cnt, candidate_idx in enumerate(pred[:RERANKING_TOP_K]):
                            candidates_features[cnt] = torch.from_numpy(
                                load_npy(f"Db_{seq_name}_{candidate_idx}")
                            ).float().cuda()
                            database_GeMs[cnt] = torch.from_numpy(
                                database_features[candidate_idx]
                            ).float().cuda()

                        # Expand query to batch: [K, 256, 768]
                        query_batch = query_enc_features.unsqueeze(0).expand(RERANKING_TOP_K, -1, -1)

                        # Bidirectional decoding
                        thermal_decoded, rgb_decoded = model.module.forward_recon_diff_decode(
                            query_batch,         # thermal [K, 256, 384]
                            candidates_features,  # RGB [K, 256, 384]
                            return_layer=RETURN_DECODER_LAYER
                        ) # output.shape = [K, 256, 384] (intra-normalized)
                        
                        query_GeM_attn_maps = torch.zeros(RERANKING_TOP_K, patch_count, device='cuda')
                        db_GeM_attn_maps = torch.zeros(RERANKING_TOP_K, patch_count, device='cuda')
                        for cnt in range(RERANKING_TOP_K):
                            query_GeM_attn_maps[cnt] = query_GeM @ thermal_decoded[cnt].transpose(-2, -1)
                            db_GeM_attn_maps[cnt] = database_GeMs[cnt] @ rgb_decoded[cnt].transpose(-2, -1)
                            
                            sqrt_d = math.sqrt(args.features_dim)
                            query_GeM_attn_maps[cnt] = F.softmax(query_GeM_attn_maps[cnt] / sqrt_d, dim=-1)
                            db_GeM_attn_maps[cnt] = F.softmax(db_GeM_attn_maps[cnt] / sqrt_d, dim=-1)

                        # Sort and reorder
                        rerank_scores_np = rerank_scores.cpu().numpy()
                        rerank_index = rerank_scores_np.argsort()[::-1]
                        rerank_scores_dict[query_index] = rerank_scores_np[rerank_index].tolist()
                        predictions.append(pred[rerank_index])

                predictions = np.array(predictions)
            elif args.use_reranking == 'reconPairVPR':
                # ReconSelaVPR: CroCo bidirectional decoder + SelaVPR-style local matching
                logging.info("Using reconPairVPR reranking (bidirectional decoder + and scoring)")

                # Verify saved patch features exist
                saved_files = os.listdir(NPY_ROOTPATH)
                db_files = [f for f in saved_files if f.startswith(f"Db_{seq_name}_")
                            and "sela" not in f and "penultimate" not in f]
                query_files = [f for f in saved_files if f.startswith(f"Query_{seq_name}_")
                               and "sela" not in f and "penultimate" not in f]

                assert len(db_files) == eval_ds.database_num, \
                    f"DB files mismatch: {len(db_files)} vs {eval_ds.database_num}"
                assert len(query_files) == eval_ds.queries_num, \
                    f"Query files mismatch: {len(query_files)} vs {eval_ds.queries_num}"
                logging.info(f"✓ Verified: {len(db_files)} DB + {len(query_files)} Query patch files")

                predictions = []
                rerank_scores_dict = {}
                candidates_features = torch.zeros(RERANKING_TOP_K, patch_count, args.features_dim, device='cuda')

                with torch.no_grad():
                    for query_index, pred in enumerate(tqdm(prev_predictions, desc="ReconPairVPR Reranking")):
                        # Load query encoder features: [256, 768]
                        query_enc_features = torch.from_numpy(
                            load_npy(f"Query_{seq_name}_{query_index}")
                        ).float().cuda()

                        # Load top-K candidate encoder features
                        for cnt, candidate_idx in enumerate(pred[:RERANKING_TOP_K]):
                            candidates_features[cnt] = torch.from_numpy(
                                load_npy(f"Db_{seq_name}_{candidate_idx}")
                            ).float().cuda()

                        # Expand query to batch: [K, 256, 768]
                        query_batch = query_enc_features.unsqueeze(0).expand(RERANKING_TOP_K, -1, -1)
                        rerank_scores = model.module.stage2_inference(query_batch, candidates_features)

                        # Sort and reorder
                        rerank_scores_np = rerank_scores.cpu().numpy()
                        rerank_index = rerank_scores_np.argsort()[::-1]
                        rerank_scores_dict[query_index] = rerank_scores_np[rerank_index].tolist()
                        predictions.append(pred[rerank_index])
                predictions = np.array(predictions)

            elif args.use_reranking == 'reconAttn':
                # Attention-based masking + Reconstruction loss reranking
                logging.info("Using reconAttn reranking (attention-based masking + reconstruction loss)")

                from torchvision.transforms import v2
                from datasets_T2R import base_transform
                from croco.models.masking import AttentionMask

                predictions_list = []
                rerank_scores_dict = {}

                with torch.no_grad():
                    for query_index, pred in enumerate(tqdm(prev_predictions, desc="ReconAttn Reranking")):
                        top_k_indices = pred[:RERANKING_TOP_K]

                        # ===== 1. Load raw images =====
                        query_img_raw = eval_ds.get_thermal_img(eval_ds.t_queries_paths[query_index])
                        query_img = base_transform(query_img_raw)
                        query_img = v2.functional.resize(query_img, args.resize).unsqueeze(0).cuda()

                        candidate_imgs = []
                        for db_idx in top_k_indices:
                            cand_img_raw = eval_ds.get_rgb_img(eval_ds.rgb_database_paths[db_idx])
                            cand_img = base_transform(cand_img_raw)
                            cand_img = v2.functional.resize(cand_img, args.resize)
                            candidate_imgs.append(cand_img)
                        candidate_imgs = torch.stack(candidate_imgs).cuda()  # [K, 3, H, W]

                        # ===== 2. Use stage2_inference_recon_batch =====
                        recon_scores = model.module.stage2_inference_recon_batch(
                            query_img,       # [1, C, H, W]
                            candidate_imgs   # [K, C, H, W]
                        )

                        # ===== 3. Rerank (lower loss = better match) =====
                        recon_scores_np = recon_scores.cpu().numpy()
                        rerank_index = recon_scores_np.argsort()  # ascending: lower loss is better
                        rerank_scores_dict[query_index] = recon_scores_np[rerank_index].tolist()
                        predictions_list.append(pred[rerank_index])

                predictions = np.array(predictions_list)

            elif args.use_reranking == 'distance':
                # Distance-based geometric matching reranking (DataLoader-based)
                logging.info("Using distance-based reranking (DataLoader-based, batched)")

                # ===== DataLoader 설정 =====
                reranking_dataset = RerankingDataset(
                    predictions=prev_predictions,
                    seq_name=seq_name,
                    npy_root_path=NPY_ROOTPATH,
                    reranking_top_k=RERANKING_TOP_K,
                    load_targets=False  # distance reranking은 target 불필요
                )

                reranking_batch_size = getattr(args, 'reranking_batch_size', 32)
                reranking_num_workers = getattr(args, 'reranking_num_workers', 8)

                reranking_dataloader = DataLoader(
                    dataset=reranking_dataset,
                    batch_size=reranking_batch_size,
                    shuffle=False,
                    num_workers=reranking_num_workers,
                    collate_fn=reranking_collate_fn,
                    pin_memory=True,
                    prefetch_factor=2
                )

                logging.info(f"Distance Reranking DataLoader: batch_size={reranking_batch_size}, num_workers={reranking_num_workers}")

                # ===== Batched Distance Reranking 실행 =====
                predictions, rerank_scores_dict = batched_distance_reranking(
                    model=model,
                    dataloader=reranking_dataloader,
                    prev_predictions=prev_predictions,
                    reranking_top_k=RERANKING_TOP_K
                )

            del queries_features
            del database_features
        
        # 4. positive query(정답)가 몇 번째 top-N에 속하는지 검사하기
        positives_per_query = eval_ds.get_positives()
        recalls = np.zeros(len(args.recall_values))
        pre_num = eval_ds.queries_num
        for query_index, pred in enumerate(predictions):
            for i, n in enumerate(args.recall_values):
                if np.any(np.in1d(pred[:n], positives_per_query[query_index])):
                    recalls[i:] += 1
                    break
        recalls = recalls / eval_ds.queries_num * 100
        
        logging.info(f"recalls: {','.join(map(str, recalls))}")
        recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(args.recall_values, recalls)])
        
        if args.use_reranking != 'none':
            prev_recalls = np.zeros(len(args.recall_values))
            for query_index, pred in enumerate(prev_predictions):
                for i, n in enumerate(args.recall_values):
                    if np.any(np.in1d(pred[:n], positives_per_query[query_index])):
                        prev_recalls[i:] += 1
                        break
            prev_recalls = prev_recalls / eval_ds.queries_num * 100
        
            logging.info(f"=================================================")
            prev_recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(args.recall_values, prev_recalls)])
            logging.info(f"Recalls before RERANKING {seq_name}: {prev_recalls_str}")
            logging.info(f"=================================================")
        
        gc.collect()
        torch.cuda.empty_cache()
        return recalls, recalls_str
    except Exception as e:
        import traceback
        print(f"ERROR caught: {e}")
        traceback.print_exc()  # 전체 stack trace 출력
        breakpoint()
        