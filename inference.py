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

from croco.models.masking import RandomMask
from croco.models.criterion import MaskedMSE

from recon_vis import *

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

def save_mnn_visualization(eval_ds, query_indices, top_k_db_indices, 
                          mutual_matches_list, rerank_scores, 
                          save_dir, epoch):
    """
    MNN matching 시각화
    
    Args:
        eval_ds: dataset
        query_indices: list of query indices
        top_k_db_indices: [num_queries, K] - Top-K DB indices
        mutual_matches_list: list of (matches_i, matches_j, conf) tuples
        rerank_scores: [num_queries, K] - Reranking scores
        save_dir: save directory
        epoch: current epoch
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    os.makedirs(save_dir, exist_ok=True)
    
    for q_idx, (query_idx, db_indices, matches, scores) in enumerate(
        zip(query_indices, top_k_db_indices, mutual_matches_list, rerank_scores)
    ):
        # Load images
        thermal_abs_idx = eval_ds.database_num + query_idx
        thermal_img = eval_ds[thermal_abs_idx][0]  # [3, 224, 224]
        
        # Denormalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        thermal_img = (thermal_img * std + mean).permute(1, 2, 0).numpy()
        thermal_img = np.clip(thermal_img, 0, 1)
        
        # Top-5 RGB images
        fig, axes = plt.subplots(2, 6, figsize=(24, 8))
        
        # Row 1: Thermal + Top-5 RGB
        axes[0, 0].imshow(thermal_img)
        axes[0, 0].set_title(f'Query {query_idx}\n(Thermal)', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        for k in range(5):
            db_idx = db_indices[k]
            rgb_img = eval_ds[int(db_idx)][0]
            rgb_img = (rgb_img * std + mean).permute(1, 2, 0).numpy()
            rgb_img = np.clip(rgb_img, 0, 1)
            
            axes[0, k+1].imshow(rgb_img)
            axes[0, k+1].set_title(f'Rank {k+1}\nScore: {scores[k]:.3f}', fontsize=10)
            axes[0, k+1].axis('off')
        
        # Row 2: Matching visualization (only Top-1)
        if len(matches[0]) > 0:
            matches_i, matches_j, matches_conf = matches
            
            # Top-1 RGB
            db_idx = db_indices[0]
            rgb_img = eval_ds[int(db_idx)][0]
            rgb_img = (rgb_img * std + mean).permute(1, 2, 0).numpy()
            rgb_img = np.clip(rgb_img, 0, 1)
            
            # Side-by-side with matches
            combined = np.hstack([thermal_img, rgb_img])
            
            ax = plt.subplot(2, 1, 2)
            ax.imshow(combined)
            
            # Draw matches
            patch_size = 14
            img_h, img_w = 224, 224
            
            # Convert patch indices to pixel coordinates
            def patch_to_pixel(patch_idx):
                row = patch_idx // 16
                col = patch_idx % 16
                y = row * patch_size + patch_size // 2
                x = col * patch_size + patch_size // 2
                return x, y
            
            # Draw lines
            num_matches = min(50, len(matches_i))  # 최대 50개
            for idx in range(num_matches):
                i = matches_i[idx]
                j = matches_j[idx]
                conf = matches_conf[idx]
                
                x1, y1 = patch_to_pixel(i)
                x2, y2 = patch_to_pixel(j)
                x2 += img_w  # RGB는 오른쪽
                
                # Color by confidence
                color = plt.cm.hot(conf / 0.1)  # 0.1 = max expected conf
                
                ax.plot([x1, x2], [y1, y2], 
                       color=color, linewidth=1, alpha=0.6)
                ax.scatter([x1], [y1], c='cyan', s=10, zorder=5)
                ax.scatter([x2], [y2], c='lime', s=10, zorder=5)
            
            ax.set_title(f'MNN Matches: {len(matches_i)} pairs (showing top {num_matches})', 
                        fontsize=12, fontweight='bold')
            ax.axis('off')
            
            # Vertical divider
            ax.axvline(x=img_w, color='white', linewidth=2, linestyle='--')
        else:
            axes[1, 0].text(0.5, 0.5, 'No matches found', 
                          ha='center', va='center', fontsize=16)
            axes[1, 0].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'epoch_{epoch:03d}_query_{query_idx:05d}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved MNN visualization: {save_path}")
    
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

# TODO: can be less memory cost
# TODO: finish the uncompleted parts
def inference(args, eval_ds, model, pca=None, k=1, use_cuda=True, verbose=True):
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
            database_patch_features = np.empty((eval_ds.database_num, 16*16, args.features_dim), dtype="float32")
            for inputs, indices, flags in tqdm(database_dataloader, ncols=100):
                outputs = model(inputs.to(args.device), flags)
                
                features = outputs[0].view(-1, args.features_dim)
                patch_features = outputs[1].view(-1, 16*16, args.features_dim)
                database_features[indices.numpy(), :] = features.cpu().numpy()
                database_patch_features[indices.numpy(), :, :] = patch_features.cpu().numpy()
                
            logging.info(f"Finished extracting {eval_ds.database_num} database features in {time.time() - start_time:.2f} s")

            ### Extract query features
            # 2. query 전부의 feature 추출
            start_time = time.time()
            queries_infer_batch_size = args.infer_batch_size
            queries_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num, len(eval_ds))))
            queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args.num_workers,
                                            batch_size=queries_infer_batch_size, pin_memory=(args.device=="cuda"))

            queries_features = np.empty((eval_ds.queries_num, args.features_dim), dtype="float32")
            queries_patch_features = np.empty((eval_ds.queries_num, 16*16, args.features_dim), dtype="float32")
            for inputs, indices, flags in tqdm(queries_dataloader, ncols=100):
                outputs = model(inputs.to(args.device), flags)
                
                features = outputs[0].view(-1, args.features_dim)
                patch_features = outputs[1].view(-1, 16*16, args.features_dim)
                queries_features[indices.numpy()-eval_ds.database_num, :] = features.cpu().numpy()
                queries_patch_features[indices.numpy()-eval_ds.database_num, :, :] = patch_features.cpu().numpy()
                # break # for fast debug
                
            logging.info(f"Finished extracting {eval_ds.queries_num} query features in {time.time() - start_time:.2f} s")

        # 3. faiss를 이용하여, L2 distance로 가까운 descriptor 찾기
        faiss_index = faiss.IndexFlatL2(args.features_dim)
        faiss_index.add(database_features)
        
        start_time = time.time()
        distances, predictions = faiss_index.search(queries_features, max(args.recall_values))
        del faiss_index
        
        #########################
        ### RERANKING Process ###
        #########################
        start_time = time.time()
        if args.use_reranking:                
            torch.cuda.empty_cache()
            ##### 시각화용 #####
            original_predictions = predictions.copy()
            
            ##################
            # rerank3. Decoder 쭉쭉 태워서 rerank 진행하기
            # masked_database_features.shape: [1197, 256, 768]
            RERANKING_TOP_K = 5
            RERANK_BATCH_SIZE = 32
            reranked_predictions = predictions.copy()  # 원본 보존
            reconstruction_losses_dict = {}
            
            total_count = 0
            top1_change_count = 0
            with torch.no_grad():
                # Batch 단위로 처리
                num_queries = eval_ds.queries_num
                num_batches = (num_queries + RERANK_BATCH_SIZE - 1) // RERANK_BATCH_SIZE
                
                try:
                    for batch_idx in tqdm(range(num_batches), desc="Reranking", ncols=100):
                        start_idx = batch_idx * RERANK_BATCH_SIZE
                        end_idx = min(start_idx + RERANK_BATCH_SIZE, num_queries)
                        batch_size = end_idx - start_idx

                        # a. Query thermal encoding (batch)
                        encoded_queries = queries_patch_features[start_idx:end_idx]  # [B, 256, 768]
                        encoded_queries = torch.tensor(encoded_queries, dtype=torch.float32).to('cuda')
                        encoded_queries += model.module.decoder_pos_embed

                        # b. Top-K database RGB encoding (batch)
                        top_k_db_indices_batch = predictions[start_idx:end_idx, :RERANKING_TOP_K]  # [B, K]

                        # c. Flatten indices to fetch all at once
                        all_db_indices = top_k_db_indices_batch.flatten() # [B*K]
                        encoded_dbs_flat = database_patch_features[all_db_indices]  # [B*K, 256, 768]
                        encoded_dbs_flat = torch.tensor(encoded_dbs_flat, dtype=torch.float32).to('cuda')
                        encoded_dbs = encoded_dbs_flat.reshape(batch_size, RERANKING_TOP_K, 256, -1)  # [B, K, 256, 768]
                        encoded_dbs += model.module.decoder_pos_embed
                        
                        # d. Query batch 생성 (각 query를 K번 반복)
                        encoded_query_batch = encoded_queries.unsqueeze(1).expand(-1, RERANKING_TOP_K, -1, -1)  # [B, K, 256, 768]
                        
                        # e. Reshape for decoder: [B*K, 256, 768]
                        encoded_query_flat = encoded_query_batch.reshape(-1, 256, encoded_queries.size(-1))
                        encoded_dbs_flat = encoded_dbs.reshape(-1, 256, encoded_dbs.size(-1))
                        
                        # f. Decoder 통과 (thermal-rgb pair)
                        thermal_dec = encoded_query_flat.clone()
                        for blk in model.module.decoder_thermal_blocks:
                            thermal_dec = blk(thermal_dec, encoded_dbs_flat, return_attention=True)
                        thermal_dec = model.module.decoder_norm(thermal_dec)
                        thermal_cross_attn_map = model.module.decoder_thermal_blocks[-1].cross_attn_weights
                        
                        rgb_dec = encoded_dbs_flat.clone()
                        for blk in model.module.decoder_rgb_blocks:
                            rgb_dec = blk(rgb_dec, encoded_query_flat, return_attention=True)
                        rgb_dec = model.module.decoder_norm(rgb_dec)
                        rgb_cross_attn_map = model.module.decoder_rgb_blocks[-1].cross_attn_weights
                        
                        # ========== LoFTR-style Mutual Agreement ==========
                        rgb_map_transposed = rgb_cross_attn_map.transpose(1, 2)
                        mutual_agreement = thermal_cross_attn_map * rgb_map_transposed  # [B*K, 256, 256]
                        
                        # ========== MNN (Mutual Nearest Neighbor) ==========
                        B_K, N_thermal, N_rgb = thermal_cross_attn_map.shape
                        
                        # 1. Best matches (argmax)
                        thermal_to_rgb = thermal_cross_attn_map.argmax(dim=2)  # [B*K, 256]
                        rgb_to_thermal = rgb_cross_attn_map.argmax(dim=2)      # [B*K, 256]
                        
                        # 2. Mutual check + Threshold
                        threshold = 0.01
                        mutual_scores = torch.zeros(B_K, device=thermal_cross_attn_map.device)
                        mutual_matches_list = []  # For visualization
                        
                        for b in range(B_K):
                            i_range = torch.arange(N_thermal, device=thermal_cross_attn_map.device)
                            j_from_i = thermal_to_rgb[b]
                            i_from_j = rgb_to_thermal[b, j_from_i]
                            
                            # Mutual mask
                            is_mutual = (i_from_j == i_range)
                            
                            # Confidence check
                            conf_thermal = thermal_cross_attn_map[b, i_range, j_from_i]
                            conf_rgb = rgb_cross_attn_map[b, j_from_i, i_range]
                            conf_avg = (conf_thermal + conf_rgb) / 2
                            
                            # Threshold
                            is_confident = conf_avg > threshold
                            
                            # Final mask
                            valid_matches = is_mutual & is_confident
                            
                            # Score
                            mutual_scores[b] = (conf_avg * valid_matches).sum()
                            
                            # Save matches for visualization (첫 5개만)
                            if b < 5:
                                matches_i = i_range[valid_matches].cpu().numpy()
                                matches_j = j_from_i[valid_matches].cpu().numpy()
                                matches_conf = conf_avg[valid_matches].cpu().numpy()
                                mutual_matches_list.append((matches_i, matches_j, matches_conf))
                        
                        # 3. Reshape
                        rerank_scores_batch = mutual_scores.view(batch_size, RERANKING_TOP_K)
                        
                        # 4. Reranking
                        for i in range(batch_size):
                            query_idx = start_idx + i
                            scores = rerank_scores_batch[i].cpu().numpy()
                            reranked_order = np.argsort(-scores)  # Descending
                            
                            top_k_db_indices = top_k_db_indices_batch[i]
                            reranked_predictions[query_idx, :RERANKING_TOP_K] = \
                                top_k_db_indices[reranked_order]
                            
                            if predictions[query_idx, 0] != reranked_predictions[query_idx, 0]:
                                top1_change_count += 1
                            total_count += 1
                        
                        # ========== Statistics Logging (첫 배치만) ==========
                        if batch_idx == 0:
                            # MNN Statistics
                            print(f"\n{'='*60}")
                            print(f"MNN Statistics (Batch {batch_idx}):")
                            print(f"{'='*60}")
                            for b in range(min(5, B_K)):
                                if b < len(mutual_matches_list):
                                    num_matches = len(mutual_matches_list[b][0])
                                    avg_conf = mutual_matches_list[b][2].mean() if num_matches > 0 else 0
                                    print(f"Sample {b}: {num_matches:3d} matches, avg conf: {avg_conf:.4f}")
                            print(f"{'='*60}\n")
                            
                            # Mutual Agreement Statistics
                            ma_flat = mutual_agreement.view(mutual_agreement.size(0), -1)
                            stats = {
                                'mean': ma_flat.mean(dim=1).cpu().numpy(),
                                'std': ma_flat.std(dim=1).cpu().numpy(),
                                'max': ma_flat.max(dim=1)[0].cpu().numpy(),
                                'min': ma_flat.min(dim=1)[0].cpu().numpy(),
                                'median': ma_flat.median(dim=1)[0].cpu().numpy(),
                            }
                            
                            thresholds = [0.00001, 0.001, 0.005, 0.01, 0.05, 0.1]
                            threshold_counts = {}
                            for th in thresholds:
                                mask = (mutual_agreement > th)
                                counts = mask.sum(dim=(1, 2)).cpu().numpy()
                                threshold_counts[th] = counts
                            
                            log_path = os.path.join(args.save_dir, 'mutual_agreement_logs', f'epoch_{args.current_epoch:03d}_batch_{batch_idx:03d}.txt')
                            os.makedirs(os.path.dirname(log_path), exist_ok=True)
                            
                            with open(log_path, 'w') as f:
                                f.write(f"Epoch {args.current_epoch} - Batch {batch_idx}\n")
                                f.write(f"Shape: {mutual_agreement.shape}\n")
                                f.write("="*60 + "\n\n")
                                
                                for b in range(min(5, mutual_agreement.size(0))):
                                    f.write(f"Sample {b} (Query {start_idx + b // RERANKING_TOP_K}, Rank {b % RERANKING_TOP_K}):\n")
                                    f.write(f"  Mean:   {stats['mean'][b]:.6f}\n")
                                    f.write(f"  Std:    {stats['std'][b]:.6f}\n")
                                    f.write(f"  Max:    {stats['max'][b]:.6f}\n")
                                    f.write(f"  Min:    {stats['min'][b]:.6f}\n")
                                    f.write(f"  Median: {stats['median'][b]:.6f}\n")
                                    f.write(f"  Threshold counts:\n")
                                    for th in thresholds:
                                        f.write(f"    > {th:.3f}: {threshold_counts[th][b]:.0f} / 65536 ({100*threshold_counts[th][b]/65536:.2f}%)\n")
                                    f.write("\n")
                                
                                f.write("="*60 + "\n")
                                f.write("Overall Statistics (all B*K samples):\n")
                                f.write(f"  Mean:   {stats['mean'].mean():.6f} ± {stats['mean'].std():.6f}\n")
                                f.write(f"  Max:    {stats['max'].mean():.6f} ± {stats['max'].std():.6f}\n")
                                f.write(f"  Median: {stats['median'].mean():.6f} ± {stats['median'].std():.6f}\n")
                                f.write(f"\n  Average threshold counts:\n")
                                for th in thresholds:
                                    avg_count = threshold_counts[th].mean()
                                    f.write(f"    > {th:.3f}: {avg_count:.1f} / 65536 ({100*avg_count/65536:.2f}%)\n")
                            
                            print(f"Saved mutual agreement log: {log_path}")
                            
                            # Visualization
                            save_mnn_visualization(
                                eval_ds=eval_ds,
                                query_indices=list(range(start_idx, min(start_idx + 5, end_idx))),
                                top_k_db_indices=top_k_db_indices_batch[:5],
                                mutual_matches_list=mutual_matches_list,
                                rerank_scores=rerank_scores_batch[:5].cpu().numpy(),
                                save_dir=os.path.join(args.save_dir, 'mnn_matches'),
                                epoch=args.current_epoch
                            )
                        break
                        # ==================================================
                        
                except Exception as e:
                    import traceback
                    print(f"ERROR caught: {e}")
                    traceback.print_exc()
                    breakpoint()

                logging.info(f"Reranking completed in {time.time() - start_time:.2f} s")
                print(f"RERANK: changed {top1_change_count} / {total_count}")
                
                # ===== 시각화 호출 =====
                if hasattr(args, 'current_epoch'):
                    positives_per_query = eval_ds.get_positives()
                    visualize_reranking_comparison(
                        args, 
                        eval_ds,
                        original_predictions=original_predictions,
                        reranked_predictions=reranked_predictions,
                        reconstruction_losses_dict=reconstruction_losses_dict,
                        positives_per_query=positives_per_query,
                        epoch=args.current_epoch,
                        distances=distances,
                        reconstructed_images=None,
                        save_dir=os.path.join(args.save_dir, 'rerank_vis'),
                        num_samples=5
                    )
                #########################
                predictions = reranked_predictions
        
        import gc; gc.collect()
        del database_features
        del queries_features # predictions: [Query, top20]
        
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
        print(pre_num, eval_ds.queries_num)
        
        return recalls, recalls_str
    except Exception as e:
        import traceback
        print(f"ERROR caught: {e}")
        traceback.print_exc()  # 전체 stack trace 출력
        breakpoint()
    
### TODO
def top_n_voting(topn, predictions, distances, maj_weight):
    if topn == 'top1':
        n = 1
        selected = 0
    elif topn == 'top5':
        n = 5
        selected = slice(0, 5)
    elif topn == 'top10':
        n = 10
        selected = slice(0, 10)
    # find predictions that repeat in the first, first five,
    # or fist ten columns for each crop
    vals, counts = np.unique(predictions[:, selected], return_counts=True)
    # for each prediction that repeats more than once,
    # subtract from its score
    for val, count in zip(vals[counts > 1], counts[counts > 1]):
        mask = (predictions[:, selected] == val)
        distances[:, selected][mask] -= maj_weight * count/n
        
def fuse_inference(args, eval_ds, models):
    model_rgb, model_t = models
    ds_rgb, ds_t = eval_ds
    model_rgb.eval()
    model_t.eval()
    
    with torch.no_grad():
        ### Extract database features
        start_time = time.time()

        database_subset_ds_rgb = Subset(ds_rgb, list(range(ds_rgb.database_num)))
        database_dataloader_rgb = DataLoader(dataset=database_subset_ds_rgb, num_workers=args.num_workers,
                                        batch_size=1, pin_memory=(args.device=="cuda"))
        
        database_subset_ds_t = Subset(ds_t, list(range(ds_t.database_num)))
        database_dataloader_t = DataLoader(dataset=database_subset_ds_t, num_workers=args.num_workers,
                                        batch_size=1, pin_memory=(args.device=="cuda"))

        assert ds_rgb.database_num == ds_t.database_num
        
        database_features_rgb = np.empty((ds_rgb.database_num, args.features_dim), dtype="float32")
        database_features_t = np.empty((ds_t.database_num, args.features_dim), dtype="float32")
        database_features_cat = np.empty((ds_rgb.database_num, 2*args.features_dim), dtype="float32")
        database_features_add = np.empty((ds_rgb.database_num, args.features_dim), dtype="float32")

        for inputs, indices in tqdm(database_dataloader_rgb, ncols=100):
            features = model_rgb(inputs.to(args.device)).view(-1, args.features_dim)
            features = features.cpu().numpy()
            database_features_rgb[indices.numpy(), :] = features
            database_features_cat[indices.numpy(), :args.features_dim] = features
            database_features_add[indices.numpy(), :] = features
        
        for inputs, indices in tqdm(database_dataloader_t, ncols=100):
            features = model_t(inputs.to(args.device)).view(-1, args.features_dim)
            features = features.cpu().numpy()
            database_features_t[indices.numpy(), :] = features
            database_features_cat[indices.numpy(), args.features_dim:] = features
            database_features_add[indices.numpy(), :] += features

        logging.info(f"Finished extracting {ds_rgb.database_num}*2 database features in {time.time() - start_time:.2f} s")

        ### Extract query features
        start_time = time.time()

        queries_infer_batch_size = 1
        queries_subset_ds_rgb = Subset(ds_rgb, list(range(ds_rgb.database_num, len(ds_rgb))))
        queries_dataloader_rgb = DataLoader(dataset=queries_subset_ds_rgb, num_workers=args.num_workers,
                                        batch_size=queries_infer_batch_size, pin_memory=(args.device=="cuda"))
        queries_subset_ds_t = Subset(ds_t, list(range(ds_t.database_num, len(ds_t))))
        queries_dataloader_t = DataLoader(dataset=queries_subset_ds_t, num_workers=args.num_workers,
                                        batch_size=queries_infer_batch_size, pin_memory=(args.device=="cuda"))
        
        assert ds_rgb.queries_num == ds_t.queries_num
        
        queries_features_rgb = np.empty((ds_rgb.queries_num, args.features_dim), dtype="float32")
        queries_features_t = np.empty((ds_t.queries_num, args.features_dim), dtype="float32")
        queries_features_cat = np.empty((ds_rgb.queries_num, 2*args.features_dim), dtype="float32")
        queries_features_add = np.empty((ds_rgb.queries_num, args.features_dim), dtype="float32")
        
        for inputs, indices in tqdm(queries_dataloader_rgb, ncols=100):
            features = model_rgb(inputs.to(args.device)).view(-1, args.features_dim)
            features = features.cpu().numpy()
            queries_features_rgb[indices.numpy()-ds_rgb.database_num, :] = features
            queries_features_cat[indices.numpy()-ds_rgb.database_num, :args.features_dim] = features
            queries_features_add[indices.numpy()-ds_rgb.database_num, :] = features
        
        for inputs, indices in tqdm(queries_dataloader_t, ncols=100):
            features = model_t(inputs.to(args.device)).view(-1, args.features_dim)
            features = features.cpu().numpy()
            queries_features_t[indices.numpy()-ds_rgb.database_num, :] = features
            queries_features_cat[indices.numpy()-ds_rgb.database_num, args.features_dim:] = features
            queries_features_add[indices.numpy()-ds_rgb.database_num, :] += features
                 
        logging.info(f"Finished extracting {ds_rgb.queries_num}*2 query features in {time.time() - start_time:.2f} s")
    
    faiss_index_rgb = faiss.IndexFlatL2(args.features_dim)
    faiss_index_t = faiss.IndexFlatL2(args.features_dim)
    faiss_index_cat = faiss.IndexFlatL2(2*args.features_dim)
    faiss_index_add = faiss.IndexFlatL2(args.features_dim)
    
    faiss_index_rgb.add(database_features_rgb)
    del database_features_rgb
    faiss_index_t.add(database_features_t)
    del database_features_t
    faiss_index_cat.add(database_features_cat)
    del database_features_cat
    faiss_index_add.add(database_features_add)
    del database_features_add

    ### Calculating recalls
    start_time = time.time()
    
    predictions_all = []
    distances_all = []
    
    distances, predictions = faiss_index_rgb.search(queries_features_rgb, max(args.recall_values))
    predictions_all.append(predictions)
    distances_all.append(distances)
    del queries_features_rgb
    
    distances, predictions = faiss_index_t.search(queries_features_t, max(args.recall_values))
    predictions_all.append(predictions)
    distances_all.append(distances)
    del queries_features_t
    
    distances, predictions = faiss_index_cat.search(queries_features_cat, max(args.recall_values))
    predictions_all.append(predictions)
    distances_all.append(distances)
    del queries_features_cat
    
    distances, predictions = faiss_index_add.search(queries_features_add, max(args.recall_values))
    predictions_all.append(predictions)
    distances_all.append(distances)
    del queries_features_add

    split = {
        'morning':list(range(0,365))+list(range(1371,1770))+list(range(2754,2869)),
        'afternoon':list(range(365,892))+list(range(1770,2231))+list(range(2869,2994)),
        'evening':list(range(892,1371))+list(range(2231,2754))+list(range(2994,3130)),
        'allday':list(range(0,3130))
    }
    
    positives_per_query = ds_rgb.get_positives()
    recalls = {'morning':np.zeros((4, len(args.recall_values))), 'afternoon':np.zeros((4, len(args.recall_values))),\
        'evening':np.zeros((4, len(args.recall_values))), 'allday':np.zeros((4, len(args.recall_values)))}
    recalls_str = {'morning':[""]*4, 'afternoon':[""]*4, 'evening':[""]*4, 'allday':[""]*4}
    
    method = ['rgb', 't', 'cat', 'add']
    
    for k, (predictions, distances) in enumerate(zip(predictions_all, distances_all)):
        for key, indices in split.items():
            for query_index in indices:
                pred = predictions[query_index]
                for i, n in enumerate(args.recall_values):
                    if np.any(np.in1d(pred[:n], positives_per_query[query_index])):
                        recalls[key][k, i:] += 1
                        break
            
            recalls[key][k] = recalls[key][k] / len(indices) * 100
            recalls_str[key][k] = ", ".join([f"{method[k]}/{key}  R@{val}: {rec:.1f}" for val, rec in zip(args.recall_values, recalls[key][k])])

    return recalls, recalls_str