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
def inference(args, eval_ds, model, scene_name="", pca=None, k=1, use_cuda=True, verbose=True):
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
            
            for inputs, indices, flags in tqdm(database_dataloader, ncols=100):
                outputs = model(inputs.to(args.device), flags)
                
                features = outputs[0].view(-1, args.features_dim)
                features = features.cpu().numpy()
                database_features[indices.numpy(), :] = features

            logging.info(f"Finished extracting {eval_ds.database_num} database features in {time.time() - start_time:.2f} s")

            ### Extract query features
            # 2. query 전부의 feature 추출
            start_time = time.time()
            queries_infer_batch_size = args.infer_batch_size
            queries_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num, len(eval_ds))))
            queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args.num_workers,
                                            batch_size=queries_infer_batch_size, pin_memory=(args.device=="cuda"))

            queries_features = np.empty((eval_ds.queries_num, args.features_dim), dtype="float32")

            for inputs, indices, flags in tqdm(queries_dataloader, ncols=100):
                outputs = model(inputs.to(args.device), flags)
                
                features = outputs[0].view(-1, args.features_dim)
                features = features.cpu().numpy()
                queries_features[indices.numpy()-eval_ds.database_num, :] = features

            logging.info(f"Finished extracting {eval_ds.queries_num} query features in {time.time() - start_time:.2f} s")

        # 3. faiss를 이용하여, L2 distance로 가까운 descriptor 찾기
        faiss_index = faiss.IndexFlatL2(args.features_dim)
        faiss_index.add(database_features)
        del database_features
        
        start_time = time.time()
        distances, predictions = faiss_index.search(queries_features, max(args.recall_values))
        del queries_features
        del faiss_index
        import gc; gc.collect()
        
        #########################
        ### RERANKING Process ###
        #########################
        # ========== 시각화용 변수 초기화 ==========
        confidence_map_for_vis = None
        top5_predictions_for_vis = predictions[:, :5].copy()  # [num_queries, 5]
        # ==========================================
        
        start_time = time.time()
        if args.use_reranking:                
            torch.cuda.empty_cache()
            ##### 시각화용 #####
            original_distances = distances.copy()
            original_predictions = predictions.copy()
            ##################
            with torch.no_grad():
                # rerank1. RGB database 전부 추출 (before decoder)
                start_time = time.time()

                database_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num)))
                database_dataloader = DataLoader(dataset=database_subset_ds, num_workers=args.num_workers,
                                                batch_size=args.infer_batch_size, pin_memory=(args.device=="cuda"))
            
                database_mask = np.empty((eval_ds.database_num, 256), dtype="bool")  # ← 수정!
                masked_database_embedding = np.empty((eval_ds.database_num, 256, args.features_dim), dtype="float32")
                for inputs, indices, flags in tqdm(database_dataloader, ncols=100):
                    rgb_visible, mask, patch_B, patch_N, patch_D = model.module.croco_like_encoder(inputs.to(args.device))
                    rgb_full = model.module.croco_encoded_mask_expension(rgb_visible, mask, patch_B, patch_N, patch_D)
                    rgb_full_dec = rgb_full + model.module.decoder_pos_embed
                    enc_features = rgb_full_dec.cpu().numpy()
                    masked_database_embedding[indices.numpy(), :, :] = enc_features
                    database_mask[indices.numpy(), :] = mask.detach().cpu().numpy()  # ← 수정!

                logging.info(f"Finished extracting (FOR RERANK) {eval_ds.database_num} database features in {time.time() - start_time:.2f} s")

                # rerank2. masked된 query 전부 추출 (before decoder)
                start_time = time.time()
                queries_infer_batch_size = args.infer_batch_size
                queries_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num, len(eval_ds))))
                queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args.num_workers,
                                                batch_size=queries_infer_batch_size, pin_memory=(args.device=="cuda"))

                thermal_embedding = np.empty((eval_ds.queries_num, 256, args.features_dim), dtype="float32")
                for inputs, indices, flags in tqdm(queries_dataloader, ncols=100):
                    features = model(inputs.to(args.device), flags, return_mask=True)
                    encoded_features = (features[1].reshape(inputs.size(0), 256, -1)+ model.module.decoder_pos_embed).cpu().numpy()
                    thermal_embedding[indices.numpy()-eval_ds.database_num,:,:] = encoded_features

                logging.info(f"Finished extracting (FOR RERANK) {eval_ds.queries_num} query features in {time.time() - start_time:.2f} s")
                        
            # rerank3. Decoder 쭉쭉 태워서 rerank 진행하기
            RERANKING_TOP_K = 5
            RERANK_BATCH_SIZE = 32
            reranked_predictions = predictions.copy()
            reconstruction_criterion = MaskedMSE(
                norm_pix_loss=False,
                masked=True,
                reduction='none',
                loss_type=args.recon_loss_fn_type
            )
            reconstruction_losses_dict = {}
            
            top1_change_count = 0
            total_count = 0
            with torch.no_grad():
                num_queries = eval_ds.queries_num
                num_batches = (num_queries + RERANK_BATCH_SIZE - 1) // RERANK_BATCH_SIZE
                
                for batch_idx in tqdm(range(num_batches), desc="Reranking", ncols=100):
                    start_idx = batch_idx * RERANK_BATCH_SIZE
                    end_idx = min(start_idx + RERANK_BATCH_SIZE, num_queries)
                    batch_size = end_idx - start_idx
                    
                    encoded_queries = thermal_embedding[start_idx:end_idx]
                    encoded_queries = torch.tensor(encoded_queries, dtype=torch.float32).to('cuda')
                    
                    top_k_db_indices_batch = predictions[start_idx:end_idx, :RERANKING_TOP_K]
                    
                    all_db_indices = top_k_db_indices_batch.flatten()
                    encoded_dbs_flat = masked_database_embedding[all_db_indices]
                    encoded_dbs_flat = torch.tensor(encoded_dbs_flat, dtype=torch.float32).to('cuda')
                    encoded_dbs = encoded_dbs_flat.reshape(batch_size, RERANKING_TOP_K, 256, -1)
                    
                    encoded_query_batch = encoded_queries.unsqueeze(1).expand(-1, RERANKING_TOP_K, -1, -1)
                    encoded_query_flat = encoded_query_batch.reshape(-1, 256, encoded_queries.size(-1))
                    encoded_dbs_flat = encoded_dbs.reshape(-1, 256, encoded_dbs.size(-1))
                    
                    encoded_dbs_flat_dec = encoded_dbs_flat
                    for blk in model.module.decoder_blocks:
                        encoded_dbs_flat_dec = blk(encoded_dbs_flat_dec, encoded_query_flat)
                    encoded_dbs_flat_dec = model.module.decoder_norm(encoded_dbs_flat_dec)
                    
                    reconstructed_patches = model.module.prediction_head(encoded_dbs_flat_dec)
                    
                    # ========== Confidence map 계산 (첫 배치만) ==========
                    if batch_idx == 0 and hasattr(model.module, 'confidence_head'):
                        confidence_map_for_vis = model.module.confidence_head(encoded_dbs_flat_dec[:5])  # [5, 256, 1]
                        confidence_map_for_vis = confidence_map_for_vis.squeeze(-1)  # [5, 256]
                    # ====================================================
                    
                    query_abs_indices = list(range(eval_ds.database_num + start_idx, eval_ds.database_num + end_idx))
                    query_imgs = torch.stack([eval_ds[idx][0] for idx in query_abs_indices]).to('cuda')
                    query_imgs_batch = query_imgs.unsqueeze(1).expand(-1, RERANKING_TOP_K, -1, -1, -1)
                    query_imgs_flat = query_imgs_batch.reshape(-1, *query_imgs.shape[1:])
                    target_patches = patchify(query_imgs_flat)
                    
                    masks_batch = database_mask[all_db_indices]  # ← 수정!
                    masks_batch = torch.tensor(masks_batch, dtype=torch.bool).to('cuda')
                    
                    loss = reconstruction_criterion(
                        pred=reconstructed_patches,
                        mask=masks_batch,
                        target=target_patches
                    )
                    
                    loss = loss.reshape(batch_size, RERANKING_TOP_K)
                    
                    for i, query_idx in enumerate(range(start_idx, end_idx)):
                        reconstruction_losses = loss[i].cpu().numpy()
                        reranked_order = np.argsort(reconstruction_losses)
                        
                        reconstruction_losses_dict[query_idx] = reconstruction_losses.tolist()
                        
                        top_k_db_indices = top_k_db_indices_batch[i]
                        reranked_predictions[query_idx, :RERANKING_TOP_K] = top_k_db_indices[reranked_order]
                        
                        if predictions[query_idx, 0] != reranked_predictions[query_idx, 0]:
                            top1_change_count += 1
                        total_count += 1

                logging.info(f"Reranking completed in {time.time() - start_time:.2f} s")
                print(f"RERANK: changed {top1_change_count} / {total_count}")
                    
                # Reranking comparison visualization
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
                
                predictions = reranked_predictions
                top5_predictions_for_vis = reranked_predictions[:, :5].copy()
        
        # 4. Recall 계산
        positives_per_query = eval_ds.get_positives()
        recalls = np.zeros(len(args.recall_values))
        for query_index, pred in enumerate(predictions):
            for i, n in enumerate(args.recall_values):
                if np.any(np.in1d(pred[:n], positives_per_query[query_index])):
                    recalls[i:] += 1
                    break
        recalls = recalls / eval_ds.queries_num * 100
        
        logging.info(f"recalls: {','.join(map(str, recalls))}")
        recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(args.recall_values, recalls)])

        # ========== 시각화 (Reranking 여부 무관) ==========
        if hasattr(args, 'current_epoch'):
            try:
                # 첫 번째 query 사용
                query_idx = 0
                
                # Thermal query
                thermal_abs_idx = eval_ds.database_num + query_idx
                thermal_img = eval_ds[thermal_abs_idx][0].to('cuda')  # [3, 224, 224]
                
                # Top-5 RGB (reranking 했으면 reranked, 안 했으면 original)
                top5_rgb_indices = top5_predictions_for_vis[query_idx]  # [5]
                top5_rgb_imgs = []
                for db_idx in top5_rgb_indices:
                    rgb_tensor = eval_ds[int(db_idx)][0]
                    top5_rgb_imgs.append(rgb_tensor)
                top5_rgb_imgs = torch.stack(top5_rgb_imgs).to('cuda')  # [5, 3, 224, 224]
                
                # Confidence map (reranking 했을 때만 존재)
                if confidence_map_for_vis is not None:
                    top5_conf = confidence_map_for_vis[:5]  # [5, 256]
                else:
                    # Dummy confidence (균일)
                    top5_conf = torch.ones(5, 256).to('cuda') * 0.5
                
                save_path = os.path.join(
                    args.save_dir,
                    'confidence_maps',
                    f'epoch_{args.current_epoch:03d}_{scene_name}_confMap.png'
                )
                
                save_confidence_vis_simple(
                    thermal_img=thermal_img,
                    rgb_imgs=top5_rgb_imgs,
                    confidence_map=top5_conf,
                    save_path=save_path
                )
                
                logging.info(f"Saved confidence visualization: {save_path}")
            except Exception as e:
                logging.warning(f"Failed to save confidence visualization: {e}")
        # ==================================================
        
        return recalls, recalls_str
        
    except Exception as e:
        import traceback
        print(f"ERROR caught: {e}")
        traceback.print_exc()
        breakpoint()      
        