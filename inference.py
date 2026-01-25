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

START_TIME = get_timestamp()
NPY_ROOTPATH = f"/home/jwkim/workspace/THR2RGB_Cross_Place_Recognition/npys_agg/{START_TIME}"
def save_npy(data, path, comment):
    np.save(os.path.join(NPY_ROOTPATH, path), data)
    
def load_npy(path, comment):
    return np.load(os.path.join(NPY_ROOTPATH, path)+".npy")
    
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
        
            database_features = np.empty((eval_ds.database_num, args.features_dim*2), dtype="float32")
            database_attn_map = np.empty((eval_ds.database_num, patch_count), dtype="float32")
            
            for inputs, indices, flags in tqdm(database_dataloader, ncols=100):
                flags_int = [1 if f == 'rgb' else 0 for f in flags]
                flags = torch.tensor(flags_int, dtype=torch.long, device=args.device)
                outputs = model(inputs.to(args.device), flags)
                features = outputs[0].view(-1, args.features_dim*2)
                breakpoint()
                patch_features = outputs[1].view(-1, patch_W*patch_H, args.features_dim*2)
                
                indices_npy = indices.numpy()
                database_features[indices_npy,:] = features.cpu().numpy() # [B, C] # 이건 저장 x
                database_attn_map[indices_npy,:] = outputs[4].cpu().numpy() # [B, N] # 이것도 저장 x
                for num, idx in enumerate(indices_npy):
                    save_npy(patch_features[num].cpu().numpy(), f"Db_{seq_name}_{idx}", args.comment)
                    if args.r2_penultimate_layer and args.use_r2former:
                        save_npy(outputs[5][num].cpu().numpy(), f"Db_{seq_name}_penultimate_{idx}", args.comment)
                
            logging.info(f"Finished extracting {eval_ds.database_num} database features in {time.time() - start_time:.2f} s")

            # 2. query 전부의 feature 추출
            start_time = time.time()
            queries_infer_batch_size = args.infer_batch_size
            queries_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num, len(eval_ds))))
            queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args.num_workers,
                                            batch_size=queries_infer_batch_size, pin_memory=(args.device=="cuda"))

            queries_features = np.empty((eval_ds.queries_num, args.features_dim*2), dtype="float32")
            queries_attn_map = np.empty((eval_ds.queries_num, patch_count), dtype="float32")
            
            for inputs, indices, flags in tqdm(queries_dataloader, ncols=100):
                flags_int = [1 if f == 'rgb' else 0 for f in flags]
                flags = torch.tensor(flags_int, dtype=torch.long, device=args.device)
                outputs = model(inputs.to(args.device), flags)
                features = outputs[0].view(-1, args.features_dim*2)
                patch_features = outputs[1].view(-1, patch_W*patch_H, args.features_dim)
                
                indices_npy = indices.numpy()-eval_ds.database_num
                queries_features[indices.numpy()-eval_ds.database_num,:] = features.cpu().numpy()
                queries_attn_map[indices.numpy()-eval_ds.database_num,:] = outputs[4].cpu().numpy()
                
                for num, idx in enumerate(indices_npy):
                    save_npy(patch_features[num].cpu().numpy(), f"Query_{seq_name}_{idx}", args.comment)
                    if args.r2_penultimate_layer and args.use_r2former:
                        save_npy(outputs[5][num].cpu().numpy(), f"Query_{seq_name}_penultimate_{idx}", args.comment)
                # if args.use_fast_track: break
                
            logging.info(f"Finished extracting {eval_ds.queries_num} query features in {time.time() - start_time:.2f} s")

        # 3. faiss를 이용하여, L2 distance로 가까운 descriptor 찾기
        faiss_index = faiss.IndexFlatL2(args.features_dim*2)
        faiss_index.add(database_features)
        
        start_time = time.time()
        _, predictions = faiss_index.search(queries_features, max(args.recall_values))
        del faiss_index
        
        import gc; gc.collect()
        torch.cuda.empty_cache()
        
        #####################################
        ############# RERANKING #############
        if args.use_reranking:
            # ========== [수정 1] 파일 개수 검증 추가 ==========
            saved_files = os.listdir(NPY_ROOTPATH)
            prefix = "penultimate" if args.r2_penultimate_layer else ""
            
            db_files = [f for f in saved_files if f.startswith(f"Db_{seq_name}") and prefix in f]
            query_files = [f for f in saved_files if f.startswith(f"Query_{seq_name}") and prefix in f]
            
            assert len(db_files) == eval_ds.database_num, \
                f"DB files mismatch: {len(db_files)} vs {eval_ds.database_num}"
            assert len(query_files) == eval_ds.queries_num, \
                f"Query files mismatch: {len(query_files)} vs {eval_ds.queries_num}"
            
            logging.info(f"✓ File count verified: {len(db_files)} DB + {len(query_files)} Query")
            # ===================================================
            
            RERANKING_TOP_K = 5
            RERANK_BATCH_SIZE = 32
            prev_predictions = predictions.copy()  # 원본 보존
            reconstruction_losses_dict = {}
            
            total_count = 0
            top1_change_count = 0
            
            try:
                with torch.no_grad():
                    RERANK_BATCH_SIZE = 4
                    num_batches = (eval_ds.queries_num + RERANK_BATCH_SIZE - 1) // RERANK_BATCH_SIZE
                    
                    for batch_idx in tqdm(range(num_batches), desc="Reranking", ncols=100):
                        start_idx = batch_idx * RERANK_BATCH_SIZE
                        end_idx = min(start_idx + RERANK_BATCH_SIZE, eval_ds.queries_num)
                        batch_size = end_idx - start_idx
                        
                        if args.use_r2former:
                            # ========== 1. Query Features 준비 (배치) ==========
                            # [수정 2] 배치 내 각 query별로 npy 파일 개별 로드
                            if args.r2_penultimate_layer:
                                query_features_list = [
                                    torch.from_numpy(
                                        load_npy(f"Query_{seq_name}_penultimate_{query_idx}", args.comment)
                                    )
                                    for query_idx in range(start_idx, end_idx)
                                ]
                            else:
                                query_features_list = [
                                    torch.from_numpy(
                                        load_npy(f"Query_{seq_name}_{query_idx}", args.comment)
                                    )
                                    for query_idx in range(start_idx, end_idx)
                                ]
                            
                            # Stack하여 배치 텐서로 변환 [B, 256, 768]
                            encoded_queries_features = torch.stack(query_features_list).float().cuda()
                            # ===================================================
                            
                            encoded_queries_attn_map = torch.from_numpy(
                                queries_attn_map[start_idx:end_idx]
                            ).float().cuda()  # [B, 256]
                            
                            encoded_queries_descriptor = torch.from_numpy(
                                queries_features[start_idx:end_idx]
                            ).float().cuda()  # [B, 768]
                            
                            # ========== 2. Top-K Database Features 준비 (배치) ==========
                            top_k_db_indices_batch = predictions[start_idx:end_idx, :RERANKING_TOP_K]  # [B, K]
                            
                            # [수정 3] 배치 내 모든 DB indices에 대해 npy 파일 개별 로드
                            all_db_indices = top_k_db_indices_batch.flatten()  # [B*K]
                            
                            if args.r2_penultimate_layer:
                                db_features_list = [
                                    torch.from_numpy(
                                        load_npy(f"Db_{seq_name}_penultimate_{db_idx}", args.comment)
                                    )
                                    for db_idx in all_db_indices
                                ]
                            else:
                                db_features_list = [
                                    torch.from_numpy(
                                        load_npy(f"Db_{seq_name}_{db_idx}", args.comment)
                                    )
                                    for db_idx in all_db_indices
                                ]
                            
                            # Stack하여 텐서로 변환 후 reshape [B*K, 256, 768] -> [B, K, 256, 768]
                            encoded_dbs_features = torch.stack(db_features_list).float().cuda()
                            encoded_dbs_features = encoded_dbs_features.reshape(batch_size, RERANKING_TOP_K, patch_count, -1)
                            # ===================================================
                            
                            encoded_dbs_attn_map = torch.from_numpy(
                                database_attn_map[all_db_indices]
                            ).float().cuda()  # [B*K, 256]
                            
                            encoded_dbs_descriptor = torch.from_numpy(
                                database_features[all_db_indices]
                            ).float().cuda()  # [B*K, 768]
                            
                            # Reshape: [B, K, 256, 768]
                            encoded_dbs_attn_map = encoded_dbs_attn_map.reshape(batch_size, RERANKING_TOP_K, patch_count)
                            encoded_dbs_descriptor = encoded_dbs_descriptor.reshape(batch_size, RERANKING_TOP_K, -1)
                            
                            # ========== 5. Reranker용 데이터 준비 ==========
                            # Concatenate [query, db1, db2, ...] → [B, K+1, 256, 768]
                            concated_db_patches = torch.cat([
                                encoded_queries_features.unsqueeze(1),  # [B, 1, 256, 768]
                                encoded_dbs_features  # [B, K, 256, 768]
                            ], dim=1)
                            
                            concated_db_attn_map = torch.cat([
                                encoded_queries_attn_map.unsqueeze(1),  # [B, 1, 256]
                                encoded_dbs_attn_map  # [B, K, 256]
                            ], dim=1)
                            
                            # Flatten: [B*(K+1), 256, 768]
                            concated_db_patches_flat = concated_db_patches.reshape(-1, patch_count, concated_db_patches.size(-1))
                            concated_db_attn_map_flat = concated_db_attn_map.reshape(-1, patch_count)
                            
                            # ========== 6. Indices 생성 ==========
                            # query_index: [B*K] - 각 query의 위치 (0, K+1, 2*(K+1), ...)
                            queries_indexes = torch.zeros(batch_size * RERANKING_TOP_K, dtype=torch.long).cuda()
                            for i in range(batch_size):
                                queries_indexes[i * RERANKING_TOP_K:(i + 1) * RERANKING_TOP_K] = i * (RERANKING_TOP_K + 1)
                            
                            # positive_index: [B*K] - 각 database의 위치 (1, 2, ..., K, K+2, K+3, ...)
                            positives_indexes = torch.arange(batch_size * (RERANKING_TOP_K + 1), dtype=torch.long).cuda()
                            positives_indexes = positives_indexes[positives_indexes % (RERANKING_TOP_K + 1) != 0]
                            
                            # ========== 7. Global Descriptors ==========
                            global_query_exp = encoded_queries_descriptor.unsqueeze(1).expand(
                                -1, RERANKING_TOP_K, -1
                            ).reshape(-1, encoded_queries_descriptor.size(-1))  # [B*K, 768]
                            
                            global_pos_flat = encoded_dbs_descriptor.reshape(-1, encoded_dbs_descriptor.size(-1))  # [B*K, 768]
                            
                            # ========== 9. Reranking ==========
                            reranker = model.module.reranker
                            reranker.global_query_cache = encoded_queries_descriptor[0].unsqueeze(0)  # [1, 768]
                            
                            rerank_scores = reranker(
                                concated_db_patches_flat,  # [B*(K+1), 256, 768]
                                concated_db_attn_map_flat,  # [B*(K+1), 256]
                                queries_indexes,  # [B*K]
                                positives_indexes,  # [B*K]
                                None,  # neg_index
                                global_query=global_query_exp,  # [B*K, 768]
                                global_pos=global_pos_flat,  # [B*K, 768]
                                global_neg=None,
                                cross_attn_matrix=None  # [B*K, 256, 256]
                            )  # [B*K]
                            
                            # ========== 10. Reshape & Reorder ==========
                            rerank_scores = rerank_scores.reshape(batch_size, RERANKING_TOP_K)  # [B, K]
                            
                            for i in range(batch_size):
                                sorted_indices = torch.argsort(rerank_scores[i], descending=True)
                                predictions[start_idx + i, :RERANKING_TOP_K] = top_k_db_indices_batch[i][sorted_indices.cpu()]
                            
                            # ========== [수정 4] 메모리 해제 추가 ==========
                            del encoded_queries_features, encoded_queries_attn_map, encoded_queries_descriptor
                            del encoded_dbs_features, encoded_dbs_attn_map, encoded_dbs_descriptor
                            del concated_db_patches, concated_db_attn_map, concated_db_patches_flat, concated_db_attn_map_flat
                            del query_features_list, db_features_list
                            # ===================================================
                        else:
                            # a. Query thermal encoding (batch)
                            if args.rerank_with_RGB:
                                encoded_queries = queries_patch_features[start_idx:end_idx]  # [B, 256, 768]
                            else:
                                encoded_queries = masked_queries_features[start_idx:end_idx]  # [B, 256, 768]
                            encoded_queries = torch.tensor(encoded_queries, dtype=torch.float32).to('cuda')                            

                            # b. Top-K database RGB encoding (batch)
                            top_k_db_indices_batch = predictions[start_idx:end_idx, :RERANKING_TOP_K]  # [B, K]

                            # c. Flatten indices to fetch all at once
                            all_db_indices = top_k_db_indices_batch.flatten() # [B*K]
                            if args.rerank_with_RGB:
                                encoded_dbs_flat = masked_database_features[all_db_indices]  # [B*K, 256, 768]
                            else:
                                encoded_dbs_flat = database_patch_features[all_db_indices]  # [B*K, 256, 768]
                            encoded_dbs_flat = torch.tensor(encoded_dbs_flat, dtype=torch.float32).to('cuda')
                            encoded_dbs = encoded_dbs_flat.reshape(batch_size, RERANKING_TOP_K, 256, -1)  # [B, K, 256, 768]
                            
                            if not args.rerank_with_RGB:
                                encoded_dbs += model.module.decoder_pos_embed
                            
                            # d. Query batch 생성 (각 query를 K번 반복)
                            encoded_query_batch = encoded_queries.unsqueeze(1).expand(-1, RERANKING_TOP_K, -1, -1)  # [B, K, 256, 768]
                            
                            # e. Reshape for decoder: [B*K, 256, 768]
                            encoded_query_flat = encoded_query_batch.reshape(-1, 256, encoded_queries.size(-1))
                            encoded_dbs_flat = encoded_dbs.reshape(-1, 256, encoded_dbs.size(-1))
                            
                            # f. Decoder 통과 (thermal-rgb pair)
                            if args.rerank_with_RGB:
                                decoded_result = encoded_dbs_flat.clone()
                                for blk in model.module.decoder_rgb_blocks:
                                    decoded_result = blk(decoded_result, encoded_query_flat, return_attention=True)
                                decoded_result = model.module.decoder_norm(decoded_result)
                                rgb_cross_attn_map = model.module.decoder_rgb_blocks[-1].cross_attn_weights
                            else:
                                decoded_result = encoded_query_flat.clone()
                                for blk in model.module.decoder_thermal_blocks:
                                    decoded_result = blk(decoded_result, encoded_dbs_flat, return_attention=True)
                                decoded_result = model.module.decoder_norm(decoded_result)
                                thermal_cross_attn_map = model.module.decoder_thermal_blocks[-1].cross_attn_weights

                            # rgb_dec = encoded_dbs_flat.clone()
                            # for blk in model.module.decoder_rgb_blocks:
                            #     rgb_dec = blk(rgb_dec, encoded_query_flat, return_attention=True)
                            # rgb_dec = model.module.decoder_norm(rgb_dec)
                            # rgb_cross_attn_map = model.module.decoder_rgb_blocks[-1].cross_attn_weights

                            # e. Reconstruction
                            reconstructed_patches_thermal = model.module.prediction_head(decoded_result)  # [B*K, 256, 588]
                            # reconstructed_patches_rgb = model.module.prediction_head(rgb_dec)  # [B*K, 256, 588]

                            # h. Target patches (batch)
                            query_abs_indices = list(range(eval_ds.database_num + start_idx, eval_ds.database_num + end_idx))
                            query_imgs = torch.stack([eval_ds[idx][0] for idx in query_abs_indices]).to('cuda') # [B, 3, 224, 224]
                            query_imgs_batch = query_imgs.unsqueeze(1).expand(-1, RERANKING_TOP_K, -1, -1, -1) # [B, K, 3, 224, 224]
                            query_imgs_flat = query_imgs_batch.reshape(-1, *query_imgs.shape[1:])  # [B*K, 3, 224, 224]
                            target_patches = patchify(query_imgs_flat)

                            # i. Masks (batch)
                            masks_batch = queries_features_mask[start_idx:end_idx]  # [B, 256]
                            masks_batch = torch.tensor(masks_batch, dtype=torch.bool).to('cuda')
                            masks_flat = masks_batch.unsqueeze(1).expand(-1, RERANKING_TOP_K, -1)  # [B, K, 256]
                            masks_flat = masks_flat.reshape(-1, 256)  # [B*K, 256]
                        
                            loss = reconstruction_criterion(
                                pred=reconstructed_patches_thermal,
                                mask=masks_flat,
                                target=target_patches
                            )  # [B*K]
                            loss = loss.reshape(batch_size, RERANKING_TOP_K)  # [B, K]
                            
                            # 4. Reranking
                            for i, query_idx in enumerate(range(start_idx, end_idx)):
                                reconstruction_losses = loss[i].cpu().numpy()
                                reranked_order = np.argsort(reconstruction_losses)
                                
                                reconstruction_losses_dict[query_idx] = reconstruction_losses.tolist()
                                
                                top_k_db_indices = top_k_db_indices_batch[i]
                                reranked_predictions[query_idx, :RERANKING_TOP_K] = top_k_db_indices[reranked_order]
                                
                                if predictions[query_idx, 0] != reranked_predictions[query_idx, 0]:
                                    top1_change_count += 1
                                total_count += 1
                                
                        if args.use_fast_track:
                            break
            except Exception as e:
                import traceback
                print(f"ERROR caught: {e}")
                traceback.print_exc()  # 전체 stack trace 출력
                breakpoint()
                
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

        if args.use_reranking:
            prev_recalls = np.zeros(len(args.recall_values))
            for query_index, pred in enumerate(prev_predictions):
                for i, n in enumerate(args.recall_values):
                    if np.any(np.in1d(pred[:n], positives_per_query[query_index])):
                        prev_recalls[i:] += 1
                        break
            prev_recalls = prev_recalls / eval_ds.queries_num * 100
            
            logging.info(f"=================================================")
            logging.info(f"recalls before RERANKING: {','.join(map(str, prev_recalls))}")
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