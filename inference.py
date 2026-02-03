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
from local_matching import *

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
NPY_ROOTPATH = None
def save_npy(data, path):
    np.save(os.path.join(NPY_ROOTPATH, path), data)
    
def load_npy(path):
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
            
            use_selaVPR = args.use_reranking in ['selaVPR', 'match_conf']
            use_penultimate = args.r2_penultimate_layer
            print(f"Using SelaVPR features: {use_selaVPR}")
            print(f"Using penultimate: {use_penultimate}")
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
                    for num, idx in enumerate(indices_npy):
                        if use_penultimate: save_npy(outputs[5][num].cpu().numpy(), f"Db_{seq_name}_penultimate_{idx}")
                        if use_selaVPR: save_npy(outputs[6][num].cpu().numpy(), f"Db_{seq_name}_sela_{idx}")
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
                    
                    for num, idx in enumerate(indices_npy):
                        if use_penultimate: save_npy(outputs[5][num].cpu().numpy(), f"Query_{seq_name}_penultimate_{idx}")
                        if use_selaVPR: save_npy(outputs[6][num].cpu().numpy(), f"Query_{seq_name}_sela_{idx}")
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
                # FIXME: 옛날 코드에서 긁어서 추가하기
                ...
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
                candidates_local_features = torch.zeros(RERANKING_TOP_K, 61, 61, args.features_dim, device='cuda')
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
                            candidates_features  # RGB [K, 256, 768]
                        )
                        # thermal_decoded_local: [K, 61, 61, 768] - thermal refined with RGB context
                        # rgb_decoded_local: [K, 61, 61, 768] - RGB refined with thermal context

                        # MNN matching between decoded features
                        query_local = thermal_decoded_local[0]  # [61, 61, 768]
                        db_local = rgb_decoded_local            # [K, 61, 61, 768]

                        rerank_scores = local_sim(
                            query_local, db_local,
                            trainflag=False,
                            method_type=args.selaVPR_rerank_score_type
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
        