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
        positives_per_query = eval_ds.get_positives()
        test_method = args.test_method
        model = model.eval()
        with torch.no_grad():
            # 1. database feature들 추출
            start_time = time.time()

            database_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num)))
            database_dataloader = DataLoader(dataset=database_subset_ds, num_workers=args.num_workers,
                                            batch_size=args.infer_batch_size, pin_memory=(args.device=="cuda"))
        
            database_features = np.empty((eval_ds.database_num, args.features_dim), dtype="float32")
            
            # NOTE:GOOD CODE
            for inputs, indices, flags in tqdm(database_dataloader, ncols=100):
                features = model(inputs.to(args.device), flags)[0].view(-1, args.features_dim)
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
                features = model(inputs.to(args.device), flags)[0].view(-1, args.features_dim)
                features = features.cpu().numpy()
                queries_features[indices.numpy()-eval_ds.database_num, :] = features

            logging.info(f"Finished extracting {eval_ds.queries_num} query features in {time.time() - start_time:.2f} s")
        
        # queries_features = all_features[eval_ds.database_num:]
        # database_features = all_features[:eval_ds.database_num]

        # 3. faiss를 이용하여, L2 distance로 가까운 descriptor 찾기
        if torch.cuda.is_available():
            res = faiss.StandardGpuResources()
            cpu_index = faiss.IndexFlatL2(args.features_dim)
            faiss_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        else:
            faiss_index = faiss.IndexFlatL2(args.features_dim)
            
        faiss_index.add(database_features)
        del database_features
        
        start_time = time.time()
        distances, predictions = faiss_index.search(queries_features, max(args.recall_values))
        del queries_features # predictions: [Query, top20]
        
        ####################################
        ### RERANKING Process
        ####################################
        start_time = time.time()
        if args.use_reranking:
            with torch.no_grad():
                # NOTE: decoder에 들어가기 완전 직전 상태를 저장해놔야함
                # rerank1. RGB database 전부 추출 (before decoder)
                start_time = time.time()

                database_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num)))
                database_dataloader = DataLoader(dataset=database_subset_ds, num_workers=args.num_workers,
                                                batch_size=args.infer_batch_size, pin_memory=(args.device=="cuda"))
            
                masked_database_features = np.empty((eval_ds.database_num, 256, args.features_dim), dtype="float32")
                model.module.use_masked_inference = True
                for inputs, indices, flags in tqdm(database_dataloader, ncols=100):
                    features = model(inputs.to(args.device), flags)
                    encoded_features = features[1].reshape(inputs.size(0), 256, -1).cpu().numpy()
                    masked_database_features[indices.numpy(), :, :] = encoded_features
                model.module.use_masked_inference = False

                logging.info(f"Finished extracting (FOR RERANK) {eval_ds.database_num} database features in {time.time() - start_time:.2f} s")

                # rerank2. masked된 query 전부 추출 (before decoder)
                start_time = time.time()
                queries_infer_batch_size = args.infer_batch_size
                queries_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num, len(eval_ds))))
                queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args.num_workers,
                                                batch_size=queries_infer_batch_size, pin_memory=(args.device=="cuda"))

                queries_features_mask = np.empty((eval_ds.queries_num, 256), dtype="bool")
                masked_queries_features = np.empty((eval_ds.queries_num, 256, args.features_dim), dtype="float32")
                model.module.use_masked_inference = True
                for inputs, indices, flags in tqdm(queries_dataloader, ncols=100):
                    features = model(inputs.to(args.device), flags, return_mask=True)
                    encoded_features = features[1].reshape(inputs.size(0), 256, -1).cpu().numpy()
                    masked_queries_features[indices.numpy()-eval_ds.database_num,:,:] = encoded_features
                    queries_features_mask[indices.numpy()-eval_ds.database_num,:] = features[3].detach().cpu().numpy()
                model.module.use_masked_inference = False

                logging.info(f"Finished extracting (FOR RERANK) {eval_ds.queries_num} query features in {time.time() - start_time:.2f} s")

            # rerank3. Decoder 쭉쭉 태워서 rerank 진행하기
            # masked_database_features.shape: [1197, 256, 768]
            RERANKING_TOP_K = 5
            reranked_predictions = predictions.copy()  # 원본 보존
            reconstruction_losses_dict = {}
            reconstruction_criterion = MaskedMSE(
                norm_pix_loss=False,
                masked=True,
                reduction='none'
            )
            
            top1_change_count = 0
            total_count = 0
            with torch.no_grad():
                for query_index in tqdm(range(eval_ds.queries_num), desc="Reranking", ncols=100):
                    # a. Query thermal encoding된 거 가져오기
                    encoded_query = masked_queries_features[query_index]
                    encoded_query = torch.tensor(encoded_query, dtype=torch.float32).to('cuda')
                    
                    # b. Top-K database RGB encoding된 거 가져오기
                    top_k_db_indices = predictions[query_index, :RERANKING_TOP_K]
                    
                    encoded_dbs = []
                    for db_idx in top_k_db_indices:
                        encoded_dbs.append(torch.tensor(masked_database_features[db_idx], dtype=torch.float32).to('cuda'))
                    encoded_dbs = torch.stack(encoded_dbs)

                    # c. query를 batch로 만들어주기
                    encoded_query_batch = encoded_query.unsqueeze(0).expand(
                        RERANKING_TOP_K, -1, -1
                    )  # [RERANKING_TOP_K, N_visible, 768]
        
                    # d. Decoder 통과
                    thermal_dec = encoded_query_batch
                    for blk in model.module.decoder_blocks:
                        thermal_dec = blk(thermal_dec, encoded_dbs)
                    thermal_full_dec = model.module.decoder_norm(thermal_dec)
                    
                    # e. Reconstruction
                    reconstructed_patches = model.module.prediction_head(thermal_full_dec)  # [1, 256, 588]
                    query_abs_idx = eval_ds.database_num + query_index
                    query_img = eval_ds[query_abs_idx][0]
                    query_img_batch = query_img.unsqueeze(0).expand(RERANKING_TOP_K, -1, -1, -1).to('cuda')
                    target_patches = patchify(query_img_batch)
                    
                    # f. Reconstruction loss 계산
                    if reconstructed_patches.shape != target_patches.shape:
                        print(f"recons_shape: {reconstructed_patches.shape}")
                        print(f"target_shape: {target_patches.shape}")
                        breakpoint()

                    mask = torch.tensor(queries_features_mask[query_index], dtype=torch.bool).to('cuda')
                    mask_batch = mask.unsqueeze(0).expand(RERANKING_TOP_K, -1)
                    loss = reconstruction_criterion(pred=reconstructed_patches, mask=mask_batch, target=target_patches)
                    
                    # g. Reconstruction loss 기반 reranking
                    reconstruction_losses = np.array(loss.detach().cpu())
                    reranked_order = np.argsort(reconstruction_losses)
                    
                    #### 시각화 ####
                    reconstruction_losses_dict[query_index] = reconstruction_losses.tolist()
                    ##############
                    
                    # h. 기존 predictions 업데이트
                    reranked_predictions[query_index, :RERANKING_TOP_K] = top_k_db_indices[reranked_order]
                    if predictions[query_index, 0] != reranked_predictions[query_index, 0]:
                        top1_change_count += 1
                    total_count += 1

                logging.info(f"Reranking completed in {time.time() - start_time:.2f} s")
                print(f"RERANK: changed {top1_change_count} / {total_count}")
                
                # ===== 시각화 추가 ===== #
                if hasattr(args, 'current_epoch'):
                    visualize_reranking_comparison(
                        args, eval_ds,
                        original_predictions=predictions,
                        reranked_predictions=reranked_predictions,
                        reconstruction_losses_dict=reconstruction_losses_dict,
                        positives_per_query=positives_per_query,
                        epoch=args.current_epoch,
                        save_dir=os.path.join(args.save_dir, 'rerank_vis'),
                        num_samples=4
                    )
        
                # Reranking 결과로 recall 계산
                predictions = reranked_predictions
                ####################################
        
        # 4. positive query(정답)가 몇번째 top-N에 속하는지 검사하기
        # positives_per_query = eval_ds.get_positives()
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