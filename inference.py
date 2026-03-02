# inference.py
# Inference script for cross-modal VPR
import faiss
import torch
import logging
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
import time
import gc
import os
import cv2
from PIL import Image
import torchvision.transforms as T

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def get_timestamp():
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_image_transform(resize):
    """Get image transform for loading images"""
    return T.Compose([
        T.Resize(resize),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def inference(args, eval_ds, model):
    orig_W = args.resize[0]
    orig_H = args.resize[1]
    patch_W = int(orig_W / 14)
    patch_H = int(orig_H / 14)
    patch_count = patch_W * patch_H

    try:
        model = model.eval()
        with torch.no_grad():
            # 1. Extract database features
            start_time = time.time()
            database_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num)))
            database_dataloader = DataLoader(
                dataset=database_subset_ds,
                num_workers=args.num_workers,
                batch_size=args.infer_batch_size,
                pin_memory=(args.device == "cuda")
            )

            database_features = np.empty((eval_ds.database_num, args.affinity_dim), dtype="float32")

            for inputs, indices, flags in tqdm(database_dataloader, ncols=100, desc="DB features"):
                flags_int = [1 if f == 'rgb' else 0 for f in flags]
                flags = torch.tensor(flags_int, dtype=torch.long, device=args.device)
                outputs = model(inputs.to(args.device), flags)
                features = outputs[0].view(-1, args.affinity_dim)

                indices_npy = indices.numpy()
                database_features[indices_npy, :] = features.cpu().numpy()

            logging.info(f"Extracted {eval_ds.database_num} database features in {time.time() - start_time:.2f}s")

            # 2. Extract query features
            start_time = time.time()
            queries_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num, len(eval_ds))))
            queries_dataloader = DataLoader(
                dataset=queries_subset_ds,
                num_workers=args.num_workers,
                batch_size=args.infer_batch_size,
                pin_memory=(args.device == "cuda")
            )

            queries_features = np.empty((eval_ds.queries_num, args.affinity_dim), dtype="float32")

            for inputs, indices, flags in tqdm(queries_dataloader, ncols=100, desc="Query features"):
                flags_int = [1 if f == 'rgb' else 0 for f in flags]
                flags = torch.tensor(flags_int, dtype=torch.long, device=args.device)
                outputs = model(inputs.to(args.device), flags)
                features = outputs[0].view(-1, args.affinity_dim)

                indices_npy = indices.numpy() - eval_ds.database_num
                queries_features[indices_npy, :] = features.cpu().numpy()

            logging.info(f"Extracted {eval_ds.queries_num} query features in {time.time() - start_time:.2f}s")

        # 3. Find nearest neighbors using FAISS
        faiss_index = faiss.IndexFlatL2(args.affinity_dim)
        faiss_index.add(database_features)

        start_time = time.time()
        _, predictions = faiss_index.search(queries_features, max(args.recall_values))
        logging.info(f"FAISS search completed in {time.time() - start_time:.2f}s")

        del faiss_index
        del queries_features
        del database_features

        gc.collect()
        torch.cuda.empty_cache()

        # 4. Compute recalls
        positives_per_query = eval_ds.get_positives()
        recalls = np.zeros(len(args.recall_values))

        for query_index, pred in enumerate(predictions):
            for i, n in enumerate(args.recall_values):
                if np.any(np.in1d(pred[:n], positives_per_query[query_index])):
                    recalls[i:] += 1
                    break

        recalls = recalls / eval_ds.queries_num * 100

        logging.info(f"Recalls: {','.join(map(str, recalls))}")
        recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(args.recall_values, recalls)])

        gc.collect()
        torch.cuda.empty_cache()

        return recalls, recalls_str

    except Exception as e:
        import traceback
        print(f"ERROR: {e}")
        traceback.print_exc()
        raise


def inference_with_reranking(args, eval_ds, model):
    """
    Inference with GMRW re-ranking.

    Pipeline:
    1. Extract global descriptors (FAISS)
    2. Get top-K candidates per query
    3. Re-rank using GMRW cycle score
    4. Compute recalls on re-ranked results
    """
    orig_W = args.resize[0]
    orig_H = args.resize[1]
    rerank_k = args.rerank_top_k
    score_method = args.score_method

    transform = get_image_transform(args.resize)

    try:
        model = model.eval()
        with torch.no_grad():
            # 1. Extract database features
            start_time = time.time()
            database_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num)))
            database_dataloader = DataLoader(
                dataset=database_subset_ds,
                num_workers=args.num_workers,
                batch_size=args.infer_batch_size,
                pin_memory=(args.device == "cuda")
            )

            database_features = np.empty((eval_ds.database_num, args.affinity_dim), dtype="float32")

            for inputs, indices, flags in tqdm(database_dataloader, ncols=100, desc="DB features"):
                flags_int = [1 if f == 'rgb' else 0 for f in flags]
                flags = torch.tensor(flags_int, dtype=torch.long, device=args.device)
                outputs = model(inputs.to(args.device), flags)
                features = outputs[0].view(-1, args.affinity_dim)

                indices_npy = indices.numpy()
                database_features[indices_npy, :] = features.cpu().numpy()

            logging.info(f"Extracted {eval_ds.database_num} database features in {time.time() - start_time:.2f}s")

            # 2. Extract query features
            start_time = time.time()
            queries_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num, len(eval_ds))))
            queries_dataloader = DataLoader(
                dataset=queries_subset_ds,
                num_workers=args.num_workers,
                batch_size=args.infer_batch_size,
                pin_memory=(args.device == "cuda")
            )

            queries_features = np.empty((eval_ds.queries_num, args.affinity_dim), dtype="float32")

            for inputs, indices, flags in tqdm(queries_dataloader, ncols=100, desc="Query features"):
                flags_int = [1 if f == 'rgb' else 0 for f in flags]
                flags = torch.tensor(flags_int, dtype=torch.long, device=args.device)
                outputs = model(inputs.to(args.device), flags)
                features = outputs[0].view(-1, args.affinity_dim)

                indices_npy = indices.numpy() - eval_ds.database_num
                queries_features[indices_npy, :] = features.cpu().numpy()

            logging.info(f"Extracted {eval_ds.queries_num} query features in {time.time() - start_time:.2f}s")

        # 3. Find initial top-K candidates using FAISS
        faiss_index = faiss.IndexFlatL2(args.affinity_dim)
        faiss_index.add(database_features)

        start_time = time.time()
        # Get top rerank_k candidates for re-ranking
        search_k = max(rerank_k, max(args.recall_values))
        _, initial_predictions = faiss_index.search(queries_features, search_k)
        logging.info(f"FAISS search (top-{search_k}) completed in {time.time() - start_time:.2f}s")

        del faiss_index
        del queries_features
        del database_features
        gc.collect()
        torch.cuda.empty_cache()

        # 4. Re-ranking with GMRW
        start_time = time.time()
        final_predictions = []

        # Get actual model (handle DataParallel)
        actual_model = model.module if hasattr(model, 'module') else model

        logging.info(f"Starting GMRW re-ranking with top-{rerank_k} candidates...")

        for q_idx in tqdm(range(eval_ds.queries_num), ncols=100, desc="GMRW Re-ranking"):
            # Get query image path and load
            # Query: thermal, Database: RGB (cross-modal)
            query_path = eval_ds.t_queries_paths[q_idx]
            query_img = Image.open(query_path).convert('RGB')
            query_tensor = transform(query_img).unsqueeze(0).to(args.device)  # (1, 3, H, W)

            # Get top-K candidate indices
            candidates = initial_predictions[q_idx][:rerank_k]

            # Load candidate RGB images
            cand_tensors = []
            for cand_idx in candidates:
                cand_path = eval_ds.rgb_database_paths[cand_idx]
                cand_img = Image.open(cand_path).convert('RGB')
                cand_tensors.append(transform(cand_img))
            cand_batch = torch.stack(cand_tensors).to(args.device)  # (K, 3, H, W)

            # Expand query to match batch size
            query_batch = query_tensor.expand(len(candidates), -1, -1, -1)  # (K, 3, H, W)

            # Compute GMRW scores
            with torch.no_grad():
                scores, _ = actual_model.stage2_inference(
                    query_batch, cand_batch, score_method=score_method
                )

            # Re-rank by score (descending - higher score = better match)
            sorted_indices = scores.argsort(descending=True).cpu().numpy()
            reranked_candidates = candidates[sorted_indices]

            # Append remaining candidates (beyond rerank_k) if needed
            if search_k > rerank_k:
                remaining = initial_predictions[q_idx][rerank_k:]
                reranked_candidates = np.concatenate([reranked_candidates, remaining])

            final_predictions.append(reranked_candidates)

        final_predictions = np.array(final_predictions)
        logging.info(f"GMRW re-ranking completed in {time.time() - start_time:.2f}s")

        gc.collect()
        torch.cuda.empty_cache()

        # 5. Compute recalls
        positives_per_query = eval_ds.get_positives()
        recalls = np.zeros(len(args.recall_values))
        recalls_before = np.zeros(len(args.recall_values))

        # Compute recalls for both before and after re-ranking
        for query_index in range(eval_ds.queries_num):
            pred_reranked = final_predictions[query_index]
            pred_original = initial_predictions[query_index]

            # After re-ranking
            for i, n in enumerate(args.recall_values):
                if np.any(np.in1d(pred_reranked[:n], positives_per_query[query_index])):
                    recalls[i:] += 1
                    break

            # Before re-ranking (for comparison)
            for i, n in enumerate(args.recall_values):
                if np.any(np.in1d(pred_original[:n], positives_per_query[query_index])):
                    recalls_before[i:] += 1
                    break

        recalls = recalls / eval_ds.queries_num * 100
        recalls_before = recalls_before / eval_ds.queries_num * 100

        logging.info(f"Recalls (before re-ranking): {','.join([f'{r:.1f}' for r in recalls_before])}")
        logging.info(f"Recalls (after re-ranking): {','.join([f'{r:.1f}' for r in recalls])}")

        recalls_str_before = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(args.recall_values, recalls_before)])
        recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(args.recall_values, recalls)])

        logging.info(f"Before: {recalls_str_before}")
        logging.info(f"After:  {recalls_str}")

        # Compute improvement
        improvement = recalls - recalls_before
        improvement_str = ", ".join([f"R@{val}: {imp:+.1f}" for val, imp in zip(args.recall_values, improvement)])
        logging.info(f"Improvement: {improvement_str}")

        gc.collect()
        torch.cuda.empty_cache()

        return recalls, recalls_str, recalls_before, recalls_str_before

    except Exception as e:
        import traceback
        print(f"ERROR: {e}")
        traceback.print_exc()
        raise
