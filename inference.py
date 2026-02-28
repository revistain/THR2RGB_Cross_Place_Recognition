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

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def get_timestamp():
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d_%H%M%S")


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

            # Descriptor dimension: affinity_dim (mixer output)
            descriptor_dim = args.affinity_dim
            database_features = np.empty((eval_ds.database_num, descriptor_dim), dtype="float32")

            for inputs, indices, flags in tqdm(database_dataloader, ncols=100, desc="DB features"):
                flags_int = [1 if f == 'rgb' else 0 for f in flags]
                flags = torch.tensor(flags_int, dtype=torch.long, device=args.device)
                outputs = model(inputs.to(args.device), flags)
                features = outputs[0].view(-1, descriptor_dim)

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

            queries_features = np.empty((eval_ds.queries_num, descriptor_dim), dtype="float32")

            for inputs, indices, flags in tqdm(queries_dataloader, ncols=100, desc="Query features"):
                flags_int = [1 if f == 'rgb' else 0 for f in flags]
                flags = torch.tensor(flags_int, dtype=torch.long, device=args.device)
                outputs = model(inputs.to(args.device), flags)
                features = outputs[0].view(-1, descriptor_dim)

                indices_npy = indices.numpy() - eval_ds.database_num
                queries_features[indices_npy, :] = features.cpu().numpy()

            logging.info(f"Extracted {eval_ds.queries_num} query features in {time.time() - start_time:.2f}s")

        # 3. Find nearest neighbors using FAISS
        faiss_index = faiss.IndexFlatL2(descriptor_dim)
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
