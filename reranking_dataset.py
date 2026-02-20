"""
Reranking Dataset and utilities for parallel batch processing.

이 모듈은 reranking 과정을 DataLoader 기반으로 병렬화하여 성능을 향상시킵니다.
- RerankingDataset: 쿼리별 feature를 병렬로 로드
- reranking_collate_fn: 여러 쿼리를 하나의 배치로 합침
- compute_batch_recon_loss: 벡터화된 reconstruction loss 계산
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset


class RerankingDataset(Dataset):
    """
    Reranking을 위한 Dataset 클래스.

    각 샘플은 1개 쿼리 + 해당 쿼리의 top-K 후보들의 feature를 포함합니다.
    DataLoader의 num_workers를 통해 파일 I/O를 병렬화합니다.

    Args:
        predictions: FAISS 검색 결과 [num_queries, max_recall]
        seq_name: 시퀀스 이름 (파일 경로용)
        npy_root_path: NPY 파일들이 저장된 디렉토리
        reranking_top_k: 각 쿼리당 고려할 후보 수 (기본값: 5)
        load_targets: target patch도 로드할지 여부 (reconstruction용)
    """

    def __init__(
        self,
        predictions: np.ndarray,
        seq_name: str,
        npy_root_path: str,
        reranking_top_k: int = 5,
        load_targets: bool = True
    ):
        self.predictions = predictions
        self.seq_name = seq_name
        self.npy_root_path = npy_root_path
        self.reranking_top_k = reranking_top_k
        self.load_targets = load_targets

    def __len__(self) -> int:
        return len(self.predictions)

    def _load_npy(self, filename: str) -> np.ndarray:
        """NPY 파일 로드 헬퍼"""
        return np.load(os.path.join(self.npy_root_path, filename + ".npy"))

    def __getitem__(self, query_index: int) -> dict:
        """
        단일 쿼리에 대한 데이터 로드.

        Returns:
            dict with:
                - query_index: 쿼리 인덱스
                - top_k_indices: top-K 후보 인덱스 [K]
                - query_feat: 쿼리 feature [N, D]
                - candidate_feats: 후보 features [K, N, D]
                - query_target: (optional) 쿼리 target [N, P]
                - candidate_targets: (optional) 후보 targets [K, N, P]
        """
        top_k_indices = self.predictions[query_index][:self.reranking_top_k]

        # 쿼리 feature 로드
        query_feat = self._load_npy(f"Query_{self.seq_name}_{query_index}")

        # 후보 features 로드
        candidate_feats = []
        for db_idx in top_k_indices:
            candidate_feats.append(self._load_npy(f"Db_{self.seq_name}_{db_idx}"))
        candidate_feats = np.stack(candidate_feats, axis=0)  # [K, N, D]

        result = {
            'query_index': query_index,
            'top_k_indices': top_k_indices.copy(),
            'query_feat': query_feat,
            'candidate_feats': candidate_feats,
        }

        # Target patches 로드 (reconstruction용)
        if self.load_targets:
            query_target = self._load_npy(f"Query_{self.seq_name}_target_{query_index}")

            candidate_targets = []
            for db_idx in top_k_indices:
                candidate_targets.append(self._load_npy(f"Db_{self.seq_name}_target_{db_idx}"))
            candidate_targets = np.stack(candidate_targets, axis=0)  # [K, N, P]

            result['query_target'] = query_target
            result['candidate_targets'] = candidate_targets

        return result


def reranking_collate_fn(batch: list) -> dict:
    """
    여러 쿼리를 하나의 큰 배치로 합치는 collate 함수.

    GPU에서 효율적인 처리를 위해 [B, K, ...] 형태를 [B*K, ...]로 flatten합니다.

    Args:
        batch: list of dicts from RerankingDataset.__getitem__

    Returns:
        dict with batched tensors:
            - query_indices: list of query indices
            - top_k_indices: [B, K] numpy array
            - thermal_feat: [B*K, N, D] tensor (쿼리를 K번 복제)
            - rgb_feat: [B*K, N, D] tensor (후보들)
            - thermal_target: [B*K, N, P] tensor (optional)
            - rgb_target: [B*K, N, P] tensor (optional)
            - batch_size: B
            - top_k: K
    """
    batch_size = len(batch)
    K = batch[0]['candidate_feats'].shape[0]  # RERANKING_TOP_K

    # 쿼리 정보
    query_indices = [b['query_index'] for b in batch]
    top_k_indices = np.stack([b['top_k_indices'] for b in batch])  # [B, K]

    # 쿼리 features: [B, N, D] -> [B, K, N, D] (K번 복제)
    query_feats = np.stack([b['query_feat'] for b in batch])  # [B, N, D]
    query_feats = np.tile(query_feats[:, np.newaxis], (1, K, 1, 1))  # [B, K, N, D]

    # 후보 features: [B, K, N, D]
    candidate_feats = np.stack([b['candidate_feats'] for b in batch])  # [B, K, N, D]

    result = {
        'query_indices': query_indices,
        'top_k_indices': top_k_indices,
        'thermal_feat': torch.from_numpy(query_feats.reshape(-1, *query_feats.shape[2:])),  # [B*K, N, D]
        'rgb_feat': torch.from_numpy(candidate_feats.reshape(-1, *candidate_feats.shape[2:])),  # [B*K, N, D]
        'batch_size': batch_size,
        'top_k': K,
    }

    # Target patches (optional)
    if 'query_target' in batch[0]:
        query_targets = np.stack([b['query_target'] for b in batch])  # [B, N, P]
        query_targets = np.tile(query_targets[:, np.newaxis], (1, K, 1, 1))  # [B, K, N, P]

        candidate_targets = np.stack([b['candidate_targets'] for b in batch])  # [B, K, N, P]

        result['thermal_target'] = torch.from_numpy(query_targets.reshape(-1, *query_targets.shape[2:]))
        result['rgb_target'] = torch.from_numpy(candidate_targets.reshape(-1, *candidate_targets.shape[2:]))

    return result


def compute_batch_recon_loss(
    thermal_recon: torch.Tensor,    # [B, K, N, D]
    rgb_recon: torch.Tensor,        # [B, K, N, D]
    thermal_masks: torch.Tensor,    # [B, K, N]
    rgb_masks: torch.Tensor,        # [B, K, N]
    thermal_target: torch.Tensor,   # [B, K, N, D]
    rgb_target: torch.Tensor,       # [B, K, N, D]
    thermal_weight: torch.Tensor = None,  # [B, K, N]
    rgb_weight: torch.Tensor = None,      # [B, K, N]
    use_weight: bool = False
) -> torch.Tensor:
    """
    배치 전체에 대해 한 번에 reconstruction loss 계산.

    모든 쿼리-후보 쌍에 대해 벡터화된 연산으로 손실을 계산합니다.

    Args:
        thermal_recon: 복원된 thermal features [B, K, N, D]
        rgb_recon: 복원된 RGB features [B, K, N, D]
        thermal_masks: thermal 마스크 (True=masked) [B, K, N]
        rgb_masks: RGB 마스크 [B, K, N]
        thermal_target: thermal ground truth [B, K, N, D]
        rgb_target: RGB ground truth [B, K, N, D]
        thermal_weight: (optional) thermal 가중치 [B, K, N]
        rgb_weight: (optional) RGB 가중치 [B, K, N]
        use_weight: 가중치 사용 여부

    Returns:
        losses: [B, K] - 각 쿼리의 각 후보에 대한 평균 손실
    """
    # MSE loss per position: [B, K, N, D] -> [B, K, N]
    thermal_mse = ((thermal_recon - thermal_target) ** 2).mean(dim=-1)
    rgb_mse = ((rgb_recon - rgb_target) ** 2).mean(dim=-1)

    # 마스크된 위치만 계산 (True인 위치가 masked)
    thermal_mse = thermal_mse * thermal_masks.float()
    rgb_mse = rgb_mse * rgb_masks.float()

    # 가중치 적용 (선택적)
    if use_weight and thermal_weight is not None and rgb_weight is not None:
        thermal_mse = thermal_mse * thermal_weight
        rgb_mse = rgb_mse * rgb_weight

    # 마스크된 패치 수로 정규화
    thermal_mask_count = thermal_masks.sum(dim=-1).clamp(min=1)  # [B, K]
    rgb_mask_count = rgb_masks.sum(dim=-1).clamp(min=1)  # [B, K]

    thermal_loss = thermal_mse.sum(dim=-1) / thermal_mask_count  # [B, K]
    rgb_loss = rgb_mse.sum(dim=-1) / rgb_mask_count  # [B, K]

    return (thermal_loss + rgb_loss) / 2  # [B, K]
