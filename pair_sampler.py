"""
Pair Sampler for Reconstruction

유사도 기반 reconstruction pair 샘플링
- 거리 기반 후보 풀 + Feature 유사도 기반 최종 선택

Reconstruction Pairs:
1. Query Thermal ← Pos Thermal (intra, 10m + feature)
2. Pos Thermal ← Query Thermal (intra, vice versa)
3. Query-similar RGB ← RGB-similar RGB (intra, same traverse)
4. Query Thermal ← Query-similar RGB (inter, same traverse)
"""

import numpy as np
import faiss
from sklearn.neighbors import NearestNeighbors
from random import randint


class PairSampler:
    """
    유사도 기반 reconstruction pair 샘플링
    """

    def __init__(self, database_utms, queries_utms,
                 traverse_indices=None,
                 distance_threshold=10.0):
        """
        Args:
            database_utms: [N_db, 2] numpy array - database 위치 좌표
            queries_utms: [N_q, 2] numpy array - query 위치 좌표
            traverse_indices: dict {img_idx: traverse_id} - 각 이미지의 traverse ID
            distance_threshold: 거리 임계값 (미터)
        """
        self.distance_threshold = distance_threshold
        self.traverse_indices = traverse_indices
        self.database_utms = database_utms
        self.queries_utms = queries_utms

        # 거리 기반 후보 풀 (distance_threshold 이내)
        self.knn = NearestNeighbors(n_jobs=4)
        self.knn.fit(database_utms)
        self.positives_per_query = list(self.knn.radius_neighbors(
            queries_utms, radius=distance_threshold, return_distance=False
        ))

    def get_similar_by_feature(self, query_feat, candidate_feats, candidate_indices, k=1):
        """
        Feature 유사도로 가장 가까운 k개 반환

        Args:
            query_feat: [D] numpy array - query feature
            candidate_feats: [N, D] numpy array - candidate features
            candidate_indices: [N] numpy array - candidate의 원래 인덱스
            k: 반환할 개수

        Returns:
            [k] numpy array - 선택된 인덱스들
        """
        if len(candidate_feats) == 0:
            return None

        feat_dim = query_feat.shape[-1]
        faiss_index = faiss.IndexFlatL2(feat_dim)
        faiss_index.add(candidate_feats.astype(np.float32).reshape(-1, feat_dim))

        query_feat_2d = query_feat.reshape(1, -1).astype(np.float32)
        _, best_nums = faiss_index.search(query_feat_2d, min(k, len(candidate_feats)))

        return candidate_indices[best_nums.reshape(-1)]

    def get_similar_thermal(self, query_idx, query_feat, all_thermal_feats,
                            same_traverse_indices=None):
        """
        Query Thermal과 유사한 Thermal 반환 (Intra-modal)
        조건: 같은 시간대(traverse) + distance_threshold 이내 + feature 최유사

        Args:
            query_idx: query index
            query_feat: [D] numpy array - query thermal feature
            all_thermal_feats: [N, D] numpy array or RAMEfficient2DMatrix - 모든 thermal features
            same_traverse_indices: [M] numpy array - 같은 traverse의 인덱스들

        Returns:
            int or None - 유사한 thermal의 인덱스
        """
        # 거리 제약
        candidates = self.positives_per_query[query_idx]
        if len(candidates) == 0:
            return None

        # 같은 시간대(traverse) 제약 (Intra-modal)
        if same_traverse_indices is not None and len(same_traverse_indices) > 0:
            candidates = np.intersect1d(candidates, same_traverse_indices)

        if len(candidates) == 0:
            return None

        # Feature 추출 시 None 체크 (sparse cache 대응)
        valid_candidates = []
        valid_feats = []
        for idx in candidates:
            feat = all_thermal_feats[idx]
            if feat is not None:
                valid_candidates.append(idx)
                valid_feats.append(feat)

        if len(valid_feats) == 0:
            return None

        candidate_feats = np.array(valid_feats)
        valid_candidates = np.array(valid_candidates)
        result = self.get_similar_by_feature(query_feat, candidate_feats, valid_candidates, k=1)

        return result[0] if result is not None else None

    def get_similar_rgb_to_thermal(self, query_idx, query_thermal_feat, all_rgb_feats):
        """
        Query Thermal과 유사한 RGB 반환 (Inter-modal)
        조건: distance_threshold 이내 + feature 최유사 (시간대 제약 X)

        Args:
            query_idx: query index (거리 제약용)
            query_thermal_feat: [D] numpy array - query thermal feature
            all_rgb_feats: [N, D] numpy array - 모든 RGB features

        Returns:
            int or None - 유사한 RGB의 인덱스
        """
        # 거리 제약만 적용 (Inter-modal이므로 traverse 제약 X)
        candidates = self.positives_per_query[query_idx]
        if len(candidates) == 0:
            return None

        candidate_feats = all_rgb_feats[candidates]
        result = self.get_similar_by_feature(query_thermal_feat, candidate_feats, candidates, k=1)

        return result[0] if result is not None else None

    def get_similar_rgb_to_rgb(self, rgb_idx, rgb_feat, all_rgb_feats,
                               database_utms, same_traverse_indices=None):
        """
        주어진 RGB와 유사한 RGB 반환
        조건: distance_threshold 이내 + 같은 traverse + feature 최유사 (자기 자신 제외)

        Args:
            rgb_idx: 현재 RGB 인덱스 (제외됨)
            rgb_feat: [D] numpy array - RGB feature
            all_rgb_feats: [N, D] numpy array - 모든 RGB features
            database_utms: [N, 2] numpy array - database 좌표
            same_traverse_indices: [M] numpy array - 같은 traverse의 인덱스들

        Returns:
            int or None - 유사한 RGB의 인덱스
        """
        # 거리 제약: rgb_idx 위치에서 distance_threshold 이내
        rgb_utm = database_utms[rgb_idx]
        distances = np.linalg.norm(database_utms - rgb_utm, axis=1)
        distance_mask = (distances < self.distance_threshold) & (distances > 0)  # 자기 자신 제외
        candidates = np.where(distance_mask)[0]

        if len(candidates) == 0:
            return None

        # traverse 제약 적용 (있는 경우)
        if same_traverse_indices is not None and len(same_traverse_indices) > 0:
            candidates = np.intersect1d(candidates, same_traverse_indices)

        # 자기 자신 제외
        candidates = candidates[candidates != rgb_idx]

        if len(candidates) == 0:
            return None

        candidate_feats = all_rgb_feats[candidates]
        result = self.get_similar_by_feature(rgb_feat, candidate_feats, candidates, k=1)

        return result[0] if result is not None else None

    def sample_all_pairs(self, query_idx, query_thermal_feat,
                         all_thermal_feats, all_rgb_feats,
                         query_traverse_id=None):
        """
        모든 reconstruction pairs 한번에 샘플링

        Args:
            query_idx: query index
            query_thermal_feat: [D] numpy array - query thermal feature
            all_thermal_feats: [N_thermal, D] numpy array - 모든 thermal features
            all_rgb_feats: [N_rgb, D] numpy array - 모든 RGB features
            query_traverse_id: query의 traverse ID (optional)

        Returns:
            dict with keys:
            - 'pos_thermal_idx': 유사한 Thermal (for intra Thermal←Thermal)
            - 'query_similar_rgb_idx': Query와 유사한 RGB (for inter)
            - 'rgb_similar_rgb_idx': 그 RGB와 유사한 RGB (for intra RGB←RGB)
        """
        result = {}

        same_traverse = self._get_same_traverse_indices(query_traverse_id)

        # 1. Query Thermal과 유사한 Thermal (Intra-modal: 같은 시간대 + 거리 + feature)
        pos_thermal_idx = self.get_similar_thermal(
            query_idx, query_thermal_feat, all_thermal_feats, same_traverse
        )
        result['pos_thermal_idx'] = pos_thermal_idx

        # 2. Query Thermal과 유사한 RGB (Inter-modal: 거리 + feature만, 시간대 제약 X)
        query_similar_rgb_idx = self.get_similar_rgb_to_thermal(
            query_idx, query_thermal_feat, all_rgb_feats
        )
        result['query_similar_rgb_idx'] = query_similar_rgb_idx

        # 3. 그 RGB와 유사한 RGB (Intra-modal: 같은 시간대 + 거리 + feature)
        if query_similar_rgb_idx is not None:
            rgb_feat = all_rgb_feats[query_similar_rgb_idx]
            rgb_similar_rgb_idx = self.get_similar_rgb_to_rgb(
                query_similar_rgb_idx, rgb_feat, all_rgb_feats,
                self.database_utms, same_traverse
            )
            result['rgb_similar_rgb_idx'] = rgb_similar_rgb_idx
        else:
            result['rgb_similar_rgb_idx'] = None

        return result

    def _get_same_traverse_indices(self, traverse_id):
        """
        같은 traverse의 인덱스들 반환

        Args:
            traverse_id: 찾을 traverse ID

        Returns:
            numpy array of indices with same traverse_id
        """
        if self.traverse_indices is None or traverse_id is None:
            return None

        indices = np.array([
            idx for idx, t_id in self.traverse_indices.items()
            if t_id == traverse_id
        ])

        return indices if len(indices) > 0 else None


class BatchPairSampler:
    """
    배치 단위 pair 샘플링 (network.py forward에서 사용)
    Features를 이미 계산한 후 호출
    """

    def __init__(self, distance_threshold=10.0):
        self.distance_threshold = distance_threshold

    def sample_pairs_from_features(self,
                                   thermal_feats,      # [B, D]
                                   rgb_feats,          # [B, D]
                                   thermal_utms=None,  # [B, 2] optional
                                   rgb_utms=None):     # [B, 2] optional
        """
        배치 내에서 feature 기반 pair 샘플링

        Args:
            thermal_feats: [B, D] - batch thermal features
            rgb_feats: [B, D] - batch RGB features
            thermal_utms: [B, 2] - batch thermal UTM coordinates (optional)
            rgb_utms: [B, 2] - batch RGB UTM coordinates (optional)

        Returns:
            dict with indices for each reconstruction pair
        """
        B = thermal_feats.shape[0]
        results = []

        for i in range(B):
            query_thermal_feat = thermal_feats[i]

            # 거리 제약이 있는 경우
            if thermal_utms is not None:
                distances = np.linalg.norm(thermal_utms - thermal_utms[i:i+1], axis=1)
                valid_thermal_mask = (distances < self.distance_threshold) & (distances > 0)
                valid_thermal_indices = np.where(valid_thermal_mask)[0]
            else:
                # 자기 자신 제외 모든 thermal
                valid_thermal_indices = np.concatenate([
                    np.arange(0, i), np.arange(i+1, B)
                ])

            # 1. 유사한 Thermal 찾기
            if len(valid_thermal_indices) > 0:
                candidate_feats = thermal_feats[valid_thermal_indices]
                similarities = self._compute_similarity(query_thermal_feat, candidate_feats)
                best_idx = valid_thermal_indices[np.argmax(similarities)]
                pos_thermal_idx = best_idx
            else:
                pos_thermal_idx = None

            # 2. 유사한 RGB 찾기 (배치 내 모든 RGB)
            rgb_similarities = self._compute_similarity(query_thermal_feat, rgb_feats)
            query_similar_rgb_idx = np.argmax(rgb_similarities)

            # 3. 그 RGB와 유사한 다른 RGB 찾기
            if query_similar_rgb_idx is not None:
                rgb_feat = rgb_feats[query_similar_rgb_idx]
                # 자기 자신 제외
                other_indices = np.concatenate([
                    np.arange(0, query_similar_rgb_idx),
                    np.arange(query_similar_rgb_idx + 1, B)
                ])
                if len(other_indices) > 0:
                    other_rgb_feats = rgb_feats[other_indices]
                    rgb_similarities = self._compute_similarity(rgb_feat, other_rgb_feats)
                    rgb_similar_rgb_idx = other_indices[np.argmax(rgb_similarities)]
                else:
                    rgb_similar_rgb_idx = None
            else:
                rgb_similar_rgb_idx = None

            results.append({
                'query_idx': i,
                'pos_thermal_idx': pos_thermal_idx,
                'query_similar_rgb_idx': query_similar_rgb_idx,
                'rgb_similar_rgb_idx': rgb_similar_rgb_idx
            })

        return results

    def _compute_similarity(self, query_feat, candidate_feats):
        """
        Cosine similarity 계산

        Args:
            query_feat: [D] numpy array
            candidate_feats: [N, D] numpy array

        Returns:
            [N] numpy array of similarities
        """
        query_norm = query_feat / (np.linalg.norm(query_feat) + 1e-8)
        candidate_norms = candidate_feats / (np.linalg.norm(candidate_feats, axis=1, keepdims=True) + 1e-8)
        similarities = np.dot(candidate_norms, query_norm)
        return similarities


def get_traverse_indices_from_paths(paths, traverse_names=['morning', 'afternoon', 'evening', 'nighttime']):
    """
    파일 경로에서 traverse 정보 추출

    Args:
        paths: list of image paths
        traverse_names: list of traverse name keywords

    Returns:
        dict {idx: traverse_id}
    """
    import re

    traverse_indices = {}
    session_to_id = {}  # 세션 문자열 -> ID 매핑
    next_session_id = 0

    for idx, path in enumerate(paths):
        path_str = path if isinstance(path, str) else str(path)
        path_lower = path_str.lower()

        # 1. 먼저 traverse name 키워드 확인
        found = False
        for t_id, t_name in enumerate(traverse_names):
            if t_name in path_lower:
                traverse_indices[idx] = t_id
                found = True
                break

        if found:
            continue

        # 2. 세션 ID 추출 (예: _2021-08-06-10-59-33)
        session_match = re.search(r'_(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})', path_str)
        if session_match:
            session_str = session_match.group(1)
            if session_str not in session_to_id:
                session_to_id[session_str] = next_session_id + 100  # 100부터 시작 (traverse name과 구분)
                next_session_id += 1
            traverse_indices[idx] = session_to_id[session_str]
        else:
            traverse_indices[idx] = -1

    return traverse_indices


# ===== Visualization =====
def visualize_pair_sampling(sampler, query_idx, query_thermal_feat,
                            all_thermal_feats, all_rgb_feats,
                            database_utms, queries_utms,
                            save_path=None):
    """
    Pair 샘플링 결과 시각화

    Args:
        sampler: PairSampler instance
        query_idx: query index
        query_thermal_feat: [D] query feature
        all_thermal_feats: [N, D] all thermal features
        all_rgb_feats: [N, D] all RGB features
        database_utms: [N, 2] database coordinates
        queries_utms: [M, 2] query coordinates
        save_path: optional save path
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    # 샘플링 실행
    result = sampler.sample_all_pairs(
        query_idx, query_thermal_feat, all_thermal_feats, all_rgb_feats
    )

    query_utm = queries_utms[query_idx]

    # Figure 설정
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # ===== Plot 1: 전체 위치 관계 =====
    ax1 = axes[0]

    # Database points (gray)
    ax1.scatter(database_utms[:, 0], database_utms[:, 1],
                c='lightgray', s=20, alpha=0.5, label='Database')

    # Query (red star)
    ax1.scatter(query_utm[0], query_utm[1],
                c='red', s=200, marker='*', edgecolors='black',
                linewidths=1, label=f'Query {query_idx}', zorder=10)

    # Distance threshold circle
    circle = Circle((query_utm[0], query_utm[1]), sampler.distance_threshold,
                    fill=False, color='red', linestyle='--', linewidth=2,
                    label=f'{sampler.distance_threshold}m threshold')
    ax1.add_patch(circle)

    # Pos Thermal (blue)
    if result['pos_thermal_idx'] is not None:
        pos_thermal_utm = database_utms[result['pos_thermal_idx']]
        dist = np.linalg.norm(query_utm - pos_thermal_utm)

        # Feature similarity
        sim = np.dot(query_thermal_feat, all_thermal_feats[result['pos_thermal_idx']]) / \
              (np.linalg.norm(query_thermal_feat) * np.linalg.norm(all_thermal_feats[result['pos_thermal_idx']]) + 1e-8)

        ax1.scatter(pos_thermal_utm[0], pos_thermal_utm[1],
                   c='blue', s=150, marker='o', edgecolors='black',
                   linewidths=2, label=f'Pos Thermal (d={dist:.1f}m, sim={sim:.3f})', zorder=9)
        ax1.plot([query_utm[0], pos_thermal_utm[0]], [query_utm[1], pos_thermal_utm[1]],
                'b--', linewidth=2, alpha=0.7)

    # Similar RGB (green)
    if result['query_similar_rgb_idx'] is not None:
        rgb_utm = database_utms[result['query_similar_rgb_idx']]
        dist = np.linalg.norm(query_utm - rgb_utm)

        sim = np.dot(query_thermal_feat, all_rgb_feats[result['query_similar_rgb_idx']]) / \
              (np.linalg.norm(query_thermal_feat) * np.linalg.norm(all_rgb_feats[result['query_similar_rgb_idx']]) + 1e-8)

        ax1.scatter(rgb_utm[0], rgb_utm[1],
                   c='green', s=150, marker='s', edgecolors='black',
                   linewidths=2, label=f'Similar RGB (d={dist:.1f}m, sim={sim:.3f})', zorder=9)
        ax1.plot([query_utm[0], rgb_utm[0]], [query_utm[1], rgb_utm[1]],
                'g--', linewidth=2, alpha=0.7)

    # RGB similar RGB (orange)
    if result['rgb_similar_rgb_idx'] is not None:
        rgb2_utm = database_utms[result['rgb_similar_rgb_idx']]

        if result['query_similar_rgb_idx'] is not None:
            rgb1_utm = database_utms[result['query_similar_rgb_idx']]
            dist = np.linalg.norm(rgb1_utm - rgb2_utm)

            rgb1_feat = all_rgb_feats[result['query_similar_rgb_idx']]
            rgb2_feat = all_rgb_feats[result['rgb_similar_rgb_idx']]
            sim = np.dot(rgb1_feat, rgb2_feat) / (np.linalg.norm(rgb1_feat) * np.linalg.norm(rgb2_feat) + 1e-8)

            ax1.scatter(rgb2_utm[0], rgb2_utm[1],
                       c='orange', s=150, marker='D', edgecolors='black',
                       linewidths=2, label=f'RGB→RGB (d={dist:.1f}m, sim={sim:.3f})', zorder=9)
            ax1.plot([rgb1_utm[0], rgb2_utm[0]], [rgb1_utm[1], rgb2_utm[1]],
                    'orange', linestyle='--', linewidth=2, alpha=0.7)

    ax1.set_xlabel('UTM X (m)', fontsize=12)
    ax1.set_ylabel('UTM Y (m)', fontsize=12)
    ax1.set_title(f'Query {query_idx}: Spatial Relationships', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_aspect('equal')
    ax1.grid(alpha=0.3)

    # ===== Plot 2: Thermal Feature Similarity =====
    ax2 = axes[1]

    # 모든 thermal과의 similarity
    thermal_sims = np.array([
        np.dot(query_thermal_feat, all_thermal_feats[i]) /
        (np.linalg.norm(query_thermal_feat) * np.linalg.norm(all_thermal_feats[i]) + 1e-8)
        for i in range(len(all_thermal_feats))
    ])

    # 거리 계산
    thermal_dists = np.linalg.norm(database_utms - query_utm, axis=1)

    # Scatter: x=distance, y=similarity
    scatter = ax2.scatter(thermal_dists, thermal_sims, c='lightblue', s=30, alpha=0.5, label='All Thermals')

    # Threshold line
    ax2.axvline(x=sampler.distance_threshold, color='red', linestyle='--',
                linewidth=2, label=f'{sampler.distance_threshold}m threshold')

    # Selected pos thermal
    if result['pos_thermal_idx'] is not None:
        idx = result['pos_thermal_idx']
        ax2.scatter(thermal_dists[idx], thermal_sims[idx],
                   c='blue', s=200, marker='o', edgecolors='black',
                   linewidths=2, label='Selected Pos Thermal', zorder=10)

    ax2.set_xlabel('Distance from Query (m)', fontsize=12)
    ax2.set_ylabel('Cosine Similarity', fontsize=12)
    ax2.set_title('Thermal: Distance vs Similarity', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(alpha=0.3)

    # ===== Plot 3: RGB Feature Similarity =====
    ax3 = axes[2]

    # Query와 모든 RGB의 similarity
    rgb_sims_to_query = np.array([
        np.dot(query_thermal_feat, all_rgb_feats[i]) /
        (np.linalg.norm(query_thermal_feat) * np.linalg.norm(all_rgb_feats[i]) + 1e-8)
        for i in range(len(all_rgb_feats))
    ])

    rgb_dists = np.linalg.norm(database_utms - query_utm, axis=1)

    scatter = ax3.scatter(rgb_dists, rgb_sims_to_query, c='lightgreen', s=30, alpha=0.5, label='All RGBs')

    # Selected similar RGB
    if result['query_similar_rgb_idx'] is not None:
        idx = result['query_similar_rgb_idx']
        ax3.scatter(rgb_dists[idx], rgb_sims_to_query[idx],
                   c='green', s=200, marker='s', edgecolors='black',
                   linewidths=2, label='Selected Similar RGB', zorder=10)

    # RGB similar RGB
    if result['rgb_similar_rgb_idx'] is not None:
        idx = result['rgb_similar_rgb_idx']
        ax3.scatter(rgb_dists[idx], rgb_sims_to_query[idx],
                   c='orange', s=200, marker='D', edgecolors='black',
                   linewidths=2, label='RGB→RGB', zorder=10)

    ax3.set_xlabel('Distance from Query (m)', fontsize=12)
    ax3.set_ylabel('Cosine Similarity (to Query)', fontsize=12)
    ax3.set_title('RGB: Distance vs Similarity', fontsize=14, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(alpha=0.3)

    plt.suptitle(f'Pair Sampling Visualization - Query {query_idx}',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()

    plt.close()

    return result


# ===== Test code =====
if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')  # For saving without display
    import matplotlib.pyplot as plt

    print("Testing PairSampler with Visualization...")

    # Dummy data
    np.random.seed(42)
    N_db = 100
    N_q = 1000
    D = 384

    database_utms = np.random.randn(N_db, 2) * 50  # 50m 범위
    queries_utms = np.random.randn(N_q, 2) * 50

    thermal_feats = np.random.randn(N_db, D).astype(np.float32)
    rgb_feats = np.random.randn(N_db, D).astype(np.float32)
    query_thermal_feats = np.random.randn(N_q, D).astype(np.float32)

    # PairSampler 테스트
    sampler = PairSampler(database_utms, queries_utms, distance_threshold=20.0)

    import os
    os.makedirs('./pair_sampler_vis', exist_ok=True)

    for query_idx in [randint(0, 1000) for _ in range(10)]:
        print(f"\nQuery {query_idx}:")
        result = sampler.sample_all_pairs(
            query_idx,
            query_thermal_feats[query_idx],
            thermal_feats,
            rgb_feats,
            query_traverse_id=None
        )
        print(f"  pos_thermal_idx: {result['pos_thermal_idx']}")
        print(f"  query_similar_rgb_idx: {result['query_similar_rgb_idx']}")
        print(f"  rgb_similar_rgb_idx: {result['rgb_similar_rgb_idx']}")

        # 시각화 저장
        save_path = f'./pair_sampler_vis/query_{query_idx:02d}.png'
        visualize_pair_sampling(
            sampler, query_idx, query_thermal_feats[query_idx],
            thermal_feats, rgb_feats,
            database_utms, queries_utms,
            save_path=save_path
        )

    # 거리별 분포 시각화
    print("\n\nGenerating Distance Distribution Analysis...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Query 0에 대해 상세 분석
    query_idx = 0
    query_utm = queries_utms[query_idx]
    query_feat = query_thermal_feats[query_idx]

    # Thermal distances & similarities
    thermal_dists = np.linalg.norm(database_utms - query_utm, axis=1)
    thermal_sims = np.array([
        np.dot(query_feat, thermal_feats[i]) /
        (np.linalg.norm(query_feat) * np.linalg.norm(thermal_feats[i]) + 1e-8)
        for i in range(N_db)
    ])

    # 거리 threshold 내부 vs 외부 비교
    inside_mask = thermal_dists < 20.0
    outside_mask = ~inside_mask

    ax1 = axes[0]
    ax1.hist(thermal_sims[inside_mask], bins=20, alpha=0.7, label=f'Inside 20m (n={inside_mask.sum()})', color='blue')
    ax1.hist(thermal_sims[outside_mask], bins=20, alpha=0.7, label=f'Outside 20m (n={outside_mask.sum()})', color='gray')
    ax1.axvline(x=thermal_sims[sampler.positives_per_query[query_idx]].max(),
                color='red', linestyle='--', linewidth=2, label='Selected (max sim)')
    ax1.set_xlabel('Cosine Similarity', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Thermal Similarity Distribution\n(Inside vs Outside threshold)', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # RGB distances & similarities
    rgb_sims = np.array([
        np.dot(query_feat, rgb_feats[i]) /
        (np.linalg.norm(query_feat) * np.linalg.norm(rgb_feats[i]) + 1e-8)
        for i in range(N_db)
    ])

    ax2 = axes[1]
    ax2.scatter(thermal_dists, rgb_sims, c='green', alpha=0.5, s=30)
    ax2.axvline(x=20.0, color='red', linestyle='--', linewidth=2, label='20m threshold')

    # Best RGB 표시
    best_rgb_idx = np.argmax(rgb_sims)
    ax2.scatter(thermal_dists[best_rgb_idx], rgb_sims[best_rgb_idx],
               c='orange', s=200, marker='*', edgecolors='black',
               linewidths=2, label=f'Best RGB (d={thermal_dists[best_rgb_idx]:.1f}m)', zorder=10)

    ax2.set_xlabel('Distance from Query (m)', fontsize=12)
    ax2.set_ylabel('Cosine Similarity (Query Thermal → RGB)', fontsize=12)
    ax2.set_title('Cross-modal Similarity: Thermal → RGB', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('./pair_sampler_vis/distribution_analysis.png', dpi=150, bbox_inches='tight')
    print("Saved distribution analysis to ./pair_sampler_vis/distribution_analysis.png")
    plt.close()

    # BatchPairSampler 테스트
    print("\n\nTesting BatchPairSampler...")
    batch_sampler = BatchPairSampler(distance_threshold=20.0)

    batch_thermal = np.random.randn(4, D).astype(np.float32)
    batch_rgb = np.random.randn(4, D).astype(np.float32)
    batch_utms = np.random.randn(4, 2) * 20

    results = batch_sampler.sample_pairs_from_features(
        batch_thermal, batch_rgb, batch_utms, batch_utms
    )

    for r in results:
        print(f"\nQuery {r['query_idx']}:")
        print(f"  pos_thermal_idx: {r['pos_thermal_idx']}")
        print(f"  query_similar_rgb_idx: {r['query_similar_rgb_idx']}")
        print(f"  rgb_similar_rgb_idx: {r['rgb_similar_rgb_idx']}")

    print("\n\nAll tests passed!")
    print(f"\nVisualization saved to ./pair_sampler_vis/")


# ===== Real Dataset Test =====
def test_with_real_dataset():
    """실제 데이터셋으로 테스트하고 이미지 시각화"""
    import sys
    import os
    import torch
    import cv2
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import argparse

    from datasets_T2R import TripletsSTheReODual

    print("\n" + "="*60)
    print("Testing with Real Dataset...")
    print("="*60)

    # Args 설정 (Parser 대신 직접 설정)
    args = argparse.Namespace()
    args.train_seq = ['Campus']
    args.test_seq = ['Urban']
    args.resize = [224, 224]
    args.hard_positives_dist_threshold = 10
    args.soft_positives_dist_threshold = 10
    args.negs_num_per_query = 10
    args.img_time = 'allday'
    args.features_dim = 384
    args.test_method = 'hard_resize'
    args.sequences = ['Campus']
    args.mining = 'partial'
    args.neg_samples_num = 1000
    args.queries_per_epoch = 2000
    args.cache_refresh_rate = 1000

    # 데이터셋 로드
    print("Loading dataset...")
    datasets_folder = '/home/jwkim/workspace/THR2RGB_Cross_Place_Recognition/Dataset/save_mat'
    triplets_ds = TripletsSTheReODual(args, datasets_folder, use_align_rgb=True)

    print(f"Database: {triplets_ds.database_num}")
    print(f"Queries: {triplets_ds.queries_num}")

    # Traverse 정보 추출 (파일 경로에서)
    print("Extracting traverse info from paths...")
    db_traverse_indices = get_traverse_indices_from_paths(
        triplets_ds.t_database_paths,
        traverse_names=['morning', 'afternoon', 'evening', 'clearsky', 'rainy', 'nighttime']
    )
    query_traverse_indices = get_traverse_indices_from_paths(
        triplets_ds.t_queries_paths,
        traverse_names=['morning', 'afternoon', 'evening', 'clearsky', 'rainy', 'nighttime']
    )

    # PairSampler 생성 (실제 좌표 + traverse 정보 사용)
    sampler = PairSampler(
        triplets_ds.database_utms,
        triplets_ds.queries_utms,
        traverse_indices=db_traverse_indices,
        distance_threshold=10.0
    )

    # 시각화 디렉토리
    os.makedirs('./pair_sampler_vis/real_images', exist_ok=True)

    # 랜덤 feature 생성 (실제로는 모델에서 추출해야 함)
    # 여기서는 dummy로 대체
    D = 384
    all_thermal_feats = np.random.randn(triplets_ds.database_num, D).astype(np.float32)
    all_rgb_feats = np.random.randn(triplets_ds.database_num, D).astype(np.float32)
    query_thermal_feats = np.random.randn(triplets_ds.queries_num, D).astype(np.float32)
    
    # 3개 query에 대해 시각화
    for query_idx in [randint(0, 1000) for _ in range(10)]:
        if query_idx >= triplets_ds.queries_num:
            continue

        print(f"\nProcessing Query {query_idx}...")

        # Query의 traverse ID
        query_traverse_id = query_traverse_indices.get(query_idx, None)
        print(f"  Query traverse: {query_traverse_id}")

        # Pair 샘플링
        result = sampler.sample_all_pairs(
            query_idx,
            query_thermal_feats[query_idx],
            all_thermal_feats,
            all_rgb_feats,
            query_traverse_id=query_traverse_id
        )

        # 이미지 로드
        query_thermal_path = triplets_ds.t_queries_paths[query_idx]
        query_rgb_path = triplets_ds.rgb_queries_paths[query_idx]

        query_thermal_img = triplets_ds.get_thermal_img(query_thermal_path)
        query_rgb_img = triplets_ds.get_rgb_img(query_rgb_path)

        # Resize
        query_thermal_img = cv2.resize(query_thermal_img, (224, 224))
        query_rgb_img = cv2.resize(query_rgb_img, (224, 224))

        # BGR to RGB
        query_thermal_img = cv2.cvtColor(query_thermal_img, cv2.COLOR_BGR2RGB)
        query_rgb_img = cv2.cvtColor(query_rgb_img, cv2.COLOR_BGR2RGB)

        # 샘플링된 이미지 로드
        images_to_show = {
            'Query Thermal': query_thermal_img,
            'Query RGB (same time)': query_rgb_img,
        }

        # Pos Thermal
        if result['pos_thermal_idx'] is not None:
            pos_thermal_path = triplets_ds.t_database_paths[result['pos_thermal_idx']]
            pos_thermal_img = triplets_ds.get_thermal_img(pos_thermal_path)
            pos_thermal_img = cv2.resize(pos_thermal_img, (224, 224))
            pos_thermal_img = cv2.cvtColor(pos_thermal_img, cv2.COLOR_BGR2RGB)

            dist = np.linalg.norm(
                triplets_ds.queries_utms[query_idx] -
                triplets_ds.database_utms[result['pos_thermal_idx']]
            )
            images_to_show[f'Pos Thermal\n(d={dist:.1f}m)'] = pos_thermal_img

        # Similar RGB
        if result['query_similar_rgb_idx'] is not None:
            similar_rgb_path = triplets_ds.rgb_database_paths[result['query_similar_rgb_idx']]
            similar_rgb_img = triplets_ds.get_rgb_img(similar_rgb_path)
            similar_rgb_img = cv2.resize(similar_rgb_img, (224, 224))
            similar_rgb_img = cv2.cvtColor(similar_rgb_img, cv2.COLOR_BGR2RGB)

            dist = np.linalg.norm(
                triplets_ds.queries_utms[query_idx] -
                triplets_ds.database_utms[result['query_similar_rgb_idx']]
            )
            images_to_show[f'Similar RGB\n(d={dist:.1f}m)'] = similar_rgb_img

        # RGB similar RGB
        if result['rgb_similar_rgb_idx'] is not None:
            rgb2_path = triplets_ds.rgb_database_paths[result['rgb_similar_rgb_idx']]
            rgb2_img = triplets_ds.get_rgb_img(rgb2_path)
            rgb2_img = cv2.resize(rgb2_img, (224, 224))
            rgb2_img = cv2.cvtColor(rgb2_img, cv2.COLOR_BGR2RGB)

            if result['query_similar_rgb_idx'] is not None:
                dist = np.linalg.norm(
                    triplets_ds.database_utms[result['query_similar_rgb_idx']] -
                    triplets_ds.database_utms[result['rgb_similar_rgb_idx']]
                )
            else:
                dist = 0
            images_to_show[f'RGB→RGB\n(d={dist:.1f}m)'] = rgb2_img

        # Figure 생성
        n_images = len(images_to_show)
        fig, axes = plt.subplots(1, n_images, figsize=(5*n_images, 5))

        for ax, (title, img) in zip(axes, images_to_show.items()):
            ax.imshow(img)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.axis('off')

        plt.suptitle(f'Query {query_idx} - Sampled Pairs', fontsize=16, fontweight='bold')
        plt.tight_layout()

        save_path = f'./pair_sampler_vis/real_images/query_{query_idx:04d}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")

    print("\nReal dataset test complete!")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--real':
        test_with_real_dataset()
