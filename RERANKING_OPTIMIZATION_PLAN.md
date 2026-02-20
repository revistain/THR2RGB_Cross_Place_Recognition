# Reranking 병렬화 최적화 계획

## 1. 현재 구현의 문제점

### 현재 코드 구조 (inference.py:511-627)

```python
for query_index, pred in enumerate(tqdm(prev_predictions)):  # ~2000 쿼리 순차 처리
    top_k_indices = pred[:RERANKING_TOP_K]  # K=5

    # 파일 I/O: 쿼리마다 개별 로드
    query_feat = load_npy(f"Query_{seq_name}_{query_index}")
    for db_idx in top_k_indices:
        candidate_feats.append(load_npy(f"Db_{seq_name}_{db_idx}"))

    # GPU 연산: 배치 크기 = 5 (매우 작음)
    thermal_dec = decoder(thermal_feat, rgb_feat)  # [5, 256, D]

    # 손실 계산 후 재정렬
    rerank_index = recon_losses.argsort()
```

### 병목 지점 분석

| 구간 | 현재 상태 | 문제점 |
|------|----------|--------|
| **파일 I/O** | 쿼리당 6개 파일 순차 로드 | 디스크 대기 시간 누적 |
| **GPU 배치** | 5개 샘플만 처리 | GPU utilization ~5% |
| **메모리 전송** | 매 반복마다 `.cuda()` | PCIe 병목 |
| **전체 루프** | 2000회 반복 | O(n) 시간 복잡도 |

### 예상 시간 분석 (쿼리 2000개 기준)

```
현재:
- 파일 I/O: ~5ms × 6파일 × 2000쿼리 = 60초
- GPU forward: ~10ms × 2000쿼리 = 20초
- 기타 오버헤드: ~10초
- 총합: ~90초

최적화 후 (예상):
- 파일 I/O (병렬): ~5ms × 6파일 × 2000쿼리 / 8workers = 7.5초
- GPU forward (배치): ~10ms × 2000쿼리 / 32배치 = 0.6초
- 기타 오버헤드: ~3초
- 총합: ~11초 (약 8배 향상)
```

---

## 2. 제안하는 해결책

### 핵심 아이디어

1. **DataLoader로 파일 I/O 병렬화**: `num_workers`로 멀티프로세스 로딩
2. **Multi-Query Batching**: 여러 쿼리를 한 번에 GPU에서 처리
3. **Vectorized Loss 계산**: 내부 for loop 제거

### 아키텍처 변경

```
[현재]
for query in queries:           # 순차
    load_features()             # 순차 I/O
    gpu_forward()               # 배치=5
    compute_loss()              # 순차

[변경 후]
RerankingDataset + DataLoader   # 병렬 I/O (num_workers=8)
    ↓
for batch in dataloader:        # 배치=32 쿼리
    gpu_forward()               # 배치=32×5=160
    compute_loss_vectorized()   # 벡터화
```

---

## 3. 구현 계획

### 3.1 RerankingDataset 클래스 생성

```python
class RerankingDataset(Dataset):
    """
    각 샘플 = 1개 쿼리 + 해당 쿼리의 top-K 후보들
    DataLoader가 여러 쿼리를 병렬로 로드
    """
    def __init__(self, predictions, seq_name, reranking_top_k=5):
        self.predictions = predictions      # [num_queries, max_recall]
        self.seq_name = seq_name
        self.reranking_top_k = reranking_top_k

    def __len__(self):
        return len(self.predictions)

    def __getitem__(self, query_index):
        top_k_indices = self.predictions[query_index][:self.reranking_top_k]

        # 쿼리 feature 로드
        query_feat = load_npy(f"Query_{self.seq_name}_{query_index}")
        query_target = load_npy(f"Query_{self.seq_name}_target_{query_index}")

        # 후보 features 로드
        candidate_feats = []
        candidate_targets = []
        for db_idx in top_k_indices:
            candidate_feats.append(load_npy(f"Db_{self.seq_name}_{db_idx}"))
            candidate_targets.append(load_npy(f"Db_{self.seq_name}_target_{db_idx}"))

        return {
            'query_index': query_index,
            'top_k_indices': top_k_indices,
            'query_feat': query_feat,           # [256, 768]
            'query_target': query_target,       # [256, 588]
            'candidate_feats': np.stack(candidate_feats),    # [K, 256, 768]
            'candidate_targets': np.stack(candidate_targets) # [K, 256, 588]
        }
```

### 3.2 Custom Collate Function

```python
def reranking_collate_fn(batch):
    """
    여러 쿼리를 하나의 큰 배치로 합침

    Input: list of dicts, len = batch_size (예: 32)
    Output: dict with batched tensors
    """
    batch_size = len(batch)
    K = batch[0]['candidate_feats'].shape[0]  # RERANKING_TOP_K

    # 쿼리 정보
    query_indices = [b['query_index'] for b in batch]
    top_k_indices = np.stack([b['top_k_indices'] for b in batch])  # [B, K]

    # 쿼리 features: [B, 256, 768] -> expand to [B, K, 256, 768]
    query_feats = np.stack([b['query_feat'] for b in batch])
    query_feats = np.tile(query_feats[:, np.newaxis], (1, K, 1, 1))  # [B, K, 256, 768]

    query_targets = np.stack([b['query_target'] for b in batch])
    query_targets = np.tile(query_targets[:, np.newaxis], (1, K, 1, 1))  # [B, K, 256, 588]

    # 후보 features: [B, K, 256, 768]
    candidate_feats = np.stack([b['candidate_feats'] for b in batch])
    candidate_targets = np.stack([b['candidate_targets'] for b in batch])

    # Flatten for GPU processing: [B*K, 256, 768]
    return {
        'query_indices': query_indices,
        'top_k_indices': torch.from_numpy(top_k_indices),
        'thermal_feat': torch.from_numpy(query_feats.reshape(-1, 256, 768)),
        'thermal_target': torch.from_numpy(query_targets.reshape(-1, 256, 588)),
        'rgb_feat': torch.from_numpy(candidate_feats.reshape(-1, 256, 768)),
        'rgb_target': torch.from_numpy(candidate_targets.reshape(-1, 256, 588)),
        'batch_size': batch_size,
        'top_k': K
    }
```

### 3.3 Vectorized Reranking Loop

```python
def batched_recon_reranking(model, dataloader, args, H_feat, W_feat):
    """
    배치 단위로 reranking 수행
    """
    predictions_list = []
    rerank_scores_dict = {}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Batched Recon Reranking"):
            B = batch['batch_size']
            K = batch['top_k']
            BK = B * K  # 총 샘플 수 (예: 32 * 5 = 160)

            # GPU로 전송 (한 번에)
            thermal_feat = batch['thermal_feat'].float().cuda()    # [BK, 256, 768]
            thermal_target = batch['thermal_target'].float().cuda()
            rgb_feat = batch['rgb_feat'].float().cuda()
            rgb_target = batch['rgb_target'].float().cuda()

            N, D = thermal_feat.shape[1], thermal_feat.shape[2]

            # ===== GeM scores =====
            thermal_spatial = thermal_feat.permute(0, 2, 1).view(BK, D, H_feat, W_feat)
            thermal_gem = model.module.thermal_aggregation(thermal_spatial)
            thermal_gem_scores = torch.einsum('bd,bnd->bn', thermal_gem, thermal_feat)
            thermal_gem_weight = F.softmax(thermal_gem_scores, dim=-1)

            rgb_spatial = rgb_feat.permute(0, 2, 1).view(BK, D, H_feat, W_feat)
            rgb_gem = model.module.rgb_aggregation(rgb_spatial)
            rgb_gem_scores = torch.einsum('bd,bnd->bn', rgb_gem, rgb_feat)
            rgb_gem_weight = F.softmax(rgb_gem_scores, dim=-1)

            # ===== Masking =====
            mask_generator = model.module.mask_generator
            thermal_masks = mask_generator(thermal_feat, gem_score=thermal_gem_scores)
            rgb_masks = mask_generator(rgb_feat, gem_score=rgb_gem_scores)

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
            thermal_recon = model.module.prediction_thermal_head(thermal_dec)
            rgb_recon = model.module.prediction_rgb_head(rgb_dec)

            # ===== Vectorized loss computation =====
            # [BK] -> [B, K]로 reshape하여 쿼리별 손실 계산
            thermal_recon = thermal_recon.view(B, K, N, -1)
            rgb_recon = rgb_recon.view(B, K, N, -1)
            thermal_masks = thermal_masks.view(B, K, N)
            rgb_masks = rgb_masks.view(B, K, N)
            thermal_target = thermal_target.view(B, K, N, -1)
            rgb_target = rgb_target.view(B, K, N, -1)
            thermal_gem_weight = thermal_gem_weight.view(B, K, N)
            rgb_gem_weight = rgb_gem_weight.view(B, K, N)

            # 배치 전체에 대해 한 번에 손실 계산
            recon_losses = compute_batch_recon_loss(
                thermal_recon, rgb_recon,
                thermal_masks, rgb_masks,
                thermal_target, rgb_target,
                thermal_gem_weight, rgb_gem_weight,
                args.use_gem_recon_weight
            )  # [B, K]

            # ===== Rerank =====
            rerank_indices = recon_losses.argsort(dim=1)  # [B, K]

            # 결과 저장
            top_k_indices = batch['top_k_indices'].numpy()  # [B, K]
            for i, query_idx in enumerate(batch['query_indices']):
                reranked = top_k_indices[i][rerank_indices[i].cpu().numpy()]
                predictions_list.append(reranked)
                rerank_scores_dict[query_idx] = recon_losses[i].cpu().numpy().tolist()

    return np.array(predictions_list), rerank_scores_dict
```

### 3.4 Vectorized Loss Function

```python
def compute_batch_recon_loss(
    thermal_recon,      # [B, K, N, D]
    rgb_recon,          # [B, K, N, D]
    thermal_masks,      # [B, K, N]
    rgb_masks,          # [B, K, N]
    thermal_target,     # [B, K, N, D]
    rgb_target,         # [B, K, N, D]
    thermal_weight,     # [B, K, N]
    rgb_weight,         # [B, K, N]
    use_weight=False
):
    """
    모든 쿼리-후보 쌍에 대해 한 번에 손실 계산

    Returns:
        losses: [B, K] - 각 쿼리의 각 후보에 대한 손실
    """
    B, K, N, D = thermal_recon.shape

    # MSE loss: [B, K, N, D] -> [B, K, N]
    thermal_mse = ((thermal_recon - thermal_target) ** 2).mean(dim=-1)
    rgb_mse = ((rgb_recon - rgb_target) ** 2).mean(dim=-1)

    # 마스크된 위치만 계산
    thermal_mse = thermal_mse * thermal_masks.float()
    rgb_mse = rgb_mse * rgb_masks.float()

    if use_weight:
        thermal_mse = thermal_mse * thermal_weight
        rgb_mse = rgb_mse * rgb_weight

    # 마스크된 패치 수로 정규화
    thermal_mask_count = thermal_masks.sum(dim=-1).clamp(min=1)  # [B, K]
    rgb_mask_count = rgb_masks.sum(dim=-1).clamp(min=1)

    thermal_loss = thermal_mse.sum(dim=-1) / thermal_mask_count  # [B, K]
    rgb_loss = rgb_mse.sum(dim=-1) / rgb_mask_count

    return (thermal_loss + rgb_loss) / 2  # [B, K]
```

---

## 4. 메인 코드 변경

### 변경 전 (inference.py)

```python
# Line 511-627
with torch.no_grad():
    for query_index, pred in enumerate(tqdm(prev_predictions)):
        # ... 개별 처리 ...
predictions = np.array(predictions_list)
```

### 변경 후

```python
# 새로운 DataLoader 기반 처리
from reranking_dataset import RerankingDataset, reranking_collate_fn

reranking_dataset = RerankingDataset(
    predictions=prev_predictions,
    seq_name=seq_name,
    reranking_top_k=RERANKING_TOP_K
)

reranking_dataloader = DataLoader(
    dataset=reranking_dataset,
    batch_size=32,              # 32개 쿼리를 한 번에
    shuffle=False,
    num_workers=8,              # 병렬 파일 로딩
    collate_fn=reranking_collate_fn,
    pin_memory=True
)

predictions, rerank_scores_dict = batched_recon_reranking(
    model=model,
    dataloader=reranking_dataloader,
    args=args,
    H_feat=H_feat,
    W_feat=W_feat
)
```

---

## 5. 파일 구조

```
THR2RGB_Cross_Place_Recognition/
├── inference.py                 # 메인 코드 (수정)
├── reranking_dataset.py         # 새로 생성
│   ├── RerankingDataset
│   ├── reranking_collate_fn
│   └── compute_batch_recon_loss
└── reranking_utils.py           # 새로 생성 (선택적)
    └── batched_recon_reranking
```

---

## 6. 예상 성능 향상

| 항목 | 현재 | 최적화 후 | 향상 |
|------|------|----------|------|
| 파일 I/O | 순차 | 8 workers 병렬 | ~8x |
| GPU 배치 크기 | 5 | 160 (32×5) | ~32x |
| GPU utilization | ~5% | ~80% | ~16x |
| 전체 시간 (2000 쿼리) | ~90초 | ~11초 | **~8x** |

---

## 7. 추가 고려사항

### 메모리 사용량
- 배치 크기 32 × K=5 = 160 샘플
- Feature: 160 × 256 × 768 × 4bytes = ~125MB
- Target: 160 × 256 × 588 × 4bytes = ~96MB
- 총 GPU 메모리 증가: ~500MB (충분히 감당 가능)

### Visualization 호환성
- `vis_query_indices`에 해당하는 쿼리는 별도 저장 로직 필요
- 배치 처리 후 해당 인덱스만 필터링

### 다른 Reranking 방법 (selaVPR, diffGeM 등)
- 동일한 패턴으로 적용 가능
- 각 방법별 Dataset 클래스 또는 통합 클래스 구현

---

## 8. 구현 순서

1. `reranking_dataset.py` 파일 생성
   - `RerankingDataset` 클래스
   - `reranking_collate_fn` 함수
   - `compute_batch_recon_loss` 함수

2. `inference.py` 수정
   - `batched_recon_reranking` 함수 추가
   - 기존 for loop을 DataLoader 기반으로 교체

3. 테스트 및 검증
   - 결과 일치 확인 (기존 vs 최적화)
   - 속도 측정

4. (선택) 다른 reranking 방법에도 적용
