
import torch
import random
from glob import glob
import torch.nn.functional as F
import datetime
import numpy as np
import cv2
import math

import matplotlib.pyplot as plt

def plot_attention_1d(attn_map):
    """
    1D Attention Map을 시각화하는 함수
    
    Args:
        attn_map (list or np.array): [256] 크기의 1D Attention Score 배열
    """
    # 입력 데이터를 numpy 배열로 변환
    breakpoint()
    data = np.array(attn_map.detach().cpu())
    
    # 시각화를 위한 Figure 생성 (위: 막대그래프, 아래: 히트맵)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), 
                                   gridspec_kw={'height_ratios': [3, 1]})
    
    # 1. Bar Plot (값의 크기 확인용)
    ax1.bar(range(len(data)), data, color='royalblue', width=1.0)
    ax1.set_title("Attention Score Distribution (Bar Plot)")
    ax1.set_ylabel("Score")
    ax1.set_xlim(0, len(data))
    ax1.grid(True, axis='y', linestyle='--', alpha=0.3)
    
    # 2. Heatmap Strip (위치 직관 확인용)
    # imshow를 쓰기 위해 차원 확장: [256] -> [1, 256]
    heatmap_data = data[np.newaxis, :]
    
    # cmap='Reds' 혹은 'Viridis' 등을 추천
    im = ax2.imshow(heatmap_data, cmap='Reds', aspect='auto', vmin=0, vmax=1)
    ax2.set_title("Attention Heatmap")
    ax2.set_xlabel("Token Index")
    ax2.set_yticks([])  # y축 눈금 제거
    
    # 컬러바 추가
    plt.colorbar(im, ax=ax2, orientation='vertical', label='Intensity')
    
    plt.tight_layout()
    plt.savefig('attn_vis.png')

def get_keypoints(img_size):
    # flaten by x 
    H,W = img_size
    patch_size = 1#14
    N_h = H//patch_size
    N_w = W//patch_size
    keypoints = np.zeros((2, N_h*N_w), dtype=int)
    keypoints[0] = np.tile(np.linspace(patch_size//2, W-patch_size//2, N_w, 
                                       dtype=int), N_h)
    keypoints[1] = np.repeat(np.linspace(patch_size//2, H-patch_size//2, N_h,
                                         dtype=int), N_w)
    return np.transpose(keypoints)

def match_batch_tensor(fm1, fm2, trainflag, grid_size, query_attn_map=None, db_attn_map=None, method_type='none'):
    '''
    fm1: (l,D)
    fm2: (N,l,D)
    mask1: (l)
    mask2: (N,l)
    '''
    M = torch.matmul(fm2, fm1.T) # (N,l,l) # Similarity Matrix # dot product로만으로도 cosine 유사도 가능 (if L2 normed)
    # M.shape: [5, 3721, 3721]
    
    max1 = torch.argmax(M, dim=1) # (N,l) # fm1이 보는 fm2에서 제일 유사한 index들
    max2 = torch.argmax(M, dim=2) # (N,l) # fm2이 보는 fm1에서 제일 유사한 index들
    # M은 (Batch, DB_Row, Query_Col)
    m = max2[torch.arange(M.shape[0]).reshape((-1,1)), max1] # (N, l) # MNN
    valid = torch.arange(M.shape[-1]).repeat((M.shape[0],1)).cuda() == m # (N, l) bool # MNN matched?
    scores = torch.zeros(fm2.shape[0]).cuda()
    
    query_attn_map = query_attn_map.reshape(-1)
    db_attn_map = db_attn_map.reshape(db_attn_map.shape[0], -1)
    
    LOW_LIMIT = 0.00 # 하위 trim
    HIGH_LIMIT = 0.97 # 상위 trim
    if method_type == 'quantile_attn' and query_attn_map is not None:
        q_limit_low = torch.quantile(query_attn_map, LOW_LIMIT)  # 하위 20% 지점
        q_limit_high = torch.quantile(query_attn_map, HIGH_LIMIT) # 상위 20% 지점 (80% 지점)
        
    for i in range(fm2.shape[0]):
        idx1 = torch.nonzero(valid[i,:]).squeeze() # matching된 fm1 index들
        idx2 = max1[i,:][idx1] # matching된 fm2 index들
        assert idx1.shape==idx2.shape

        if trainflag:
            if len(idx1.shape)>0:      
                similarity = torch.mean(torch.sum(fm1[idx1] * fm2[i][idx2],dim=1),dim=0)
            else:
                print("No mutual nearest neighbors!")
                similarity = torch.mean(torch.sum(fm1 * fm2[i],dim=1),dim=0)
            return similarity
        else:
            if len(idx1.shape)<1:
                scores[i] = 0
            else:
                if method_type == 'quantile_attn' and query_attn_map is not None and db_attn_map is not None:
                    # 1. 현재 DB 이미지(i) 전체 분포에서 상/하위 20% 기준값 계산
                    d_limit_low = torch.quantile(db_attn_map[i], LOW_LIMIT) # 하위 20퍼 거르기
                    d_limit_high = torch.quantile(db_attn_map[i], HIGH_LIMIT) # 상위 20퍼 거르기
                    
                    # 2. 매칭된 포인트들이 '각자의 이미지'에서 정상 범위(중위 60%)에 있는지 확인
                    # Query 쪽 조건: Query 전체 맵 기준 중간 60%
                    valid_q = (query_attn_map[idx1] >= q_limit_low) & (query_attn_map[idx1] <= q_limit_high)
                    # DB 쪽 조건: DB 전체 맵 기준 중간 60%
                    valid_d = (db_attn_map[i][idx2] >= d_limit_low) & (db_attn_map[i][idx2] <= d_limit_high)
                    
                    # 3. 교집합(AND): 양쪽 다 정상 범위인 매칭만 유효한 것으로 인정
                    final_mask = valid_q & valid_d
                    
                    # 4. Counting (살아남은 매칭 개수)
                    scores[i] = final_mask.sum()
                elif method_type == 'mul_cossim':
                    matched_sims = M[i, idx2, idx1] 
                    scores[i] = torch.sum(matched_sims)
                elif method_type == 'none':
                    scores[i] = len(idx1)
    return scores

def local_sim(features_1, features_2, trainflag=False, query_attn_map=None, db_attn_map=None, method_type='none'):
    B, H, W, C = features_2.shape
    if trainflag:
        queries = features_1
        preds = features_2
        queries,preds = queries.view(B, H*W, C),preds.view(B, H*W, C)
        similarity = torch.zeros(B).cuda()
        for i in range(B):
            query, pred = queries[i], preds[i].unsqueeze(0)
            similarity[i] = match_batch_tensor(query, pred, trainflag, grid_size=(H, W), method_type=method_type)
        return similarity
    else:
        query = features_1
        preds = features_2
        query,preds = query.view(H*W, C),preds.view(B, H*W, C)
        # query: [3721, 128]
        # preds: [5, 3721, 128]
        if method_type == 'quantile_attn':
            # reshape and interpolate attn_map 61->16
            query_attn_map = query_attn_map.reshape(16, 16).unsqueeze(0).unsqueeze(0) # NOTE: HARDCODE
            db_attn_map = db_attn_map.reshape(B, 16, 16).unsqueeze(1) # NOTE: HARDCODE
            query_attn_map = F.interpolate(query_attn_map, size=(H, W), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
            db_attn_map = F.interpolate(db_attn_map, size=(H, W), mode='bilinear', align_corners=False).squeeze(1)
        
        scores = match_batch_tensor(query, preds, trainflag, grid_size=(H, W),
                    query_attn_map=query_attn_map, db_attn_map=db_attn_map, method_type=method_type)
        return scores

class LocalFeatureLoss(torch.nn.Module):
    def __init__(self):
        super(LocalFeatureLoss,self).__init__()
        return
    def forward(self, feature_data):
        anchor, positive, negative = feature_data[0], feature_data[1], feature_data[2]
        simP = local_sim(anchor,positive,trainflag=True)
        simN = local_sim(anchor,negative,trainflag=True)
        loss = torch.sum(torch.clamp(-simP+simN+0., min=0.))
        return loss