# recon_vis.py
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
from utils import get_timestamp
import cv2

def visualize_reconstruction(model, thermal_img, paired_rgb, device='cuda', save_path=None):
    """
    Args:
        model: CrossModalVPR_Net
        thermal_img: [1, 3, 224, 224] - single thermal image
        paired_rgb: [1, 3, 224, 224] - aligned RGB image
        device: 'cuda' or 'cpu'
        save_path: Optional path to save the figure
    """
    model.train()
    with torch.no_grad():
        _, patch_thermal, recon_loss, mask_thermal, cls_attn_map, masked_patch_thermal = model.module.forward_model(thermal_img, modality='thermal', paired_rgb=paired_rgb)
    model.eval()
    
    reconstructed_pixels = model.module.prediction_head(patch_thermal)  # [1, 256, 588]
    reconstructed_img = unpatchify_visual(reconstructed_pixels, patch_size=14) # [1, 3, 224, 224]
    
    # Denormalize (ImageNet stats 사용했다고 가정)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    
    thermal_img_denorm = thermal_img * std + mean
    aligned_rgb_denorm = paired_rgb * std + mean
    reconstructed_img_denorm = reconstructed_img * std + mean
    
    # Visible patches는 원본, Masked patches는 reconstruction 사용
    original_patches = model.module.patchify(thermal_img)
    hybrid_patches = original_patches.clone()  # [1, 256, 588]
    hybrid_patches[mask_thermal] = reconstructed_pixels[mask_thermal]  # Masked 위치만 reconstruction으로 교체
    
    hybrid_img = unpatchify_visual(hybrid_patches, patch_size=14)  # [1, 3, 224, 224]
    hybrid_img_denorm = hybrid_img * std + mean
    
    hybrid_img_denorm = hybrid_img_denorm.detach().cpu()
    hybrid_img = hybrid_img.detach().cpu()
    hybrid_patches = hybrid_patches.detach().cpu()
    original_patches = original_patches.detach().cpu()
    reconstructed_img = reconstructed_img.detach().cpu()
    reconstructed_pixels = reconstructed_pixels.detach().cpu()
    
    thermal_img_denorm = thermal_img_denorm.detach().cpu()
    reconstructed_img_denorm = reconstructed_img_denorm.detach().cpu()
    aligned_rgb_denorm = aligned_rgb_denorm.detach().cpu()
    
    # Mask 시각화 (16x16 grid)
    mask_2d = mask_thermal.reshape(1, 16, 16).float()  # [1, 16, 16]
    mask_img = torch.nn.functional.interpolate(
        mask_2d.unsqueeze(1), size=(224, 224), mode='nearest'
    ).squeeze(1)  # [1, 224, 224]
    
    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1
    axes[0, 0].imshow(thermal_img_denorm[0].cpu().permute(1, 2, 0).clip(0, 1))
    axes[0, 0].set_title('Original Thermal', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(mask_img[0].cpu(), cmap='RdYlGn_r', vmin=0, vmax=1)
    axes[0, 1].set_title(f'Mask (Masked={mask_thermal.float().mean()*100:.1f}%)', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # ★ 핵심: Hybrid 이미지 (Visible + Reconstructed)
    axes[0, 2].imshow(hybrid_img_denorm[0].cpu().permute(1, 2, 0).clip(0, 1))
    axes[0, 2].set_title('Hybrid (Vis+Recon)', fontsize=14, fontweight='bold', color='red')
    axes[0, 2].axis('off')
    
    # Row 2
    axes[1, 0].imshow(aligned_rgb_denorm[0].cpu().permute(1, 2, 0).clip(0, 1))
    axes[1, 0].set_title('Aligned RGB (Reference)', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Reconstruction only (masked 영역만)
    recon_only = reconstructed_img_denorm.clone()
    mask_expanded = mask_img.unsqueeze(1).expand(-1, 3, -1, -1)  # [1, 3, 224, 224]
    recon_only[mask_expanded < 0.5] = 0  # Visible region = black
    axes[1, 1].imshow(recon_only[0].cpu().permute(1, 2, 0).clip(0, 1))
    axes[1, 1].set_title('Reconstructed (Masked Only)', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Reconstruction error (masked 영역에서만)
    error = (thermal_img_denorm - reconstructed_img_denorm).abs().mean(dim=1)  # [1, 224, 224]
    error_masked = error.clone()
    error_masked[mask_img < 0.5] = 0  # Visible 영역은 0으로
    im = axes[1, 2].imshow(error_masked[0].cpu(), cmap='hot', vmin=0, vmax=0.3)
    axes[1, 2].set_title('Error (Masked Only)', fontsize=14, fontweight='bold')
    axes[1, 2].axis('off')
    plt.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
        plt.close()
    else:
        plt.show()
    
    return hybrid_img_denorm

def unpatchify_visual(patches, patch_size=14):
    """
    patches: [B, N, patch_size**2 * 3]
    return: [B, 3, H, W]
    """
    B = patches.shape[0]
    h = w = int(patches.shape[1] ** 0.5)  # 16
    
    patches = patches.reshape(B, h, w, patch_size, patch_size, 3)
    patches = torch.einsum('nhwpqc->nchpwq', patches)
    imgs = patches.reshape(B, 3, h * patch_size, w * patch_size)
    
    return imgs


# ===== Training Loop에서 사용 =====
def visualize_during_training(args, model, triplets_dl, device, epoch, save_dir='./visualizations', comment="default"):
    """Training 중간에 주기적으로 시각화"""
    import os
    timestamp = get_timestamp()
    save_subdir = os.path.join(save_dir, comment, timestamp)
    os.makedirs(save_subdir, exist_ok=True)
    
    model.eval()
    
    # 첫 번째 batch 가져오기
    images, _, _, aligned_rgbs = next(iter(triplets_dl))
    # Thermal query 1개만 추출 (첫 번째 thermal)
    thermal_img = images[0:1].to(device)  # [1, 3, 224, 224]
    paired_rgb = aligned_rgbs[0:1].to(device)  # [1, 3, 224, 224]
    
    save_path = f"{save_subdir}/epoch_{epoch:03d}.png"
    visualize_reconstruction(model, thermal_img, paired_rgb, device, save_path)
    
    model.train()
    
def visualize_reranking_comparison(args, eval_ds, 
                                   original_predictions, 
                                   reranked_predictions,
                                   reconstruction_losses_dict,
                                   positives_per_query,
                                   epoch,
                                   distances=None,
                                   reconstructed_images=None,
                                   save_dir='./rerank_visualizations',
                                   num_samples=2):
    """
    Reranking 전/후를 비교하는 시각화 + Loss 통계
    """
    import os
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import csv
    
    os.makedirs(save_dir, exist_ok=True)
    
    # ===== Loss 통계 계산 =====
    all_losses = []
    
    for query_idx, losses in reconstruction_losses_dict.items():
        all_losses.extend(losses)
    
    all_losses = np.array(all_losses)
    
    # 통계량
    stats = {
        'epoch': epoch,
        'mean': all_losses.mean(),
        'std': all_losses.std(),
    }
    
    # ===== CSV 저장 =====
    csv_path = os.path.join(save_dir, 'loss_stats.csv')
    file_exists = os.path.exists(csv_path)
    
    with open(csv_path, 'a', newline='') as f:
        fieldnames = ['epoch', 'mean', 'std']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        writer.writerow(stats)
    
    # ===== 그래프 =====
    import pandas as pd
    df = pd.read_csv(csv_path)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.errorbar(df['epoch'], df['mean'], yerr=df['std'],
                fmt='o-', linewidth=2, capsize=5, color='blue')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Reconstruction Loss', fontsize=12)
    ax.set_title('Loss Mean ± Std per Epoch', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_trend.png'), dpi=120)
    plt.close()
    
    # ===== 콘솔 출력 =====
    print(f"\n{'='*60}")
    print(f"Epoch {epoch} Loss: {stats['mean']:.6f} ± {stats['std']:.6f}")
    print(f"{'='*60}\n")
    
    # ===== 샘플링 전략 =====
    improved_queries = []
    worsened_queries = []
    unchanged_queries = []
    
    for q_idx in range(eval_ds.queries_num):
        orig_top1 = original_predictions[q_idx, 0]
        rerank_top1 = reranked_predictions[q_idx, 0]
        positives = positives_per_query[q_idx]
        
        if orig_top1 != rerank_top1:
            if rerank_top1 in positives:
                improved_queries.append(q_idx)
            else:
                worsened_queries.append(q_idx)
        else:
            unchanged_queries.append(q_idx)
    
    # 샘플링
    sample_indices = []
    
    if len(improved_queries) > 0:
        n_improved = min(num_samples // 2, len(improved_queries))
        sample_indices.extend(np.random.choice(improved_queries, n_improved, replace=False))
    
    if len(sample_indices) < num_samples and len(worsened_queries) > 0:
        n_worsened = min(num_samples - len(sample_indices), len(worsened_queries))
        sample_indices.extend(np.random.choice(worsened_queries, n_worsened, replace=False))
    
    if len(sample_indices) < num_samples:
        remaining = num_samples - len(sample_indices)
        all_remaining = list(set(range(eval_ds.queries_num)) - set(sample_indices))
        sample_indices.extend(np.random.choice(all_remaining, remaining, replace=False))
    
    # ===== 시각화 =====
    for sample_num, query_idx in enumerate(sample_indices):
        n_rows = 4 if reconstructed_images is not None else 3
        fig = plt.figure(figsize=(22, 6*n_rows//3))
        gs = fig.add_gridspec(n_rows, 7, hspace=0.35, wspace=0.3)
        
        # Query 이미지
        query_img = eval_ds.get_thermal_img(eval_ds.t_queries_paths[query_idx])
        query_img = cv2.resize(query_img, (224, 224))
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
        
        positives = positives_per_query[query_idx]
        
        # Row 0: Query
        ax_query = fig.add_subplot(gs[0, :])
        ax_query.imshow(query_img)
        ax_query.set_title(f'Query #{query_idx} (Thermal)\nPositives in DB: {len(positives)}', 
                          fontsize=14, fontweight='bold')
        ax_query.axis('off')
        
        # Row 1: Original (Faiss)
        for rank in range(5):
            ax = fig.add_subplot(gs[1, rank+1])
            
            pred_idx = original_predictions[query_idx, rank]
            db_img = eval_ds.get_rgb_img(eval_ds.rgb_database_paths[pred_idx])
            db_img = cv2.resize(db_img, (224, 224))
            db_img = cv2.cvtColor(db_img, cv2.COLOR_BGR2RGB)
            
            is_correct = pred_idx in positives
            
            # GPS 거리
            if hasattr(eval_ds, 'database_utms') and hasattr(eval_ds, 'queries_utms'):
                query_gps = eval_ds.queries_utms[query_idx]
                db_gps = eval_ds.database_utms[pred_idx]
                gps_distance = np.linalg.norm(query_gps - db_gps)
            else:
                gps_distance = None
            
            # Border
            border_color = 'green' if is_correct else 'red'
            border_width = 4 if is_correct else 2
            
            ax.imshow(db_img)
            rect = Rectangle((0, 0), 223, 223, linewidth=border_width, 
                           edgecolor=border_color, facecolor='none')
            ax.add_patch(rect)
            
            # Title
            title = f'Rank {rank+1}'
            if is_correct:
                title += ' ✓'
            
            if distances is not None:
                title += f'\nL2: {distances[query_idx, rank]:.2f}'
            
            if gps_distance is not None:
                title += f'\nGPS: {gps_distance:.1f}m'
            
            ax.set_title(title, fontsize=10, fontweight='bold', color=border_color)
            ax.axis('off')
        
        # Legend
        ax_legend1 = fig.add_subplot(gs[1, 6])
        ax_legend1.text(0.1, 0.7, 'Before\nReranking', fontsize=14, 
                       fontweight='bold', va='center')
        ax_legend1.text(0.1, 0.3, '(Faiss L2)', fontsize=11, 
                       style='italic', va='center')
        ax_legend1.axis('off')
        
        # Label
        ax_label1 = fig.add_subplot(gs[1, 0])
        ax_label1.text(0.5, 0.5, 'Original\nRetrieval', fontsize=12,
                      fontweight='bold', ha='center', va='center')
        ax_label1.axis('off')
        
        # Row 2: Reranked
        for rank in range(5):
            ax = fig.add_subplot(gs[2, rank+1])
            
            pred_idx = reranked_predictions[query_idx, rank]
            db_img = eval_ds.get_rgb_img(eval_ds.rgb_database_paths[pred_idx])
            db_img = cv2.resize(db_img, (224, 224))
            db_img = cv2.cvtColor(db_img, cv2.COLOR_BGR2RGB)
            
            is_correct = pred_idx in positives
            
            # GPS 거리
            if hasattr(eval_ds, 'database_utms') and hasattr(eval_ds, 'queries_utms'):
                query_gps = eval_ds.queries_utms[query_idx]
                db_gps = eval_ds.database_utms[pred_idx]
                gps_distance = np.linalg.norm(query_gps - db_gps)
            else:
                gps_distance = None
            
            # 원래 순위
            orig_rank = np.where(original_predictions[query_idx, :5] == pred_idx)[0]
            if len(orig_rank) > 0:
                rank_change = f"(R{orig_rank[0]+1}→R{rank+1})"
            else:
                rank_change = "(new)"
            
            # Border
            border_color = 'green' if is_correct else 'red'
            border_width = 4 if is_correct else 2
            
            ax.imshow(db_img)
            rect = Rectangle((0, 0), 223, 223, linewidth=border_width,
                           edgecolor=border_color, facecolor='none')
            ax.add_patch(rect)
            
            # Title
            title = f'Rank {rank+1}'
            if is_correct:
                title += ' ✓'
            title += f' {rank_change}'
            
            # Reconstruction loss
            recon_losses = reconstruction_losses_dict.get(query_idx, [0]*5)
            title += f'\nRecon: {recon_losses[rank]:.3f}'
            
            if gps_distance is not None:
                title += f'\nGPS: {gps_distance:.1f}m'
            
            ax.set_title(title, fontsize=10, fontweight='bold', color=border_color)
            ax.axis('off')
        
        # Legend
        ax_legend2 = fig.add_subplot(gs[2, 6])
        ax_legend2.text(0.1, 0.7, 'After\nReranking', fontsize=14,
                       fontweight='bold', va='center')
        ax_legend2.text(0.1, 0.3, '(Recon Loss)', fontsize=11,
                       style='italic', va='center')
        ax_legend2.axis('off')
        
        # Label
        ax_label2 = fig.add_subplot(gs[2, 0])
        ax_label2.text(0.5, 0.5, 'After\nReranking', fontsize=12,
                      fontweight='bold', ha='center', va='center')
        ax_label2.axis('off')
        
        # Row 3: Reconstructed (optional)
        if reconstructed_images is not None and query_idx in reconstructed_images:
            for rank in range(5):
                ax = fig.add_subplot(gs[3, rank+1])
                
                recon_img = reconstructed_images[query_idx][rank]
                
                if isinstance(recon_img, torch.Tensor):
                    recon_img = recon_img.cpu().numpy()
                
                if recon_img.shape[0] == 3:
                    recon_img = recon_img.transpose(1, 2, 0)
                
                recon_img = np.clip(recon_img, 0, 1)
                
                ax.imshow(recon_img)
                ax.set_title(f'Reconstructed R{rank+1}', fontsize=10)
                ax.axis('off')
            
            ax_label3 = fig.add_subplot(gs[3, 0])
            ax_label3.text(0.5, 0.5, 'Reconstructed\nThermal', fontsize=12,
                          fontweight='bold', ha='center', va='center')
            ax_label3.axis('off')
            
            ax_legend3 = fig.add_subplot(gs[3, 6])
            ax_legend3.text(0.1, 0.5, 'Decoder\nOutput', fontsize=11,
                           style='italic', va='center')
            ax_legend3.axis('off')
        
        # Overall title
        orig_top1 = original_predictions[query_idx, 0]
        rerank_top1 = reranked_predictions[query_idx, 0]
        
        if orig_top1 != rerank_top1:
            if rerank_top1 in positives:
                status = "🎉 IMPROVED (Wrong → Correct)"
                color = 'darkgreen'
            elif orig_top1 in positives:
                status = "⚠️ WORSENED (Correct → Wrong)"
                color = 'darkred'
            else:
                status = "🔄 CHANGED (Wrong → Wrong)"
                color = 'orange'
        else:
            if orig_top1 in positives:
                status = "✓ MAINTAINED (Correct → Correct)"
                color = 'blue'
            else:
                status = "− MAINTAINED (Wrong → Wrong)"
                color = 'gray'
        
        fig.suptitle(f'Query #{query_idx} - {status}', 
                    fontsize=16, fontweight='bold', color=color)
        
        # Save
        save_path = os.path.join(save_dir, f'epoch_{epoch:03d}_query_{query_idx:05d}.png')
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {save_path}")
    
    # ===== Summary =====
    summary_path = os.path.join(save_dir, f'epoch_{epoch:03d}_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Reranking Summary - Epoch {epoch}\n")
        f.write("="*50 + "\n\n")
        f.write(f"Total queries: {eval_ds.queries_num}\n")
        f.write(f"Improved (wrong→correct): {len(improved_queries)}\n")
        f.write(f"Worsened (correct→wrong): {len(worsened_queries)}\n")
        f.write(f"Unchanged: {len(unchanged_queries)}\n\n")
        
        # Top-1 accuracy
        orig_correct = sum([1 for q in range(eval_ds.queries_num) if original_predictions[q,0] in positives_per_query[q]])
        rerank_correct = sum([1 for q in range(eval_ds.queries_num) if reranked_predictions[q,0] in positives_per_query[q]])
        
        f.write(f"Top-1 Accuracy:\n")
        f.write(f"  Before: {orig_correct}/{eval_ds.queries_num} ({100*orig_correct/eval_ds.queries_num:.2f}%)\n")
        f.write(f"  After:  {rerank_correct}/{eval_ds.queries_num} ({100*rerank_correct/eval_ds.queries_num:.2f}%)\n")
        f.write(f"  Delta:  {rerank_correct-orig_correct:+d} ({100*(rerank_correct-orig_correct)/eval_ds.queries_num:+.2f}%)\n")
    
    print(f"Saved summary: {summary_path}")

def save_simple_cross_attn(attn_map, save_path, query_idx=None):
    """
    Args:
        attn_map: [B, N, M] 또는 [N, M] 텐서 (CPU/GPU 상관없음)
        save_path: 저장할 파일 경로 (예: './test.png')
        query_idx: (선택) 보고 싶은 Query 패치 번호. 안 넣으면 정중앙을 봅니다.
    """
    # 1. 텐서 정리 (GPU -> CPU, Batch 차원 제거)
    if isinstance(attn_map, torch.Tensor):
        attn_map = attn_map.detach().cpu().numpy()
    
    # [B, N, M]인 경우 첫 번째 배치를 선택
    if attn_map.ndim == 3:
        attn_map = attn_map[0]  # [N, M]
        
    N, M = attn_map.shape
    
    # 2. Grid 크기 자동 계산 (정사각형 가정)
    # M (Key 개수) = H * W
    grid_size = int(np.sqrt(M))
    
    # 3. Query 선택 (기본값: 정중앙 패치)
    if query_idx is None:
        query_idx = N // 2  # 중앙 인덱스
        
    # 4. 해당 Query가 바라보는 Attention Map 추출 [M] -> [H, W]
    heatmap = attn_map[query_idx, :]
    heatmap = heatmap.reshape(grid_size, grid_size)
    
    # 5. 보기 좋게 해상도 키우기 (Interpolation)
    # 16x16 같은 저해상도를 256x256으로 부드럽게 키움
    heatmap = cv2.resize(heatmap, (256, 256), interpolation=cv2.INTER_NEAREST)
    
    # 6. Min-Max 정규화 (0~1) - 선명하게 보기 위해
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    # 7. 이미지 저장 (Jet colormap 적용)
    # plt.imsave는 자동으로 컬러맵을 입혀서 저장해줍니다.
    plt.imsave(save_path, heatmap, cmap='jet')
    
    print(f"Saved attention map to {save_path} (Query Index: {query_idx})")

def save_mnn_visualization(eval_ds, query_indices, top_k_db_indices, 
                          mutual_matches_list, rerank_scores,
                          thermal_cross_attn_maps, rgb_cross_attn_maps,
                          save_dir, epoch, top_k=5):
    """
    MNN matching + Cross-Attention 시각화 (Top-K 전체)
    
    Args:
        eval_ds: dataset
        query_indices: list of query indices to visualize
        top_k_db_indices: [num_queries, K] - Top-K DB indices
        mutual_matches_list: list of (matches_i, matches_j, conf) tuples
        rerank_scores: [num_queries, K] - Reranking scores
        thermal_cross_attn_maps: [num_queries*K, 256, 256] - Thermal→RGB attention
        rgb_cross_attn_maps: [num_queries*K, 256, 256] - RGB→Thermal attention
        save_dir: save directory
        epoch: current epoch
        top_k: number of top retrievals to show (default: 5)
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.gridspec import GridSpec
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Denormalization helper
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    def denormalize(img_tensor):
        """[3, H, W] → [H, W, 3] numpy, range [0, 1]"""
        img = (img_tensor * std + mean).permute(1, 2, 0).cpu().numpy()
        return np.clip(img, 0, 1)
    
    def patch_to_pixel(patch_idx, patch_size=14, grid_size=16):
        """Patch index → pixel coordinates (center)"""
        row, col = divmod(patch_idx, grid_size)
        y = row * patch_size + patch_size // 2
        x = col * patch_size + patch_size // 2
        return x, y
    
    def aggregate_attention(attn_map, method='max'):
        """
        Aggregate [256, 256] attention → [16, 16] heatmap
        
        Args:
            attn_map: [num_queries, num_keys] = [256, 256]
            method: 'max' (각 key의 max attention) or 'mean' (평균)
        """
        if method == 'max':
            aggregated = attn_map.max(dim=0)[0]  # [256]
        elif method == 'mean':
            aggregated = attn_map.mean(dim=0)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        heatmap = aggregated.cpu().numpy().reshape(16, 16)
        return heatmap
    
    # Process each query
    for list_idx, query_idx in enumerate(query_indices):
        # === Load query image ===
        thermal_abs_idx = eval_ds.database_num + query_idx
        thermal_img = denormalize(eval_ds[thermal_abs_idx][0])
        
        db_indices = top_k_db_indices[list_idx]
        scores = rerank_scores[list_idx]
        matches = mutual_matches_list[list_idx]
        
        # Load all Top-K RGB images
        rgb_imgs = [denormalize(eval_ds[int(db_idx)][0]) for db_idx in db_indices[:top_k]]
        
        # === Create large figure ===
        # Layout: 
        # Row 0: Query + Top-5 RGB images (6 cols)
        # Row 1-5: Each rank's attention maps (6 cols each)
        # Row 6: MNN matching visualization (full width)
        
        fig = plt.figure(figsize=(30, 36))
        gs = GridSpec(7, 6, figure=fig, hspace=0.4, wspace=0.3)
        
        # ==========================================
        # Row 0: Query + Top-K Retrieved Images
        # ==========================================
        ax_query = fig.add_subplot(gs[0, 0])
        ax_query.imshow(thermal_img)
        ax_query.set_title(f'Query {query_idx}\n(Thermal)', fontsize=14, fontweight='bold')
        ax_query.axis('off')
        
        for k in range(min(top_k, 5)):
            ax = fig.add_subplot(gs[0, k+1])
            ax.imshow(rgb_imgs[k])
            ax.set_title(f'Rank {k+1}\nScore: {scores[k]:.3f}', fontsize=12, fontweight='bold')
            ax.axis('off')
            
            # Highlight if it's the best score
            if k == 0:
                for spine in ax.spines.values():
                    spine.set_edgecolor('lime')
                    spine.set_linewidth(4)
        
        # ==========================================
        # Row 1-5: Attention Maps for Each Rank
        # ==========================================
        for k in range(min(top_k, 5)):
            row_idx = k + 1
            
            # Get attention maps for this rank
            attn_idx = list_idx * top_k + k
            thermal_attn_raw = thermal_cross_attn_maps[attn_idx]  # [256, 256]
            rgb_attn_raw = rgb_cross_attn_maps[attn_idx]
            
            # Aggregate to spatial heatmap
            thermal_heatmap = aggregate_attention(thermal_attn_raw, method='max')  # [16, 16]
            rgb_heatmap = aggregate_attention(rgb_attn_raw, method='max')
            
            # Mutual agreement
            mutual_attn = (thermal_attn_raw * rgb_attn_raw.T).cpu().numpy()  # [256, 256]
            mutual_heatmap = mutual_attn.mean(axis=0).reshape(16, 16)
            
            # Resize for visualization
            thermal_heatmap_vis = cv2.resize(thermal_heatmap, (224, 224), interpolation=cv2.INTER_LINEAR)
            rgb_heatmap_vis = cv2.resize(rgb_heatmap, (224, 224), interpolation=cv2.INTER_LINEAR)
            mutual_heatmap_vis = cv2.resize(mutual_heatmap, (224, 224), interpolation=cv2.INTER_LINEAR)
            
            # Col 0: Thermal→RGB attention
            ax1 = fig.add_subplot(gs[row_idx, 0])
            im1 = ax1.imshow(thermal_heatmap_vis, cmap='hot', vmin=0, vmax=thermal_heatmap.max())
            ax1.set_title(f'Rank {k+1}: Thermal→RGB\n(Max Attn)', fontsize=11, fontweight='bold')
            ax1.axis('off')
            plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
            
            # Col 1: RGB→Thermal attention
            ax2 = fig.add_subplot(gs[row_idx, 1])
            im2 = ax2.imshow(rgb_heatmap_vis, cmap='hot', vmin=0, vmax=rgb_heatmap.max())
            ax2.set_title(f'Rank {k+1}: RGB→Thermal\n(Max Attn)', fontsize=11, fontweight='bold')
            ax2.axis('off')
            plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
            
            # Col 2: Mutual agreement
            ax3 = fig.add_subplot(gs[row_idx, 2])
            im3 = ax3.imshow(mutual_heatmap_vis, cmap='hot', vmin=0, vmax=mutual_heatmap.max())
            ax3.set_title(f'Rank {k+1}: Mutual\n(T→R × R→T)', fontsize=11, fontweight='bold')
            ax3.axis('off')
            plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
            
            # Col 3: Thermal→RGB overlay on RGB image
            ax4 = fig.add_subplot(gs[row_idx, 3])
            heatmap_overlay = plt.cm.hot(thermal_heatmap_vis / (thermal_heatmap.max() + 1e-8))[:, :, :3]
            overlay_img = 0.5 * rgb_imgs[k] + 0.5 * heatmap_overlay
            ax4.imshow(overlay_img)
            ax4.set_title(f'Rank {k+1}: T→R Overlay\non RGB', fontsize=11, fontweight='bold')
            ax4.axis('off')
            
            # Col 4: RGB→Thermal overlay on Thermal image
            ax5 = fig.add_subplot(gs[row_idx, 4])
            rgb_heatmap_overlay = plt.cm.hot(rgb_heatmap_vis / (rgb_heatmap.max() + 1e-8))[:, :, :3]
            thermal_overlay_img = 0.5 * thermal_img + 0.5 * rgb_heatmap_overlay
            ax5.imshow(thermal_overlay_img)
            ax5.set_title(f'Rank {k+1}: R→T Overlay\non Thermal', fontsize=11, fontweight='bold')
            ax5.axis('off')
            
            # Col 5: Top-3 confident patch matches for this rank
            ax6 = fig.add_subplot(gs[row_idx, 5])
            
            # For Rank 1 (k=0), show MNN matches
            if k == 0 and len(matches[0]) > 0:
                matches_i, matches_j, matches_conf = matches
                
                # Get top-3 confident matches
                top_3_idx = np.argsort(matches_conf)[-3:][::-1] if len(matches_conf) >= 3 else range(len(matches_conf))
                
                # Show combined image with match lines
                combined_small = np.hstack([
                    cv2.resize(thermal_img, (112, 112)),
                    cv2.resize(rgb_imgs[0], (112, 112))
                ])
                ax6.imshow(combined_small)
                
                # Draw top matches
                for idx in top_3_idx:
                    i, j, conf = matches_i[idx], matches_j[idx], matches_conf[idx]
                    
                    x1, y1 = patch_to_pixel(i, patch_size=7, grid_size=16)
                    x2, y2 = patch_to_pixel(j, patch_size=7, grid_size=16)
                    x2 += 112  # Offset
                    
                    color = plt.cm.hot(min(conf / 0.1, 1.0))
                    ax6.plot([x1, x2], [y1, y2], color=color, linewidth=2, alpha=0.8)
                    ax6.scatter([x1, x2], [y1, y2], c=['cyan', 'lime'], 
                               s=20, zorder=5, edgecolors='white', linewidths=0.8)
                
                ax6.axvline(x=112, color='white', linewidth=2, linestyle='--', alpha=0.6)
                ax6.set_title(f'Top-3 MNN Matches\n({len(matches_i)} total)', fontsize=11, fontweight='bold')
                ax6.axis('off')
            else:
                # For other ranks, show attention distribution histogram
                attn_values = thermal_attn_raw.cpu().numpy().flatten()
                ax6.hist(attn_values, bins=50, color=f'C{k}', alpha=0.7, edgecolor='black')
                ax6.set_title(f'Rank {k+1}: Attention\nDistribution', fontsize=11, fontweight='bold')
                ax6.set_xlabel('Attention Value', fontsize=9)
                ax6.set_ylabel('Frequency', fontsize=9)
                ax6.grid(alpha=0.3)
        
        # ==========================================
        # Row 6: Detailed MNN Matching (Full Width)
        # ==========================================
        ax_mnn = fig.add_subplot(gs[6, :])
        
        if len(matches[0]) > 0:
            matches_i, matches_j, matches_conf = matches
            
            # Side-by-side images
            combined = np.hstack([thermal_img, rgb_imgs[0]])
            ax_mnn.imshow(combined)
            
            # Draw all matches (or top N)
            num_draw = min(100, len(matches_i))
            sorted_idx = np.argsort(matches_conf)[-num_draw:][::-1]  # Top confident matches
            
            for idx in sorted_idx:
                i, j, conf = matches_i[idx], matches_j[idx], matches_conf[idx]
                
                x1, y1 = patch_to_pixel(i)
                x2, y2 = patch_to_pixel(j)
                x2 += 224  # Offset for right image
                
                # Color by confidence
                color = plt.cm.hot(min(conf / 0.1, 1.0))
                alpha = 0.3 + 0.7 * (conf / max(matches_conf))  # More confident = more opaque
                
                ax_mnn.plot([x1, x2], [y1, y2], color=color, linewidth=1.5, alpha=alpha)
                ax_mnn.scatter([x1], [y1], c='cyan', s=20, zorder=5, edgecolors='white', linewidths=0.5)
                ax_mnn.scatter([x2], [y2], c='lime', s=20, zorder=5, edgecolors='white', linewidths=0.5)
            
            # Divider line
            ax_mnn.axvline(x=224, color='white', linewidth=4, linestyle='--', alpha=0.9)
            
            # Statistics
            avg_conf = np.mean(matches_conf)
            max_conf = np.max(matches_conf)
            title_str = (f'Mutual Nearest Neighbor Matches\n'
                        f'Total: {len(matches_i)} pairs | Shown: Top {num_draw} | '
                        f'Avg Conf: {avg_conf:.3f} | Max Conf: {max_conf:.3f}')
            ax_mnn.set_title(title_str, fontsize=14, fontweight='bold')
            ax_mnn.axis('off')
        else:
            ax_mnn.text(0.5, 0.5, 'No Mutual Nearest Neighbor Matches Found', 
                       ha='center', va='center', fontsize=24, color='red', 
                       transform=ax_mnn.transAxes, fontweight='bold')
            ax_mnn.axis('off')
        
        # ==========================================
        # Overall title
        # ==========================================
        fig.suptitle(f'Query {query_idx} - Complete Cross-Modal Matching Analysis (Epoch {epoch})', 
                    fontsize=18, fontweight='bold', y=0.995)
        
        # Save
        save_path = os.path.join(save_dir, f'epoch{epoch:03d}_query{query_idx:05d}_full.png')
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close()
        
        print(f"[{list_idx+1}/{len(query_indices)}] Saved: {save_path}")
    
    print(f"✓ Visualization complete: {len(query_indices)} queries saved to {save_dir}")
