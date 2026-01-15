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
        _, patch_thermal, recon_loss, mask_thermal, masked_patch_thermal = model.module.forward_model(thermal_img, modality='thermal', paired_rgb=paired_rgb)
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
    if args.use_contrastive_recon_loss:
        assert images.size(0) % args.train_batch_size == 0
        size_of_batch = int(images.size(0) / args.train_batch_size)
        train_batch_size = args.train_batch_size
        
        pos_rgbs = [images[idx] for idx in range(1, images.size(0), size_of_batch)]
        pos_rgbs = torch.stack(pos_rgbs)
        aligned_rgbs = pos_rgbs
        
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
                          thermal_cross_attn_maps, rgb_cross_attn_maps,  # ← 추가
                          save_dir, epoch):
    """
    MNN matching + Cross-Attention 시각화
    
    Args:
        eval_ds: dataset
        query_indices: list of query indices
        top_k_db_indices: [num_queries, K] - Top-K DB indices
        mutual_matches_list: list of (matches_i, matches_j, conf) tuples
        rerank_scores: [num_queries, K] - Reranking scores
        thermal_cross_attn_maps: [num_queries*K, 256, 256] - Thermal→RGB attention
        rgb_cross_attn_maps: [num_queries*K, 256, 256] - RGB→Thermal attention
        save_dir: save directory
        epoch: current epoch
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    os.makedirs(save_dir, exist_ok=True)
    
    for q_idx, (query_idx, db_indices, matches, scores) in enumerate(
        zip(query_indices, top_k_db_indices, mutual_matches_list, rerank_scores)
    ):
        # Load images
        thermal_abs_idx = eval_ds.database_num + query_idx
        thermal_img = eval_ds[thermal_abs_idx][0]  # [3, 224, 224]
        
        # Denormalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        thermal_img = (thermal_img * std + mean).permute(1, 2, 0).numpy()
        thermal_img = np.clip(thermal_img, 0, 1)
        
        # Top-1 RGB
        db_idx = db_indices[0]
        rgb_img = eval_ds[int(db_idx)][0]
        rgb_img = (rgb_img * std + mean).permute(1, 2, 0).numpy()
        rgb_img = np.clip(rgb_img, 0, 1)
        
        # ========== Figure with 3 rows ==========
        fig = plt.figure(figsize=(24, 18))
        gs = fig.add_gridspec(3, 6, hspace=0.3, wspace=0.3)
        
        # ===== Row 1: Top-5 RGB Retrieval =====
        ax_query = fig.add_subplot(gs[0, 0])
        ax_query.imshow(thermal_img)
        ax_query.set_title(f'Query {query_idx}\n(Thermal)', fontsize=12, fontweight='bold')
        ax_query.axis('off')
        
        for k in range(5):
            ax = fig.add_subplot(gs[0, k+1])
            db_idx_k = db_indices[k]
            rgb_k = eval_ds[int(db_idx_k)][0]
            rgb_k = (rgb_k * std + mean).permute(1, 2, 0).numpy()
            rgb_k = np.clip(rgb_k, 0, 1)
            
            ax.imshow(rgb_k)
            ax.set_title(f'Rank {k+1}\nScore: {scores[k]:.3f}', fontsize=10)
            ax.axis('off')
        
        # ===== Row 2: Cross-Attention Maps =====
        # Thermal → RGB attention (Top-1)
        ax_thermal_attn = fig.add_subplot(gs[1, 0])
        thermal_attn = thermal_cross_attn_maps[q_idx * 5]  # Top-1
        # Average over all query patches
        thermal_attn_avg = thermal_attn.max(dim=0)[0].cpu().numpy().reshape(16, 16) # .mean(dim=0)
        thermal_attn_resized = cv2.resize(thermal_attn_avg, (224, 224), interpolation=cv2.INTER_NEAREST)
        
        im1 = ax_thermal_attn.imshow(thermal_attn_resized, cmap='hot', vmin=0, vmax=thermal_attn_avg.max())
        ax_thermal_attn.set_title('Thermal→RGB\n(Avg Attention)', fontsize=11, fontweight='bold')
        ax_thermal_attn.axis('off')
        plt.colorbar(im1, ax=ax_thermal_attn, fraction=0.046, pad=0.04)
        
        # RGB → Thermal attention (Top-1)
        ax_rgb_attn = fig.add_subplot(gs[1, 1])
        rgb_attn = rgb_cross_attn_maps[q_idx * 5]  # Top-1
        # Average over all RGB patches
        rgb_attn_avg = rgb_attn.max(dim=0)[0].cpu().numpy().reshape(16, 16) # .mean(dim=0)
        rgb_attn_resized = cv2.resize(rgb_attn_avg, (224, 224), interpolation=cv2.INTER_NEAREST)
        
        im2 = ax_rgb_attn.imshow(rgb_attn_resized, cmap='hot', vmin=0, vmax=rgb_attn_avg.max())
        ax_rgb_attn.set_title('RGB→Thermal\n(Avg Attention)', fontsize=11, fontweight='bold')
        ax_rgb_attn.axis('off')
        plt.colorbar(im2, ax=ax_rgb_attn, fraction=0.046, pad=0.04)
        
        # Mutual Agreement (element-wise product)
        ax_mutual = fig.add_subplot(gs[1, 2])
        mutual_attn = (thermal_attn * rgb_attn.transpose(0, 1)).cpu().numpy()
        mutual_attn_avg = mutual_attn.mean(axis=0).reshape(16, 16)
        mutual_attn_resized = cv2.resize(mutual_attn_avg, (224, 224), interpolation=cv2.INTER_NEAREST)
        
        im3 = ax_mutual.imshow(mutual_attn_resized, cmap='hot', vmin=0, vmax=mutual_attn_avg.max())
        ax_mutual.set_title('Mutual Agreement\n(T→R × R→T)', fontsize=11, fontweight='bold')
        ax_mutual.axis('off')
        plt.colorbar(im3, ax=ax_mutual, fraction=0.046, pad=0.04)
        
        # Patch-level attention heatmaps (sample patches)
        # Select 3 sample patches with high attention
        if len(matches[0]) > 0:
            matches_i, matches_j, matches_conf = matches
            
            # Top-3 confident matches
            top_3_indices = np.argsort(matches_conf)[-3:][::-1]
            
            for idx, sample_idx in enumerate(top_3_indices):
                if sample_idx >= len(matches_i):
                    continue
                    
                ax = fig.add_subplot(gs[1, 3+idx])
                
                i = matches_i[sample_idx]
                j = matches_j[sample_idx]
                conf = matches_conf[sample_idx]
                
                # Show attention for this specific patch
                patch_attn = thermal_attn[i].cpu().numpy().reshape(16, 16)
                patch_attn_resized = cv2.resize(patch_attn, (224, 224), interpolation=cv2.INTER_NEAREST)
                
                # Overlay on RGB
                rgb_overlay = rgb_img.copy()
                heatmap = plt.cm.hot(patch_attn_resized)[:, :, :3]
                rgb_overlay = 0.6 * rgb_overlay + 0.4 * heatmap
                
                ax.imshow(rgb_overlay)
                
                # Mark the matched patch
                row_j = j // 16
                col_j = j % 16
                y_j = row_j * 14
                x_j = col_j * 14
                rect = patches.Rectangle((x_j, y_j), 14, 14, linewidth=2, edgecolor='cyan', facecolor='none')
                ax.add_patch(rect)
                
                ax.set_title(f'Patch {i}→{j}\nConf: {conf:.3f}', fontsize=9)
                ax.axis('off')
        
        # ===== Row 3: MNN Matching =====
        if len(matches[0]) > 0:
            matches_i, matches_j, matches_conf = matches
            
            # Side-by-side with matches
            combined = np.hstack([thermal_img, rgb_img])
            
            ax = fig.add_subplot(gs[2, :])
            ax.imshow(combined)
            
            # Draw matches
            patch_size = 14
            img_w = 224
            
            def patch_to_pixel(patch_idx):
                row = patch_idx // 16
                col = patch_idx % 16
                y = row * patch_size + patch_size // 2
                x = col * patch_size + patch_size // 2
                return x, y
            
            # Draw lines
            num_matches = min(50, len(matches_i))
            for idx in range(num_matches):
                i = matches_i[idx]
                j = matches_j[idx]
                conf = matches_conf[idx]
                
                x1, y1 = patch_to_pixel(i)
                x2, y2 = patch_to_pixel(j)
                x2 += img_w  # RGB는 오른쪽
                
                # Color by confidence
                color = plt.cm.hot(conf / 0.1)
                
                ax.plot([x1, x2], [y1, y2], 
                       color=color, linewidth=1.5, alpha=0.7)
                ax.scatter([x1], [y1], c='cyan', s=15, zorder=5, edgecolors='white', linewidths=0.5)
                ax.scatter([x2], [y2], c='lime', s=15, zorder=5, edgecolors='white', linewidths=0.5)
            
            ax.set_title(f'MNN Matches: {len(matches_i)} pairs (showing top {num_matches})', 
                        fontsize=14, fontweight='bold')
            ax.axis('off')
            
            # Vertical divider
            ax.axvline(x=img_w, color='white', linewidth=3, linestyle='--', alpha=0.8)
        else:
            ax = fig.add_subplot(gs[2, :])
            ax.text(0.5, 0.5, 'No mutual matches found', 
                   ha='center', va='center', fontsize=20, color='red')
            ax.axis('off')
        
        plt.suptitle(f'Query {query_idx} - Cross-Modal Matching Analysis', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'epoch_{epoch:03d}_query_{query_idx:05d}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved MNN visualization: {save_path}")
