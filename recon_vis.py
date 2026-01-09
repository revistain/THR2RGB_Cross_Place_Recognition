# recon_vis.py
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
from utils import get_timestamp
import cv2

# recon_vis.py

def visualize_reconstruction(model, thermal_img, aligned_rgb, device='cuda', save_path=None, return_attention=False):
    """
    Args:
        model: CrossModalVPR_Net
        thermal_img: [1, 3, 224, 224] - single thermal image
        aligned_rgb: [1, 3, 224, 224] - aligned RGB image
        device: 'cuda' or 'cpu'
        save_path: Optional path to save the figure
        return_attention: bool - Attention map 시각화 여부
    """
    model.eval()
    
    with torch.no_grad():
        # 1. Patch embedding
        thermal_patch = model.module.thermal_backbone.patch_embed(thermal_img)
        B, N, D = thermal_patch.shape  # [1, 256, 768]
        
        # 2. Positional embedding
        pos_tokens = model.module.thermal_backbone.pos_embed[:, 1:, :]
        pos_embed_grid = pos_tokens.reshape(1, 37, 37, 768).permute(0, 3, 1, 2)
        pos_embed_resized = torch.nn.functional.interpolate(
            pos_embed_grid, size=(16, 16), mode='bicubic', align_corners=False
        )
        pos_embed_final = pos_embed_resized.permute(0, 2, 3, 1).flatten(1, 2)
        thermal_patch = thermal_patch + pos_embed_final
        
        # 3. Masking
        mask = model.module.mask_generator(thermal_patch)  # [1, 256]
        thermal_visible = thermal_patch[~mask].reshape(B, -1, D)
        
        # 4. Encoder (visible only)
        for blk in model.module.thermal_backbone.blocks:
            thermal_visible = blk(thermal_visible)
        thermal_visible = model.module.thermal_backbone.norm(thermal_visible)
        
        # 5. RGB features
        rgb_full = model.module.rgb_backbone(aligned_rgb)
        rgb_full = rgb_full["x_norm_patchtokens"]
        
        # 6. Mask token expansion
        mask_tokens = model.module.mask_token.expand(B, N, -1)
        thermal_full = mask_tokens.clone()
        thermal_full[0, ~mask[0]] = thermal_visible[0]
        
        # 7. Decoder positional encoding
        thermal_full = thermal_full + model.module.decoder_pos_embed
        rgb_full = rgb_full + model.module.decoder_pos_embed
        
        # ========== 8. Decoder (Attention 수집) ==========
        attention_maps = []
        for layer_idx, blk in enumerate(model.module.decoder_blocks):
            thermal_full = blk(thermal_full, rgb_full, return_attention=return_attention)
            
            # 마지막 3개 layer의 attention만 저장
            if return_attention and layer_idx >= len(model.module.decoder_blocks) - 3:
                if hasattr(blk, 'cross_attn_weights') and blk.cross_attn_weights is not None:
                    attention_maps.append(blk.cross_attn_weights.detach().cpu())
        
        thermal_full = model.module.decoder_norm(thermal_full)
        # =================================================

        # ========== Confidence Map 추가 ==========
        if hasattr(model.module, 'confidence_head'):
            confidence_map = model.module.confidence_head(thermal_full).squeeze().cpu()  # [256]
        else:
            confidence_map = None
        # ==========================================

        # 9. Prediction
        reconstructed_patches = model.module.prediction_head(thermal_full)  # [1, 256, 588]
        
        # 10. Unpatchify
        reconstructed_img = unpatchify_visual(reconstructed_patches, patch_size=14)  # [1, 3, 224, 224]
        
        # 11. 원본 이미지도 patchify
        original_patches = model.module.patchify(thermal_img)  # [1, 256, 588]
        
    # Denormalize (ImageNet stats 사용했다고 가정)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    
    thermal_img_denorm = thermal_img * std + mean
    aligned_rgb_denorm = aligned_rgb * std + mean
    reconstructed_img_denorm = reconstructed_img * std + mean
    
    # ===== 핵심: Visible + Reconstructed 합치기 =====
    hybrid_patches = original_patches.clone()  # [1, 256, 588]
    hybrid_patches[mask] = reconstructed_patches[mask]
    
    hybrid_img = unpatchify_visual(hybrid_patches, patch_size=14)
    hybrid_img_denorm = hybrid_img * std + mean
    
    # Mask 시각화 (16x16 grid)
    mask_2d = mask.reshape(1, 16, 16).float()
    mask_img = torch.nn.functional.interpolate(
        mask_2d.unsqueeze(1), size=(224, 224), mode='nearest'
    ).squeeze(1)
    
    # ========== Plot (Attention 포함 여부에 따라) ==========
    if return_attention and len(attention_maps) > 0:
        # Attention 있으면 3x4 grid
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Row 1: 기본 시각화
        ax00 = fig.add_subplot(gs[0, 0])
        ax00.imshow(thermal_img_denorm[0].cpu().permute(1, 2, 0).clip(0, 1))
        ax00.set_title('Original Thermal', fontsize=14, fontweight='bold')
        ax00.axis('off')
        
        ax01 = fig.add_subplot(gs[0, 1])
        ax01.imshow(mask_img[0].cpu(), cmap='RdYlGn_r', vmin=0, vmax=1)
        ax01.set_title(f'Mask ({mask.float().mean()*100:.1f}%)', fontsize=14, fontweight='bold')
        ax01.axis('off')
        
        ax02 = fig.add_subplot(gs[0, 2])
        ax02.imshow(hybrid_img_denorm[0].cpu().permute(1, 2, 0).clip(0, 1))
        ax02.set_title('Hybrid (Vis+Recon)', fontsize=14, fontweight='bold', color='red')
        ax02.axis('off')
        
        ax03 = fig.add_subplot(gs[0, 3])
        ax03.imshow(aligned_rgb_denorm[0].cpu().permute(1, 2, 0).clip(0, 1))
        ax03.set_title('Aligned RGB', fontsize=14, fontweight='bold')
        ax03.axis('off')
        
        # Row 2: Reconstruction + Error + Token PCA
        ax10 = fig.add_subplot(gs[1, 0])
        recon_only = reconstructed_img_denorm.clone()
        mask_expanded = mask_img.unsqueeze(1).expand(-1, 3, -1, -1)
        recon_only[mask_expanded < 0.5] = 0
        ax10.imshow(recon_only[0].cpu().permute(1, 2, 0).clip(0, 1))
        ax10.set_title('Reconstructed (Masked)', fontsize=14, fontweight='bold')
        ax10.axis('off')
        
        ax11 = fig.add_subplot(gs[1, 1])
        error = (thermal_img_denorm - reconstructed_img_denorm).abs().mean(dim=1)
        error_masked = error.clone()
        error_masked[mask_img < 0.5] = 0
        im = ax11.imshow(error_masked[0].cpu(), cmap='hot', vmin=0, vmax=0.3)
        ax11.set_title('Error (Masked)', fontsize=14, fontweight='bold')
        ax11.axis('off')
        plt.colorbar(im, ax=ax11, fraction=0.046, pad=0.04)
        
        # Token PCA
        ax12 = fig.add_subplot(gs[1, 2])
        tokens = thermal_full[0].cpu().numpy()  # [256, 768]
        from sklearn.decomposition import PCA
        pca = PCA(n_components=3)
        tokens_pca = pca.fit_transform(tokens)
        tokens_pca = (tokens_pca - tokens_pca.min()) / (tokens_pca.max() - tokens_pca.min() + 1e-8)
        tokens_pca_img = tokens_pca.reshape(16, 16, 3)
        tokens_pca_img = cv2.resize(tokens_pca_img, (224, 224), interpolation=cv2.INTER_NEAREST)
        ax12.imshow(tokens_pca_img)
        ax12.set_title(f'Token PCA\n(Var: {pca.explained_variance_ratio_.sum():.2f})', fontsize=12)
        ax12.axis('off')
        
        # Confidence
        if confidence_map is not None:
            ax13 = fig.add_subplot(gs[1, 3])
            from matplotlib import cm as mpl_cm
            conf_map_2d = confidence_map.numpy().reshape(16, 16)
            conf_resized = cv2.resize(conf_map_2d, (224, 224), interpolation=cv2.INTER_NEAREST)
            conf_colored = (mpl_cm.jet(conf_resized)[:, :, :3] * 255).astype(np.uint8)
            ax13.imshow(conf_colored)
            ax13.set_title(f'Confidence\n(μ={conf_map_2d.mean():.3f})', fontsize=12)
            ax13.axis('off')
        
        # Row 3: Attention Maps (마지막 3 layers)
        for i, attn in enumerate(attention_maps):
            ax = fig.add_subplot(gs[2, i])
            
            # [1, 256, 256] → [256, 256]
            attn_map = attn[0].numpy()
            
            # Average over query positions
            attn_avg = attn_map.mean(axis=0).reshape(16, 16)
            attn_resized = cv2.resize(attn_avg, (224, 224), interpolation=cv2.INTER_NEAREST)
            
            im = ax.imshow(attn_resized, cmap='hot', vmin=0, vmax=attn_avg.max())
            layer_num = len(model.module.decoder_blocks) - 3 + i
            ax.set_title(f'Cross-Attn L{layer_num+1}\n(Thermal→RGB)', fontsize=12)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.suptitle('Decoder Analysis with Attention Maps', fontsize=16, fontweight='bold')
        
    else:
        # Attention 없으면 기존 2x3 grid
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Row 1
        axes[0, 0].imshow(thermal_img_denorm[0].cpu().permute(1, 2, 0).clip(0, 1))
        axes[0, 0].set_title('Original Thermal', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(mask_img[0].cpu(), cmap='RdYlGn_r', vmin=0, vmax=1)
        axes[0, 1].set_title(f'Mask (Masked={mask.float().mean()*100:.1f}%)', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(hybrid_img_denorm[0].cpu().permute(1, 2, 0).clip(0, 1))
        axes[0, 2].set_title('Hybrid (Vis+Recon)', fontsize=14, fontweight='bold', color='red')
        axes[0, 2].axis('off')
        
        # Row 2
        axes[1, 0].imshow(aligned_rgb_denorm[0].cpu().permute(1, 2, 0).clip(0, 1))
        axes[1, 0].set_title('Aligned RGB (Reference)', fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')
        
        recon_only = reconstructed_img_denorm.clone()
        mask_expanded = mask_img.unsqueeze(1).expand(-1, 3, -1, -1)
        recon_only[mask_expanded < 0.5] = 0
        axes[1, 1].imshow(recon_only[0].cpu().permute(1, 2, 0).clip(0, 1))
        axes[1, 1].set_title('Reconstructed (Masked Only)', fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')
        
        error = (thermal_img_denorm - reconstructed_img_denorm).abs().mean(dim=1)
        error_masked = error.clone()
        error_masked[mask_img < 0.5] = 0
        im = axes[1, 2].imshow(error_masked[0].cpu(), cmap='hot', vmin=0, vmax=0.3)
        axes[1, 2].set_title('Error (Masked Only)', fontsize=14, fontweight='bold')
        axes[1, 2].axis('off')
        plt.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.04)
    # ======================================================
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
        plt.close()
    else:
        plt.show()
    
    # ========== Confidence 시각화 추가 (맨 끝) ==========
    if hasattr(model.module, 'confidence_head') and save_path is not None and not return_attention:
        conf_save_path = save_path.replace('.png', '_confidence.png')
        save_confidence_vis_simple(
            thermal_img=thermal_img[0],
            rgb_img=aligned_rgb[0],
            confidence_map=confidence_map,
            save_path=conf_save_path
        )
    # ==================================================
    
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
    images, _, _, paired_rgbs = next(iter(triplets_dl))
    if args.use_pos_as_aligned_rgb:
        assert images.size(0) % args.train_batch_size == 0
        size_of_batch = int(images.size(0) / args.train_batch_size)
        pos_rgbs = [images[idx] for idx in range(1, images.size(0), size_of_batch)]
        pos_rgbs = torch.stack(pos_rgbs)
        paired_rgbs = pos_rgbs
        
    # Thermal query 1개만 추출 (첫 번째 thermal)
    thermal_img = images[0:1].to(device)  # [1, 3, 224, 224]
    aligned_rgb = paired_rgbs[0:1].to(device)  # [1, 3, 224, 224]
    
    save_path = f"{save_subdir}/epoch_{epoch:03d}.png"
    visualize_reconstruction(
        model, 
        thermal_img, 
        aligned_rgb, 
        device, 
        save_path,
        return_attention=True
    )
    
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
    
def save_confidence_vis_simple(thermal_img, rgb_img, confidence_map, save_path):
    """
    최소 코드로 3개 이미지 시각화
    
    Args:
        thermal_img: [3, 224, 224] normalized tensor
        rgb_img: [3, 224, 224] normalized tensor
        confidence_map: [256] tensor
        save_path: str
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm
    import os
    
    # ========== 수정: device 맞추기 ==========
    device = thermal_img.device
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
    # =========================================
    
    thermal = ((thermal_img * std + mean).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    rgb = ((rgb_img * std + mean).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    
    # Confidence map
    conf_map = confidence_map.cpu().numpy().reshape(16, 16)
    conf_resized = cv2.resize(conf_map, (224, 224))
    conf_colored = (cm.jet(conf_resized)[:, :, :3] * 255).astype(np.uint8)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(thermal); axes[0].set_title('Query Thermal'); axes[0].axis('off')
    axes[1].imshow(conf_colored); axes[1].set_title(f'Confidence (mean: {conf_map.mean():.3f})'); axes[1].axis('off')
    axes[2].imshow(rgb); axes[2].set_title('Top-1 RGB'); axes[2].axis('off')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()