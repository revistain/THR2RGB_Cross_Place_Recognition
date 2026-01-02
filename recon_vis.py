# recon_vis.py
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
from utils import get_timestamp
import cv2

def visualize_reconstruction(model, thermal_img, aligned_rgb, device='cuda', save_path=None):
    """
    Args:
        model: CrossModalVPR_Net
        thermal_img: [1, 3, 224, 224] - single thermal image
        aligned_rgb: [1, 3, 224, 224] - aligned RGB image
        device: 'cuda' or 'cpu'
        save_path: Optional path to save the figure
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
        
        # 8. Decoder
        for blk in model.module.decoder_blocks:
            thermal_full = blk(thermal_full, rgb_full)
        thermal_full = model.module.decoder_norm(thermal_full)
        
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
    # Visible patches는 원본, Masked patches는 reconstruction 사용
    hybrid_patches = original_patches.clone()  # [1, 256, 588]
    hybrid_patches[mask] = reconstructed_patches[mask]  # Masked 위치만 reconstruction으로 교체
    
    hybrid_img = unpatchify_visual(hybrid_patches, patch_size=14)  # [1, 3, 224, 224]
    hybrid_img_denorm = hybrid_img * std + mean
    
    # Mask 시각화 (16x16 grid)
    mask_2d = mask.reshape(1, 16, 16).float()  # [1, 16, 16]
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
    axes[0, 1].set_title(f'Mask (Masked={mask.float().mean()*100:.1f}%)', fontsize=14, fontweight='bold')
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
def visualize_during_training(model, triplets_dl, device, epoch, save_dir='./visualizations', comment="default"):
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
    aligned_rgb = aligned_rgbs[0:1].to(device)  # [1, 3, 224, 224]
    
    save_path = f"{save_subdir}/epoch_{epoch:03d}.png"
    visualize_reconstruction(model, thermal_img, aligned_rgb, device, save_path)
    
    model.train()
    
def visualize_reranking_comparison(args, eval_ds, 
                                   original_predictions, 
                                   reranked_predictions,
                                   reconstruction_losses_dict,
                                   positives_per_query,
                                   epoch,
                                   save_dir='./rerank_visualizations',
                                   num_samples=2):
    # FIXME: 현재 이미지를 랜덤하게 가져오지 않음
    # 이미지의 거리계산 없음
    # 정답인 이미지를 제대로 체크하는지 모르겠음
    # reconstruction된 이미지를 그려야할지도 결정못함
    """
    Reranking 전/후를 비교하는 시각화
    
    Args:
        original_predictions: Faiss 초기 predictions [Q, K]
        reranked_predictions: Reranking 후 predictions [Q, K]
        reconstruction_losses_dict: {query_idx: [loss1, loss2, ...]} 
        num_samples: 시각화할 query 개수
    """
    import os
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Top-1이 바뀐 query들 중에서 샘플링
    changed_queries = []
    for q_idx in range(eval_ds.queries_num):
        if original_predictions[q_idx, 0] != reranked_predictions[q_idx, 0]:
            changed_queries.append(q_idx)
    
    if len(changed_queries) < num_samples:
        # Top-1 안 바뀐 것도 포함
        sample_indices = np.random.choice(eval_ds.queries_num, num_samples, replace=False)
    else:
        # Top-1 바뀐 것만
        sample_indices = np.random.choice(changed_queries, num_samples, replace=False)
    
    for sample_num, query_idx in enumerate(sample_indices):
        fig = plt.figure(figsize=(20, 8))
        gs = fig.add_gridspec(2, 7, hspace=0.3, wspace=0.3)
        
        # Query 이미지
        query_img = eval_ds.get_thermal_img(eval_ds.t_queries_paths[query_idx])
        query_img = cv2.resize(query_img, (224, 224))
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
        
        positives = positives_per_query[query_idx]
        
        # ===== Row 1: Original (Faiss) =====
        ax_query1 = fig.add_subplot(gs[0, 0])
        ax_query1.imshow(query_img)
        ax_query1.set_title('Query\n(Thermal)', fontsize=12, fontweight='bold')
        ax_query1.axis('off')
        
        for rank in range(5):
            ax = fig.add_subplot(gs[0, rank+1])
            
            pred_idx = original_predictions[query_idx, rank]
            db_img = eval_ds.get_rgb_img(eval_ds.rgb_database_paths[pred_idx])
            db_img = cv2.resize(db_img, (224, 224))
            db_img = cv2.cvtColor(db_img, cv2.COLOR_BGR2RGB)
            
            is_correct = pred_idx in positives
            
            # Border 색상
            if is_correct:
                border_color = 'green'
                border_width = 4
            else:
                border_color = 'red'
                border_width = 2
            
            ax.imshow(db_img)
            
            # Border 그리기
            rect = Rectangle((0, 0), 223, 223, linewidth=border_width, 
                           edgecolor=border_color, facecolor='none')
            ax.add_patch(rect)
            
            # Title
            title = f'Rank {rank+1}'
            if is_correct:
                title += ' Yes'
            recon_losses = reconstruction_losses_dict.get(query_idx, [0]*5)
            title += f'\nLoss: {recon_losses[rank]:.3f}'
            ax.set_title(title, fontsize=11, fontweight='bold', 
                        color=border_color)
            ax.axis('off')
        
        # Legend for row 1
        ax_legend1 = fig.add_subplot(gs[0, 6])
        ax_legend1.text(0.1, 0.7, 'Before\nReranking', fontsize=14, 
                       fontweight='bold', va='center')
        ax_legend1.text(0.1, 0.3, '(Faiss L2)', fontsize=11, 
                       style='italic', va='center')
        ax_legend1.axis('off')
        
        # ===== Row 2: Reranked =====
        ax_query2 = fig.add_subplot(gs[1, 0])
        ax_query2.imshow(query_img)
        ax_query2.set_title('Query\n(Thermal)', fontsize=12, fontweight='bold')
        ax_query2.axis('off')
        
        for rank in range(5):
            ax = fig.add_subplot(gs[1, rank+1])
            
            pred_idx = reranked_predictions[query_idx, rank]
            db_img = eval_ds.get_rgb_img(eval_ds.rgb_database_paths[pred_idx])
            db_img = cv2.resize(db_img, (224, 224))
            db_img = cv2.cvtColor(db_img, cv2.COLOR_BGR2RGB)
            
            is_correct = pred_idx in positives
            
            # 원래 순위 찾기
            orig_rank = np.where(original_predictions[query_idx, :5] == pred_idx)[0]
            if len(orig_rank) > 0:
                rank_change = f"(was R{orig_rank[0]+1})"
            else:
                rank_change = "(new)"
            
            # Border 색상
            if is_correct:
                border_color = 'green'
                border_width = 4
            else:
                border_color = 'red'
                border_width = 2
            
            ax.imshow(db_img)
            
            # Border
            rect = Rectangle((0, 0), 223, 223, linewidth=border_width,
                           edgecolor=border_color, facecolor='none')
            ax.add_patch(rect)
            
            # Title with loss
            title = f'Rank {rank+1}'
            if is_correct:
                title += ' Yes'
            title += f'\n{rank_change}'
            
            ax.set_title(title, fontsize=10, fontweight='bold',
                        color=border_color)
            ax.axis('off')
        
        # Legend for row 2
        ax_legend2 = fig.add_subplot(gs[1, 6])
        ax_legend2.text(0.1, 0.7, 'After\nReranking', fontsize=14,
                       fontweight='bold', va='center')
        ax_legend2.text(0.1, 0.3, '(Recon Loss)', fontsize=11,
                       style='italic', va='center')
        ax_legend2.axis('off')
        
        # Overall title
        top1_changed = original_predictions[query_idx, 0] != reranked_predictions[query_idx, 0]
        change_marker = "🔄 TOP-1 CHANGED" if top1_changed else "✓ TOP-1 SAME"
        
        fig.suptitle(f'Query #{query_idx} - {change_marker}', 
                    fontsize=16, fontweight='bold',
                    color='red' if top1_changed else 'blue')
        
        # Save
        save_path = os.path.join(save_dir, f'epoch_{epoch:03d}_sample_{sample_num+1}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {save_path}")