# recon_vis.py
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
from utils import get_timestamp
import cv2

def visualize_reconstruction(model, args, thermal_img, paired_rgb, device='cuda', save_path=None):
    """
    Visualize bidirectional reconstruction (Thermal→RGB and RGB→Thermal).

    NOTE: This function now matches stage2_forward_recon exactly:
    - Uses AttentionMask (CLS attention based, 50% masking)
    - Uses full encoder output with mask token replacement
    - Uses CroCo decoder for cross-modal reconstruction

    Args:
        model: CrossModalVPR_Net (wrapped in DataParallel)
        args: training arguments
        thermal_img: [1, 3, 224, 224] - single thermal image
        paired_rgb: [1, 3, 224, 224] - aligned RGB image
        device: 'cuda' or 'cpu'
        save_path: Optional path to save the figure
    """
    from croco.models.masking import AttentionMask

    orig_W = args.resize[0]
    orig_H = args.resize[1]

    # Get the underlying model (handle DataParallel)
    net = model.module if hasattr(model, 'module') else model

    # Set to eval mode for visualization
    was_training = net.training
    net.eval()

    B = 1  # Single image
    patch_N = net.patch_count

    with torch.no_grad():
        # ========== 1. Forward without masking to get attention maps ==========
        # (Same as stage2_forward_recon)
        thermal_out = net.shared_backbone(thermal_img, return_attention=True)
        rgb_out = net.shared_backbone(paired_rgb, return_attention=True)

        # CLS attention: [B, num_heads, N] -> [B, N] (sum over heads)
        thermal_cls_attn = thermal_out["cls_attention"].sum(dim=1)  # [1, 256]
        rgb_cls_attn = rgb_out["cls_attention"].sum(dim=1)  # [1, 256]

        # Full features (no masking)
        thermal_full = thermal_out["x_norm_patchtokens"]  # [1, 256, D]
        rgb_full = rgb_out["x_norm_patchtokens"]  # [1, 256, D]

        # ========== 2. Create attention-based masks (mask HIGH attention patches) ==========
        # (Same as stage2_forward_recon - AttentionMask with 50% ratio)
        attn_masker = AttentionMask(patch_N, mask_ratio=0.5)
        mask_thermal = attn_masker(thermal_cls_attn)  # [1, 256] True=masked
        mask_rgb = attn_masker(rgb_cls_attn)  # [1, 256] True=masked

        # ========== 3. Apply masking (replace masked positions with mask token) ==========
        thermal_masked = net.mask_token.expand(B, patch_N, -1).clone()
        thermal_masked[~mask_thermal] = thermal_full[~mask_thermal]

        rgb_masked = net.mask_token.expand(B, patch_N, -1).clone()
        rgb_masked[~mask_rgb] = rgb_full[~mask_rgb]

        # ========== 4. Add positional embedding ==========
        thermal_masked = thermal_masked + net.decoder_pos_embed
        rgb_masked = rgb_masked + net.decoder_pos_embed

        # Reference features (full, with positional embedding)
        thermal_ref = thermal_full + net.decoder_pos_embed
        rgb_ref = rgb_full + net.decoder_pos_embed

        # ========== 5. CroCo Decoder forward (cross-modal reconstruction) ==========
        # Thermal masked -> decode with RGB reference -> reconstruct thermal
        thermal_decoded = thermal_masked
        for blk in net.decoder_blocks:
            thermal_decoded = blk(thermal_decoded, rgb_ref)
        thermal_decoded = net.decoder_norm(thermal_decoded)

        # RGB masked -> decode with thermal reference -> reconstruct RGB
        rgb_decoded = rgb_masked
        for blk in net.decoder_blocks:
            rgb_decoded = blk(rgb_decoded, thermal_ref)
        rgb_decoded = net.decoder_norm(rgb_decoded)

        # ========== 6. Prediction heads ==========
        reconstructed_thermal_pixels = net.prediction_thermal_head(thermal_decoded)  # [1, 256, 588]
        reconstructed_rgb_pixels = net.prediction_rgb_head(rgb_decoded)  # [1, 256, 588]

    # Restore training mode
    if was_training:
        net.train()

    # ========== Unpatchify to images ==========
    reconstructed_thermal_img = unpatchify_visual(reconstructed_thermal_pixels, orig_H, orig_W, patch_size=14)
    reconstructed_rgb_img = unpatchify_visual(reconstructed_rgb_pixels, orig_H, orig_W, patch_size=14)

    # ========== Denormalize ==========
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    thermal_img_denorm = thermal_img * std + mean
    paired_rgb_denorm = paired_rgb * std + mean
    reconstructed_thermal_denorm = reconstructed_thermal_img * std + mean
    reconstructed_rgb_denorm = reconstructed_rgb_img * std + mean

    # ========== Create hybrid images (visible + reconstructed) ==========
    # For thermal: original visible patches + reconstructed masked patches
    original_thermal_patches = net.patchify(thermal_img)
    hybrid_thermal_patches = original_thermal_patches.clone()
    hybrid_thermal_patches[mask_thermal] = reconstructed_thermal_pixels[mask_thermal]
    hybrid_thermal_img = unpatchify_visual(hybrid_thermal_patches, orig_H, orig_W, patch_size=14)
    hybrid_thermal_denorm = hybrid_thermal_img * std + mean

    # For RGB: original visible patches + reconstructed masked patches
    original_rgb_patches = net.patchify(paired_rgb)
    hybrid_rgb_patches = original_rgb_patches.clone()
    hybrid_rgb_patches[mask_rgb] = reconstructed_rgb_pixels[mask_rgb]
    hybrid_rgb_img = unpatchify_visual(hybrid_rgb_patches, orig_H, orig_W, patch_size=14)
    hybrid_rgb_denorm = hybrid_rgb_img * std + mean

    # Move to CPU for visualization
    thermal_img_denorm = thermal_img_denorm.detach().cpu()
    paired_rgb_denorm = paired_rgb_denorm.detach().cpu()
    reconstructed_thermal_denorm = reconstructed_thermal_denorm.detach().cpu()
    reconstructed_rgb_denorm = reconstructed_rgb_denorm.detach().cpu()
    hybrid_thermal_denorm = hybrid_thermal_denorm.detach().cpu()
    hybrid_rgb_denorm = hybrid_rgb_denorm.detach().cpu()

    # ========== Create mask visualizations ==========
    h_patches = int(orig_H / 14)
    w_patches = int(orig_W / 14)

    mask_thermal_2d = mask_thermal.reshape(1, h_patches, w_patches).float()
    mask_thermal_img = torch.nn.functional.interpolate(
        mask_thermal_2d.unsqueeze(1), size=(orig_H, orig_W), mode='nearest'
    ).squeeze(1).cpu()

    mask_rgb_2d = mask_rgb.reshape(1, h_patches, w_patches).float()
    mask_rgb_img = torch.nn.functional.interpolate(
        mask_rgb_2d.unsqueeze(1), size=(orig_H, orig_W), mode='nearest'
    ).squeeze(1).cpu()

    # ========== Plot ==========
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))

    # Row 0: Thermal reconstruction (Thermal → reconstructed using RGB context)
    axes[0, 0].imshow(thermal_img_denorm[0].permute(1, 2, 0).clip(0, 1))
    axes[0, 0].set_title('Original Thermal', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(mask_thermal_img[0], cmap='RdYlGn_r', vmin=0, vmax=1)
    axes[0, 1].set_title(f'Thermal Mask ({mask_thermal.float().mean()*100:.1f}% masked)', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(hybrid_thermal_denorm[0].permute(1, 2, 0).clip(0, 1))
    axes[0, 2].set_title('Thermal: Visible + Reconstructed', fontsize=14, fontweight='bold', color='darkgreen')
    axes[0, 2].axis('off')

    axes[0, 3].imshow(paired_rgb_denorm[0].permute(1, 2, 0).clip(0, 1))
    axes[0, 3].set_title('RGB (Context for Thermal Recon)', fontsize=14, fontweight='bold', color='blue')
    axes[0, 3].axis('off')

    # Row 1: RGB reconstruction (RGB → reconstructed using Thermal context)
    axes[1, 0].imshow(paired_rgb_denorm[0].permute(1, 2, 0).clip(0, 1))
    axes[1, 0].set_title('Original RGB', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(mask_rgb_img[0], cmap='RdYlGn_r', vmin=0, vmax=1)
    axes[1, 1].set_title(f'RGB Mask ({mask_rgb.float().mean()*100:.1f}% masked)', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(hybrid_rgb_denorm[0].permute(1, 2, 0).clip(0, 1))
    axes[1, 2].set_title('RGB: Visible + Reconstructed', fontsize=14, fontweight='bold', color='darkgreen')
    axes[1, 2].axis('off')

    axes[1, 3].imshow(thermal_img_denorm[0].permute(1, 2, 0).clip(0, 1))
    axes[1, 3].set_title('Thermal (Context for RGB Recon)', fontsize=14, fontweight='bold', color='blue')
    axes[1, 3].axis('off')

    # Row 2: Error analysis
    # Thermal reconstruction error (on masked regions only)
    thermal_error = (thermal_img_denorm - reconstructed_thermal_denorm).abs().mean(dim=1)
    thermal_error_masked = thermal_error.clone()
    thermal_error_masked[mask_thermal_img < 0.5] = 0
    im0 = axes[2, 0].imshow(thermal_error_masked[0], cmap='hot', vmin=0, vmax=0.3)
    axes[2, 0].set_title('Thermal Recon Error (Masked Only)', fontsize=14, fontweight='bold')
    axes[2, 0].axis('off')
    plt.colorbar(im0, ax=axes[2, 0], fraction=0.046, pad=0.04)

    # RGB reconstruction error (on masked regions only)
    rgb_error = (paired_rgb_denorm - reconstructed_rgb_denorm).abs().mean(dim=1)
    rgb_error_masked = rgb_error.clone()
    rgb_error_masked[mask_rgb_img < 0.5] = 0
    im1 = axes[2, 1].imshow(rgb_error_masked[0], cmap='hot', vmin=0, vmax=0.3)
    axes[2, 1].set_title('RGB Recon Error (Masked Only)', fontsize=14, fontweight='bold')
    axes[2, 1].axis('off')
    plt.colorbar(im1, ax=axes[2, 1], fraction=0.046, pad=0.04)

    # Show reconstructed only (masked regions)
    recon_thermal_only = reconstructed_thermal_denorm.clone()
    mask_thermal_expanded = mask_thermal_img.unsqueeze(1).expand(-1, 3, -1, -1)
    recon_thermal_only[mask_thermal_expanded < 0.5] = 0.5  # Gray for visible regions
    axes[2, 2].imshow(recon_thermal_only[0].permute(1, 2, 0).clip(0, 1))
    axes[2, 2].set_title('Thermal Recon (Masked Only)', fontsize=14, fontweight='bold')
    axes[2, 2].axis('off')

    recon_rgb_only = reconstructed_rgb_denorm.clone()
    mask_rgb_expanded = mask_rgb_img.unsqueeze(1).expand(-1, 3, -1, -1)
    recon_rgb_only[mask_rgb_expanded < 0.5] = 0.5  # Gray for visible regions
    axes[2, 3].imshow(recon_rgb_only[0].permute(1, 2, 0).clip(0, 1))
    axes[2, 3].set_title('RGB Recon (Masked Only)', fontsize=14, fontweight='bold')
    axes[2, 3].axis('off')

    # Overall title
    fig.suptitle('Bidirectional Cross-Modal Reconstruction\n'
                 '(Top: Thermal reconstructed from RGB context | Middle: RGB reconstructed from Thermal context)',
                 fontsize=16, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved reconstruction visualization to {save_path}")
        plt.close()
    else:
        plt.show()

    return hybrid_thermal_denorm, hybrid_rgb_denorm

def unpatchify_visual(patches, orig_H, orig_W, patch_size=14):
    """
    patches: [B, N, patch_size**2 * 3]
    return: [B, 3, H, W]
    """
    B = patches.shape[0]
    h = int(orig_H / patch_size)
    w = int(orig_W / patch_size)

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
    visualize_reconstruction(model, args, thermal_img, paired_rgb, device, save_path)

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
                status = "IMPROVED (Wrong -> Correct)"
                color = 'darkgreen'
            elif orig_top1 in positives:
                status = "WORSENED (Correct -> Wrong)"
                color = 'darkred'
            else:
                status = "CHANGED (Wrong -> Wrong)"
                color = 'orange'
        else:
            if orig_top1 in positives:
                status = "MAINTAINED (Correct -> Correct)"
                color = 'blue'
            else:
                status = "MAINTAINED (Wrong -> Wrong)"
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
        f.write(f"Improved (wrong->correct): {len(improved_queries)}\n")
        f.write(f"Worsened (correct->wrong): {len(worsened_queries)}\n")
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
                          save_dir, epoch, top_k=5, positives_per_query=None):
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
        positives_per_query: dict mapping query_idx -> set of positive db indices
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
            ax1.set_title(f'Rank {k+1}: Thermal->RGB\n(Max Attn)', fontsize=11, fontweight='bold')
            ax1.axis('off')
            plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

            # Col 1: RGB→Thermal attention
            ax2 = fig.add_subplot(gs[row_idx, 1])
            im2 = ax2.imshow(rgb_heatmap_vis, cmap='hot', vmin=0, vmax=rgb_heatmap.max())
            ax2.set_title(f'Rank {k+1}: RGB->Thermal\n(Max Attn)', fontsize=11, fontweight='bold')
            ax2.axis('off')
            plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

            # Col 2: Mutual agreement
            ax3 = fig.add_subplot(gs[row_idx, 2])
            im3 = ax3.imshow(mutual_heatmap_vis, cmap='hot', vmin=0, vmax=mutual_heatmap.max())
            ax3.set_title(f'Rank {k+1}: Mutual\n(T->R x R->T)', fontsize=11, fontweight='bold')
            ax3.axis('off')
            plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

            # Col 3: Thermal→RGB overlay on RGB image
            ax4 = fig.add_subplot(gs[row_idx, 3])
            heatmap_overlay = plt.cm.hot(thermal_heatmap_vis / (thermal_heatmap.max() + 1e-8))[:, :, :3]
            overlay_img = 0.5 * rgb_imgs[k] + 0.5 * heatmap_overlay
            ax4.imshow(overlay_img)
            ax4.set_title(f'Rank {k+1}: T->R Overlay\non RGB', fontsize=11, fontweight='bold')
            ax4.axis('off')

            # Col 4: RGB→Thermal overlay on Thermal image
            ax5 = fig.add_subplot(gs[row_idx, 4])
            rgb_heatmap_overlay = plt.cm.hot(rgb_heatmap_vis / (rgb_heatmap.max() + 1e-8))[:, :, :3]
            thermal_overlay_img = 0.5 * thermal_img + 0.5 * rgb_heatmap_overlay
            ax5.imshow(thermal_overlay_img)
            ax5.set_title(f'Rank {k+1}: R->T Overlay\non Thermal', fontsize=11, fontweight='bold')
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

            # Check if top-1 prediction is correct
            top1_db_idx = int(db_indices[0])
            is_correct = False
            if positives_per_query is not None and query_idx in positives_per_query:
                is_correct = top1_db_idx in positives_per_query[query_idx]

            # Side-by-side images
            combined = np.hstack([thermal_img, rgb_imgs[0]])
            ax_mnn.imshow(combined)

            # Add border to indicate correct/wrong
            border_color = 'lime' if is_correct else 'red'
            for spine in ax_mnn.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(6)
                spine.set_visible(True)

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
            status_str = 'CORRECT' if is_correct else 'WRONG'
            status_color = 'green' if is_correct else 'red'
            title_str = (f'Mutual Nearest Neighbor Matches [{status_str}]\n'
                        f'Total: {len(matches_i)} pairs | Shown: Top {num_draw} | '
                        f'Avg Conf: {avg_conf:.3f} | Max Conf: {max_conf:.3f}')
            ax_mnn.set_title(title_str, fontsize=14, fontweight='bold', color=status_color)
            ax_mnn.axis('off')
        else:
            ax_mnn.text(0.5, 0.5, 'No Mutual Nearest Neighbor Matches Found',
                       ha='center', va='center', fontsize=24, color='red',
                       transform=ax_mnn.transAxes, fontweight='bold')
            ax_mnn.axis('off')

        # ==========================================
        # Overall title
        # ==========================================
        # Check correctness for title
        top1_db_idx = int(db_indices[0])
        is_correct_title = False
        if positives_per_query is not None and query_idx in positives_per_query:
            is_correct_title = top1_db_idx in positives_per_query[query_idx]
        status_str = 'CORRECT' if is_correct_title else 'WRONG'
        title_color = 'darkgreen' if is_correct_title else 'darkred'
        fig.suptitle(f'Query {query_idx} - [{status_str}] Cross-Modal Matching Analysis (Epoch {epoch})',
                    fontsize=18, fontweight='bold', y=0.995, color=title_color)

        # Save
        save_path = os.path.join(save_dir, f'epoch{epoch:03d}_query{query_idx:05d}_full.png')
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close()

        print(f"[{list_idx+1}/{len(query_indices)}] Saved: {save_path}")

    print(f"Visualization complete: {len(query_indices)} queries saved to {save_dir}")


def visualize_selaVPR_reranking(args, eval_ds,
                                 original_predictions,
                                 reranked_predictions,
                                 rerank_scores_dict,
                                 positives_per_query,
                                 epoch,
                                 distances=None,
                                 npy_root_path=None,
                                 seq_name="",
                                 save_dir='./selaVPR_visualizations',
                                 num_samples=2,
                                 reranking_method='selaVPR'):
    """
    Visualize selaVPR/match_conf reranking with MNN matching visualization.

    Args:
        args: training arguments
        eval_ds: evaluation dataset
        original_predictions: [num_queries, K] - original Faiss predictions
        reranked_predictions: [num_queries, K] - reranked predictions
        rerank_scores_dict: dict mapping query_idx -> list of scores (MNN count or confidence)
        positives_per_query: dict mapping query_idx -> set of positive db indices
        epoch: current epoch
        distances: [num_queries, K] - L2 distances (optional)
        npy_root_path: path to NPY files (for loading sela features on demand)
        seq_name: sequence name for loading sela features
        save_dir: save directory
        num_samples: number of samples to visualize
        reranking_method: 'selaVPR' or 'match_conf'
    """
    import os
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import torch.nn.functional as F
    import csv

    os.makedirs(save_dir, exist_ok=True)

    # ===== Score statistics =====
    all_scores = []
    for query_idx, scores in rerank_scores_dict.items():
        all_scores.extend(scores)
    all_scores = np.array(all_scores)

    stats = {
        'epoch': epoch,
        'mean': all_scores.mean(),
        'std': all_scores.std(),
        'method': reranking_method
    }

    # ===== CSV logging =====
    csv_path = os.path.join(save_dir, 'score_stats.csv')
    file_exists = os.path.exists(csv_path)

    with open(csv_path, 'a', newline='') as f:
        fieldnames = ['epoch', 'mean', 'std', 'method']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(stats)

    # ===== Sampling strategy =====
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

    # Sample
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
        if len(all_remaining) > 0:
            sample_indices.extend(np.random.choice(all_remaining, min(remaining, len(all_remaining)), replace=False))

    # ===== NPY loading helper =====
    def load_sela_npy(npy_path):
        """Load sela features from NPY file."""
        return np.load(npy_path)

    # ===== MNN extraction helper =====
    def extract_mnn_matches(fm1, fm2, top_n=50):
        """
        Extract MNN matches between two feature maps.
        fm1, fm2: [H, W, D] numpy arrays
        Returns: (matches_i, matches_j, similarities)
        """
        H, W, D = fm1.shape
        fm1_flat = fm1.reshape(-1, D)  # [L, D]
        fm2_flat = fm2.reshape(-1, D)  # [L, D]

        # Normalize
        fm1_norm = fm1_flat / (np.linalg.norm(fm1_flat, axis=1, keepdims=True) + 1e-8)
        fm2_norm = fm2_flat / (np.linalg.norm(fm2_flat, axis=1, keepdims=True) + 1e-8)

        # Similarity matrix
        sim_matrix = np.dot(fm1_norm, fm2_norm.T)  # [L, L]

        # Mutual nearest neighbors
        max1 = np.argmax(sim_matrix, axis=1)  # [L] - best match in fm2 for each fm1
        max2 = np.argmax(sim_matrix, axis=0)  # [L] - best match in fm1 for each fm2

        # Check mutual consistency
        mutual_mask = max2[max1] == np.arange(len(max1))

        matches_i = np.where(mutual_mask)[0]
        matches_j = max1[matches_i]
        similarities = sim_matrix[matches_i, matches_j]

        # Sort by similarity and take top_n
        sorted_idx = np.argsort(similarities)[::-1][:top_n]
        return matches_i[sorted_idx], matches_j[sorted_idx], similarities[sorted_idx]

    def patch_to_pixel(patch_idx, patch_size=4, grid_h=61, grid_w=61):
        """Convert patch index to pixel coordinates (for 61x61 grid on 224x224 image)."""
        row = patch_idx // grid_w
        col = patch_idx % grid_w
        # Map to 224x224 image
        y = int((row + 0.5) * 224 / grid_h)
        x = int((col + 0.5) * 224 / grid_w)
        return x, y

    # Check if we can load sela features
    has_sela = npy_root_path is not None and os.path.exists(npy_root_path)

    # ===== Visualization =====
    for sample_num, query_idx in enumerate(sample_indices):
        # Decide number of rows based on whether we have sela features
        # Check if sela file exists for this query
        query_has_sela = has_sela and os.path.exists(
            os.path.join(npy_root_path, f"Query_{seq_name}_sela_{query_idx}.npy")
        ) if has_sela else False
        # Row 3: MNN visualization, Row 4: similarity graphs
        n_rows = 5 if query_has_sela else 3

        fig = plt.figure(figsize=(24, 6 * n_rows // 2))
        gs = fig.add_gridspec(n_rows, 7, hspace=0.4, wspace=0.3)

        # Load query image
        query_img = eval_ds.get_thermal_img(eval_ds.t_queries_paths[query_idx])
        query_img = cv2.resize(query_img, (224, 224))
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)

        positives = positives_per_query[query_idx]

        # === Row 0: Query ===
        ax_query = fig.add_subplot(gs[0, :])
        ax_query.imshow(query_img)
        ax_query.set_title(f'Query #{query_idx} (Thermal)\nPositives in DB: {len(positives)}',
                          fontsize=14, fontweight='bold')
        ax_query.axis('off')

        # === Row 1: Original (Before Reranking) ===
        for rank in range(5):
            ax = fig.add_subplot(gs[1, rank + 1])

            pred_idx = original_predictions[query_idx, rank]
            db_img = eval_ds.get_rgb_img(eval_ds.rgb_database_paths[pred_idx])
            db_img = cv2.resize(db_img, (224, 224))
            db_img = cv2.cvtColor(db_img, cv2.COLOR_BGR2RGB)

            is_correct = pred_idx in positives

            # GPS distance
            if hasattr(eval_ds, 'database_utms') and hasattr(eval_ds, 'queries_utms'):
                query_gps = eval_ds.queries_utms[query_idx]
                db_gps = eval_ds.database_utms[pred_idx]
                gps_distance = np.linalg.norm(query_gps - db_gps)
            else:
                gps_distance = None

            border_color = 'green' if is_correct else 'red'
            border_width = 4 if is_correct else 2

            ax.imshow(db_img)
            rect = Rectangle((0, 0), 223, 223, linewidth=border_width,
                             edgecolor=border_color, facecolor='none')
            ax.add_patch(rect)

            title = f'Rank {rank + 1}'
            if is_correct:
                title += ' ✓'
            if distances is not None:
                title += f'\nL2: {distances[query_idx, rank]:.2f}'
            if gps_distance is not None:
                title += f'\nGPS: {gps_distance:.1f}m'

            ax.set_title(title, fontsize=10, fontweight='bold', color=border_color)
            ax.axis('off')

        # Row 1 labels
        ax_label1 = fig.add_subplot(gs[1, 0])
        ax_label1.text(0.5, 0.5, 'Before\nReranking', fontsize=12,
                      fontweight='bold', ha='center', va='center')
        ax_label1.axis('off')

        ax_legend1 = fig.add_subplot(gs[1, 6])
        ax_legend1.text(0.1, 0.7, 'Faiss', fontsize=14, fontweight='bold', va='center')
        ax_legend1.text(0.1, 0.3, '(L2 Distance)', fontsize=11, style='italic', va='center')
        ax_legend1.axis('off')

        # === Row 2: Reranked ===
        for rank in range(5):
            ax = fig.add_subplot(gs[2, rank + 1])

            pred_idx = reranked_predictions[query_idx, rank]
            db_img = eval_ds.get_rgb_img(eval_ds.rgb_database_paths[pred_idx])
            db_img = cv2.resize(db_img, (224, 224))
            db_img = cv2.cvtColor(db_img, cv2.COLOR_BGR2RGB)

            is_correct = pred_idx in positives

            # GPS distance
            if hasattr(eval_ds, 'database_utms') and hasattr(eval_ds, 'queries_utms'):
                query_gps = eval_ds.queries_utms[query_idx]
                db_gps = eval_ds.database_utms[pred_idx]
                gps_distance = np.linalg.norm(query_gps - db_gps)
            else:
                gps_distance = None

            # Rank change
            orig_rank = np.where(original_predictions[query_idx, :20] == pred_idx)[0]
            if len(orig_rank) > 0:
                rank_change = f"(R{orig_rank[0] + 1}→R{rank + 1})"
            else:
                rank_change = "(new)"

            border_color = 'green' if is_correct else 'red'
            border_width = 4 if is_correct else 2

            ax.imshow(db_img)
            rect = Rectangle((0, 0), 223, 223, linewidth=border_width,
                             edgecolor=border_color, facecolor='none')
            ax.add_patch(rect)

            title = f'Rank {rank + 1}'
            if is_correct:
                title += ' ✓'
            title += f' {rank_change}'

            # Reranking score
            rerank_scores = rerank_scores_dict.get(query_idx, [0] * 5)
            if rank < len(rerank_scores):
                if reranking_method == 'selaVPR':
                    title += f'\nMNN: {int(rerank_scores[rank])}'
                else:  # match_conf
                    title += f'\nConf: {rerank_scores[rank]:.3f}'

            if gps_distance is not None:
                title += f'\nGPS: {gps_distance:.1f}m'

            ax.set_title(title, fontsize=10, fontweight='bold', color=border_color)
            ax.axis('off')

        # Row 2 labels
        ax_label2 = fig.add_subplot(gs[2, 0])
        ax_label2.text(0.5, 0.5, 'After\nReranking', fontsize=12,
                      fontweight='bold', ha='center', va='center')
        ax_label2.axis('off')

        ax_legend2 = fig.add_subplot(gs[2, 6])
        if reranking_method == 'selaVPR':
            ax_legend2.text(0.1, 0.7, 'selaVPR', fontsize=14, fontweight='bold', va='center')
            ax_legend2.text(0.1, 0.3, '(MNN Count)', fontsize=11, style='italic', va='center')
        else:
            ax_legend2.text(0.1, 0.7, 'MatchConf', fontsize=14, fontweight='bold', va='center')
            ax_legend2.text(0.1, 0.3, '(Confidence)', fontsize=11, style='italic', va='center')
        ax_legend2.axis('off')

        # === Row 3: MNN Matching Visualization (if sela features available) ===
        # === Row 4: Similarity Distribution Graphs ===
        if query_has_sela:
            # Load query sela features on demand
            query_sela_path = os.path.join(npy_root_path, f"Query_{seq_name}_sela_{query_idx}.npy")
            query_feat = load_sela_npy(query_sela_path)  # [H, W, D]

            # Store similarity data for graphs
            all_sims_data = []

            for rank in range(min(3, 5)):  # Show top 3 matches
                ax = fig.add_subplot(gs[3, rank * 2 + 1: rank * 2 + 3])

                pred_idx = reranked_predictions[query_idx, rank]
                db_img = eval_ds.get_rgb_img(eval_ds.rgb_database_paths[pred_idx])
                db_img = cv2.resize(db_img, (224, 224))
                db_img = cv2.cvtColor(db_img, cv2.COLOR_BGR2RGB) / 255.0

                query_img_norm = query_img.astype(np.float32) / 255.0

                # Combined image
                combined = np.hstack([query_img_norm, db_img])
                ax.imshow(combined)

                n_matches = 0
                sims_for_graph = []

                # Load DB sela features on demand
                db_sela_path = os.path.join(npy_root_path, f"Db_{seq_name}_sela_{pred_idx}.npy")
                if os.path.exists(db_sela_path):
                    db_feat = load_sela_npy(db_sela_path)

                    matches_i, matches_j, sims = extract_mnn_matches(query_feat, db_feat, top_n=100)
                    n_matches = len(matches_i)
                    sims_for_graph = sims.tolist()

                    H, W = query_feat.shape[:2]
                    for idx in range(min(50, len(matches_i))):  # Draw top 50
                        i, j, sim = matches_i[idx], matches_j[idx], sims[idx]

                        x1, y1 = patch_to_pixel(i, grid_h=H, grid_w=W)
                        x2, y2 = patch_to_pixel(j, grid_h=H, grid_w=W)
                        x2 += 224  # Offset for right image

                        color = plt.cm.hot(min(sim, 1.0))
                        alpha = 0.3 + 0.5 * sim
                        ax.plot([x1, x2], [y1, y2], color=color, linewidth=1, alpha=alpha)

                all_sims_data.append((pred_idx, n_matches, sims_for_graph))

                ax.axvline(x=224, color='white', linewidth=2, linestyle='--', alpha=0.7)

                is_correct = pred_idx in positives
                border_color = 'lime' if is_correct else 'red'
                status_str = 'CORRECT' if is_correct else 'WRONG'
                # Show match count in title
                ax.set_title(f'Rank {rank + 1} [{status_str}]\n{n_matches} MNN matches',
                            fontsize=11, fontweight='bold', color=border_color)
                ax.axis('off')

            ax_label3 = fig.add_subplot(gs[3, 0])
            ax_label3.text(0.5, 0.5, 'MNN\nMatches', fontsize=12,
                          fontweight='bold', ha='center', va='center')
            ax_label3.axis('off')

            # === Row 4: Similarity Distribution Graphs ===
            for rank in range(min(3, 5)):
                ax = fig.add_subplot(gs[4, rank * 2 + 1: rank * 2 + 3])

                pred_idx, n_matches, sims_list = all_sims_data[rank]

                if len(sims_list) > 0:
                    # Histogram of cosine similarities
                    ax.hist(sims_list, bins=20, color='steelblue', edgecolor='white', alpha=0.8)
                    ax.axvline(x=np.mean(sims_list), color='red', linestyle='--',
                              linewidth=2, label=f'Mean: {np.mean(sims_list):.3f}')
                    ax.set_xlabel('Cosine Similarity', fontsize=9)
                    ax.set_ylabel('Count', fontsize=9)
                    ax.set_xlim(0, 1)
                    ax.legend(fontsize=8)
                    ax.grid(alpha=0.3)

                    is_correct = pred_idx in positives
                    border_color = 'green' if is_correct else 'red'
                    ax.set_title(f'Rank {rank + 1} Similarity Dist.\n'
                                f'Mean={np.mean(sims_list):.3f}, Max={np.max(sims_list):.3f}',
                                fontsize=10, fontweight='bold', color=border_color)
                else:
                    ax.text(0.5, 0.5, 'No MNN\nMatches', ha='center', va='center',
                           fontsize=12, color='gray')
                    ax.axis('off')

            ax_label4 = fig.add_subplot(gs[4, 0])
            ax_label4.text(0.5, 0.5, 'Similarity\nDistribution', fontsize=12,
                          fontweight='bold', ha='center', va='center')
            ax_label4.axis('off')

        # === Overall title ===
        orig_top1 = original_predictions[query_idx, 0]
        rerank_top1 = reranked_predictions[query_idx, 0]

        if orig_top1 != rerank_top1:
            if rerank_top1 in positives:
                status = "IMPROVED (Wrong -> Correct)"
                color = 'darkgreen'
            elif orig_top1 in positives:
                status = "WORSENED (Correct -> Wrong)"
                color = 'darkred'
            else:
                status = "CHANGED (Wrong -> Wrong)"
                color = 'orange'
        else:
            if orig_top1 in positives:
                status = "MAINTAINED (Correct -> Correct)"
                color = 'blue'
            else:
                status = "MAINTAINED (Wrong -> Wrong)"
                color = 'gray'

        method_name = 'selaVPR' if reranking_method == 'selaVPR' else 'MatchConf'
        fig.suptitle(f'Query #{query_idx} - {status} [{method_name} Reranking]',
                    fontsize=16, fontweight='bold', color=color)

        # Save
        save_path = os.path.join(save_dir, f'epoch_{epoch:03d}_query_{query_idx:05d}.png')
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")

    # ===== Summary =====
    summary_path = os.path.join(save_dir, f'epoch_{epoch:03d}_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"{reranking_method} Reranking Summary - Epoch {epoch}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total queries: {eval_ds.queries_num}\n")
        f.write(f"Improved (wrong->correct): {len(improved_queries)}\n")
        f.write(f"Worsened (correct->wrong): {len(worsened_queries)}\n")
        f.write(f"Unchanged: {len(unchanged_queries)}\n\n")

        orig_correct = sum([1 for q in range(eval_ds.queries_num)
                           if original_predictions[q, 0] in positives_per_query[q]])
        rerank_correct = sum([1 for q in range(eval_ds.queries_num)
                             if reranked_predictions[q, 0] in positives_per_query[q]])

        f.write(f"Top-1 Accuracy:\n")
        f.write(f"  Before: {orig_correct}/{eval_ds.queries_num} ({100*orig_correct/eval_ds.queries_num:.2f}%)\n")
        f.write(f"  After:  {rerank_correct}/{eval_ds.queries_num} ({100*rerank_correct/eval_ds.queries_num:.2f}%)\n")
        f.write(f"  Delta:  {rerank_correct-orig_correct:+d} ({100*(rerank_correct-orig_correct)/eval_ds.queries_num:+.2f}%)\n")
        f.write(f"\nScore Stats: Mean={stats['mean']:.4f}, Std={stats['std']:.4f}\n")

    print(f"Saved summary: {summary_path}")

    return improved_queries, worsened_queries, unchanged_queries
