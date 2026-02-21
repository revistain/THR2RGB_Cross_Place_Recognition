# visual.py
"""
Visualize penultimate and last layer attention maps using PCA
at the end of each training epoch.

Supports DINOv2 with register tokens.
Token order in attention: [CLS, register_tokens (if any), patch_tokens]
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import random
import cv2


def denormalize(tensor, device='cpu'):
    """
    Denormalize ImageNet normalized tensor.
    Args:
        tensor: [B, 3, H, W] or [3, H, W]
    Returns:
        Denormalized tensor with values in [0, 1]
    """
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
        tensor = tensor * std + mean
        tensor = tensor.squeeze(0)
    else:
        tensor = tensor * std + mean

    return torch.clamp(tensor, 0, 1)


def apply_pca_to_patches(patch_tokens, n_components=3):
    """
    Apply PCA to patch tokens for visualization.

    Args:
        patch_tokens: [B, N, D] patch embeddings (N=256 for 16x16 grid, D=768)
        n_components: number of PCA components (3 for RGB visualization)

    Returns:
        pca_features: [B, N, n_components] reduced features
    """
    B, N, D = patch_tokens.shape

    # Flatten all patches for PCA fitting
    all_patches = patch_tokens.reshape(-1, D)  # [B*N, D]

    # Fit PCA
    pca = PCA(n_components=n_components)
    pca_features = pca.fit_transform(all_patches)  # [B*N, n_components]

    # Reshape back
    pca_features = pca_features.reshape(B, N, n_components)

    return pca_features


def normalize_for_vis(features):
    """
    Normalize features to [0, 1] for visualization.

    Args:
        features: numpy array of any shape

    Returns:
        Normalized features in [0, 1]
    """
    min_val = features.min()
    max_val = features.max()
    if max_val - min_val > 1e-8:
        return (features - min_val) / (max_val - min_val)
    else:
        return np.zeros_like(features)


def reshape_to_grid(features, grid_h, grid_w):
    """
    Reshape flattened patch features to 2D grid.

    Args:
        features: [N, C] where N = grid_h * grid_w
        grid_h, grid_w: grid dimensions

    Returns:
        [grid_h, grid_w, C]
    """
    return features.reshape(grid_h, grid_w, -1)


def get_attention_pca_visualization(attention_map, grid_h=16, grid_w=16, num_register_tokens=0):
    """
    Apply PCA to attention maps for visualization.

    Args:
        attention_map: [num_heads, N_total, N_total] or [N, N] attention weights
                       where N_total = 1 (CLS) + num_register_tokens + num_patches
        grid_h, grid_w: spatial grid dimensions
        num_register_tokens: number of register tokens in DINOv2

    Returns:
        pca_vis: [grid_h, grid_w, 3] RGB visualization
    """
    # Token order: [CLS, register_tokens, patch_tokens]
    start_idx = 1 + num_register_tokens  # Skip CLS and register tokens

    if attention_map.dim() == 3:
        # [num_heads, N_total, N_total] -> average over heads, extract patch-to-patch
        attn = attention_map.mean(dim=0)  # [N_total, N_total]
        attn = attn[start_idx:, start_idx:]  # [num_patches, num_patches]
    else:
        attn = attention_map

    attn = attn.cpu().numpy()
    N = attn.shape[0]

    # Apply PCA to get 3 components
    if N > 3:
        pca = PCA(n_components=3)
        pca_features = pca.fit_transform(attn)  # [N, 3]
    else:
        pca_features = attn[:, :3] if attn.shape[1] >= 3 else np.zeros((N, 3))

    # Normalize to [0, 1]
    pca_features = normalize_for_vis(pca_features)

    # Reshape to grid
    pca_vis = reshape_to_grid(pca_features, grid_h, grid_w)

    return pca_vis


def visualize_attention_maps_pca(
    args,
    model,
    dataloader,
    device,
    epoch,
    num_samples=4,
    save_dir='./attention_visualizations',
    comment='default'
):
    """
    Visualize penultimate and last layer attention maps using PCA.

    Args:
        args: training arguments
        model: the model
        dataloader: data loader to sample images from
        device: cuda/cpu device
        epoch: current epoch number
        num_samples: number of random images to visualize
        save_dir: directory to save visualizations
        comment: experiment comment for subfolder
    """
    from utils import get_timestamp

    timestamp = get_timestamp()
    save_subdir = os.path.join(save_dir, comment, f'epoch_{epoch:03d}_{timestamp}')
    os.makedirs(save_subdir, exist_ok=True)

    model.eval()

    # Get a batch of images
    images, _, _, aligned_rgbs = next(iter(dataloader))

    # Calculate batch structure
    # images shape: [batch_size * (1 + 1 + negs_num_per_query), 3, H, W]
    # where each sample = [thermal_query, rgb_positive, rgb_neg1, ..., rgb_negN]
    batch_size = args.train_batch_size
    sample_size = 1 + 1 + args.negs_num_per_query  # thermal + pos + negs

    # Collect thermal and RGB indices
    thermal_indices = [i * sample_size for i in range(batch_size)]
    rgb_pos_indices = [i * sample_size + 1 for i in range(batch_size)]

    # Select random samples to visualize
    num_samples = min(num_samples, batch_size)
    selected_batch_indices = random.sample(range(batch_size), num_samples)

    with torch.no_grad():
        for sample_idx, batch_idx in enumerate(selected_batch_indices):
            # Get thermal image
            thermal_idx = thermal_indices[batch_idx]
            thermal_img = images[thermal_idx:thermal_idx+1].to(device)

            # Get RGB positive image
            rgb_idx = rgb_pos_indices[batch_idx]
            rgb_img = images[rgb_idx:rgb_idx+1].to(device)

            # Get aligned RGB if available
            if aligned_rgbs is not None:
                aligned_rgb_img = aligned_rgbs[batch_idx:batch_idx+1].to(device)
            else:
                aligned_rgb_img = rgb_img

            # Forward pass for thermal
            thermal_out = model.module.shared_backbone(thermal_img, return_attention=True)

            # Forward pass for RGB
            rgb_out = model.module.shared_backbone(rgb_img, return_attention=True)

            # Get number of register tokens (for DINOv2 with registers)
            num_register_tokens = model.module.shared_backbone.num_register_tokens

            # Extract attention maps
            # Full attention: [B, num_heads, N_total, N_total]
            # where N_total = 1 (CLS) + num_register_tokens + num_patches
            thermal_last_attn = thermal_out.get("attention", None)
            thermal_penultimate_attn = thermal_out.get("penultimate_attention", None)

            rgb_last_attn = rgb_out.get("attention", None)
            rgb_penultimate_attn = rgb_out.get("penultimate_attention", None)

            # Pre-computed CLS -> patches attention (already skips register tokens)
            # Shape: [B, num_heads, num_patches]
            thermal_cls_attn_last = thermal_out.get("cls_attention", None)
            thermal_cls_attn_penult = thermal_out.get("cls_penultimate_attention", None)

            rgb_cls_attn_last = rgb_out.get("cls_attention", None)
            rgb_cls_attn_penult = rgb_out.get("cls_penultimate_attention", None)

            # Extract patch tokens for PCA visualization
            # These already exclude CLS and register tokens
            thermal_last_patches = thermal_out["x_norm_patchtokens"]  # [1, N, D]
            thermal_penultimate_patches = thermal_out.get("penultimate_norm_patchtokens", None)

            rgb_last_patches = rgb_out["x_norm_patchtokens"]
            rgb_penultimate_patches = rgb_out.get("penultimate_norm_patchtokens", None)

            # Calculate grid size
            H, W = thermal_img.shape[2], thermal_img.shape[3]
            grid_h = H // 14
            grid_w = W // 14

            # Create figure
            fig, axes = plt.subplots(4, 5, figsize=(25, 20))

            # Row 0: Original images
            # Thermal original
            thermal_denorm = denormalize(thermal_img, device).cpu()[0].permute(1, 2, 0).numpy()
            axes[0, 0].imshow(thermal_denorm)
            axes[0, 0].set_title('Thermal (Query)', fontsize=12, fontweight='bold')
            axes[0, 0].axis('off')

            # RGB original
            rgb_denorm = denormalize(rgb_img, device).cpu()[0].permute(1, 2, 0).numpy()
            axes[0, 1].imshow(rgb_denorm)
            axes[0, 1].set_title('RGB (Positive)', fontsize=12, fontweight='bold')
            axes[0, 1].axis('off')

            # Aligned RGB
            aligned_rgb_denorm = denormalize(aligned_rgb_img, device).cpu()[0].permute(1, 2, 0).numpy()
            axes[0, 2].imshow(aligned_rgb_denorm)
            axes[0, 2].set_title('Aligned RGB', fontsize=12, fontweight='bold')
            axes[0, 2].axis('off')

            # Empty spaces
            axes[0, 3].axis('off')
            axes[0, 4].axis('off')
            axes[0, 3].text(0.5, 0.5, f'Epoch: {epoch}\nSample: {sample_idx+1}/{num_samples}',
                           ha='center', va='center', fontsize=14, fontweight='bold',
                           transform=axes[0, 3].transAxes)

            # Row 1: Last layer attention (CLS attention to patches)
            # Use pre-computed cls_attention which correctly skips register tokens
            if thermal_cls_attn_last is not None:
                # cls_attention shape: [B, num_heads, num_patches]
                # Average over heads: [num_patches]
                thermal_cls_attn = thermal_cls_attn_last[0].mean(dim=0)  # [num_patches]
                thermal_cls_attn_2d = thermal_cls_attn.cpu().numpy().reshape(grid_h, grid_w)
                im1 = axes[1, 0].imshow(thermal_cls_attn_2d, cmap='hot')
                axes[1, 0].set_title(f'Thermal Last CLS Attn\n(reg_tokens={num_register_tokens})', fontsize=11)
                axes[1, 0].axis('off')
                plt.colorbar(im1, ax=axes[1, 0], fraction=0.046, pad=0.04)
            else:
                axes[1, 0].axis('off')

            if rgb_cls_attn_last is not None:
                rgb_cls_attn = rgb_cls_attn_last[0].mean(dim=0)
                rgb_cls_attn_2d = rgb_cls_attn.cpu().numpy().reshape(grid_h, grid_w)
                im2 = axes[1, 1].imshow(rgb_cls_attn_2d, cmap='hot')
                axes[1, 1].set_title(f'RGB Last CLS Attn\n(reg_tokens={num_register_tokens})', fontsize=11)
                axes[1, 1].axis('off')
                plt.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)
            else:
                axes[1, 1].axis('off')

            # Overlay attention on images
            if thermal_cls_attn_last is not None:
                thermal_cls_attn_resized = np.kron(thermal_cls_attn_2d, np.ones((14, 14)))
                thermal_cls_attn_resized = (thermal_cls_attn_resized - thermal_cls_attn_resized.min()) / \
                                           (thermal_cls_attn_resized.max() - thermal_cls_attn_resized.min() + 1e-8)
                heatmap = plt.cm.hot(thermal_cls_attn_resized)[:, :, :3]
                overlay = 0.5 * thermal_denorm + 0.5 * heatmap
                axes[1, 2].imshow(overlay)
                axes[1, 2].set_title('Thermal + Last Attn', fontsize=11)
                axes[1, 2].axis('off')
            else:
                axes[1, 2].axis('off')

            if rgb_cls_attn_last is not None:
                rgb_cls_attn_resized = np.kron(rgb_cls_attn_2d, np.ones((14, 14)))
                rgb_cls_attn_resized = (rgb_cls_attn_resized - rgb_cls_attn_resized.min()) / \
                                       (rgb_cls_attn_resized.max() - rgb_cls_attn_resized.min() + 1e-8)
                heatmap = plt.cm.hot(rgb_cls_attn_resized)[:, :, :3]
                overlay = 0.5 * rgb_denorm + 0.5 * heatmap
                axes[1, 3].imshow(overlay)
                axes[1, 3].set_title('RGB + Last Attn', fontsize=11)
                axes[1, 3].axis('off')
            else:
                axes[1, 3].axis('off')

            axes[1, 4].text(0.5, 0.5, 'Last Layer\nAttention\n(CLS -> Patches)',
                           ha='center', va='center', fontsize=12,
                           transform=axes[1, 4].transAxes)
            axes[1, 4].axis('off')

            # Row 2: Penultimate layer attention
            # Use pre-computed cls_penultimate_attention which correctly skips register tokens
            if thermal_cls_attn_penult is not None:
                thermal_pen_cls_attn = thermal_cls_attn_penult[0].mean(dim=0)  # [num_patches]
                thermal_pen_cls_attn_2d = thermal_pen_cls_attn.cpu().numpy().reshape(grid_h, grid_w)
                im3 = axes[2, 0].imshow(thermal_pen_cls_attn_2d, cmap='viridis')
                axes[2, 0].set_title('Thermal Penult CLS Attn', fontsize=11)
                axes[2, 0].axis('off')
                plt.colorbar(im3, ax=axes[2, 0], fraction=0.046, pad=0.04)
            else:
                axes[2, 0].axis('off')

            if rgb_cls_attn_penult is not None:
                rgb_pen_cls_attn = rgb_cls_attn_penult[0].mean(dim=0)
                rgb_pen_cls_attn_2d = rgb_pen_cls_attn.cpu().numpy().reshape(grid_h, grid_w)
                im4 = axes[2, 1].imshow(rgb_pen_cls_attn_2d, cmap='viridis')
                axes[2, 1].set_title('RGB Penult CLS Attn', fontsize=11)
                axes[2, 1].axis('off')
                plt.colorbar(im4, ax=axes[2, 1], fraction=0.046, pad=0.04)
            else:
                axes[2, 1].axis('off')

            # Overlay penultimate attention
            if thermal_cls_attn_penult is not None:
                thermal_pen_attn_resized = np.kron(thermal_pen_cls_attn_2d, np.ones((14, 14)))
                thermal_pen_attn_resized = normalize_for_vis(thermal_pen_attn_resized)
                heatmap = plt.cm.viridis(thermal_pen_attn_resized)[:, :, :3]
                overlay = 0.5 * thermal_denorm + 0.5 * heatmap
                axes[2, 2].imshow(overlay)
                axes[2, 2].set_title('Thermal + Penult Attn', fontsize=11)
                axes[2, 2].axis('off')
            else:
                axes[2, 2].axis('off')

            if rgb_cls_attn_penult is not None:
                rgb_pen_attn_resized = np.kron(rgb_pen_cls_attn_2d, np.ones((14, 14)))
                rgb_pen_attn_resized = normalize_for_vis(rgb_pen_attn_resized)
                heatmap = plt.cm.viridis(rgb_pen_attn_resized)[:, :, :3]
                overlay = 0.5 * rgb_denorm + 0.5 * heatmap
                axes[2, 3].imshow(overlay)
                axes[2, 3].set_title('RGB + Penult Attn', fontsize=11)
                axes[2, 3].axis('off')
            else:
                axes[2, 3].axis('off')

            axes[2, 4].text(0.5, 0.5, 'Penultimate Layer\nAttention\n(CLS -> Patches)',
                           ha='center', va='center', fontsize=12,
                           transform=axes[2, 4].transAxes)
            axes[2, 4].axis('off')

            # Row 3: PCA of patch tokens
            # Last layer patches PCA
            thermal_last_pca = apply_pca_to_patches(thermal_last_patches.cpu().numpy(), n_components=3)
            thermal_last_pca_vis = normalize_for_vis(thermal_last_pca[0]).reshape(grid_h, grid_w, 3)
            thermal_last_pca_resized = np.kron(thermal_last_pca_vis, np.ones((14, 14, 1)))
            axes[3, 0].imshow(thermal_last_pca_resized)
            axes[3, 0].set_title('Thermal Last PCA', fontsize=11)
            axes[3, 0].axis('off')

            rgb_last_pca = apply_pca_to_patches(rgb_last_patches.cpu().numpy(), n_components=3)
            rgb_last_pca_vis = normalize_for_vis(rgb_last_pca[0]).reshape(grid_h, grid_w, 3)
            rgb_last_pca_resized = np.kron(rgb_last_pca_vis, np.ones((14, 14, 1)))
            axes[3, 1].imshow(rgb_last_pca_resized)
            axes[3, 1].set_title('RGB Last PCA', fontsize=11)
            axes[3, 1].axis('off')

            # Penultimate layer patches PCA
            if thermal_penultimate_patches is not None:
                thermal_pen_pca = apply_pca_to_patches(thermal_penultimate_patches.cpu().numpy(), n_components=3)
                thermal_pen_pca_vis = normalize_for_vis(thermal_pen_pca[0]).reshape(grid_h, grid_w, 3)
                thermal_pen_pca_resized = np.kron(thermal_pen_pca_vis, np.ones((14, 14, 1)))
                axes[3, 2].imshow(thermal_pen_pca_resized)
                axes[3, 2].set_title('Thermal Penult PCA', fontsize=11)
                axes[3, 2].axis('off')
            else:
                axes[3, 2].axis('off')

            if rgb_penultimate_patches is not None:
                rgb_pen_pca = apply_pca_to_patches(rgb_penultimate_patches.cpu().numpy(), n_components=3)
                rgb_pen_pca_vis = normalize_for_vis(rgb_pen_pca[0]).reshape(grid_h, grid_w, 3)
                rgb_pen_pca_resized = np.kron(rgb_pen_pca_vis, np.ones((14, 14, 1)))
                axes[3, 3].imshow(rgb_pen_pca_resized)
                axes[3, 3].set_title('RGB Penult PCA', fontsize=11)
                axes[3, 3].axis('off')
            else:
                axes[3, 3].axis('off')

            axes[3, 4].text(0.5, 0.5, 'Patch Tokens\nPCA (3 components)\n-> RGB Visualization',
                           ha='center', va='center', fontsize=12,
                           transform=axes[3, 4].transAxes)
            axes[3, 4].axis('off')

            # Overall title
            fig.suptitle(f'Attention Map Visualization (Epoch {epoch}, Sample {sample_idx+1})',
                        fontsize=16, fontweight='bold')

            plt.tight_layout(rect=[0, 0, 1, 0.97])

            # Save
            save_path = os.path.join(save_subdir, f'Epoch_{epoch}_{sample_idx:02d}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"Saved attention visualization: {save_path}")

    model.train()
    print(f"Attention visualization complete for epoch {epoch}")


def visualize_attention_comparison(
    args,
    model,
    dataloader,
    device,
    epoch,
    num_samples=2,
    save_dir='./attention_visualizations',
    comment='default'
):
    """
    Simpler visualization comparing last vs penultimate attention side by side.

    Args:
        args: training arguments
        model: the model
        dataloader: data loader to sample images from
        device: cuda/cpu device
        epoch: current epoch number
        num_samples: number of images to visualize
        save_dir: directory to save visualizations
        comment: experiment comment
    """
    from utils import get_timestamp

    timestamp = get_timestamp()
    save_subdir = os.path.join(save_dir, comment, f'epoch_{epoch:03d}_{timestamp}')
    os.makedirs(save_subdir, exist_ok=True)

    model.eval()

    # Get batch
    images, _, _, aligned_rgbs = next(iter(dataloader))
    batch_size = args.train_batch_size
    sample_size = 1 + 1 + args.negs_num_per_query

    # Thermal indices (first in each sample group)
    thermal_indices = [i * sample_size for i in range(batch_size)]

    num_samples = min(num_samples, batch_size)
    selected = random.sample(range(batch_size), num_samples)

    with torch.no_grad():
        # Get number of register tokens
        num_register_tokens = model.module.shared_backbone.num_register_tokens

        for sample_idx, batch_idx in enumerate(selected):
            thermal_idx = thermal_indices[batch_idx]
            thermal_img = images[thermal_idx:thermal_idx+1].to(device)

            # Forward
            out = model.module.shared_backbone(thermal_img, return_attention=True)

            # Get dimensions
            H, W = thermal_img.shape[2], thermal_img.shape[3]
            grid_h, grid_w = H // 14, W // 14

            # Original image
            thermal_denorm = denormalize(thermal_img, device).cpu()[0].permute(1, 2, 0).numpy()

            # Pre-computed CLS -> patches attention (already skips register tokens)
            cls_attn_last = out.get("cls_attention", None)  # [B, num_heads, num_patches]
            cls_attn_penult = out.get("cls_penultimate_attention", None)

            # Patch tokens (already exclude CLS and register tokens)
            last_patches = out["x_norm_patchtokens"]
            penult_patches = out.get("penultimate_norm_patchtokens", None)

            # Create figure
            fig, axes = plt.subplots(2, 4, figsize=(20, 10))

            # Top row: Last layer
            axes[0, 0].imshow(thermal_denorm)
            axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
            axes[0, 0].axis('off')

            if cls_attn_last is not None:
                # cls_attention: [B, num_heads, num_patches] -> average over heads
                cls_attn = cls_attn_last[0].mean(dim=0).cpu().numpy().reshape(grid_h, grid_w)
                im = axes[0, 1].imshow(cls_attn, cmap='hot')
                axes[0, 1].set_title(f'Last Layer CLS Attn\n(reg_tokens={num_register_tokens})', fontsize=12)
                axes[0, 1].axis('off')
                plt.colorbar(im, ax=axes[0, 1], fraction=0.046)

                # Overlay
                cls_attn_resized = np.kron(cls_attn, np.ones((14, 14)))
                cls_attn_resized = normalize_for_vis(cls_attn_resized)
                heatmap = plt.cm.hot(cls_attn_resized)[:, :, :3]
                axes[0, 2].imshow(0.5 * thermal_denorm + 0.5 * heatmap)
                axes[0, 2].set_title('Last Attn Overlay', fontsize=12)
                axes[0, 2].axis('off')
            else:
                axes[0, 1].axis('off')
                axes[0, 2].axis('off')

            # PCA of last patches
            last_pca = apply_pca_to_patches(last_patches.cpu().numpy(), 3)
            last_pca_vis = normalize_for_vis(last_pca[0]).reshape(grid_h, grid_w, 3)
            axes[0, 3].imshow(np.kron(last_pca_vis, np.ones((14, 14, 1))))
            axes[0, 3].set_title('Last Layer PCA', fontsize=12)
            axes[0, 3].axis('off')

            # Bottom row: Penultimate layer
            axes[1, 0].imshow(thermal_denorm)
            axes[1, 0].set_title('Original Image', fontsize=12, fontweight='bold')
            axes[1, 0].axis('off')

            if cls_attn_penult is not None:
                # cls_penultimate_attention: [B, num_heads, num_patches] -> average over heads
                pen_cls_attn = cls_attn_penult[0].mean(dim=0).cpu().numpy().reshape(grid_h, grid_w)
                im = axes[1, 1].imshow(pen_cls_attn, cmap='viridis')
                axes[1, 1].set_title(f'Penultimate CLS Attn\n(reg_tokens={num_register_tokens})', fontsize=12)
                axes[1, 1].axis('off')
                plt.colorbar(im, ax=axes[1, 1], fraction=0.046)

                # Overlay
                pen_attn_resized = np.kron(pen_cls_attn, np.ones((14, 14)))
                pen_attn_resized = normalize_for_vis(pen_attn_resized)
                heatmap = plt.cm.viridis(pen_attn_resized)[:, :, :3]
                axes[1, 2].imshow(0.5 * thermal_denorm + 0.5 * heatmap)
                axes[1, 2].set_title('Penult Attn Overlay', fontsize=12)
                axes[1, 2].axis('off')
            else:
                axes[1, 1].axis('off')
                axes[1, 2].axis('off')

            # PCA of penultimate patches
            if penult_patches is not None:
                pen_pca = apply_pca_to_patches(penult_patches.cpu().numpy(), 3)
                pen_pca_vis = normalize_for_vis(pen_pca[0]).reshape(grid_h, grid_w, 3)
                axes[1, 3].imshow(np.kron(pen_pca_vis, np.ones((14, 14, 1))))
                axes[1, 3].set_title('Penultimate PCA', fontsize=12)
                axes[1, 3].axis('off')
            else:
                axes[1, 3].axis('off')

            fig.suptitle(f'Last vs Penultimate Attention (Epoch {epoch})',
                        fontsize=14, fontweight='bold')

            plt.tight_layout(rect=[0, 0, 1, 0.96])

            save_path = os.path.join(save_subdir, f'comparison_{sample_idx:02d}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"Saved: {save_path}")

    model.train()


def get_mnn_matches(fm1, fm2):
    """
    Get mutual nearest neighbor matches between two feature maps.

    Args:
        fm1: [H*W, C] features from image 1 (e.g., thermal)
        fm2: [H*W, C] features from image 2 (e.g., RGB)

    Returns:
        idx1: indices of matched keypoints in fm1
        idx2: indices of matched keypoints in fm2
    """
    # Compute similarity matrix
    M = torch.matmul(fm2, fm1.T)  # [l2, l1]

    # Find nearest neighbors in both directions
    max1 = torch.argmax(M, dim=0)  # For each fm1 point, best match in fm2: [l1]
    max2 = torch.argmax(M, dim=1)  # For each fm2 point, best match in fm1: [l2]

    # Check mutual nearest neighbors
    # For each point i in fm1, check if its match in fm2 also matches back to i
    mutual_check = max2[max1]  # [l1] - for each fm1 point, get the fm1 index that fm2 matches to
    valid_mask = torch.arange(fm1.shape[0], device=fm1.device) == mutual_check

    idx1 = torch.nonzero(valid_mask).squeeze(-1)  # Valid indices in fm1
    idx2 = max1[idx1]  # Corresponding indices in fm2

    return idx1, idx2


def visualize_cls_gem_attention(
    model,
    images,
    device,
    save_path,
    modality='thermal'
):
    """
    Visualize CLS token attention and GeM pooling attention maps with PCA visualization.
    - CLS: PCA across attention heads -> RGB
    - GeM: PCA across feature dimensions (patch_tokens * global_desc) -> RGB

    Args:
        model: the model (already in eval mode)
        images: [B, 3, H, W] input images tensor
        device: cuda/cpu device
        save_path: path to save the visualization
        modality: 'thermal' or 'rgb'
    """
    model.eval()

    with torch.no_grad():
        images = images.to(device)
        B = images.shape[0]
        H, W = images.shape[2], images.shape[3]
        grid_h, grid_w = H // 14, W // 14

        # Forward pass through model
        flags = torch.zeros(B, dtype=torch.long, device=device) if modality == 'thermal' else torch.ones(B, dtype=torch.long, device=device)
        outputs = model(images, flags)

        global_desc = outputs[0]   # [B, D]
        patch_tokens = outputs[1]  # [B, N, D]

        # Get multi-head attention from backbone
        backbone_out = model.module.shared_backbone(images, return_attention=True)
        cls_attn_multihead = backbone_out.get("cls_attention", None)  # [B, num_heads, N]

        # Create figure: 5 columns (Original, CLS PCA, CLS overlay, GeM PCA, GeM overlay)
        num_samples = min(B, 4)
        fig, axes = plt.subplots(num_samples, 5, figsize=(25, 5 * num_samples))

        if num_samples == 1:
            axes = axes.reshape(1, -1)

        for i in range(num_samples):
            img_denorm = denormalize(images[i:i+1], images.device).cpu()[0].permute(1, 2, 0).numpy()

            # Col 0: Original image
            axes[i, 0].imshow(img_denorm)
            axes[i, 0].set_title(f'Original ({modality.capitalize()})', fontsize=12, fontweight='bold')
            axes[i, 0].axis('off')

            # CLS PCA: [num_heads, N] -> PCA -> [N, 3] -> RGB
            if cls_attn_multihead is not None:
                cls_heads = cls_attn_multihead[i].cpu().numpy().T  # [N, num_heads]
                pca = PCA(n_components=3)
                cls_pca = pca.fit_transform(cls_heads)  # [N, 3]
                cls_pca_norm = normalize_for_vis(cls_pca)
                cls_pca_vis = cls_pca_norm.reshape(grid_h, grid_w, 3)
                cls_pca_resized = np.kron(cls_pca_vis, np.ones((14, 14, 1)))

                axes[i, 1].imshow(cls_pca_resized)
                axes[i, 1].set_title('CLS Attention PCA', fontsize=12)
                axes[i, 1].axis('off')

                # CLS heatmap overlay (sum over heads)
                cls_attn_sum = cls_attn_multihead[i].sum(dim=0).cpu().numpy().reshape(grid_h, grid_w)
                cls_attn_resized = np.kron(cls_attn_sum, np.ones((14, 14)))
                cls_attn_norm = normalize_for_vis(cls_attn_resized)
                heatmap_cls = plt.cm.hot(cls_attn_norm)[:, :, :3]
                cls_overlay = 0.5 * img_denorm + 0.5 * heatmap_cls
                cls_overlay = np.clip(cls_overlay, 0, 1)
                axes[i, 2].imshow(cls_overlay)
                axes[i, 2].set_title('CLS Overlay', fontsize=12)
                axes[i, 2].axis('off')
            else:
                axes[i, 1].axis('off')
                axes[i, 2].axis('off')

            # GeM PCA: patch_tokens * global_desc -> [N, D] -> PCA -> [N, 3] -> RGB
            pt = patch_tokens[i].cpu().numpy()  # [N, D]
            gd = global_desc[i].cpu().numpy()   # [D]
            gem_contrib = pt * gd[np.newaxis, :]  # [N, D] element-wise contribution

            pca = PCA(n_components=3)
            gem_pca = pca.fit_transform(gem_contrib)  # [N, 3]
            gem_pca_norm = normalize_for_vis(gem_pca)
            gem_pca_vis = gem_pca_norm.reshape(grid_h, grid_w, 3)
            gem_pca_resized = np.kron(gem_pca_vis, np.ones((14, 14, 1)))

            axes[i, 3].imshow(gem_pca_resized)
            axes[i, 3].set_title('GeM Attention PCA', fontsize=12)
            axes[i, 3].axis('off')

            # GeM heatmap overlay (dot product)
            gem_attn = (pt * gd[np.newaxis, :]).sum(axis=1).reshape(grid_h, grid_w)  # [N] -> [H, W]
            gem_attn_resized = np.kron(gem_attn, np.ones((14, 14)))
            gem_attn_norm = normalize_for_vis(gem_attn_resized)
            heatmap_gem = plt.cm.hot(gem_attn_norm)[:, :, :3]
            gem_overlay = 0.5 * img_denorm + 0.5 * heatmap_gem
            gem_overlay = np.clip(gem_overlay, 0, 1)
            axes[i, 4].imshow(gem_overlay)
            axes[i, 4].set_title('GeM Overlay', fontsize=12)
            axes[i, 4].axis('off')

        fig.suptitle('CLS Token PCA vs GeM Pooling PCA', fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.97])

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved attention visualization: {save_path}")


def visualize_cls_gem_attention_from_outputs(
    images,
    cls_attn_map,
    gem_attn_map,
    device,
    save_path,
    modality='thermal',
    patch_tokens=None,
    cls_attn_multihead=None,
    global_desc=None
):
    """
    Visualize CLS token attention and GeM pooling attention maps with PCA visualization.

    Args:
        images: [B, 3, H, W] input images tensor
        cls_attn_map: [B, N] CLS attention map (sum over heads)
        gem_attn_map: [B, N] GeM attention map
        device: cuda/cpu device
        save_path: path to save the visualization
        modality: 'thermal' or 'rgb'
        patch_tokens: [B, N, D] patch token embeddings for GeM PCA
        cls_attn_multihead: [B, num_heads, N] multi-head CLS attention for PCA
        global_desc: [B, D] global descriptor for GeM PCA
    """
    B = images.shape[0]
    H, W = images.shape[2], images.shape[3]
    grid_h, grid_w = H // 14, W // 14

    # Create figure: 5 columns (Original, CLS PCA, CLS overlay, GeM PCA, GeM overlay)
    num_samples = min(B, 4)
    fig, axes = plt.subplots(num_samples, 5, figsize=(25, 5 * num_samples))

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        # Denormalize image
        img_denorm = denormalize(images[i:i+1], images.device).cpu()[0].permute(1, 2, 0).numpy()

        # Col 0: Original image
        axes[i, 0].imshow(img_denorm)
        axes[i, 0].set_title(f'Original ({modality.capitalize()})', fontsize=12, fontweight='bold')
        axes[i, 0].axis('off')

        # CLS PCA visualization
        if cls_attn_multihead is not None:
            # cls_attn_multihead: [B, num_heads, N] -> [N, num_heads] for PCA
            cls_heads = cls_attn_multihead[i].cpu().numpy().T  # [N, num_heads]
            if cls_heads.shape[1] >= 3:
                pca = PCA(n_components=3)
                cls_pca = pca.fit_transform(cls_heads)  # [N, 3]
            else:
                cls_pca = np.zeros((cls_heads.shape[0], 3))
                cls_pca[:, :cls_heads.shape[1]] = cls_heads
            cls_pca_norm = normalize_for_vis(cls_pca)
            cls_pca_vis = cls_pca_norm.reshape(grid_h, grid_w, 3)
            cls_pca_resized = np.kron(cls_pca_vis, np.ones((14, 14, 1)))

            axes[i, 1].imshow(cls_pca_resized)
            axes[i, 1].set_title('CLS Attention PCA', fontsize=12)
            axes[i, 1].axis('off')

            # CLS heatmap overlay (sum over heads)
            cls_attn_sum = cls_attn_multihead[i].sum(dim=0).cpu().numpy().reshape(grid_h, grid_w)
            cls_attn_resized = np.kron(cls_attn_sum, np.ones((14, 14)))
            cls_attn_norm = normalize_for_vis(cls_attn_resized)
            heatmap_cls = plt.cm.hot(cls_attn_norm)[:, :, :3]
            cls_overlay = 0.5 * img_denorm + 0.5 * heatmap_cls
            cls_overlay = np.clip(cls_overlay, 0, 1)
            axes[i, 2].imshow(cls_overlay)
            axes[i, 2].set_title('CLS Overlay', fontsize=12)
            axes[i, 2].axis('off')
        else:
            # Fallback to heatmap if multi-head not available
            cls_attn = cls_attn_map[i].cpu().numpy().reshape(grid_h, grid_w)
            cls_attn_resized = np.kron(cls_attn, np.ones((14, 14)))
            cls_attn_norm = normalize_for_vis(cls_attn_resized)
            axes[i, 1].imshow(cls_attn_norm, cmap='hot')
            axes[i, 1].set_title('CLS Attention (no PCA)', fontsize=12)
            axes[i, 1].axis('off')

            heatmap_cls = plt.cm.hot(cls_attn_norm)[:, :, :3]
            cls_overlay = 0.5 * img_denorm + 0.5 * heatmap_cls
            cls_overlay = np.clip(cls_overlay, 0, 1)
            axes[i, 2].imshow(cls_overlay)
            axes[i, 2].set_title('CLS Overlay', fontsize=12)
            axes[i, 2].axis('off')

        # GeM PCA visualization
        if patch_tokens is not None and global_desc is not None:
            # Compute per-dimension contribution: patch_tokens * global_desc
            # patch_tokens: [B, N, D], global_desc: [B, D]
            pt = patch_tokens[i].cpu().numpy()  # [N, D]
            gd = global_desc[i].cpu().numpy()   # [D]
            # Element-wise contribution of each dimension for each patch
            gem_contrib = pt * gd[np.newaxis, :]  # [N, D]

            # Apply PCA to reduce D -> 3
            pca = PCA(n_components=3)
            gem_pca = pca.fit_transform(gem_contrib)  # [N, 3]
            gem_pca_norm = normalize_for_vis(gem_pca)
            gem_pca_vis = gem_pca_norm.reshape(grid_h, grid_w, 3)
            gem_pca_resized = np.kron(gem_pca_vis, np.ones((14, 14, 1)))

            axes[i, 3].imshow(gem_pca_resized)
            axes[i, 3].set_title('GeM Attention PCA', fontsize=12)
            axes[i, 3].axis('off')

            # GeM heatmap overlay (dot product)
            gem_attn = (pt * gd[np.newaxis, :]).sum(axis=1).reshape(grid_h, grid_w)  # [N] -> [H, W]
            gem_attn_resized = np.kron(gem_attn, np.ones((14, 14)))
            gem_attn_norm = normalize_for_vis(gem_attn_resized)
            heatmap_gem = plt.cm.hot(gem_attn_norm)[:, :, :3]
            gem_overlay = 0.5 * img_denorm + 0.5 * heatmap_gem
            gem_overlay = np.clip(gem_overlay, 0, 1)
            axes[i, 4].imshow(gem_overlay)
            axes[i, 4].set_title('GeM Overlay', fontsize=12)
            axes[i, 4].axis('off')
        else:
            # Fallback to heatmap if patch_tokens not available
            gem_attn = gem_attn_map[i].cpu()
            gem_attn_softmax = torch.softmax(gem_attn, dim=0).numpy().reshape(grid_h, grid_w)
            gem_attn_resized = np.kron(gem_attn_softmax, np.ones((14, 14)))
            gem_attn_norm = normalize_for_vis(gem_attn_resized)
            axes[i, 3].imshow(gem_attn_norm, cmap='hot')
            axes[i, 3].set_title('GeM Attention (no PCA)', fontsize=12)
            axes[i, 3].axis('off')

            heatmap_gem = plt.cm.hot(gem_attn_norm)[:, :, :3]
            gem_overlay = 0.5 * img_denorm + 0.5 * heatmap_gem
            gem_overlay = np.clip(gem_overlay, 0, 1)
            axes[i, 4].imshow(gem_overlay)
            axes[i, 4].set_title('GeM Overlay', fontsize=12)
            axes[i, 4].axis('off')

    fig.suptitle('CLS Token PCA vs GeM Pooling PCA', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved attention visualization: {save_path}")


def visualize_mnn_matches(
    args,
    model,
    dataloader,
    device,
    epoch,
    num_samples=4,
    save_dir='./mnn_visualizations',
    comment='default',
    max_matches=50
):
    """
    Visualize MNN (Mutual Nearest Neighbor) matched lines between thermal and RGB images,
    along with the encoder's last layer attention map.

    Args:
        args: training arguments (must have use_selaVPR_loss or use_reranking=='selaVPR')
        model: the model with LocalAdapt module
        dataloader: data loader to sample images from
        device: cuda/cpu device
        epoch: current epoch number
        num_samples: number of random image pairs to visualize
        save_dir: directory to save visualizations
        comment: experiment comment for subfolder
        max_matches: maximum number of MNN lines to draw
    """
    from utils import get_timestamp

    # Check if selaVPR features are available
    if not (args.use_reranking == 'selaVPR'):
        print("MNN visualization requires use_selaVPR_loss or use_reranking='selaVPR'")
        return

    timestamp = get_timestamp()
    save_subdir = os.path.join(save_dir, comment, f'epoch_{epoch:03d}_{timestamp}')
    os.makedirs(save_subdir, exist_ok=True)

    model.eval()

    # Get a batch of images
    images, _, _, aligned_rgbs = next(iter(dataloader))

    # Calculate batch structure
    batch_size = args.train_batch_size
    sample_size = 1 + 1 + args.negs_num_per_query  # thermal + pos + negs

    # Collect indices
    thermal_indices = [i * sample_size for i in range(batch_size)]
    rgb_pos_indices = [i * sample_size + 1 for i in range(batch_size)]

    # Select random samples
    num_samples = min(num_samples, batch_size)
    selected_batch_indices = random.sample(range(batch_size), num_samples)

    with torch.no_grad():
        for sample_idx, batch_idx in enumerate(selected_batch_indices):
            # Get thermal and RGB images
            thermal_idx = thermal_indices[batch_idx]
            rgb_idx = rgb_pos_indices[batch_idx]

            thermal_img = images[thermal_idx:thermal_idx+1].to(device)
            rgb_img = images[rgb_idx:rgb_idx+1].to(device)

            # Forward pass through backbone to get attention
            thermal_out = model.module.shared_backbone(thermal_img, return_attention=True)
            rgb_out = model.module.shared_backbone(rgb_img, return_attention=True)

            # Get patch tokens for local features
            thermal_patch_tokens = thermal_out["x_norm_patchtokens"]  # [1, N, D]
            rgb_patch_tokens = rgb_out["x_norm_patchtokens"]  # [1, N, D]

            # Get image dimensions
            H, W = thermal_img.shape[2], thermal_img.shape[3]
            H_feat = H // 14
            W_feat = W // 14

            # Compute selaVPR local features through LocalAdapt
            # Reshape patch tokens: [1, N, D] -> [1, D, H_feat, W_feat]
            thermal_x = thermal_patch_tokens.permute(0, 2, 1).view(1, -1, H_feat, W_feat)
            rgb_x = rgb_patch_tokens.permute(0, 2, 1).view(1, -1, H_feat, W_feat)

            # Pass through LocalAdapt
            thermal_local = model.module.local_adapt(thermal_x)  # [1, 128, H_local, W_local]
            rgb_local = model.module.local_adapt(rgb_x)

            # Get local feature dimensions
            _, C_local, H_local, W_local = thermal_local.shape

            # Reshape to [H*W, C] and normalize
            thermal_local_flat = thermal_local.permute(0, 2, 3, 1).view(-1, C_local)  # [H*W, C]
            rgb_local_flat = rgb_local.permute(0, 2, 3, 1).view(-1, C_local)
            thermal_local_flat = torch.nn.functional.normalize(thermal_local_flat, p=2, dim=-1)
            rgb_local_flat = torch.nn.functional.normalize(rgb_local_flat, p=2, dim=-1)

            # Get MNN matches
            idx1, idx2 = get_mnn_matches(thermal_local_flat, rgb_local_flat)

            # Convert to numpy
            idx1_np = idx1.cpu().numpy()
            idx2_np = idx2.cpu().numpy()

            # Get CLS attention maps
            thermal_cls_attn = thermal_out.get("cls_attention", None)
            rgb_cls_attn = rgb_out.get("cls_attention", None)

            # Denormalize images
            thermal_denorm = denormalize(thermal_img, device).cpu()[0].permute(1, 2, 0).numpy()
            rgb_denorm = denormalize(rgb_img, device).cpu()[0].permute(1, 2, 0).numpy()

            # Convert to uint8 for OpenCV
            thermal_vis = (thermal_denorm * 255).astype(np.uint8)
            rgb_vis = (rgb_denorm * 255).astype(np.uint8)

            # Create figure: 2 rows
            # Row 0: MNN matches visualization
            # Row 1: Last layer attention maps
            fig, axes = plt.subplots(2, 3, figsize=(24, 16))

            # ============ Row 0: MNN Matches ============
            # Create side-by-side image
            gap = 20  # gap between images
            combined_img = np.ones((H, 2*W + gap, 3), dtype=np.uint8) * 255
            combined_img[:, :W] = thermal_vis
            combined_img[:, W+gap:] = rgb_vis

            # Convert idx to (x, y) coordinates in the local feature grid
            # Local features are H_local x W_local
            thermal_kp_y = (idx1_np // W_local) * (H / H_local) + (H / H_local / 2)
            thermal_kp_x = (idx1_np % W_local) * (W / W_local) + (W / W_local / 2)

            rgb_kp_y = (idx2_np // W_local) * (H / H_local) + (H / H_local / 2)
            rgb_kp_x = (idx2_np % W_local) * (W / W_local) + (W / W_local / 2) + W + gap

            # Randomly sample matches if too many
            num_matches = len(idx1_np)
            if num_matches > max_matches:
                sample_indices = np.random.choice(num_matches, max_matches, replace=False)
            else:
                sample_indices = np.arange(num_matches)

            # Draw matches
            combined_with_lines = combined_img.copy()
            colors = plt.cm.hsv(np.linspace(0, 1, len(sample_indices)))[:, :3] * 255

            for i, sidx in enumerate(sample_indices):
                pt1 = (int(thermal_kp_x[sidx]), int(thermal_kp_y[sidx]))
                pt2 = (int(rgb_kp_x[sidx]), int(rgb_kp_y[sidx]))
                color = tuple(map(int, colors[i]))
                cv2.line(combined_with_lines, pt1, pt2, color, 1, cv2.LINE_AA)
                cv2.circle(combined_with_lines, pt1, 3, color, -1)
                cv2.circle(combined_with_lines, pt2, 3, color, -1)

            # Plot MNN matches
            axes[0, 0].imshow(combined_with_lines)
            axes[0, 0].set_title(f'MNN Matches: {num_matches} total ({len(sample_indices)} shown)', fontsize=14, fontweight='bold')
            axes[0, 0].axis('off')

            # Plot thermal image with keypoints
            thermal_with_kp = thermal_vis.copy()
            for sidx in sample_indices:
                pt = (int(thermal_kp_x[sidx]), int(thermal_kp_y[sidx]))
                cv2.circle(thermal_with_kp, pt, 4, (0, 255, 0), -1)
            axes[0, 1].imshow(thermal_with_kp)
            axes[0, 1].set_title(f'Thermal Query ({len(sample_indices)} keypoints)', fontsize=12)
            axes[0, 1].axis('off')

            # Plot RGB image with keypoints
            rgb_with_kp = rgb_vis.copy()
            for sidx in sample_indices:
                pt = (int(rgb_kp_x[sidx] - W - gap), int(rgb_kp_y[sidx]))
                cv2.circle(rgb_with_kp, pt, 4, (0, 255, 0), -1)
            axes[0, 2].imshow(rgb_with_kp)
            axes[0, 2].set_title(f'RGB Positive ({len(sample_indices)} keypoints)', fontsize=12)
            axes[0, 2].axis('off')

            # ============ Row 1: Attention Maps ============
            grid_h, grid_w = H // 14, W // 14

            # Thermal attention
            if thermal_cls_attn is not None:
                thermal_attn = thermal_cls_attn[0].mean(dim=0).cpu().numpy().reshape(grid_h, grid_w)
                thermal_attn_resized = np.kron(thermal_attn, np.ones((14, 14)))
                thermal_attn_resized = normalize_for_vis(thermal_attn_resized)
                heatmap = plt.cm.hot(thermal_attn_resized)[:, :, :3]
                overlay = 0.5 * thermal_denorm + 0.5 * heatmap
                axes[1, 0].imshow(overlay)
                axes[1, 0].set_title('Thermal Last Layer Attention', fontsize=12)
            else:
                axes[1, 0].imshow(thermal_denorm)
                axes[1, 0].set_title('Thermal (no attention)', fontsize=12)
            axes[1, 0].axis('off')

            # RGB attention
            if rgb_cls_attn is not None:
                rgb_attn = rgb_cls_attn[0].mean(dim=0).cpu().numpy().reshape(grid_h, grid_w)
                rgb_attn_resized = np.kron(rgb_attn, np.ones((14, 14)))
                rgb_attn_resized = normalize_for_vis(rgb_attn_resized)
                heatmap = plt.cm.hot(rgb_attn_resized)[:, :, :3]
                overlay = 0.5 * rgb_denorm + 0.5 * heatmap
                axes[1, 1].imshow(overlay)
                axes[1, 1].set_title('RGB Last Layer Attention', fontsize=12)
            else:
                axes[1, 1].imshow(rgb_denorm)
                axes[1, 1].set_title('RGB (no attention)', fontsize=12)
            axes[1, 1].axis('off')

            # Info panel
            axes[1, 2].axis('off')
            info_text = (
                f"Epoch: {epoch}\n"
                f"Sample: {sample_idx+1}/{num_samples}\n"
                f"Local Feature Grid: {H_local}x{W_local}\n"
                f"Total MNN Matches: {num_matches}\n"
                f"Displayed: {len(sample_indices)}\n"
                f"Feature Dim: {C_local}"
            )
            axes[1, 2].text(0.5, 0.5, info_text, ha='center', va='center',
                          fontsize=14, transform=axes[1, 2].transAxes,
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            # Overall title
            fig.suptitle(f'MNN Matching Visualization (Epoch {epoch}, Sample {sample_idx+1})',
                        fontsize=16, fontweight='bold')

            plt.tight_layout(rect=[0, 0, 1, 0.97])

            # Save
            save_path = os.path.join(save_subdir, f'mnn_epoch_{epoch}_{sample_idx:02d}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"Saved MNN visualization: {save_path}")

    model.train()
    print(f"MNN visualization complete for epoch {epoch}")


def visualize_sinkhorn_assignment(
    thermal_img,
    rgb_img,
    P,
    save_path,
    top_k=20,
    grid_size=16,
    patch_size=14,
    title="Sinkhorn Assignment",
    gt_distance=None,
    pred_score=None
):
    """
    Sinkhorn assignment matrix를 시각화합니다.

    Args:
        thermal_img: [3, H, W] thermal image tensor (normalized)
        rgb_img: [3, H, W] RGB image tensor (normalized)
        P: [257, 257] or [256, 256] Sinkhorn assignment matrix
        save_path: 저장 경로
        top_k: 시각화할 top-k correspondence 수
        grid_size: patch grid size (16x16 = 256 patches)
        patch_size: 각 patch의 pixel size (14x14 for DINOv2)
        title: 시각화 제목
        gt_distance: ground truth distance (meters)
        pred_score: predicted matching score
    """
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D

    # Denormalize images
    thermal_np = denormalize(thermal_img.unsqueeze(0)).cpu()[0].permute(1, 2, 0).numpy()
    rgb_np = denormalize(rgb_img.unsqueeze(0)).cpu()[0].permute(1, 2, 0).numpy()

    # Remove dustbin if present
    if P.shape[0] == 257:
        P_core = P[:256, :256].cpu().numpy()  # [256, 256]
        dustbin_row = P[:256, 256].cpu().numpy()  # [256] thermal patches' dustbin assignment
        dustbin_col = P[256, :256].cpu().numpy()  # [256] RGB patches' dustbin assignment
    else:
        P_core = P.cpu().numpy()
        dustbin_row = None
        dustbin_col = None

    # Create figure with 2x3 subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # ===== Row 1: Images and Heatmap =====

    # 1-1: Thermal image
    axes[0, 0].imshow(thermal_np)
    axes[0, 0].set_title('Thermal (Query)', fontsize=12)
    axes[0, 0].axis('off')

    # 1-2: RGB image
    axes[0, 1].imshow(rgb_np)
    axes[0, 1].set_title('RGB (Database)', fontsize=12)
    axes[0, 1].axis('off')

    # 1-3: Assignment matrix heatmap
    im = axes[0, 2].imshow(P_core, cmap='hot', aspect='auto')
    axes[0, 2].set_title('Sinkhorn Assignment Matrix (256x256)', fontsize=12)
    axes[0, 2].set_xlabel('RGB Patch Index')
    axes[0, 2].set_ylabel('Thermal Patch Index')
    plt.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)

    # ===== Row 2: Correspondence Visualization =====

    # 2-1: Side-by-side with correspondence lines
    H, W = thermal_np.shape[:2]
    combined = np.concatenate([thermal_np, rgb_np], axis=1)  # [H, 2W, 3]
    axes[1, 0].imshow(combined)
    axes[1, 0].set_title(f'Top-{top_k} Correspondences', fontsize=12)

    # Find top-k correspondences
    flat_indices = np.argsort(P_core.flatten())[-top_k:][::-1]
    thermal_indices = flat_indices // grid_size**2
    rgb_indices = flat_indices % (grid_size**2)

    # Actually, P_core is [256, 256] so:
    thermal_patch_indices = flat_indices // 256
    rgb_patch_indices = flat_indices % 256

    # Draw correspondence lines
    colors = plt.cm.viridis(np.linspace(0, 1, top_k))
    for i, (t_idx, r_idx) in enumerate(zip(thermal_patch_indices, rgb_patch_indices)):
        # Thermal patch center
        t_row, t_col = t_idx // grid_size, t_idx % grid_size
        t_y = t_row * patch_size + patch_size // 2
        t_x = t_col * patch_size + patch_size // 2

        # RGB patch center (offset by image width)
        r_row, r_col = r_idx // grid_size, r_idx % grid_size
        r_y = r_row * patch_size + patch_size // 2
        r_x = r_col * patch_size + patch_size // 2 + W  # offset by thermal image width

        # Draw line
        axes[1, 0].plot([t_x, r_x], [t_y, r_y], color=colors[i], linewidth=1.5, alpha=0.7)
        # Draw points
        axes[1, 0].scatter([t_x], [t_y], c=[colors[i]], s=30, marker='o', edgecolors='white', linewidths=0.5)
        axes[1, 0].scatter([r_x], [r_y], c=[colors[i]], s=30, marker='s', edgecolors='white', linewidths=0.5)

    axes[1, 0].axis('off')

    # 2-2: Thermal with best match overlay
    axes[1, 1].imshow(thermal_np)
    # For each thermal patch, show where its best match is (as heatmap overlay)
    best_matches = P_core.argmax(axis=1)  # [256] - best RGB patch for each thermal patch
    match_confidence = P_core.max(axis=1)  # [256] - confidence

    # Create overlay
    overlay = np.zeros((grid_size, grid_size))
    for t_idx in range(256):
        t_row, t_col = t_idx // grid_size, t_idx % grid_size
        overlay[t_row, t_col] = match_confidence[t_idx]

    overlay_resized = np.kron(overlay, np.ones((patch_size, patch_size)))
    axes[1, 1].imshow(overlay_resized, cmap='jet', alpha=0.5)
    axes[1, 1].set_title('Match Confidence per Thermal Patch', fontsize=12)
    axes[1, 1].axis('off')

    # 2-3: Statistics and info
    axes[1, 2].axis('off')

    # Compute statistics
    diagonal_sum = np.trace(P_core) / 256  # Diagonal dominance (identity-like)
    max_per_row = P_core.max(axis=1).mean()  # Average max confidence
    entropy = -np.sum(P_core * np.log(P_core + 1e-10)) / 256  # Average entropy per row

    # Dustbin usage
    if dustbin_row is not None:
        dustbin_thermal = dustbin_row.mean()
        dustbin_rgb = dustbin_col.mean()
        dustbin_info = f"Dustbin (Thermal→): {dustbin_thermal:.4f}\nDustbin (RGB→): {dustbin_rgb:.4f}"
    else:
        dustbin_info = "Dustbin: N/A"

    info_text = (
        f"=== Sinkhorn Statistics ===\n\n"
        f"Diagonal Dominance: {diagonal_sum:.4f}\n"
        f"(1.0 = perfect identity matching)\n\n"
        f"Avg Max Confidence: {max_per_row:.4f}\n"
        f"Avg Entropy: {entropy:.2f}\n\n"
        f"{dustbin_info}\n\n"
    )

    if gt_distance is not None:
        info_text += f"GT Distance: {gt_distance:.1f}m\n"
    if pred_score is not None:
        info_text += f"Pred Score: {pred_score:.4f}\n"

    # Check if matching is meaningful
    if diagonal_sum > 0.01:
        info_text += "\n✓ Diagonal structure detected\n(spatially coherent matching)"
    else:
        info_text += "\n✗ No diagonal structure\n(random or semantic matching)"

    axes[1, 2].text(0.1, 0.9, info_text, ha='left', va='top',
                    fontsize=11, transform=axes[1, 2].transAxes,
                    family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Overall title
    fig.suptitle(title, fontsize=14, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved Sinkhorn visualization: {save_path}")
