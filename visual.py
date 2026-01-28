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
    if not (args.use_selaVPR_loss or args.use_reranking == 'selaVPR'):
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
