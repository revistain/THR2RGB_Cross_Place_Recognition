# recon_vis.py
"""
Reconstruction Visualization Module for Cross-Modal VPR

모든 시각화 관련 기능을 통합:
- Bidirectional reconstruction visualization
- Attention map visualization
- Training 중 visualization
"""

import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from torchvision.utils import save_image, make_grid


# ImageNet normalization constants
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


def denormalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """
    Denormalize a tensor image with mean and standard deviation.

    Args:
        tensor: [C, H, W] or [B, C, H, W] normalized tensor
        mean: normalization mean
        std: normalization std

    Returns:
        Denormalized tensor in [0, 1] range
    """
    if tensor.dim() == 3:
        # [C, H, W]
        mean = mean.view(3, 1, 1).to(tensor.device)
        std = std.view(3, 1, 1).to(tensor.device)
    elif tensor.dim() == 4:
        # [B, C, H, W]
        mean = mean.view(1, 3, 1, 1).to(tensor.device)
        std = std.view(1, 3, 1, 1).to(tensor.device)

    return (tensor * std + mean).clamp(0, 1)


def unpatchify(patches, h_feat, w_feat, patch_size=14):
    """
    Convert patchified tensor back to image.

    Args:
        patches: [B, N, patch_size**2 * 3]
        h_feat: height in patches
        w_feat: width in patches
        patch_size: size of each patch

    Returns:
        [B, 3, H, W] image tensor
    """
    B = patches.shape[0]
    patches = patches.reshape(B, h_feat, w_feat, patch_size, patch_size, 3)
    patches = torch.einsum('nhwpqc->nchpwq', patches)
    imgs = patches.reshape(B, 3, h_feat * patch_size, w_feat * patch_size)
    return imgs


def apply_mask_overlay(img, mask, h_feat, w_feat, color=(0.5, 0.5, 0.5)):
    """
    Apply mask overlay to image (masked regions shown in gray).

    Args:
        img: [B, C, H, W] image tensor
        mask: [B, N] boolean mask (True = masked)
        h_feat, w_feat: feature map dimensions
        color: RGB tuple for masked regions

    Returns:
        [B, C, H, W] image with mask overlay
    """
    B, C, H, W = img.shape

    # mask: [B, N] -> [B, 1, h_feat, w_feat]
    mask_2d = mask.view(B, h_feat, w_feat).unsqueeze(1).float()
    # upscale to image size
    mask_img = F.interpolate(mask_2d, size=(H, W), mode='nearest')

    # Create color tensor
    color_tensor = torch.tensor(color, device=img.device).view(1, 3, 1, 1)

    # Apply mask: original where not masked, color where masked
    masked_img = img * (1 - mask_img) + color_tensor * mask_img
    return masked_img


def visualize_reconstruction_grid(vis_data, save_path=None, show=False, title=None):
    """
    Create a grid visualization of reconstruction results.

    Args:
        vis_data: dict from model.visualize_reconstruction()
            Keys: 'intra_thermal', 'intra_rgb', 'inter_t2r', 'inter_r2t'
            Each contains: 'input', 'masked', 'recon', 'ref'
        save_path: path to save the figure
        show: whether to display the figure
        title: optional title for the figure

    Returns:
        matplotlib figure
    """
    # Count valid pairs
    valid_pairs = [k for k in vis_data.keys() if vis_data[k] is not None]
    n_pairs = len(valid_pairs)

    if n_pairs == 0:
        return None

    # Create figure
    fig, axes = plt.subplots(n_pairs, 4, figsize=(16, 4 * n_pairs))
    if n_pairs == 1:
        axes = axes.reshape(1, -1)

    pair_titles = {
        'intra_thermal': 'Intra-Thermal (Query Thermal <- Pos Thermal)',
        'intra_rgb': 'Intra-RGB (Similar RGB <- RGB-similar RGB)',
        'inter_t2r': 'Inter T2R (Query Thermal <- Similar RGB)',
        'inter_r2t': 'Inter R2T (Similar RGB <- Query Thermal)',
    }

    col_titles = ['Input', 'Masked', 'Reconstructed', 'Reference']

    for row_idx, pair_name in enumerate(valid_pairs):
        data = vis_data[pair_name]

        # Denormalize all images
        input_img = denormalize(data['input'])
        masked_img = denormalize(data['masked'])
        recon_img = denormalize(data['recon'])
        ref_img = denormalize(data['ref'])

        images = [input_img, masked_img, recon_img, ref_img]

        for col_idx, (img, col_title) in enumerate(zip(images, col_titles)):
            ax = axes[row_idx, col_idx]

            # Convert to numpy for display
            if img.dim() == 3:
                img_np = img.permute(1, 2, 0).cpu().numpy()
            else:
                img_np = img[0].permute(1, 2, 0).cpu().numpy()

            ax.imshow(img_np.clip(0, 1))
            ax.axis('off')

            if row_idx == 0:
                ax.set_title(col_title, fontsize=12, fontweight='bold')

        # Row label
        axes[row_idx, 0].set_ylabel(pair_titles.get(pair_name, pair_name),
                                     fontsize=10, rotation=0, ha='right', va='center')

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def save_reconstruction_images(vis_data, save_dir, epoch=None, prefix=""):
    """
    Save reconstruction images as individual files.

    Args:
        vis_data: dict from model.visualize_reconstruction()
        save_dir: directory to save images
        epoch: optional epoch number for filename
        prefix: optional prefix for filename
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for pair_name, data in vis_data.items():
        if data is None:
            continue

        # Denormalize all images
        input_img = denormalize(data['input'])
        masked_img = denormalize(data['masked'])
        recon_img = denormalize(data['recon'])
        ref_img = denormalize(data['ref'])

        # Create horizontal grid: input | masked | recon | ref
        grid = torch.cat([input_img, masked_img, recon_img, ref_img], dim=2)

        # Build filename
        if epoch is not None:
            filename = f"{prefix}epoch{epoch:03d}_{pair_name}.png"
        else:
            filename = f"{prefix}{pair_name}.png"

        save_path = save_dir / filename
        save_image(grid, save_path)


def visualize_during_training(model, dataloader, device, epoch, save_dir, args=None):
    """
    Training 중 주기적으로 시각화 수행.

    Args:
        model: CrossModalVPR_Net (wrapped in DataParallel)
        dataloader: triplets dataloader
        device: cuda or cpu
        epoch: current epoch number
        save_dir: directory to save visualizations
        args: training arguments (optional)
    """
    # Get the underlying model
    net = model.module if hasattr(model, 'module') else model

    was_training = net.training
    net.eval()

    with torch.no_grad():
        # Get a sample batch
        sample_batch = next(iter(dataloader))
        images, _, _, _, recon_images = sample_batch

        if recon_images is None:
            if was_training:
                net.train()
            return

        # Move to device
        images = images.to(device)
        recon_images_device = {}
        for key in recon_images:
            if recon_images[key] is not None:
                recon_images_device[key] = recon_images[key].to(device)
            else:
                recon_images_device[key] = None

        # Get visualization data
        vis_data = net.visualize_reconstruction(
            images,
            recon_images_device,
            batch_idx=0
        )

        # Save images
        vis_dir = Path(save_dir) / "recon_vis"
        save_reconstruction_images(vis_data, vis_dir, epoch=epoch)

        # Also save a combined grid figure
        grid_path = vis_dir / f"epoch{epoch:03d}_grid.png"
        visualize_reconstruction_grid(
            vis_data,
            save_path=grid_path,
            title=f"Epoch {epoch} Reconstruction"
        )

    if was_training:
        net.train()


def visualize_attention_maps(model, thermal_img, rgb_img, device='cuda', save_path=None):
    """
    Visualize attention maps from the model.

    Args:
        model: CrossModalVPR_Net
        thermal_img: [1, 3, H, W] thermal image
        rgb_img: [1, 3, H, W] RGB image
        device: cuda or cpu
        save_path: path to save the figure
    """
    net = model.module if hasattr(model, 'module') else model
    net.eval()

    with torch.no_grad():
        thermal_img = thermal_img.to(device)
        rgb_img = rgb_img.to(device)

        # Get attention from backbone
        thermal_out = net.shared_backbone(thermal_img, return_attention=True)
        rgb_out = net.shared_backbone(rgb_img, return_attention=True)

        # CLS attention: [B, num_heads, N] -> [B, N]
        thermal_cls_attn = thermal_out.get("cls_attention")
        rgb_cls_attn = rgb_out.get("cls_attention")

        if thermal_cls_attn is None or rgb_cls_attn is None:
            print("Attention maps not available")
            return

        # Average over heads
        thermal_attn = thermal_cls_attn.mean(dim=1)  # [1, N]
        rgb_attn = rgb_cls_attn.mean(dim=1)  # [1, N]

        # Reshape to 2D
        H, W = thermal_img.shape[2], thermal_img.shape[3]
        h_feat, w_feat = H // 14, W // 14

        thermal_attn_2d = thermal_attn.view(1, h_feat, w_feat)
        rgb_attn_2d = rgb_attn.view(1, h_feat, w_feat)

        # Upsample to image size
        thermal_attn_up = F.interpolate(
            thermal_attn_2d.unsqueeze(1),
            size=(H, W),
            mode='bilinear',
            align_corners=False
        ).squeeze(1)
        rgb_attn_up = F.interpolate(
            rgb_attn_2d.unsqueeze(1),
            size=(H, W),
            mode='bilinear',
            align_corners=False
        ).squeeze(1)

        # Denormalize images
        thermal_denorm = denormalize(thermal_img[0]).cpu()
        rgb_denorm = denormalize(rgb_img[0]).cpu()

        # Plot
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Thermal row
        axes[0, 0].imshow(thermal_denorm.permute(1, 2, 0).numpy())
        axes[0, 0].set_title('Thermal Image', fontweight='bold')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(thermal_attn_up[0].cpu().numpy(), cmap='hot')
        axes[0, 1].set_title('Thermal CLS Attention', fontweight='bold')
        axes[0, 1].axis('off')

        # Overlay
        axes[0, 2].imshow(thermal_denorm.permute(1, 2, 0).numpy())
        axes[0, 2].imshow(thermal_attn_up[0].cpu().numpy(), cmap='hot', alpha=0.5)
        axes[0, 2].set_title('Thermal + Attention Overlay', fontweight='bold')
        axes[0, 2].axis('off')

        # RGB row
        axes[1, 0].imshow(rgb_denorm.permute(1, 2, 0).numpy())
        axes[1, 0].set_title('RGB Image', fontweight='bold')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(rgb_attn_up[0].cpu().numpy(), cmap='hot')
        axes[1, 1].set_title('RGB CLS Attention', fontweight='bold')
        axes[1, 1].axis('off')

        axes[1, 2].imshow(rgb_denorm.permute(1, 2, 0).numpy())
        axes[1, 2].imshow(rgb_attn_up[0].cpu().numpy(), cmap='hot', alpha=0.5)
        axes[1, 2].set_title('RGB + Attention Overlay', fontweight='bold')
        axes[1, 2].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved attention visualization to {save_path}")
            plt.close()
        else:
            plt.show()

        return fig


def visualize_mask_pattern(mask, h_feat, w_feat, save_path=None, title="Mask Pattern"):
    """
    Visualize the masking pattern.

    Args:
        mask: [B, N] boolean mask (True = masked)
        h_feat, w_feat: feature map dimensions
        save_path: path to save the figure
        title: figure title
    """
    B = mask.shape[0]

    fig, axes = plt.subplots(1, min(B, 4), figsize=(4 * min(B, 4), 4))
    if B == 1:
        axes = [axes]

    for i in range(min(B, 4)):
        mask_2d = mask[i].view(h_feat, w_feat).cpu().numpy()
        axes[i].imshow(mask_2d, cmap='RdYlGn_r', vmin=0, vmax=1)
        mask_ratio = mask[i].float().mean().item() * 100
        axes[i].set_title(f'Sample {i} ({mask_ratio:.1f}% masked)')
        axes[i].axis('off')

    plt.suptitle(title, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    return fig
