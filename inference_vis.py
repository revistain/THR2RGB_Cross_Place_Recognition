import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import random
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader

# Local imports
import network
import datasets_T2R
from visual import (
    denormalize,
    normalize_for_vis,
    get_mnn_matches,
    apply_pca_to_patches
)
from utils import get_timestamp
import matplotlib.pyplot as plt
import cv2


def load_config_from_checkpoint(checkpoint_path):
    """Load config.yaml from the same directory as checkpoint."""
    checkpoint_dir = os.path.dirname(checkpoint_path)
    config_path = os.path.join(checkpoint_dir, 'config.yaml')

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config, checkpoint_dir


def config_to_args(config):
    """Convert config dict to argparse.Namespace."""
    args = argparse.Namespace(**config)
    return args


def load_model(args, checkpoint_path, device):
    """Load model from checkpoint."""
    # Create model
    model = network.CrossModalVPR_Net(
        args,
        pretrained_foundation=True,
        foundation_model_path=args.foundation_model_path
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Remove 'module.' prefix if present (from DataParallel)
    if list(state_dict.keys())[0].startswith('module'):
        from collections import OrderedDict
        state_dict = OrderedDict({k.replace('module.', ''): v for k, v in state_dict.items()})

    model.load_state_dict(state_dict)
    model = model.to(device)

    # Wrap with DataParallel for compatibility with visualization functions
    model = nn.DataParallel(model)
    model.eval()

    return model, checkpoint


def create_triplet_dataloader(args, dataset_folder='./Dataset'):
    """Create triplet dataloader for visualization."""
    # Create dataset
    triplets_ds = datasets_T2R.TripletsSTheReODual(
        args,
        dataset_folder,
        use_align_rgb=args.use_recon_loss or args.use_pos_as_aligned_rgb
    )

    # Compute simple random triplets for visualization
    # We need to set up triplets_global_indexes
    queries_num = triplets_ds.queries_num
    database_num = triplets_ds.database_num
    negs_num = args.negs_num_per_query

    # Create random triplets
    num_triplets = min(args.queries_per_epoch, queries_num)
    triplets_global_indexes = []

    for _ in range(num_triplets):
        query_idx = random.randint(0, queries_num - 1)
        hard_positives = triplets_ds.hard_positives_per_query[query_idx]

        if len(hard_positives) == 0:
            continue

        pos_idx = random.choice(hard_positives)

        # Random negatives (simplified - just random database images)
        neg_indices = random.sample(range(database_num), min(negs_num, database_num))

        triplets_global_indexes.append(
            torch.tensor([query_idx, pos_idx] + neg_indices)
        )

    if len(triplets_global_indexes) == 0:
        raise ValueError("No valid triplets found")

    triplets_ds.triplets_global_indexes = torch.stack(triplets_global_indexes)
    triplets_ds.is_inference = False

    # Create dataloader
    triplets_dl = DataLoader(
        dataset=triplets_ds,
        num_workers=args.num_workers,
        batch_size=args.train_batch_size,
        collate_fn=datasets_T2R.collate_fn,
        pin_memory=True,
        drop_last=True,
        shuffle=True
    )

    return triplets_dl


def visualize_single_pair(
    args,
    model,
    thermal_img,
    rgb_img,
    device,
    save_path,
    sample_idx=0,
    max_matches=50
):
    """
    Visualize attention maps and MNN matches for a single thermal-RGB pair.

    Args:
        args: arguments namespace
        model: the model
        thermal_img: [1, 3, H, W] thermal image tensor
        rgb_img: [1, 3, H, W] RGB image tensor
        device: cuda/cpu device
        save_path: path to save visualization
        sample_idx: sample index for title
        max_matches: maximum number of MNN lines to draw
    """
    model.eval()

    with torch.no_grad():
        thermal_img = thermal_img.to(device)
        rgb_img = rgb_img.to(device)

        # Forward pass through backbone
        thermal_out = model.module.shared_backbone(thermal_img, return_attention=True)
        rgb_out = model.module.shared_backbone(rgb_img, return_attention=True)

        # Get patch tokens
        thermal_patch_tokens = thermal_out["x_norm_patchtokens"]
        rgb_patch_tokens = rgb_out["x_norm_patchtokens"]

        # Get image dimensions
        H, W = thermal_img.shape[2], thermal_img.shape[3]
        H_feat = H // 14
        W_feat = W // 14
        grid_h, grid_w = H_feat, W_feat

        # Get CLS attention maps
        thermal_cls_attn = thermal_out.get("cls_attention", None)
        rgb_cls_attn = rgb_out.get("cls_attention", None)
        thermal_cls_attn_penult = thermal_out.get("cls_penultimate_attention", None)
        rgb_cls_attn_penult = rgb_out.get("cls_penultimate_attention", None)

        # Denormalize images
        thermal_denorm = denormalize(thermal_img, device).cpu()[0].permute(1, 2, 0).numpy()
        rgb_denorm = denormalize(rgb_img, device).cpu()[0].permute(1, 2, 0).numpy()
        thermal_vis = (thermal_denorm * 255).astype(np.uint8)
        rgb_vis = (rgb_denorm * 255).astype(np.uint8)

        # Check if selaVPR is available
        has_sela = args.use_reranking == 'selaVPR'

        if has_sela and hasattr(model.module, 'local_adapt'):
            # Compute selaVPR local features
            thermal_x = thermal_patch_tokens.permute(0, 2, 1).view(1, -1, H_feat, W_feat)
            rgb_x = rgb_patch_tokens.permute(0, 2, 1).view(1, -1, H_feat, W_feat)

            thermal_local = model.module.local_adapt(thermal_x)
            rgb_local = model.module.local_adapt(rgb_x)

            _, C_local, H_local, W_local = thermal_local.shape

            thermal_local_flat = thermal_local.permute(0, 2, 3, 1).view(-1, C_local)
            rgb_local_flat = rgb_local.permute(0, 2, 3, 1).view(-1, C_local)
            thermal_local_flat = torch.nn.functional.normalize(thermal_local_flat, p=2, dim=-1)
            rgb_local_flat = torch.nn.functional.normalize(rgb_local_flat, p=2, dim=-1)

            idx1, idx2 = get_mnn_matches(thermal_local_flat, rgb_local_flat)
            idx1_np = idx1.cpu().numpy()
            idx2_np = idx2.cpu().numpy()
            num_matches = len(idx1_np)
        else:
            has_sela = False
            num_matches = 0

        # Create figure
        if has_sela:
            fig, axes = plt.subplots(3, 4, figsize=(24, 18))
        else:
            fig, axes = plt.subplots(2, 4, figsize=(24, 12))

        # ============ Row 0: Original Images and Last Layer Attention ============
        axes[0, 0].imshow(thermal_denorm)
        axes[0, 0].set_title('Thermal Query', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(rgb_denorm)
        axes[0, 1].set_title('RGB Positive', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')

        # Last layer attention
        if thermal_cls_attn is not None:
            thermal_attn = thermal_cls_attn[0].mean(dim=0).cpu().numpy().reshape(grid_h, grid_w)
            thermal_attn_resized = np.kron(thermal_attn, np.ones((14, 14)))
            thermal_attn_resized = normalize_for_vis(thermal_attn_resized)
            heatmap = plt.cm.hot(thermal_attn_resized)[:, :, :3]
            overlay = 0.5 * thermal_denorm + 0.5 * heatmap
            axes[0, 2].imshow(overlay)
            axes[0, 2].set_title('Thermal Last Attn', fontsize=12)
        else:
            axes[0, 2].imshow(thermal_denorm)
            axes[0, 2].set_title('Thermal (no attn)', fontsize=12)
        axes[0, 2].axis('off')

        if rgb_cls_attn is not None:
            rgb_attn = rgb_cls_attn[0].mean(dim=0).cpu().numpy().reshape(grid_h, grid_w)
            rgb_attn_resized = np.kron(rgb_attn, np.ones((14, 14)))
            rgb_attn_resized = normalize_for_vis(rgb_attn_resized)
            heatmap = plt.cm.hot(rgb_attn_resized)[:, :, :3]
            overlay = 0.5 * rgb_denorm + 0.5 * heatmap
            axes[0, 3].imshow(overlay)
            axes[0, 3].set_title('RGB Last Attn', fontsize=12)
        else:
            axes[0, 3].imshow(rgb_denorm)
            axes[0, 3].set_title('RGB (no attn)', fontsize=12)
        axes[0, 3].axis('off')

        # ============ Row 1: Penultimate Layer Attention + PCA ============
        if thermal_cls_attn_penult is not None:
            thermal_pen_attn = thermal_cls_attn_penult[0].mean(dim=0).cpu().numpy().reshape(grid_h, grid_w)
            thermal_pen_attn_resized = np.kron(thermal_pen_attn, np.ones((14, 14)))
            thermal_pen_attn_resized = normalize_for_vis(thermal_pen_attn_resized)
            heatmap = plt.cm.viridis(thermal_pen_attn_resized)[:, :, :3]
            overlay = 0.5 * thermal_denorm + 0.5 * heatmap
            axes[1, 0].imshow(overlay)
            axes[1, 0].set_title('Thermal Penult Attn', fontsize=12)
        else:
            axes[1, 0].axis('off')
        axes[1, 0].axis('off')

        if rgb_cls_attn_penult is not None:
            rgb_pen_attn = rgb_cls_attn_penult[0].mean(dim=0).cpu().numpy().reshape(grid_h, grid_w)
            rgb_pen_attn_resized = np.kron(rgb_pen_attn, np.ones((14, 14)))
            rgb_pen_attn_resized = normalize_for_vis(rgb_pen_attn_resized)
            heatmap = plt.cm.viridis(rgb_pen_attn_resized)[:, :, :3]
            overlay = 0.5 * rgb_denorm + 0.5 * heatmap
            axes[1, 1].imshow(overlay)
            axes[1, 1].set_title('RGB Penult Attn', fontsize=12)
        else:
            axes[1, 1].axis('off')
        axes[1, 1].axis('off')

        # PCA of patch tokens
        thermal_pca = apply_pca_to_patches(thermal_patch_tokens.cpu().numpy(), n_components=3)
        thermal_pca_vis = normalize_for_vis(thermal_pca[0]).reshape(grid_h, grid_w, 3)
        thermal_pca_resized = np.kron(thermal_pca_vis, np.ones((14, 14, 1)))
        axes[1, 2].imshow(thermal_pca_resized)
        axes[1, 2].set_title('Thermal PCA', fontsize=12)
        axes[1, 2].axis('off')

        rgb_pca = apply_pca_to_patches(rgb_patch_tokens.cpu().numpy(), n_components=3)
        rgb_pca_vis = normalize_for_vis(rgb_pca[0]).reshape(grid_h, grid_w, 3)
        rgb_pca_resized = np.kron(rgb_pca_vis, np.ones((14, 14, 1)))
        axes[1, 3].imshow(rgb_pca_resized)
        axes[1, 3].set_title('RGB PCA', fontsize=12)
        axes[1, 3].axis('off')

        # ============ Row 2: MNN Matches (if available) ============
        if has_sela:
            # Create side-by-side image
            gap = 20
            combined_img = np.ones((H, 2*W + gap, 3), dtype=np.uint8) * 255
            combined_img[:, :W] = thermal_vis
            combined_img[:, W+gap:] = rgb_vis

            # Convert idx to coordinates
            thermal_kp_y = (idx1_np // W_local) * (H / H_local) + (H / H_local / 2)
            thermal_kp_x = (idx1_np % W_local) * (W / W_local) + (W / W_local / 2)
            rgb_kp_y = (idx2_np // W_local) * (H / H_local) + (H / H_local / 2)
            rgb_kp_x = (idx2_np % W_local) * (W / W_local) + (W / W_local / 2) + W + gap

            # Sample matches
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

            # Plot MNN matches (spans 2 columns)
            axes[2, 0].remove()
            axes[2, 1].remove()
            ax_mnn = fig.add_subplot(3, 2, 5)
            ax_mnn.imshow(combined_with_lines)
            ax_mnn.set_title(f'MNN Matches: {num_matches} total ({len(sample_indices)} shown)',
                           fontsize=14, fontweight='bold')
            ax_mnn.axis('off')

            # Thermal keypoints
            thermal_with_kp = thermal_vis.copy()
            for sidx in sample_indices:
                pt = (int(thermal_kp_x[sidx]), int(thermal_kp_y[sidx]))
                cv2.circle(thermal_with_kp, pt, 4, (0, 255, 0), -1)
            axes[2, 2].imshow(thermal_with_kp)
            axes[2, 2].set_title(f'Thermal Keypoints ({len(sample_indices)})', fontsize=12)
            axes[2, 2].axis('off')

            # RGB keypoints
            rgb_with_kp = rgb_vis.copy()
            for sidx in sample_indices:
                pt = (int(rgb_kp_x[sidx] - W - gap), int(rgb_kp_y[sidx]))
                cv2.circle(rgb_with_kp, pt, 4, (0, 255, 0), -1)
            axes[2, 3].imshow(rgb_with_kp)
            axes[2, 3].set_title(f'RGB Keypoints ({len(sample_indices)})', fontsize=12)
            axes[2, 3].axis('off')

        # Title
        fig.suptitle(f'Visualization Sample {sample_idx}', fontsize=16, fontweight='bold')

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved visualization: {save_path}")

from pathlib import Path
def main():
    parser = argparse.ArgumentParser(description='Inference Visualization')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='Path to checkpoint (.pth file)')
    parser.add_argument('--num_samples', type=int, default=4,
                       help='Number of samples to visualize')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: checkpoint_dir/visualizations)')
    parser.add_argument('--dataset_folder', type=str, default='./Dataset',
                       help='Path to dataset folder')
    parser.add_argument('--max_matches', type=int, default=50,
                       help='Maximum number of MNN lines to draw')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    vis_args = parser.parse_args()

    # Set seed
    random.seed(vis_args.seed)
    np.random.seed(vis_args.seed)
    torch.manual_seed(vis_args.seed)

    # Load config from checkpoint directory
    print(f"Loading config from checkpoint directory...")
    config, checkpoint_dir = load_config_from_checkpoint(vis_args.checkpoint_path)
    args = config_to_args(config)

    # Set adapter_dim based on ViT type (must be done before model creation)
    import backbone.dinov2.block as dinoblock
    model_path = Path(args.foundation_model_path)
    model_name = model_path.parts[-1].lower()
    args.features_dim = 768 if 'vitb' in model_name else 384
    dinoblock.adapter_dim = args.features_dim
    print(f"ViT type: {'ViT-B' if 'vitb' in model_name else 'ViT-S'}, features_dim: {args.features_dim}")

    # Override device
    args.device = vis_args.device

    # Set output directory
    if vis_args.output_dir is None:
        timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        output_dir = os.path.join(checkpoint_dir, 'visualizations', timestamp)
    else:
        output_dir = vis_args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"Output directory: {output_dir}")
    print(f"Checkpoint: {vis_args.checkpoint_path}")
    print(f"selaVPR enabled: {args.use_reranking == 'selaVPR'}")

    # Load model
    print(f"Loading model...")
    device = torch.device(args.device)
    model, checkpoint = load_model(args, vis_args.checkpoint_path, device)

    if 'epoch_num' in checkpoint:
        print(f"Loaded checkpoint from epoch {checkpoint['epoch_num']}")
    if 'best_r1' in checkpoint:
        print(f"Best R@1: {checkpoint['best_r1']:.2f}")

    # Create dataloader
    print(f"Creating dataloader...")
    triplets_dl = create_triplet_dataloader(args, vis_args.dataset_folder)

    # Get a batch
    images, triplets_local_indexes, triplets_global_indexes, aligned_rgbs = next(iter(triplets_dl))

    # Calculate batch structure
    batch_size = args.train_batch_size
    sample_size = 1 + 1 + args.negs_num_per_query

    thermal_indices = [i * sample_size for i in range(batch_size)]
    rgb_pos_indices = [i * sample_size + 1 for i in range(batch_size)]

    # Visualize samples
    num_samples = min(vis_args.num_samples, batch_size)
    selected_indices = random.sample(range(batch_size), num_samples)

    print(f"Visualizing {num_samples} samples...")
    for i, batch_idx in enumerate(selected_indices):
        thermal_idx = thermal_indices[batch_idx]
        rgb_idx = rgb_pos_indices[batch_idx]

        thermal_img = images[thermal_idx:thermal_idx+1]
        rgb_img = images[rgb_idx:rgb_idx+1]

        save_path = os.path.join(output_dir, f'visualization_{i:02d}.png')

        visualize_single_pair(
            args,
            model,
            thermal_img,
            rgb_img,
            device,
            save_path,
            sample_idx=i,
            max_matches=vis_args.max_matches
        )

    print(f"\nVisualization complete!")
    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
