"""
Stage 1 Cross-Attention 시각화 스크립트
- Stage 1 checkpoint 로드 후 decoder cross-attention이 sharp한지 확인
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from network import CrossModalVPR_Net
from Parser import Parser
from datasets_T2R import BaseSTheReODual
from backbone.dinov2 import block as dinoblock


def load_model(checkpoint_path, args):
    """Load model from checkpoint"""
    model = CrossModalVPR_Net(args, pretrained_foundation=True, foundation_model_path=args.foundation_model_path)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint

    # Remove 'module.' prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "")
        new_state_dict[name] = v

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Missing keys: {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")

    return model


def get_cross_attention(model, thermal_img, rgb_img, device):
    """
    Forward pass to extract cross-attention weights from decoder
    """
    model.eval()

    with torch.no_grad():
        # Encode images
        thermal_out = model.shared_backbone(thermal_img, return_attention=True)
        rgb_out = model.shared_backbone(rgb_img, return_attention=True)

        thermal_feat = thermal_out["x_norm_patchtokens"]  # [1, 256, D]
        rgb_feat = rgb_out["x_norm_patchtokens"]  # [1, 256, D]

        B = thermal_feat.shape[0]

        # Prepare decoder input (similar to stage2_forward_distance_v2)
        cls_tokens = model.decoder_cls_token.expand(2*B, -1, -1)
        query_feat = torch.cat([thermal_feat, rgb_feat], dim=0)  # [2B, 256, D]
        ref_feat = torch.cat([rgb_feat, thermal_feat], dim=0)    # [2B, 256, D]

        query_with_cls = torch.cat([cls_tokens, query_feat], dim=1)  # [2B, 257, D]
        query_with_cls = query_with_cls + model.decoder_pos_embed_with_cls
        ref_with_pos = ref_feat + model.decoder_pos_embed

        # Forward through decoder blocks, collecting attention from each layer
        x = query_with_cls
        all_cross_attn = []

        for idx, blk in enumerate(model.decoder_blocks):
            x, cross_attn_weights = blk(x, ref_with_pos, return_attention=True)
            all_cross_attn.append(cross_attn_weights.cpu())

        # Use last layer's attention
        final_attn = all_cross_attn[-1]  # [2B, 257, 256]

        # Thermal→RGB attention (exclude CLS query)
        attn_t2r = final_attn[:B, 1:, :]  # [B, 256, 256]
        # RGB→Thermal attention
        attn_r2t = final_attn[B:, 1:, :]  # [B, 256, 256]

    return attn_t2r[0].numpy(), attn_r2t[0].numpy(), all_cross_attn


def compute_attention_stats(attn):
    """Compute statistics for attention matrix"""
    # Entropy (lower = sharper)
    entropy = -np.sum(attn * np.log(attn + 1e-9), axis=-1).mean()

    # Max confidence (higher = sharper)
    max_conf = attn.max(axis=-1).mean()

    # Diagonal dominance
    diag = np.diag(attn)
    diag_dom = diag.mean()

    # Top-k concentration (what % of attention is in top-k)
    k = 5
    top_k_sum = np.sort(attn, axis=-1)[:, -k:].sum(axis=-1).mean()

    return {
        'entropy': entropy,
        'max_conf': max_conf,
        'diag_dom': diag_dom,
        'top_k_concentration': top_k_sum,
        'uniform_entropy': np.log(256)  # Reference: uniform distribution entropy
    }


def visualize_cross_attention(thermal_img, rgb_img, attn_t2r, attn_r2t,
                               save_path, title_prefix="", stats_t2r=None, stats_r2t=None):
    """
    Visualize cross-attention weights
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # Row 1: Thermal → RGB
    # Original images
    axes[0, 0].imshow(thermal_img)
    axes[0, 0].set_title('Thermal (Query)')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(rgb_img)
    axes[0, 1].set_title('RGB (Reference)')
    axes[0, 1].axis('off')

    # Full attention matrix
    im = axes[0, 2].imshow(attn_t2r, cmap='hot', aspect='auto')
    axes[0, 2].set_title(f'T→R Attention (256x256)')
    axes[0, 2].set_xlabel('RGB Patch Index')
    axes[0, 2].set_ylabel('Thermal Patch Index')
    plt.colorbar(im, ax=axes[0, 2])

    # Attention for specific patches (center patch = 128)
    center_patch = 128
    attn_row = attn_t2r[center_patch, :].reshape(16, 16)
    im2 = axes[0, 3].imshow(attn_row, cmap='hot')
    axes[0, 3].set_title(f'T→R: Patch {center_patch} attention')
    plt.colorbar(im2, ax=axes[0, 3])

    # Row 2: RGB → Thermal
    axes[1, 0].imshow(rgb_img)
    axes[1, 0].set_title('RGB (Query)')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(thermal_img)
    axes[1, 1].set_title('Thermal (Reference)')
    axes[1, 1].axis('off')

    im3 = axes[1, 2].imshow(attn_r2t, cmap='hot', aspect='auto')
    axes[1, 2].set_title(f'R→T Attention (256x256)')
    axes[1, 2].set_xlabel('Thermal Patch Index')
    axes[1, 2].set_ylabel('RGB Patch Index')
    plt.colorbar(im3, ax=axes[1, 2])

    attn_row_r2t = attn_r2t[center_patch, :].reshape(16, 16)
    im4 = axes[1, 3].imshow(attn_row_r2t, cmap='hot')
    axes[1, 3].set_title(f'R→T: Patch {center_patch} attention')
    plt.colorbar(im4, ax=axes[1, 3])

    # Add statistics text
    stats_text = f"""T→R Stats:
Entropy: {stats_t2r['entropy']:.4f} (uniform: {stats_t2r['uniform_entropy']:.2f})
Max Conf: {stats_t2r['max_conf']:.4f} (uniform: {1/256:.4f})
Diag Dom: {stats_t2r['diag_dom']:.4f}
Top-5 Conc: {stats_t2r['top_k_concentration']:.4f}

R→T Stats:
Entropy: {stats_r2t['entropy']:.4f}
Max Conf: {stats_r2t['max_conf']:.4f}
Diag Dom: {stats_r2t['diag_dom']:.4f}
Top-5 Conc: {stats_r2t['top_k_concentration']:.4f}"""

    fig.text(0.02, 0.02, stats_text, fontsize=10, family='monospace',
             verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat'))

    plt.suptitle(f'{title_prefix} Cross-Attention Visualization', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def visualize_attention_per_layer(all_cross_attn, save_path):
    """Visualize attention entropy across decoder layers"""
    num_layers = len(all_cross_attn)
    entropies_t2r = []
    entropies_r2t = []
    max_confs_t2r = []
    max_confs_r2t = []

    for layer_attn in all_cross_attn:
        # layer_attn: [2B, 257, 256]
        attn_t2r = layer_attn[0, 1:, :].numpy()  # [256, 256]
        attn_r2t = layer_attn[1, 1:, :].numpy()  # [256, 256]

        stats_t2r = compute_attention_stats(attn_t2r)
        stats_r2t = compute_attention_stats(attn_r2t)

        entropies_t2r.append(stats_t2r['entropy'])
        entropies_r2t.append(stats_r2t['entropy'])
        max_confs_t2r.append(stats_t2r['max_conf'])
        max_confs_r2t.append(stats_r2t['max_conf'])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    layers = range(1, num_layers + 1)

    axes[0].plot(layers, entropies_t2r, 'b-o', label='T→R')
    axes[0].plot(layers, entropies_r2t, 'r-o', label='R→T')
    axes[0].axhline(y=np.log(256), color='gray', linestyle='--', label='Uniform')
    axes[0].set_xlabel('Decoder Layer')
    axes[0].set_ylabel('Entropy')
    axes[0].set_title('Attention Entropy per Layer (lower = sharper)')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(layers, max_confs_t2r, 'b-o', label='T→R')
    axes[1].plot(layers, max_confs_r2t, 'r-o', label='R→T')
    axes[1].axhline(y=1/256, color='gray', linestyle='--', label='Uniform')
    axes[1].set_xlabel('Decoder Layer')
    axes[1].set_ylabel('Max Confidence')
    axes[1].set_title('Max Attention Confidence per Layer (higher = sharper)')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    # Paths
    stage1_checkpoint = "/home/jwkim/workspace/THR2RGB_Cross_Place_Recognition/logs/biRecon(w10,l1,dep8,r0.6)-attnMask(stage1, 1e-4)/260217_062526/last_model.pth"
    save_dir = "/home/jwkim/workspace/THR2RGB_Cross_Place_Recognition/logs/stage1_crossattn_vis"
    os.makedirs(save_dir, exist_ok=True)

    # Load config from stage1 training
    import yaml
    import argparse

    config_path = "/home/jwkim/workspace/THR2RGB_Cross_Place_Recognition/logs/biRecon(w10,l1,dep8,r0.6)-attnMask(stage1, 1e-4)/260217_062526/config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    args = argparse.Namespace(**config)

    # Add missing args for stage2 compatibility
    args.use_distance_loss = False
    args.use_distance_loss_v2 = True  # Need this for decoder_blocks with return_attention
    args.train_with_negatives = False
    args.sinkhorn_iters = 100
    args.use_warmup = False
    args.warmup_epochs = 5
    args.aux_loss_weight = 0.3
    args.distance_tau = 10.0
    args.inference_cls_only = False
    args.reranking_batch_size = 32
    args.reranking_num_workers = 8
    args.visualize_sinkhorn = False
    args.sinkhorn_vis_samples = 0
    args.use_gem_recon_weight = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set adapter_dim before creating model
    dinoblock.adapter_dim = args.features_dim

    # Load model
    model = load_model(stage1_checkpoint, args)
    model = model.to(device)
    model.eval()

    # Load dataset
    test_transform = transforms.Compose([
        transforms.Resize(args.resize),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Setup args for dataset
    args.sequences = ['Urban']
    args.test_method = 'hard_resize'
    args.img_time = 'allday'

    # Load test dataset
    dataset_folder = "/home/jwkim/workspace/THR2RGB_Cross_Place_Recognition/Dataset/save_mat"
    test_dataset = BaseSTheReODual(args, dataset_folder, split='test')

    # Get some positive pairs
    print("\nProcessing samples...")
    num_samples = 10

    for i in range(num_samples):
        # Get a query and its positive
        query_idx = i * 100  # Sample every 100th query
        if query_idx >= len(test_dataset.t_queries_paths):
            break

        query_path = test_dataset.t_queries_paths[query_idx]  # Thermal query

        # Find positive (closest database image)
        query_utm = test_dataset.queries_utms[query_idx]

        # Calculate distances to all database images
        db_utms = np.array(test_dataset.database_utms)
        distances = np.linalg.norm(db_utms - query_utm, axis=1)
        closest_db_idx = np.argmin(distances)
        min_distance = distances[closest_db_idx]

        db_path = test_dataset.rgb_database_paths[closest_db_idx]  # RGB database

        print(f"\nSample {i}: Query {query_idx}, DB {closest_db_idx}, Distance: {min_distance:.2f}m")

        # Load images
        thermal_img_pil = Image.open(query_path).convert('RGB')
        rgb_img_pil = Image.open(db_path).convert('RGB')

        thermal_tensor = test_transform(thermal_img_pil).unsqueeze(0).to(device)
        rgb_tensor = test_transform(rgb_img_pil).unsqueeze(0).to(device)

        # Get cross-attention
        attn_t2r, attn_r2t, all_cross_attn = get_cross_attention(model, thermal_tensor, rgb_tensor, device)

        # Compute stats
        stats_t2r = compute_attention_stats(attn_t2r)
        stats_r2t = compute_attention_stats(attn_r2t)

        print(f"  T→R: entropy={stats_t2r['entropy']:.4f}, max_conf={stats_t2r['max_conf']:.4f}")
        print(f"  R→T: entropy={stats_r2t['entropy']:.4f}, max_conf={stats_r2t['max_conf']:.4f}")
        print(f"  Uniform reference: entropy={np.log(256):.4f}, max_conf={1/256:.4f}")

        # Visualize
        thermal_img_np = np.array(thermal_img_pil.resize((224, 224)))
        rgb_img_np = np.array(rgb_img_pil.resize((224, 224)))

        save_path = os.path.join(save_dir, f"crossattn_{i:03d}_dist{min_distance:.1f}m.png")
        visualize_cross_attention(
            thermal_img_np, rgb_img_np, attn_t2r, attn_r2t,
            save_path, f"Sample {i} (dist={min_distance:.1f}m)",
            stats_t2r, stats_r2t
        )

        # Per-layer visualization (only for first sample)
        if i == 0:
            layer_save_path = os.path.join(save_dir, "layer_analysis.png")
            visualize_attention_per_layer(all_cross_attn, layer_save_path)

    print(f"\n=== Done! Results saved to {save_dir} ===")


if __name__ == "__main__":
    main()
