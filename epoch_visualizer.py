# epoch_visualizer.py
# Per-epoch visualization for cross-modal VPR (thermal ↔ RGB)
import os
import random
import gc

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torchvision.transforms import v2


# ── ImageNet denormalization ─────────────────────────────────────────────────
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def denormalize(tensor):
    """(1,3,H,W) or (3,H,W) tensor → (H,W,3) uint8 numpy."""
    t = tensor.squeeze(0).cpu().float()
    t = torch.clamp(t * _STD + _MEAN, 0, 1)
    return (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


# ── Helper: masking overlay ──────────────────────────────────────────────────
def mask_to_overlay(img_np, mask_1d, H, W, patch_size=14, alpha=0.55):
    """
    Overlay semi-transparent red on masked patches.
    img_np:  (H, W, 3) uint8
    mask_1d: (N,) bool tensor — True = masked
    """
    H_p = H // patch_size
    W_p = W // patch_size
    mask_grid = mask_1d.cpu().numpy().reshape(H_p, W_p)
    overlay = img_np.copy().astype(np.float32)
    red = np.array([220, 50, 50], dtype=np.float32)
    for r in range(H_p):
        for c in range(W_p):
            if mask_grid[r, c]:
                y0, y1 = r * patch_size, (r + 1) * patch_size
                x0, x1 = c * patch_size, (c + 1) * patch_size
                overlay[y0:y1, x0:x1] = (1 - alpha) * overlay[y0:y1, x0:x1] + alpha * red
    return overlay.astype(np.uint8)


# ── Helper: backbone attention heatmap ───────────────────────────────────────
def _attn_to_spatial(attn_1d, H, W, patch_size=14):
    """(N,) float tensor → (H,W) numpy, bilinear upsampled."""
    H_p, W_p = H // patch_size, W // patch_size
    heat = attn_1d.float().reshape(1, 1, H_p, W_p)
    heat = F.interpolate(heat, size=(H, W), mode='bilinear', align_corners=False)
    heat = heat.squeeze().numpy()
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    return heat


def attn_to_heatmap(attn, H, W, patch_size=14, colormap='turbo'):
    """
    attn: (num_heads, N) → average over heads → (H,W,3) uint8.
    """
    avg  = attn.mean(dim=0).cpu()
    heat = _attn_to_spatial(avg, H, W, patch_size)
    cmap = plt.get_cmap(colormap)
    return (cmap(heat)[:, :, :3] * 255).astype(np.uint8)


def best_head_attn_heatmap(attn, H, W, patch_size=14, colormap='turbo'):
    """
    Best head = highest variance (most structured).
    attn: (num_heads, N) → (H,W,3) uint8.
    """
    best_idx = attn.var(dim=-1).argmax().item()
    best     = attn[best_idx].cpu()
    heat     = _attn_to_spatial(best, H, W, patch_size)
    cmap     = plt.get_cmap(colormap)
    return (cmap(heat)[:, :, :3] * 255).astype(np.uint8)


# ── Helper: decoder cross-attention ─────────────────────────────────────────
def cross_attn_avg_heatmap(attn, H, W, patch_size=14, colormap='plasma'):
    """
    attn: (1, N_q, N_ref) or (N_q, N_ref) tensor.
    Average over query patches → reference attention distribution → heatmap.
    """
    if attn.dim() == 3:
        attn = attn.squeeze(0)      # (N_q, N_ref)
    N_ref = attn.shape[1]
    H_p, W_p = H // patch_size, W // patch_size
    avg  = attn.mean(dim=0).cpu().float()   # (N_ref,)
    # If N_ref == patch_count, reshape to spatial
    if avg.shape[0] == H_p * W_p:
        avg = avg.reshape(1, 1, H_p, W_p)
        heat = F.interpolate(avg, size=(H, W), mode='bilinear', align_corners=False)
        heat = heat.squeeze().numpy()
    else:
        # fallback: just show as 1D bar
        heat = avg.numpy().reshape(1, -1)
        heat = np.repeat(heat, H, axis=0)
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    cmap = plt.get_cmap(colormap)
    return (cmap(heat)[:, :, :3] * 255).astype(np.uint8)


# ── Helper: NxN matrix → square thumbnail ────────────────────────────────────
def matrix_to_img(mat_np, colormap='viridis', thumb=128):
    """
    (N, N) float numpy → (thumb, thumb, 3) uint8 heatmap.
    Reshapes to (g, g, g, g) and averages to get (g, g), then upsample.
    """
    N = mat_np.shape[0]
    g = int(N ** 0.5)
    if g * g == N:
        small = mat_np.reshape(g, g, g, g).mean(axis=(2, 3))  # (g, g)
    else:
        small = mat_np
    small = (small - small.min()) / (small.max() - small.min() + 1e-8)
    # upsample with nearest for crisp blocks
    small_t = torch.from_numpy(small).float().unsqueeze(0).unsqueeze(0)
    big = F.interpolate(small_t, size=(thumb, thumb), mode='nearest').squeeze().numpy()
    cmap = plt.get_cmap(colormap)
    return (cmap(big)[:, :, :3] * 255).astype(np.uint8)


# ── Figure builder ────────────────────────────────────────────────────────────
def make_figure(viz, H, W):
    """
    Build 6-row visualization figure.

    Row 1 [2 col]: Original Thermal | Original RGB
    Row 2 [4 col]: T→R recon | T masking overlay | R→T recon | R masking overlay
    Row 3 [4 col]: Backbone attn T avg | best head | Backbone attn R avg | best head
    Row 4 [2 col]: Decoder cross-attn T decoder | Decoder cross-attn R decoder
    Row 5 [2 col]: Self-Affinity T | Self-Affinity R
    Row 6 [2 col]: A_T2R | Cycle matrix
    """
    BG = '#1a1a2e'
    TC = 'white'
    FS = 8

    fig = plt.figure(figsize=(16, 22), facecolor=BG)
    gs  = gridspec.GridSpec(6, 4, figure=fig, hspace=0.4, wspace=0.05)

    thermal_np = denormalize(viz['thermal_orig'])
    rgb_np     = denormalize(viz['rgb_orig'])
    recon_t_np = denormalize(viz['recon_thermal'])
    recon_r_np = denormalize(viz['recon_rgb'])

    def _ax(row, col_slice, title, img):
        ax = fig.add_subplot(gs[row, col_slice])
        ax.imshow(img)
        ax.set_title(title, color=TC, fontsize=FS, pad=3)
        ax.axis('off')
        return ax

    # ── Row 1: originals ─────────────────────────────────────────────────────
    _ax(0, slice(0, 2), 'Thermal Query (input)',   thermal_np)
    _ax(0, slice(2, 4), 'RGB Reference (aligned)', rgb_np)

    # ── Row 2: reconstruction + masking overlay ──────────────────────────────
    _ax(1, 0, 'Recon Thermal\n(RGB→T)',    recon_t_np)
    mask_t_overlay = mask_to_overlay(thermal_np, viz['mask_thermal'][0], H, W)
    _ax(1, 1, 'Thermal Masking\n(red=masked)', mask_t_overlay)
    _ax(1, 2, 'Recon RGB\n(T→RGB)',        recon_r_np)
    mask_r_overlay = mask_to_overlay(rgb_np, viz['mask_rgb'][0], H, W)
    _ax(1, 3, 'RGB Masking\n(red=masked)', mask_r_overlay)

    # ── Row 3: backbone attention ────────────────────────────────────────────
    if viz['backbone_attn_t'] is not None:
        attn_t = viz['backbone_attn_t'][0]   # (num_heads, N)
        _ax(2, 0, 'Backbone Attn T\n(avg heads)',  attn_to_heatmap(attn_t, H, W))
        _ax(2, 1, 'Backbone Attn T\n(best head)',  best_head_attn_heatmap(attn_t, H, W))
    if viz['backbone_attn_r'] is not None:
        attn_r = viz['backbone_attn_r'][0]
        _ax(2, 2, 'Backbone Attn RGB\n(avg heads)', attn_to_heatmap(attn_r, H, W))
        _ax(2, 3, 'Backbone Attn RGB\n(best head)', best_head_attn_heatmap(attn_r, H, W))

    # ── Row 4: decoder cross-attention ──────────────────────────────────────
    if viz['decoder_cross_attn_t'] is not None:
        _ax(3, slice(0, 2),
            'Decoder Cross-Attn\nT-decoder → RGB-ref (avg query)',
            cross_attn_avg_heatmap(viz['decoder_cross_attn_t'], H, W))
    if viz['decoder_cross_attn_r'] is not None:
        _ax(3, slice(2, 4),
            'Decoder Cross-Attn\nRGB-decoder → T-ref (avg query)',
            cross_attn_avg_heatmap(viz['decoder_cross_attn_r'], H, W))

    # ── Row 5: self-affinity ─────────────────────────────────────────────────
    _ax(4, slice(0, 2), 'Thermal Self-Affinity',
        matrix_to_img(viz['affinity_t'].cpu().numpy()))
    _ax(4, slice(2, 4), 'RGB Self-Affinity',
        matrix_to_img(viz['affinity_r'].cpu().numpy()))

    # ── Row 6: A_T2R + cycle ─────────────────────────────────────────────────
    _ax(5, slice(0, 2), 'A_T2R  (Thermal→RGB transition)',
        matrix_to_img(viz['A_T2R'].cpu().numpy(), colormap='hot'))
    _ax(5, slice(2, 4), 'Cycle Matrix  (T→R→T)',
        matrix_to_img(viz['cycle'].cpu().numpy(), colormap='hot'))

    return fig


# ── Public API ────────────────────────────────────────────────────────────────
def save_epoch_visualizations(args, model, test_ds, epoch_num, save_dir, n_samples=5):
    """
    Save n_samples PNG visualizations for a given epoch.

    Args:
        args:       training args (needs .device, .resize)
        model:      the CrossModalVPR_Net (may be DataParallel-wrapped)
        test_ds:    BaseSTheReODual dataset instance
        epoch_num:  current epoch index
        save_dir:   root directory; files go to save_dir/epoch_XX/sample_YY.png
        n_samples:  number of random query samples to visualize
    """
    save_path = os.path.join(save_dir, f'epoch_{epoch_num:02d}')
    os.makedirs(save_path, exist_ok=True)

    H, W = args.resize[0], args.resize[1]

    # Preprocessing transform (same as training, no augmentation)
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        v2.Resize([H, W]),
    ])

    # Unwrap DataParallel
    actual_model = model.module if hasattr(model, 'module') else model
    was_training = actual_model.training
    actual_model.eval()

    # Sample random query indices
    sampled = random.sample(range(test_ds.queries_num), min(n_samples, test_ds.queries_num))

    saved_count = 0
    with torch.no_grad():
        for i, q_idx in enumerate(sampled):
            try:
                # Thermal query path  (t_img_paths[database_num + q_idx])
                t_path = str(test_ds.t_img_paths[test_ds.database_num + q_idx])
                # Aligned RGB path (same location, same capture time as thermal query)
                r_path = str(test_ds.rgb_img_paths[test_ds.database_num + q_idx])

                t_raw = test_ds.get_thermal_img(t_path)
                r_raw = test_ds.get_rgb_img(r_path)

                t_tensor = transform(t_raw).unsqueeze(0).to(args.device)
                r_tensor = transform(r_raw).unsqueeze(0).to(args.device)

                viz = actual_model.forward_for_viz(t_tensor, r_tensor)

                fig = make_figure(viz, H, W)
                out_path = os.path.join(save_path, f'sample_{i:02d}.png')
                fig.savefig(out_path, dpi=100, bbox_inches='tight', facecolor='#1a1a2e')
                plt.close(fig)

                del viz, t_tensor, r_tensor
                saved_count += 1

            except Exception as e:
                import traceback
                print(f"[Visualizer] sample {i} (q_idx={q_idx}) failed: {e}")
                traceback.print_exc()

    if was_training:
        actual_model.train()

    gc.collect()
    torch.cuda.empty_cache()
    print(f"[Visualizer] epoch {epoch_num:02d}: saved {saved_count}/{n_samples} samples → {save_path}")
