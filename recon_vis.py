import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
from utils import get_timestamp

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
        
    # Denormalize (ImageNet stats 사용했다고 가정)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    
    thermal_img_denorm = thermal_img * std + mean
    aligned_rgb_denorm = aligned_rgb * std + mean
    reconstructed_img_denorm = reconstructed_img * std + mean
    
    # Mask 시각화 (16x16 grid)
    mask_img = mask.reshape(1, 16, 16).float()  # [1, 16, 16]
    mask_img = torch.nn.functional.interpolate(
        mask_img.unsqueeze(1), size=(224, 224), mode='nearest'
    ).squeeze(1)  # [1, 224, 224]
    
    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Thermal
    axes[0, 0].imshow(thermal_img_denorm[0].cpu().permute(1, 2, 0).clip(0, 1))
    axes[0, 0].set_title('Original Thermal', fontsize=14)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(mask_img[0].cpu(), cmap='gray')
    axes[0, 1].set_title(f'Mask (Ratio: {mask.float().mean():.2f})', fontsize=14)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(reconstructed_img_denorm[0].cpu().permute(1, 2, 0).clip(0, 1))
    axes[0, 2].set_title('Reconstructed Thermal', fontsize=14)
    axes[0, 2].axis('off')
    
    # Row 2: RGB + Comparison
    axes[1, 0].imshow(aligned_rgb_denorm[0].cpu().permute(1, 2, 0).clip(0, 1))
    axes[1, 0].set_title('Aligned RGB (Reference)', fontsize=14)
    axes[1, 0].axis('off')
    
    # Masked thermal (visible만 보이게)
    masked_thermal = thermal_img_denorm.clone()
    mask_expanded = mask_img.unsqueeze(1).expand(-1, 3, -1, -1)  # [1, 3, 224, 224]
    masked_thermal[mask_expanded > 0.5] = 0  # Masked region = black
    axes[1, 1].imshow(masked_thermal[0].cpu().permute(1, 2, 0).clip(0, 1))
    axes[1, 1].set_title('Visible Patches Only', fontsize=14)
    axes[1, 1].axis('off')
    
    # Reconstruction error (MSE)
    error = (thermal_img_denorm - reconstructed_img_denorm).abs().mean(dim=1)  # [1, 224, 224]
    im = axes[1, 2].imshow(error[0].cpu(), cmap='hot')
    axes[1, 2].set_title('Reconstruction Error', fontsize=14)
    axes[1, 2].axis('off')
    plt.colorbar(im, ax=axes[1, 2], fraction=0.046)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()
    
    return reconstructed_img_denorm

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
def visualize_during_training(model, triplets_dl, device, epoch, save_dir='./visualizations',comment="default"):
    """Training 중간에 주기적으로 시각화"""
    import os
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, comment), exist_ok=True)
    os.makedirs(os.path.join(save_dir, comment, get_timestamp()), exist_ok=True)
    
    model.eval()
    
    # 첫 번째 batch 가져오기
    images, _, _, aligned_rgbs = next(iter(triplets_dl))
    
    # Thermal query 1개만 추출 (첫 번째 thermal)
    thermal_img = images[0:1].to(device)  # [1, 3, 224, 224]
    aligned_rgb = aligned_rgbs[0:1].to(device)  # [1, 3, 224, 224]
    
    save_path = f"{os.path.join(save_dir, comment, get_timestamp())}/epoch_{epoch:03d}.png"
    visualize_reconstruction(model, thermal_img, aligned_rgb, device, save_path)
    
    model.train()