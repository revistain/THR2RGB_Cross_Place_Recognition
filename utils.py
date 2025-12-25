import os
import yaml
import torch
from collections import OrderedDict
from datetime import datetime
import shutil

def save_to_yaml(args, filename='config.yaml'):
    file_path = os.path.join(args.save_dir, filename)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as file:
        yaml.dump(vars(args), file, default_flow_style=False)
        
def save_checkpoint(args, state, is_best, filename):
    model_path = os.path.join(args.save_dir, filename)
    torch.save(state, model_path)
    if is_best:
        shutil.copyfile(model_path, os.path.join(args.save_dir, "best_model.pth"))

def resume_model(resume_path, model, optimizer=None, strict=False):
    checkpoint = torch.load(resume_path, map_location='cuda')
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        # The pre-trained models that we provide in the README do not have 'state_dict' in the keys as
        # the checkpoint is directly the state dict
        state_dict = checkpoint
    # if the model contains the prefix "module" which is appendend by
    # DataParallel, remove it to avoid errors when loading dict
    if list(state_dict.keys())[0].startswith('module'):
        state_dict = OrderedDict({k.replace('module.', ''): v for (k, v) in state_dict.items()})
    model.load_state_dict(state_dict)
    return model

def resume_train(args, model, optimizer=None, strict=False):
    """Load model, optimizer, and other training parameters"""
    checkpoint = torch.load(args.resume)
    start_epoch_num = checkpoint["epoch_num"]
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    best_r5 = checkpoint["best_r5"]
    not_improved_num = checkpoint["not_improved_num"]
    if args.resume.endswith("last_model.pth"):  # Copy best model to current save_dir
        shutil.copy(args.resume.replace("last_model.pth", "best_model.pth"), args.save_dir)
    return model, optimizer, best_r5, start_epoch_num, not_improved_num

cached_timestamp = None
def get_timestamp():
    global cached_timestamp
    if cached_timestamp is None:
        cached_timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    return cached_timestamp

# visualize_reconstruction.py

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

def get_model(model):
    """DataParallel wrapper 제거"""
    if isinstance(model, torch.nn.DataParallel):
        return model.module
    return model

def visualize_reconstruction(model, save_path='reconstruction_vis.png'):
    """
    Reconstruction 결과 시각화
    """
    # DataParallel wrapper 제거
    model = get_model(model)
    
    if model.vis_data is None:
        print("No visualization data available. Train at least one batch first.")
        return
    
    vis_data = model.vis_data
    
    # 1. 데이터 추출
    thermal_input = vis_data['thermal_input']  # [3, 224, 224]
    target_features = vis_data['target_features']  # [256, 768]
    pred_features = vis_data['pred_features']  # [256, 768]
    masks = vis_data['masks']  # [256]
    rgb_reference = vis_data['rgb_reference_img']  # [3, 224, 224]
    
    # 2. Feature를 2D로 reshape
    target_norm = target_features.norm(dim=-1).view(16, 16)  # [16, 16]
    pred_norm = pred_features.norm(dim=-1).view(16, 16)  # [16, 16]
    masks_2d = masks.view(16, 16)  # [16, 16]
    
    # 3. Thermal 이미지 준비
    thermal_img = thermal_input.permute(1, 2, 0).numpy()  # [224, 224, 3]
    thermal_img = (thermal_img - thermal_img.min()) / (thermal_img.max() - thermal_img.min() + 1e-8)
    
    # RGB 이미지 준비
    if rgb_reference is not None:
        rgb_img = rgb_reference.permute(1, 2, 0).numpy()
        rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min() + 1e-8)
    else:
        rgb_img = np.zeros_like(thermal_img)
    
    # 4. Masked thermal 생성
    masked_thermal = thermal_img.copy()
    # 16x16 mask를 224x224로 upsample
    masks_upsampled = F.interpolate(
        masks_2d.unsqueeze(0).unsqueeze(0).float(),
        size=(224, 224),
        mode='nearest'
    ).squeeze().numpy()
    # ============ 수정: mask ratio 표시 업데이트 ============
    mask_ratio_actual = masks.float().mean().item()
    masked_thermal[masks_upsampled > 0.5] = 0.5  # Masked regions를 회색으로
    
    # 5. Feature map 시각화
    target_img = target_norm.numpy()
    pred_img = pred_norm.numpy()
    
    # 6. Difference map
    diff = np.abs(pred_norm.numpy() - target_norm.numpy())
    diff = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
    
    # 7. Mask 시각화
    mask_vis = masks_2d.numpy().astype(float)
    
    # 8. 플롯 생성
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Row 1: 입력 이미지들
    axes[0, 0].imshow(thermal_img)
    axes[0, 0].set_title('Original Thermal', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(masked_thermal)
    # ============ 수정: 실제 mask ratio 표시 ============
    axes[0, 1].set_title(f'Masked Thermal ({mask_ratio_actual*100:.1f}% masked)', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(rgb_img)
    axes[0, 2].set_title('RGB Reference (Cross-Attention)', fontsize=14, fontweight='bold')
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(mask_vis, cmap='gray')
    axes[0, 3].set_title('Mask (White=Masked)', fontsize=14, fontweight='bold')
    axes[0, 3].axis('off')
    
    # Row 2: Feature maps
    im1 = axes[1, 0].imshow(target_img, cmap='viridis')
    axes[1, 0].set_title('Target Features (GT)', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im1, ax=axes[1, 0], fraction=0.046)
    
    im2 = axes[1, 1].imshow(pred_img, cmap='viridis')
    axes[1, 1].set_title('Reconstructed Features', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    plt.colorbar(im2, ax=axes[1, 1], fraction=0.046)
    
    im3 = axes[1, 2].imshow(diff, cmap='hot')
    axes[1, 2].set_title('Absolute Difference', fontsize=14, fontweight='bold')
    axes[1, 2].axis('off')
    plt.colorbar(im3, ax=axes[1, 2], fraction=0.046)
    
    # ============ 수정: L2 distance map으로 변경 ============
    # Cosine similarity 대신 L2 distance 시각화
    l2_dist_map = (pred_features - target_features).norm(dim=-1).view(16, 16).numpy()
    
    im4 = axes[1, 3].imshow(l2_dist_map, cmap='hot')  # hot: 빨강=높은 오차
    axes[1, 3].set_title('L2 Distance', fontsize=14, fontweight='bold')
    axes[1, 3].axis('off')
    plt.colorbar(im4, ax=axes[1, 3], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Visualization saved to {save_path}")
    
    # ============ 수정: 통계 출력 개선 ============
    # L2 distance 계산 (per-patch)
    l2_dist_per_patch = (pred_features - target_features).norm(dim=-1)  # [256]
    
    # Cosine similarity 계산
    cos_sim_per_patch = F.cosine_similarity(
        pred_features.unsqueeze(0),
        target_features.unsqueeze(0),
        dim=-1
    ).squeeze()  # [256]
    
    print(f"\n{'='*50}")
    print(f"Reconstruction Statistics")
    print(f"{'='*50}")
    print(f"Mask ratio: {mask_ratio_actual*100:.2f}% ({masks.sum()}/{len(masks)} patches)")
    print(f"")
    print(f"Target  - mean: {target_features.mean():.4f}, std: {target_features.std():.4f}")
    print(f"Pred    - mean: {pred_features.mean():.4f}, std: {pred_features.std():.4f}")
    print(f"")
    print(f"L2 Distance (전체):")
    print(f"  Mean: {l2_dist_per_patch.mean():.4f}")
    print(f"  Std:  {l2_dist_per_patch.std():.4f}")
    print(f"  Min:  {l2_dist_per_patch.min():.4f}")
    print(f"  Max:  {l2_dist_per_patch.max():.4f}")
    print(f"")
    print(f"L2 Distance (masked only):")
    print(f"  Mean: {l2_dist_per_patch[masks].mean():.4f}")
    print(f"  Std:  {l2_dist_per_patch[masks].std():.4f}")
    print(f"")
    print(f"L2 Distance (visible only):")
    print(f"  Mean: {l2_dist_per_patch[~masks].mean():.4f}")
    print(f"  Std:  {l2_dist_per_patch[~masks].std():.4f}")
    print(f"")
    print(f"Cosine Similarity:")
    print(f"  Masked:  {cos_sim_per_patch[masks].mean():.4f}")
    print(f"  Visible: {cos_sim_per_patch[~masks].mean():.4f}")
    print(f"{'='*50}\n")
    """
    Reconstruction 결과 시각화
    """
    # DataParallel wrapper 제거
    model = get_model(model)
    
    if model.vis_data is None:
        print("No visualization data available. Train at least one batch first.")
        return
    
    vis_data = model.vis_data
    
    # 1. 데이터 추출
    thermal_input = vis_data['thermal_input']  # [3, 224, 224]
    target_features = vis_data['target_features']  # [256, 768]
    pred_features = vis_data['pred_features']  # [256, 768]
    masks = vis_data['masks']  # [256]
    rgb_reference = vis_data['rgb_reference_img']  # [3, 224, 224]
    
    # 2. Feature를 2D로 reshape
    target_norm = target_features.norm(dim=-1).view(16, 16)  # [16, 16]
    pred_norm = pred_features.norm(dim=-1).view(16, 16)  # [16, 16]
    masks_2d = masks.view(16, 16)  # [16, 16]
    
    # 3. Thermal 이미지 준비
    thermal_img = thermal_input.permute(1, 2, 0).numpy()  # [224, 224, 3]
    # ImageNet normalization 역변환 (필요시)
    # mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
    thermal_img = (thermal_img - thermal_img.min()) / (thermal_img.max() - thermal_img.min() + 1e-8)
    
    # RGB 이미지 준비
    if rgb_reference is not None:
        rgb_img = rgb_reference.permute(1, 2, 0).numpy()
        rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min() + 1e-8)
    else:
        rgb_img = np.zeros_like(thermal_img)
    
    # 4. Masked thermal 생성
    masked_thermal = thermal_img.copy()
    # 16x16 mask를 224x224로 upsample
    masks_upsampled = F.interpolate(
        masks_2d.unsqueeze(0).unsqueeze(0).float(),
        size=(224, 224),
        mode='nearest'
    ).squeeze().numpy()
    masked_thermal[masks_upsampled > 0.5] = 0.5  # Masked regions를 회색으로
    
    # 5. Feature map 시각화
    target_img = target_norm.numpy()
    # target_img = (target_img - target_img.min()) / (target_img.max() - target_img.min() + 1e-8)
    
    pred_img = pred_norm.numpy()
    # pred_img = (pred_img - pred_img.min()) / (pred_img.max() - pred_img.min() + 1e-8)
    
    # 6. Difference map
    diff = np.abs(pred_norm.numpy() - target_norm.numpy())
    diff = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
    
    # 7. Mask 시각화
    mask_vis = masks_2d.numpy().astype(float)
    
    # 8. 플롯 생성
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Row 1: 입력 이미지들
    axes[0, 0].imshow(thermal_img)
    axes[0, 0].set_title('Original Thermal', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(masked_thermal)
    axes[0, 1].set_title('Masked Thermal (90% masked)', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(rgb_img)
    axes[0, 2].set_title('RGB Reference (Cross-Attention)', fontsize=14, fontweight='bold')
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(mask_vis, cmap='gray')
    axes[0, 3].set_title('Mask (White=Masked)', fontsize=14, fontweight='bold')
    axes[0, 3].axis('off')
    
    # Row 2: Feature maps
    im1 = axes[1, 0].imshow(target_img, cmap='viridis')
    axes[1, 0].set_title('Target Features (GT)', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im1, ax=axes[1, 0], fraction=0.046)
    
    im2 = axes[1, 1].imshow(pred_img, cmap='viridis')
    axes[1, 1].set_title('Reconstructed Features', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    plt.colorbar(im2, ax=axes[1, 1], fraction=0.046)
    
    im3 = axes[1, 2].imshow(diff, cmap='hot')
    axes[1, 2].set_title('Absolute Difference', fontsize=14, fontweight='bold')
    axes[1, 2].axis('off')
    plt.colorbar(im3, ax=axes[1, 2], fraction=0.046)
    
    # Cosine similarity map
    cos_sim_map = F.cosine_similarity(
        target_features.unsqueeze(0),
        pred_features.unsqueeze(0),
        dim=-1
    ).view(16, 16).numpy()
    
    im4 = axes[1, 3].imshow(cos_sim_map, cmap='RdYlGn', vmin=-1, vmax=1)
    axes[1, 3].set_title('Cosine Similarity', fontsize=14, fontweight='bold')
    axes[1, 3].axis('off')
    plt.colorbar(im4, ax=axes[1, 3], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Visualization saved to {save_path}")
    
    # 통계 출력
    print(f"\n{'='*50}")
    print(f"Reconstruction Statistics")
    print(f"{'='*50}")
    print(f"Target  - mean: {target_features.mean():.4f}, std: {target_features.std():.4f}")
    print(f"Pred    - mean: {pred_features.mean():.4f}, std: {pred_features.std():.4f}")
    print(f"L2 dist - mean: {(pred_features - target_features).norm(dim=-1).mean():.4f}")
    print(f"Cos sim (masked)  : {cos_sim_map[masks_2d.bool()].mean():.4f}")
    print(f"Cos sim (visible) : {cos_sim_map[~masks_2d.bool()].mean():.4f}")
    print(f"{'='*50}\n")