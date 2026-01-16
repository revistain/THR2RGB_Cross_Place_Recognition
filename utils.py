import os
import yaml
import torch
import shutil
import numpy as np
from einops import rearrange
from datetime import datetime
from skimage.feature import hog
from skimage.color import rgb2gray
from collections import OrderedDict

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

def extract_hog_simple(image):
    """
    Args: image: [3, 224, 224] torch.Tensor (RGB)
    Returns: hog_features: [256, 9] torch.Tensor
    """
    # 1. Numpy로 변환 & [C,H,W] -> [H,W,C]
    img = image.cpu().numpy().transpose(1, 2, 0)
    gray = rgb2gray(img)
    
    # 2. HoG 추출
    hog_feat = hog(
        gray,
        orientations=9,
        pixels_per_cell=(7, 7),
        cells_per_block=(1, 1),
        block_norm='L2-Hys',
        feature_vector=False
    )
    
    # 4. Reshape: [16, 16, 1, 1, 9] -> [256, 9]
    hog_features = rearrange(
        hog_feat,
        '(h p1) (w p2) 1 1 c -> (h w) (p1 p2 c)',
        h=16, w=16, p1=2, p2=2
    )
    
    
    # visualize_hog_simple(image)
    
    return torch.tensor(hog_features, dtype=torch.float32)

import matplotlib.pyplot as plt
def visualize_hog_simple(image, save_path=None):
    """
    HOG feature 시각화 (간단 버전)
    
    Args:
        image: [3, 224, 224] torch.Tensor
        save_path: 저장 경로 (optional)
    """
    # 1. Prepare
    img = image.cpu().numpy().transpose(1, 2, 0)
    gray = rgb2gray(img)
    
    # 2. HOG with visualization
    hog_feat, hog_image = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        feature_vector=False,
        visualize=True  # ← 시각화!
    )
    
    # 3. Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original RGB
    axes[0].imshow(img)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Grayscale
    axes[1].imshow(gray, cmap='gray')
    axes[1].set_title('Grayscale', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # HOG visualization
    axes[2].imshow(hog_image, cmap='hot')
    axes[2].set_title('HOG Features', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()

def extract_hog_batch(images):
    """
    Args:
        images: [B, 3, 224, 224] torch.Tensor
    
    Returns:
        hog_features: [B, 256, 9] torch.Tensor
    """
    B = images.size(0)
    device = images.device
    
    hog_list = []
    for b in range(B):
        hog_feat = extract_hog_simple(images[b])
        hog_list.append(hog_feat)
    
    return torch.stack(hog_list, dim=0).to(device)