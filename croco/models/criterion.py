# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
# 
# --------------------------------------------------------
# Criterion to train CroCo
# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae``
# --------------------------------------------------------

import torch
from info_nce import InfoNCE, info_nce
from pytorch_msssim import ms_ssim

# croco의 코드는 github에서 관리 안됨

# class MaskedMSE(torch.nn.Module):
#     def __init__(self, norm_pix_loss=False, masked=True, reduction='mean'):
#         super().__init__()
#         self.norm_pix_loss = norm_pix_loss
#         self.masked = masked
#         self.reduction = reduction
        
#     def forward(self, pred, mask, target):
#         if self.norm_pix_loss:
#             mean = target.mean(dim=-1, keepdim=True)
#             var = target.var(dim=-1, keepdim=True)
#             target = (target - mean) / (var + 1.e-6)**.5
            
#         loss = (pred - target) ** 2
#         loss = loss.mean(dim=-1)  # [B, 256]
        
#         if self.masked:
#             loss = (loss * mask).sum(dim=-1) / mask.sum(dim=-1)  # [B]
#         else:
#             loss = loss.mean(dim=-1)  # [B]
        
#         if self.reduction == 'none':
#             return loss  # [B]
#         elif self.reduction == 'mean':
#             return loss.mean()  # scalar
#         else:
#             return loss.sum()  # scalar

def unpatchify(x, patch_size=14, channels=3):
    """
    x: (N, L, patch_size**2 *channels)
    imgs: (N, 3, H, W)
    """
    h = w = int(x.shape[1]**.5)
    assert h * w == x.shape[1]
    x = x.reshape(shape=(x.shape[0], h, w, patch_size, patch_size, channels))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], channels, h * patch_size, h * patch_size))
    return imgs
    
class MaskedMSE(torch.nn.Module):
    def __init__(self, norm_pix_loss=False, masked=True, reduction='mean', confidence=None, use_ssim=False):
        super().__init__()
        self.norm_pix_loss = norm_pix_loss
        self.masked = masked
        self.reduction = reduction
        self.confidence_map = None
        self.use_ssim = use_ssim
        self.internal_count = 0
        
    def forward(self, pred, mask, target, confidence_map=None):
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
        
        loss = (pred - target) ** 2  # [B, 256, 768]
        loss = loss.mean(dim=-1)     # [B, 256]
        
        # 일단 시각화만 해보기
        confidence_map = None

        # MSE
        if self.masked:
            loss = (loss * mask).sum(dim=-1) / mask.sum(dim=-1)  # [B]
        else:
            loss = loss.mean(dim=-1)  # [B]
            
        # SSIM loss
        if self.use_ssim:
            pred_unpatched = unpatchify(pred)
            target_unpatched = unpatchify(target)
            ssim_losses = ms_ssim(pred_unpatched, target_unpatched, data_range=1.0, size_average=False) # [B]
            ssim_losses = 1 - ssim_losses # 1.0일수록 좋음
            if self.internal_count % 100 == 0:
                print(f"loss: {loss} / SSIM_losses: {ssim_losses}")
            self.internal_count += 1
            loss = (loss + ssim_losses) / 2
        
        if self.reduction == 'none':
            return loss  # [B]
        elif self.reduction == 'mean':
            return loss.mean()  # scalar
        else:
            return loss.sum()  # scalar
        
        
        
'''
        # Confidence weighting with normalization
        if confidence_map is not None:
            confidence_map = confidence_map.squeeze(-1)
            loss = loss * confidence_map  # [B, 256]
            
            if self.masked:
                # Scale-preserving: normalize by confidence sum
                conf_sum = (confidence_map * mask).sum(dim=-1, keepdim=True) + 1e-8  # [B, 1]
                loss = (loss * mask).sum(dim=-1, keepdim=True) / conf_sum  # [B, 1]
                loss = loss.squeeze(-1)  # [B]
            else:
                conf_sum = confidence_map.sum(dim=-1, keepdim=True) + 1e-8  # [B, 1]
                loss = loss.sum(dim=-1, keepdim=True) / conf_sum  # [B, 1]
                loss = loss.squeeze(-1)  # [B]
        else:
'''