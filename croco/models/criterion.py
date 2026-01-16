# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
# 
# --------------------------------------------------------
# Criterion to train CroCo
# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

import torch
from info_nce import InfoNCE, info_nce
from pytorch_msssim import ms_ssim

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
    def __init__(self, norm_pix_loss=False, masked=True, reduction='mean', loss_type='mse'):
        """
        Args:
            norm_pix_loss: bool - normalize target
            masked: bool - apply mask
            reduction: str - 'none', 'mean', 'sum'
            loss_type: str - 'mse', 'l1', 'ssim', 'mse+ssim'
        """
        super().__init__()
        self.norm_pix_loss = norm_pix_loss
        self.masked = masked
        self.reduction = reduction
        self.loss_type = loss_type
        
    def forward(self, pred, mask, target):
        """
        Args:
            pred: [B, 256, 588] - predicted patches
            mask: [B, 256] - binary mask
            target: [B, 256, 588] - target patches
        Returns:
            loss: scalar or [B] depending on reduction
        """
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
        
        # ========== Loss 계산 (type별로) ==========
        if self.loss_type == 'mse':
            # MSE Loss (기존)
            loss = (pred - target) ** 2  # [B, 256, 588]
            loss = loss.mean(dim=-1)     # [B, 256]
            
        elif self.loss_type == 'l1':
            # L1 Loss
            loss = torch.abs(pred - target)  # [B, 256, 588]
            loss = loss.mean(dim=-1)         # [B, 256]
            
        elif self.loss_type == 'ssim':
            # SSIM Loss (이미지로 변환 필요)
            pred_img = unpatchify(pred)    # [B, 3, 224, 224]
            target_img = unpatchify(target)
            
            # MS-SSIM (1 - SSIM, higher is worse)
            ssim_value = ms_ssim(
                pred_img, 
                target_img, 
                data_range=1.0,  # normalized to [0, 1]
                size_average=False  # [B]
            )
            
            loss = 1 - ssim_value  # [B]
            
            # Masked 영역만 (patch level로 변환)
            if self.masked:
                # SSIM은 이미지 전체에 대한 loss이므로
                # mask 비율로 scaling
                mask_ratio = mask.float().mean(dim=1)  # [B]
                loss = loss * mask_ratio
            
            # reduction 처리
            if self.reduction == 'none':
                return loss
            elif self.reduction == 'mean':
                return loss.mean()
            else:
                return loss.sum()
            
        elif self.loss_type == 'mse+ssim':
            # MSE + SSIM 조합
            mse_loss = (pred - target) ** 2
            mse_loss = mse_loss.mean(dim=-1)  # [B, 256]
            
            # SSIM
            pred_img = unpatchify(pred)
            target_img = unpatchify(target)
            ssim_value = ms_ssim(
                pred_img, 
                target_img, 
                data_range=1.0,
                size_average=False
            )
            ssim_loss = 1 - ssim_value  # [B]
            
            # MSE는 patch-level, SSIM은 image-level
            # MSE 먼저 처리
            if self.masked:
                mse_loss = (mse_loss * mask).sum(dim=-1) / mask.sum(dim=-1)  # [B]
            else:
                mse_loss = mse_loss.mean(dim=-1)  # [B]
            
            # SSIM에 mask 비율 적용
            if self.masked:
                mask_ratio = mask.float().mean(dim=1)
                ssim_loss = ssim_loss * mask_ratio
            
            # 조합 (0.5 : 0.5)
            # ssim_ratio = 0.84
            ssim_ratio = 0.5
            loss = ssim_ratio * mse_loss + (1-ssim_ratio) * ssim_loss  # [B]
            
            # reduction
            if self.reduction == 'none':
                return loss
            elif self.reduction == 'mean':
                return loss.mean()
            else:
                return loss.sum()
        
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")
        # =========================================
        
        # MSE, L1은 patch-level이므로 mask 적용
        if self.loss_type in ['mse', 'l1']:
            if self.masked:
                loss = (loss * mask).sum(dim=-1) / mask.sum(dim=-1)  # [B]
            else:
                loss = loss.mean(dim=-1)  # [B]

            if self.reduction == 'none':
                return loss  # [B]
            elif self.reduction == 'mean':
                return loss.mean()  # scalar
            else:
                return loss.sum()  # scalar
            

# # Copyright (C) 2022-present Naver Corporation. All rights reserved.
# # Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
# # 
# # --------------------------------------------------------
# # Criterion to train CroCo
# # --------------------------------------------------------
# # References:
# # MAE: https://github.com/facebookresearch/mae``
# # --------------------------------------------------------

# import torch
# from info_nce import InfoNCE, info_nce
# from pytorch_msssim import ms_ssim

# # croco의 코드는 github에서 관리 안됨

# # class MaskedMSE(torch.nn.Module):
# #     def __init__(self, norm_pix_loss=False, masked=True, reduction='mean'):
# #         super().__init__()
# #         self.norm_pix_loss = norm_pix_loss
# #         self.masked = masked
# #         self.reduction = reduction
        
# #     def forward(self, pred, mask, target):
# #         if self.norm_pix_loss:
# #             mean = target.mean(dim=-1, keepdim=True)
# #             var = target.var(dim=-1, keepdim=True)
# #             target = (target - mean) / (var + 1.e-6)**.5
            
# #         loss = (pred - target) ** 2
# #         loss = loss.mean(dim=-1)  # [B, 256]
        
# #         if self.masked:
# #             loss = (loss * mask).sum(dim=-1) / mask.sum(dim=-1)  # [B]
# #         else:
# #             loss = loss.mean(dim=-1)  # [B]
        
# #         if self.reduction == 'none':
# #             return loss  # [B]
# #         elif self.reduction == 'mean':
# #             return loss.mean()  # scalar
# #         else:
# #             return loss.sum()  # scalar

# def unpatchify(x, patch_size=14, channels=3):
#     """
#     x: (N, L, patch_size**2 *channels)
#     imgs: (N, 3, H, W)
#     """
#     h = w = int(x.shape[1]**.5)
#     assert h * w == x.shape[1]
#     x = x.reshape(shape=(x.shape[0], h, w, patch_size, patch_size, channels))
#     x = torch.einsum('nhwpqc->nchpwq', x)
#     imgs = x.reshape(shape=(x.shape[0], channels, h * patch_size, h * patch_size))
#     return imgs
    
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
        
#         loss = (pred - target) ** 2  # [B, 256, 768]
#         loss = loss.mean(dim=-1)     # [B, 256]
        
#         # MSE
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
        
# '''
#         # Confidence weighting with normalization
#         if confidence_map is not None:
#             confidence_map = confidence_map.squeeze(-1)
#             loss = loss * confidence_map  # [B, 256]
            
#             if self.masked:
#                 # Scale-preserving: normalize by confidence sum
#                 conf_sum = (confidence_map * mask).sum(dim=-1, keepdim=True) + 1e-8  # [B, 1]
#                 loss = (loss * mask).sum(dim=-1, keepdim=True) / conf_sum  # [B, 1]
#                 loss = loss.squeeze(-1)  # [B]
#             else:
#                 conf_sum = confidence_map.sum(dim=-1, keepdim=True) + 1e-8  # [B, 1]
#                 loss = loss.sum(dim=-1, keepdim=True) / conf_sum  # [B, 1]
#                 loss = loss.squeeze(-1)  # [B]
#         else:
# '''