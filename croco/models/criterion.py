# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
# 
# --------------------------------------------------------
# Criterion to train CroCo
# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------
import math
import torch
from torch import nn
import torch.nn.functional as F
from pytorch_msssim import ms_ssim
from info_nce import InfoNCE, info_nce

class GradientVariance(nn.Module):
    """Class for calculating GV loss between to RGB images
       :parameter
       patch_size : int, scalar, size of the patches extracted from the gt and predicted images
       cpu : bool,  whether to run calculation on cpu or gpu
        """
    def __init__(self, patch_size, cpu=False):
        super(GradientVariance, self).__init__()
        self.patch_size = patch_size
        # Sobel kernel for the gradient map calculation
        self.kernel_x = torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).unsqueeze(0).unsqueeze(0)
        self.kernel_y = torch.FloatTensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).unsqueeze(0).unsqueeze(0)
        if not cpu:
            self.kernel_x = self.kernel_x.cuda()
            self.kernel_y = self.kernel_y.cuda()
        # operation for unfolding image into non overlapping patches
        self.unfold = torch.nn.Unfold(kernel_size=(self.patch_size, self.patch_size), stride=self.patch_size)

    def forward(self, output, target):
        # converting RGB image to grayscale
        gray_output = 0.2989 * output[:, 0:1, :, :] + 0.5870 * output[:, 1:2, :, :] + 0.1140 * output[:, 2:, :, :]
        gray_target = 0.2989 * target[:, 0:1, :, :] + 0.5870 * target[:, 1:2, :, :] + 0.1140 * target[:, 2:, :, :]

        # calculation of the gradient maps of x and y directions
        gx_target = F.conv2d(gray_target, self.kernel_x, stride=1, padding=1)
        gy_target = F.conv2d(gray_target, self.kernel_y, stride=1, padding=1)
        gx_output = F.conv2d(gray_output, self.kernel_x, stride=1, padding=1)
        gy_output = F.conv2d(gray_output, self.kernel_y, stride=1, padding=1)

        # unfolding image to patches
        gx_target_patches = self.unfold(gx_target)
        gy_target_patches = self.unfold(gy_target)
        gx_output_patches = self.unfold(gx_output)
        gy_output_patches = self.unfold(gy_output)

        # calculation of variance of each patch
        var_target_x = torch.var(gx_target_patches, dim=1)
        var_output_x = torch.var(gx_output_patches, dim=1)
        var_target_y = torch.var(gy_target_patches, dim=1)
        var_output_y = torch.var(gy_output_patches, dim=1)

        # loss function as a MSE between variances of patches extracted from gradient maps
        gradvar_loss_x = F.mse_loss(var_target_x, var_output_x, reduction='none')
        gradvar_loss_y = F.mse_loss(var_target_y, var_output_y, reduction='none')
        
        return gradvar_loss_x + gradvar_loss_y

def unpatchify(x, orig_h, orig_w, patch_size=14, channels=3):
    """
    x: (N, L, patch_size**2 *channels)
    imgs: (N, 3, H, W)
    """
    h = int(orig_h / 14)
    w = int(orig_w / 14)
    x = x.reshape(shape=(x.shape[0], h, w, patch_size, patch_size, channels))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], channels, h * patch_size, w * patch_size))
    return imgs
    
class MaskedMSE(torch.nn.Module):
    def __init__(self, args, norm_pix_loss=False, masked=True, reduction='mean', loss_type='mse'):
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
        self.args = args
        if 'GV' in args.recon_loss_type:
            self.grad_criterion = GradientVariance(patch_size=14).to('cuda')
        
    def forward(self, pred, mask, target, weight=None):
        """
        Args:
            pred: [B, 256, 588] - predicted patches
            mask: [B, 256] - binary mask
            target: [B, 256, 588] - target patches
            weight: [B, 256] - optional per-patch weight (e.g., GeM attention score)
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
            
        elif self.loss_type == 'GV':
            pred_img = unpatchify(pred, self.args.resize[0], self.args.resize[1])
            target_img = unpatchify(target, self.args.resize[0], self.args.resize[1])
            loss = self.grad_criterion(pred_img, target_img)  # [B, 256]

        elif self.loss_type == 'GV+l1':
            # L1 loss: [B, 256, 588] -> [B, 256]
            l1_loss = torch.abs(pred - target).mean(dim=-1)  # [B, 256]

            # GV loss: [B, 256]
            pred_img = unpatchify(pred, self.args.resize[0], self.args.resize[1])
            target_img = unpatchify(target, self.args.resize[0], self.args.resize[1])
            GV_loss = self.grad_criterion(pred_img, target_img)  # [B, 256]

            # 합산: [B, 256]
            loss = l1_loss + GV_loss * 0.1
            # mask 적용은 아래에서 처리
               
        elif self.loss_type == 'ssim':
            # SSIM Loss (이미지로 변환 필요)
            pred_img = unpatchify(pred, self.args.resize[0], self.args.resize[1])    # [B, 3, 224, 224]
            target_img = unpatchify(target, self.args.resize[0], self.args.resize[1])
            
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
            pred_img = unpatchify(pred, self.args.resize[0], self.args.resize[1])
            target_img = unpatchify(target, self.args.resize[0], self.args.resize[1])
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
        elif self.loss_type == 'l1+ssim':
            # l1 + SSIM 조합
            l1_loss = torch.abs(pred - target)
            l1_loss = l1_loss.mean(dim=-1)  # [B, 256]
            
            # SSIM
            pred_img = unpatchify(pred, self.args.resize[0], self.args.resize[1])
            target_img = unpatchify(target, self.args.resize[0], self.args.resize[1])
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
                l1_loss = (l1_loss * mask).sum(dim=-1) / mask.sum(dim=-1)  # [B]
            else:
                l1_loss = l1_loss.mean(dim=-1)  # [B]
            
            # SSIM에 mask 비율 적용
            if self.masked:
                mask_ratio = mask.float().mean(dim=1)
                ssim_loss = ssim_loss * mask_ratio
            
            # 조합 (0.5 : 0.5)
            # ssim_ratio = 0.84
            ssim_ratio = 0.5
            loss = ssim_ratio * l1_loss + (1-ssim_ratio) * ssim_loss  # [B]
            
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
        if self.loss_type in ['mse', 'l1', 'GV', 'GV+l1']:
            if self.masked:
                if weight is not None:
                    # Apply weight (e.g., GeM attention score) to loss
                    # Normalize weight to avoid scale issues
                    weight_masked = weight * mask.float()  # [B, 256]
                    weight_norm = weight_masked / (weight_masked.sum(dim=-1, keepdim=True) + 1e-8)  # [B, 256]
                    loss = (loss * weight_norm).sum(dim=-1)  # [B] - weighted sum
                else:
                    loss = (loss * mask).sum(dim=-1) / mask.sum(dim=-1)  # [B]
            else:
                if weight is not None:
                    weight_norm = weight / (weight.sum(dim=-1, keepdim=True) + 1e-8)
                    loss = (loss * weight_norm).sum(dim=-1)  # [B]
                else:
                    loss = loss.mean(dim=-1)  # [B]

            if self.reduction == 'none':
                return loss  # [B]
            elif self.reduction == 'mean':
                return loss.mean()  # scalar
            else:
                return loss.sum()  # scalar