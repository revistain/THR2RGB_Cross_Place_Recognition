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
        
    def forward(self, pred, mask, target, cls_attn_map=None, exclude_ratio=0.0):
            if self.norm_pix_loss:
                mean = target.mean(dim=-1, keepdim=True)
                var = target.var(dim=-1, keepdim=True)
                target = (target - mean) / (var + 1.e-6)**.5

            # ========== Attention 기반 하위 N% 패치 제외 ==========
            # cls_attn_map: [B, N] - 원본 attention map
            # exclude_ratio > 0이면 하위 N% 패치를 mask에서 제외
            if cls_attn_map is not None and exclude_ratio > 0:
                # masked 패치 중 하위 exclude_ratio%를 제외
                # mask: [B, N] (True = masked = loss 계산 대상)
                masked_count = mask.sum(dim=-1)  # [B]
                k = (masked_count * exclude_ratio).int()  # 제외할 패치 수

                # 배치별로 처리
                exclude_mask = torch.zeros_like(mask)
                for b in range(mask.shape[0]):
                    if k[b] > 0:
                        # masked 패치들의 attention score만 추출
                        masked_indices = mask[b].nonzero(as_tuple=True)[0]  # masked 패치 인덱스
                        masked_attn = cls_attn_map[b, masked_indices]  # 해당 패치들의 attention

                        # 하위 k[b]개의 attention을 가진 패치 인덱스 찾기
                        _, low_attn_idx = masked_attn.topk(k[b].item(), largest=False)
                        exclude_indices = masked_indices[low_attn_idx]
                        exclude_mask[b, exclude_indices] = True

                # 하위 attention 패치들을 mask에서 제외
                mask = mask & ~exclude_mask

            # ========== Loss 계산 ==========
            if self.loss_type == 'mse':
                loss = (pred - target) ** 2  # [B, 256, 588]
                loss = loss.mean(dim=-1)     # [B, 256] - 차원을 먼저 맞춰줍니다!

            elif self.loss_type == 'l1':
                loss = torch.abs(pred - target)  # [B, 256, 588]
                loss = loss.mean(dim=-1)         # [B, 256]

            elif self.loss_type == 'ssim':
                pred_img = unpatchify(pred, self.args.resize[0], self.args.resize[1])
                target_img = unpatchify(target, self.args.resize[0], self.args.resize[1])
                ssim_value = ms_ssim(pred_img, target_img, data_range=1.0, size_average=False)

                loss = 1 - ssim_value  # [B] (이미지 단위 스칼라)

                if self.masked:
                    mask_ratio = mask.float().mean(dim=1)
                    loss = loss * mask_ratio

                if self.reduction == 'none': return loss
                elif self.reduction == 'mean': return loss.mean()
                else: return loss.sum()

            elif self.loss_type == 'mse+ssim':
                mse_loss = (pred - target) ** 2
                mse_loss = mse_loss.mean(dim=-1)  # [B, 256]

                if self.masked:
                    mse_loss = (mse_loss * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1e-6)
                else:
                    mse_loss = mse_loss.mean(dim=-1)

                # SSIM 파트
                pred_img = unpatchify(pred, self.args.resize[0], self.args.resize[1])
                target_img = unpatchify(target, self.args.resize[0], self.args.resize[1])
                ssim_value = ms_ssim(pred_img, target_img, data_range=1.0, size_average=False)
                ssim_loss = 1 - ssim_value  # [B]

                if self.masked:
                    ssim_loss = ssim_loss * mask.float().mean(dim=1)

                ssim_ratio = 0.5
                loss = ssim_ratio * mse_loss + (1-ssim_ratio) * ssim_loss  # [B]

                if self.reduction == 'none': return loss
                elif self.reduction == 'mean': return loss.mean()
                else: return loss.sum()

            else:
                raise ValueError(f"Unknown loss_type: {self.loss_type}")

            # =========================================
            # 단일 패치 레벨 Loss (MSE, L1) 최종 정리 (위에 해당 안 된 것들)
            if self.loss_type in ['mse', 'l1']:
                if self.masked:
                    loss = (loss * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1e-6)
                else:
                    loss = loss.mean(dim=-1)

                if self.reduction == 'none': return loss
                elif self.reduction == 'mean': return loss.mean()
                else: return loss.sum()