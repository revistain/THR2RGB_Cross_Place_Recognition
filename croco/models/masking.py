# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).


# --------------------------------------------------------
# Masking utils
# --------------------------------------------------------

import torch
import torch.nn as nn    
    
class RandomMask(nn.Module):
    """
    random masking
    """

    def __init__(self, num_patches, mask_ratio):
        super().__init__()
        self.num_patches = num_patches
        self.num_mask = int(mask_ratio * self.num_patches)
    
    def __call__(self, x):
        noise = torch.rand(x.size(0), self.num_patches, device=x.device) 
        argsort = torch.argsort(noise, dim=1) 
        return argsort < self.num_mask

class AttentiveMask(nn.Module):
    def __init__(self, num_patches, mask_ratio):
        super().__init__()
        self.num_patches = num_patches
        self.num_mask = int(mask_ratio * self.num_patches)

    def forward(self, x, attn_map=None, attention_mask_type=None):
        """
        Returns:
            mask: [B, N] bool (True=masked, False=visible)
        """
        B = x.size(0)
        device = x.device
        
        if attn_map is not None and attention_mask_type != 'none':
            assert attention_mask_type is not None and attention_mask_type in ['high', 'low', 'none']
                 
            # 낮은 attention 패치를 마스킹 (descending=False)
            if attention_mask_type == 'high': ids_shuffle = torch.argsort(attn_map, dim=1, descending=False)
            elif attention_mask_type == 'low': ids_shuffle = torch.argsort(attn_map, dim=1, descending=True)
            ids_to_mask = ids_shuffle[:, :self.num_mask]
            
            # Bool mask로 변환
            mask = torch.zeros(B, self.num_patches, dtype=torch.bool, device=device)
            mask.scatter_(1, ids_to_mask, True)
            return mask
        else:
            # Random masking
            noise = torch.rand(B, self.num_patches, device=device)
            mask = torch.argsort(noise, dim=1) < self.num_mask
            return mask

