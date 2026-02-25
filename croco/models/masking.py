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

    def __init__(self, num_patches, mask_ratio, masking_method='random'):
        super().__init__()
        self.num_patches = num_patches
        self.num_mask = int(mask_ratio * self.num_patches)
        self.masking_method = masking_method
    
    def __call__(self, x, cls_score=None):
        if self.masking_method == 'random':
            noise = torch.rand(x.size(0), self.num_patches, device=x.device) 
            argsort = torch.argsort(noise, dim=1) 
            return argsort < self.num_mask
        elif self.masking_method == 'CLSDown':
            # cls_score: [B,N] / 신기하다
            assert cls_score is not None
            ids_shuffle = torch.argsort(cls_score, dim=1, descending=True)
            ranks = torch.argsort(ids_shuffle, dim=1)
            return ranks < self.num_mask
        elif self.masking_method == 'CLSTop':
            # cls_score: [B,N] / 신기하다
            assert cls_score is not None
            ids_shuffle = torch.argsort(cls_score, dim=1, descending=False)
            ranks = torch.argsort(ids_shuffle, dim=1)
            return ranks < self.num_mask
                                