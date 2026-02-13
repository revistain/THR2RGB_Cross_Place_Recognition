import torch
import torch.nn as nn    
    
class RandomMask(nn.Module):
    """
    random masking
    """

    def __init__(self, num_patches, mask_ratio, method='random'):
        super().__init__()
        self.num_patches = num_patches
        self.num_mask = int(mask_ratio * self.num_patches)
        self.method = method
    
    def __call__(self, x, cls_score=None, gem_score=None):
        result = None
        if self.method == 'random':
            noise = torch.rand(x.size(0), self.num_patches, device=x.device) # [B, N]
            argsort = torch.argsort(noise, dim=1, descending=False) 
        elif self.method == 'CLS':
            noise = -cls_score
            argsort = torch.argsort(noise, dim=1, descending=True) 
        elif self.method == 'GeM':
            noise = gem_score
            argsort = torch.argsort(noise, dim=1, descending=True) 
        else:
            raise Exception("maybe typo in RandomMask method")
        
        return argsort < self.num_mask 