import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from backbone.vision_transformer import vit_small, vit_base, vit_large, vit_giant2
import math
import numpy as np
from sklearn.neighbors import NearestNeighbors
import torchvision.models as models

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, work_with_tokens=False):
        super().__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps
        self.work_with_tokens=work_with_tokens
    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps, work_with_tokens=self.work_with_tokens)
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

def gem(x, p=3, eps=1e-6, work_with_tokens=False):
    if work_with_tokens:
        x = x.permute(0, 2, 1)
        # unseqeeze to maintain compatibility with Flatten
        return F.avg_pool1d(x.clamp(min=eps).pow(p), (x.size(-1))).pow(1./p).unsqueeze(3)
    else:
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

class Flatten(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x): assert x.shape[2] == x.shape[3] == 1; return x[:,:,0,0]

class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim)

class AggregationHead(nn.Module):
    # residue + bottleneck구조 사용함
    # 1. residue 사용이유: 초반에 GeM을 사용하기 위해(triplet을 구하는 과정을 조금이라도 초반에 안정적으로 하기 위함)
    # 2. bottleneck 사용이유: 너무 parameter가 많아지면 overfitting 우려가 있어 줄이기 위해
    def __init__(self, dim=768, bottleneck=192):
        super().__init__()
        self.gem = nn.Sequential(L2Norm(), GeM(), Flatten())
        self.mlp = nn.Sequential(
            nn.Linear(dim, bottleneck),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck, dim)
        )
        # 0 초기화
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
    
    def forward(self, x):
        x = self.gem(x)
        return x + self.mlp(x)

class CrossModalVPR_Net(nn.Module):
    def __init__(self, pretrained_foundation=False, foundation_model_path=None, use_alignment_proj=False, use_GeMAdditionalLayer=False):
        super().__init__()

        # 1. 두 개의 독립적인 Backbone 생성 (Weights Unshared)
        # Cross-modal에서는 모달리티 간 특성이 다르므로 가중치를 공유하지 않는 것이 일반적입니다.
        self.rgb_backbone = get_backbone(pretrained_foundation, foundation_model_path)
        self.thermal_backbone = get_backbone(pretrained_foundation, foundation_model_path)
        self.output_dim = 768

        # 2. Aggregation Layer (각각 따로 두는 것을 추천)
        # GeM의 파라미터 p가 모달리티별로 다르게 학습될 수 있도록 분리합니다.
        if use_GeMAdditionalLayer:
            print("="*30)
            print("USING GEM ADDITIONAL LAYER !!!!!")
            print("="*30)
            self.rgb_aggregation = AggregationHead(dim=768, bottleneck=192)
            self.thermal_aggregation = AggregationHead(dim=768, bottleneck=192)
        else:
            self.rgb_aggregation = nn.Sequential(
                L2Norm(), 
                GeM(work_with_tokens=None), 
                Flatten(),
            )
            self.thermal_aggregation = nn.Sequential(
                L2Norm(), 
                GeM(work_with_tokens=None), 
                Flatten()
            )


    def forward_model(self, x, modality='rgb'):
        """단일 모달리티에 대한 Forward"""
        if modality == 'rgb':
            out = self.rgb_backbone(x, return_attention=True)
            agg_layer = self.rgb_aggregation
        elif modality == 'thermal':
            out = self.thermal_backbone(x, return_attention=True)
            agg_layer = self.thermal_aggregation
        else:
            raise ValueError("Modality must be 'rgb' or 'thermal'")
            
        # Backbone 출력 처리 (ViT 기준)
        # x['x_norm_patchtokens']: (B, num_patchs, D)
        patch_tokens    = out["x_norm_patchtokens"]
        cls_token       = out["x_norm_clstoken"]
        cls_attn_map    = out["cls_attention"]
        
        # attnetion_dict_keys(['x_norm_clstoken', 'x_norm_patchtokens', 'x_prenorm', 'masks'])
        B, N, D = patch_tokens.shape
        
        # 224,224 정방 이미지 입력 가정(patch 2D 복원)
        H_feat = W_feat = int(math.sqrt(N)) 
        x_feat = patch_tokens.permute(0, 2, 1).view(B, D, H_feat, W_feat)
        
        # Aggregation -> Descriptor
        global_desc = agg_layer(x_feat) # [B, D]
        
        return global_desc, patch_tokens, cls_token, cls_attn_map

    def forward(self, x, flags):
        is_rgb = torch.tensor([f == 'rgb' for f in flags], device=x.device)
        final_emb = torch.zeros((x.size(0), self.output_dim), device=x.device)
        attn_maps = []
        
        patch_emb = torch.zeros((x.size(0), 256, 768), device=x.device)
        cls_emb = torch.zeros((x.size(0), 768), device=x.device)  
        
        if is_rgb.any():
            final_emb[is_rgb], patch_rgb, cls_rgb, attn_rgb = self.forward_model(x[is_rgb], 'rgb')
            patch_emb[is_rgb] = patch_rgb
            cls_emb[is_rgb] = cls_rgb
            attn_maps.append(attn_rgb)
        if (~is_rgb).any():
            final_emb[~is_rgb], patch_thermal, cls_thermal, attn_thermal = self.forward_model(x[~is_rgb], 'thermal')
            patch_emb[~is_rgb] = patch_thermal
            cls_emb[~is_rgb] = cls_thermal
            attn_maps.append(attn_thermal)
        
        return final_emb, patch_emb, cls_emb, torch.cat(attn_maps, dim=0)

def get_backbone(pretrained_foundation, foundation_model_path):
    backbone = vit_base(patch_size=14,img_size=518,init_values=1,block_chunks=0)
    if pretrained_foundation:
        assert foundation_model_path is not None, "Please specify foundation model path."
        model_dict = backbone.state_dict()
        state_dict = torch.load(foundation_model_path)
        model_dict.update(state_dict.items())
        backbone.load_state_dict(model_dict)
    return backbone
