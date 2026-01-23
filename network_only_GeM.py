# network.py
import math
import torch
import random
import numpy as np
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn.parameter import Parameter
from backbone.vision_transformer import vit_small, vit_base, vit_large, vit_giant2
from timm.models.vision_transformer import VisionTransformer, _cfg, PatchEmbed, Block
from sklearn.neighbors import NearestNeighbors
from timm.models.layers import trunc_normal_

from croco.models.masking import RandomMask
from croco.models.criterion import MaskedMSE
  
class CroCoDecoderBlock(nn.Module):
    def __init__(self, dim=768, num_heads=12, mlp_ratio=4.0, drop_path=0.0):
        super().__init__()
        
        # Self-Attention components
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        
        # Cross-Attention components
        self.norm2 = nn.LayerNorm(dim)
        self.norm_cross = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        
        # MLP components
        self.norm3 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )
        
        # ========== Attention 저장용 ==========
        self.self_attn_weights = None
        self.cross_attn_weights = None
        # ======================================
        
    def forward(self, x, y, return_attention=False):
        """
        Args:
            x: [B, N, D] - decoder input (RGB masked)
            y: [B, M, D] - encoder output (Thermal reference)
            return_attention: bool - attention map 반환 여부
        Returns:
            x: [B, N, D] - updated decoder features
        """
        # Step 1: Self-Attention
        x_norm = self.norm1(x)
        if return_attention:
            self_out, self_attn_weights = self.self_attn(
                x_norm, x_norm, x_norm, 
                need_weights=True, 
                average_attn_weights=True  # [B, N, N]
            )
            self.self_attn_weights = self_attn_weights
        else:
            self_out = self.self_attn(x_norm, x_norm, x_norm)[0]
        x = x + self_out
        
        # Step 2: Cross-Attention
        x_norm = self.norm2(x)
        encoder_norm = self.norm_cross(y)
        if return_attention:
            cross_out, cross_attn_weights = self.cross_attn(
                query=x_norm,
                key=encoder_norm,
                value=encoder_norm,
                need_weights=True,
                average_attn_weights=True  # [B, N, M]
            )
            self.cross_attn_weights = cross_attn_weights
        else:
            cross_out = self.cross_attn(
                query=x_norm,
                key=encoder_norm,
                value=encoder_norm
            )[0]
        x = x + cross_out
        
        # Step 3: MLP
        x = x + self.mlp(self.norm3(x))
        
        return x
     
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
    def __init__(self, args, pretrained_foundation=False, foundation_model_path=None):
        # NOTE: 그냥 args를 넘기는게 편하다는건 알지만, 이미 늦어버렸습니다...
        super().__init__()

        # 1. 두 개의 독립적인 Backbone 생성 (Weights Unshared)
        # Cross-modal에서는 모달리티 간 특성이 다르므로 가중치를 공유하지 않는 것이 일반적입니다.
        self.args = args
        self.rgb_backbone = get_backbone(pretrained_foundation, foundation_model_path)
        self.thermal_backbone = get_backbone(pretrained_foundation, foundation_model_path)
        self.output_dim = args.features_dim

        # 2. Aggregation Layer (각각 따로 두는 것을 추천)
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
        
    def forward_model(self, x, paired_rgb=None, modality='rgb', return_masked_patch=False):
        """단일 모달리티에 대한 Forward"""
        if modality == 'rgb':
            out = self.rgb_backbone(x)
            agg_layer = self.rgb_aggregation
        elif modality == 'thermal':
            out = self.thermal_backbone(x)
            agg_layer = self.thermal_aggregation
        else:
            raise ValueError("Modality must be 'rgb' or 'thermal'")
            
        # Backbone 출력 처리 (ViT 기준)
        # x['x_norm_patchtokens']: (B, num_patchs, D)
        patch_tokens = out["x_norm_patchtokens"]
        B, N, D = patch_tokens.shape
        
        # 224,224 정방 이미지 입력 가정
        H_feat = int(self.args.resize[0] / 14)
        W_feat = int(self.args.resize[1] / 14)
        
        x_feat = patch_tokens.permute(0, 2, 1).view(B, D, H_feat, W_feat)
        
        # Aggregation -> Descriptor
        global_desc = agg_layer(x_feat) # [B, D]
        
        return global_desc

    def forward(self, x, flags, paired_rgb=None, return_mask=False, return_masked_patch=False):
        is_rgb = torch.tensor([f == 'rgb' for f in flags], device=x.device)
        final_emb = torch.zeros((x.size(0), self.output_dim), device=x.device)
        
        if is_rgb.any():  final_emb[is_rgb] = self.forward_model(x[is_rgb], 'rgb')
        if (~is_rgb).any(): final_emb[~is_rgb] = self.forward_model(x[~is_rgb], 'thermal')
        
        return [final_emb]

def get_backbone(pretrained_foundation, foundation_model_path):
    backbone = vit_small(patch_size=14,img_size=518,init_values=1,block_chunks=0)
    if pretrained_foundation:
        assert foundation_model_path is not None, "Please specify foundation model path."
        model_dict = backbone.state_dict()
        state_dict = torch.load(foundation_model_path)
        model_dict.update(state_dict.items())
        backbone.load_state_dict(model_dict)
    return backbone

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[:,:, 0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[:,:, 1])  # (H*W, D/2)

    emb = torch.cat([emb_h, emb_w], dim=2) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32).cuda()
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    # pos = pos.reshape(-1)  # (M,)
    out = torch.einsum('bm,d->bmd', pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out) # (M, D/2)
    emb_cos = torch.cos(out) # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=2)  # (M, D)
    return emb

cached_patch_count = None
def get_image_patch_count(x):
    assert type(x) is torch.Tensor
    global cached_patch_count
    if cached_patch_count is None:
         cached_patch_count = int(x.shape[2] / 14) * int(x.shape[3] / 14)
    return cached_patch_count