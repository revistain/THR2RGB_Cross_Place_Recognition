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
from pathlib import Path
  
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

class RerankingModule(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.features_dim = args.features_dim
        self.num_classes = 2
        self.decoder_embed_dim = 32
        self.r2_decoder_norm = nn.LayerNorm(self.decoder_embed_dim)
        
        self.local_head_rgb = nn.Linear(self.features_dim, 128, bias=True)
        self.local_head_thermal = nn.Linear(self.features_dim, 128, bias=True)
        self.local_head_rgb.weight.data.normal_(mean=0.0, std=0.01)
        self.local_head_thermal.weight.data.normal_(mean=0.0, std=0.01)
        self.local_head_rgb.bias.data.zero_()
        self.local_head_thermal.bias.data.zero_()
        
        self.pair_head = nn.Linear(7, self.decoder_embed_dim, bias=True)
        self.pair_head_2 = nn.Linear(self.decoder_embed_dim, self.decoder_embed_dim, bias=True)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.decoder_embed_dim))
        self.cls_token_2 = nn.Parameter(torch.zeros(1, 1, self.decoder_embed_dim))
        self.decoder_pred = nn.Linear(self.decoder_embed_dim, self.num_classes, bias=True)
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_token_2, std=.02)
        
        self.num_corr = 5
        decoder_num_heads = 4
        decoder_mlp_ratio = 4.
        decoder_depth = 6
        self.blocks = nn.ModuleList([
            Block(self.decoder_embed_dim, decoder_num_heads, decoder_mlp_ratio, qkv_bias=True, norm_layer=nn.LayerNorm)
            for i in range(decoder_depth)])

        self.blocks_2 = nn.ModuleList([
            Block(self.decoder_embed_dim, decoder_num_heads, decoder_mlp_ratio, qkv_bias=True, norm_layer=nn.LayerNorm)
            for i in range(2)])
        
        self.CE = torch.nn.CrossEntropyLoss(ignore_index=-100).cuda()
        self.cos = nn.CosineSimilarity(dim=1)
        self.sm = torch.nn.Softmax(dim=1)
    
    # 1. decoder에 masking안하고 통과
    # 2. attention map 기반 각각, token 100개 선택
    # 3. cross attn matrix 기반 top 5 선택
    def _process_pair(self, paired_thermal_full, paired_thermal_cls_attn, 
                    current_target_full, current_target_cls_attn, current_global=None, cross_attn_matrix=None):
        """공통 처리 로직"""
        TOP_PATCH_COUNT = 100
        thermal_order = torch.argsort(paired_thermal_cls_attn, dim=1, descending=True)
        thermal_order = thermal_order[:, :TOP_PATCH_COUNT]
        rgb_order = torch.argsort(current_target_cls_attn, dim=1, descending=True)
        rgb_order = rgb_order[:, :TOP_PATCH_COUNT]
        rgb_idx = rgb_order.unsqueeze(2).expand(-1, -1, self.features_dim)
        thermal_idx = thermal_order.unsqueeze(2).expand(-1, -1, self.features_dim)
        selected_rgb_patches = torch.gather(current_target_full, axis=1, index=rgb_idx)
        selected_thermal_patches = torch.gather(paired_thermal_full, axis=1, index=thermal_idx)
        
        # 2. 선택된 patch를 각각 linear을 태워서 368 -> 128 dimension
        local_rgb_features = self.local_head_rgb(selected_rgb_patches)
        local_thermal_features = self.local_head_thermal(selected_thermal_patches)
        
        # 3. linear에 추가정보 넣어서 128 -> 131 차원 만들어주기
        B_sz, _, W, H = paired_thermal_full.shape, None, 256, 256
        patch_size = 16
        grid_W = int(np.ceil(W / patch_size))
        HW = max(H, W)

        # --- [RGB] x_xy (좌표) 계산 ---
        rgb_col = (rgb_order % grid_W) * patch_size + (patch_size // 2)
        rgb_row = (rgb_order // grid_W) * patch_size + (patch_size // 2)
        x_xy_rgb = torch.stack([rgb_col / float(HW), rgb_row / float(HW)], dim=2)

        # --- [Thermal] x_xy (좌표) 계산 ---
        thermal_col = (thermal_order % grid_W) * patch_size + (patch_size // 2)
        thermal_row = (thermal_order // grid_W) * patch_size + (patch_size // 2)
        x_xy_thermal = torch.stack([thermal_col / float(HW), thermal_row / float(HW)], dim=2)

        # --- [RGB] x_attention (중요도) 계산 ---
        rgb_att_val = torch.gather(current_target_cls_attn, axis=1, index=rgb_order)
        rgb_att_norm = rgb_att_val / torch.max(rgb_att_val, dim=1, keepdim=True)[0]
        rgb_att_norm = rgb_att_norm.unsqueeze(2)

        # --- [Thermal] x_attention (중요도) 계산 ---
        thermal_att_val = torch.gather(paired_thermal_cls_attn, axis=1, index=thermal_order)
        thermal_att_norm = thermal_att_val / torch.max(thermal_att_val, dim=1, keepdim=True)[0]
        thermal_att_norm = thermal_att_norm.unsqueeze(2)

        # 4. Final Concatenation (Feature + Coord + Score)
        rgb_rerank_input = torch.cat([x_xy_rgb, rgb_att_norm, local_rgb_features], dim=2)
        thermal_rerank_input = torch.cat([x_xy_thermal, thermal_att_norm, local_thermal_features], dim=2)
        
        # 4. correlation matrix 만들기
        B = rgb_rerank_input.shape[0]
        N = rgb_rerank_input.shape[1]
        rgb_rerank_token = F.normalize(rgb_rerank_input[:, :, 3:], p=2, dim=2)
        thermal_rerank_token = F.normalize(thermal_rerank_input[:, :, 3:], p=2, dim=2)
        rgb_coordinate = rgb_rerank_input[:, :, :3].detach().clamp(min=0, max=1)
        thermal_coordinate = thermal_rerank_input[:, :, :3].detach().clamp(min=0, max=1)

        # ========== Cross-Attention Matrix 활용 ==========
        if cross_attn_matrix is not None:
            # cross_attn_matrix: [B, 256, 256] - (thermal query, rgb database)
            # Top-100 patches에 해당하는 부분만 추출
            # thermal_order: [B, 100], rgb_order: [B, 100]
            
            # Advanced indexing으로 [B, 100, 100] 추출
            batch_indices = torch.arange(B, device=cross_attn_matrix.device).view(B, 1, 1)
            thermal_indices = thermal_order.unsqueeze(2)  # [B, 100, 1]
            rgb_indices = rgb_order.unsqueeze(1)  # [B, 1, 100]
            
            # [B, 100, 100] - thermal top-100 x rgb top-100
            correlation = cross_attn_matrix[
                batch_indices.expand(-1, TOP_PATCH_COUNT, TOP_PATCH_COUNT),
                thermal_indices.expand(-1, -1, TOP_PATCH_COUNT),
                rgb_indices.expand(-1, TOP_PATCH_COUNT, -1)
            ]
        else:
            # Fallback: 기존 cosine similarity 방식
            correlation = torch.matmul(rgb_rerank_token, thermal_rerank_token.permute((0, 2, 1)))
        
        # ========== 이하 동일 ==========
        xy_matrix = torch.cat(
            [rgb_coordinate.unsqueeze(2).repeat(1, 1, thermal_rerank_token.shape[1], 1),
            thermal_coordinate.unsqueeze(1).repeat(1, rgb_rerank_token.shape[1], 1, 1),
            correlation.unsqueeze(3)],
            dim=3)
        
        # correlation: [B, 100, 100, 7]
        order_q = torch.argsort(correlation.unsqueeze(3), dim=2, descending=True).repeat(1, 1, 1, 7)
        order_k = torch.argsort(correlation.unsqueeze(3), dim=1, descending=True).repeat(1, 1, 1, 7)
        select_q = torch.gather(input=xy_matrix, index=order_q[:, :, :self.num_corr, :], dim=2)
        select_k = torch.gather(input=xy_matrix, index=order_k[:, :self.num_corr, :, :], dim=1)
        select_k_copy = select_k.clone()
        select_k_copy[:,:,:,:6] = torch.flip(
            select_k[:,:,:,:6].reshape(select_k.shape[0], select_k.shape[1], select_k.shape[2], 2, 3),
            dims=(3,)
        ).reshape(select_k.shape[0], select_k.shape[1], select_k.shape[2], 6)

        # Random Sample Selection
        RANDOM_SAMPLE = 0
        if self.args.r2_add_random_patch:
            RANDOM_SAMPLE = 30
            select_q_random_index = random.sample(range(self.num_corr, correlation.shape[2]), RANDOM_SAMPLE)
            select_k_random_index = random.sample(range(self.num_corr, correlation.shape[1]), RANDOM_SAMPLE)
            
            select_q_random = torch.gather(input=xy_matrix, index=order_q[:, :, select_q_random_index, :], dim=2)
            select_k_random = torch.gather(input=xy_matrix, index=order_k[:, select_k_random_index, :, :], dim=1)
            
            select_q = torch.cat([select_q, select_q_random], axis=2)
            select_k = torch.cat([select_k, select_k_random], axis=1)
        
        select = torch.cat([select_q, select_k.permute((0, 2, 1, 3))], dim=1)
        select_copy = torch.cat([select_q, select_k.permute((0, 2, 1, 3))], dim=1)
        N_select = select.shape[1]
        
        # Linear1
        pair_matrix = self.pair_head(
            select.reshape(B * N_select * (self.num_corr + RANDOM_SAMPLE), 7)
        ).reshape(B * N_select, (self.num_corr + RANDOM_SAMPLE), self.decoder_embed_dim)
        
        pair_matrix += get_2d_sincos_pos_embed_from_grid(
            self.decoder_embed_dim,
            select_copy.reshape(B * N_select, (self.num_corr + RANDOM_SAMPLE), 7)[:, :, 3:5]
        )
        concatedTop5Pairs = torch.cat([self.cls_token_2.repeat(B * N_select, 1, 1), pair_matrix], dim=1)
        
        # Transformer1
        for blk in self.blocks_2:
            concatedTop5Pairs = blk(concatedTop5Pairs)
        concatedTop5Pairs = self.r2_decoder_norm(concatedTop5Pairs)

        # Linear2
        concatedTop5Pairs = self.pair_head_2(
            concatedTop5Pairs[:, 0, :].reshape(B * N_select, self.decoder_embed_dim)
        ).reshape(B, N_select, self.decoder_embed_dim)
        
        concatedTop5Pairs = concatedTop5Pairs + get_2d_sincos_pos_embed_from_grid(
            self.decoder_embed_dim,
            select_copy[:, :, 0, 0:2]
        )
        concatedTop5Pairs = torch.cat([self.cls_token.repeat(B, 1, 1), concatedTop5Pairs], dim=1)

        # Transformer2
        for blk in self.blocks:
            concatedTop5Pairs = blk(concatedTop5Pairs)
        concatedTop5Pairs = self.r2_decoder_norm(concatedTop5Pairs)
        
        # predictor projection
        if self.num_classes == 1:
            local_score = self.decoder_pred(concatedTop5Pairs[:, 0]).reshape(-1)
            local_score = torch.sigmoid(local_score)
            if not self.training:
                if self.global_query_cache.shape[0] == 1:
                    global_query = self.global_query_cache.expand(current_global.shape[0], -1)
                else:
                    global_query = self.global_query_cache
                global_score = self.cos(global_query.detach(), current_global.detach())
                final_score = global_score.detach() * 0.5 + local_score.detach() * 0.5
            else:
                final_score = local_score
        elif self.num_classes == 2:
            local_score = self.decoder_pred(concatedTop5Pairs[:, 0])
            if not self.training:
                if self.global_query_cache.shape[0] == 1:
                    global_query = self.global_query_cache.expand(current_global.shape[0], -1)
                else:
                    global_query = self.global_query_cache
                global_score = self.cos(global_query.detach(), current_global.detach())
                final_score = global_score.detach() * 0.5 + self.sm(local_score).detach()[:, 1] * 0.5
            else:
                final_score = local_score
        
        return final_score
    
    def forward(self, patch_embeddings, cls_attn_map,
                query_index, pos_index, neg_index,
                global_query=None, global_pos=None, global_neg=None,
                cross_attn_matrix=None):
        '''
        patch_embeddings: [48, 256, 768]
        cls_attn_map: [48, 256]
        paired_thermal_cls_attn: [4, 256]
        paired_thermal_full: [4, 256, 768]
        global_query: [4, 768]
        global_pos  : [4, 768]
        global_neg  : [4, 768]
        '''
        if self.training:
            # 5. paired RGB도 feature tokens 추출하기
            paired_thermal_full = patch_embeddings[query_index]
            paired_thermal_cls_attn = cls_attn_map[query_index]
            
            target_pos_full = patch_embeddings[pos_index]
            target_neg_full = patch_embeddings[neg_index]
            target_pos_cls_attn = cls_attn_map[pos_index]
            target_neg_cls_attn = cls_attn_map[neg_index]

            # global_query를 캐시 (global_score 계산용)
            self.global_query_cache = global_query
            
            # R2Former Reranking module 학습 구현부
            rerank_loss_pos = self._process_pair(
                paired_thermal_full, paired_thermal_cls_attn,
                target_pos_full, target_pos_cls_attn, global_pos
            )
            rerank_loss_neg = self._process_pair(
                paired_thermal_full, paired_thermal_cls_attn,
                target_neg_full, target_neg_cls_attn, global_neg
            )
            
            # CE 기반
            target = torch.zeros(rerank_loss_pos.shape[0] * 2, dtype=torch.long).cuda()
            target[:rerank_loss_pos.shape[0]] = 1
            rerank_loss = self.CE(torch.cat([rerank_loss_pos, rerank_loss_neg], dim=0), target)
            
            return rerank_loss
        else:
            # 5. paired RGB도 feature tokens 추출하기
            paired_thermal_full = patch_embeddings[query_index]
            paired_thermal_cls_attn = cls_attn_map[query_index]
            
            target_pos_full = patch_embeddings[pos_index]
            target_pos_cls_attn = cls_attn_map[pos_index]

            local_score = self._process_pair(
                paired_thermal_full, paired_thermal_cls_attn,
                target_pos_full, target_pos_cls_attn, global_pos,
                cross_attn_matrix=cross_attn_matrix
            )
                
            return local_score

class LocalAdapt(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.upconv1 = torch.nn.ConvTranspose2d(in_channels=feature_dim, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.upconv2 = torch.nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)
    def forward(self,x):
        x = self.upconv1(x)
        x = self.relu(x)
        x = self.upconv2(x)
        return x    

class CrossModalVPR_Net(nn.Module):
    def __init__(self, args, pretrained_foundation=False, foundation_model_path=None):
        # NOTE: 그냥 args를 넘기는게 편하다는건 알지만, 이미 늦어버렸습니다...
        super().__init__()

        # 1. 두 개의 독립적인 Backbone 생성 (Weights Unshared)
        # Cross-modal에서는 모달리티 간 특성이 다르므로 가중치를 공유하지 않는 것이 일반적입니다.
        self.args = args
        self.shared_backbone = get_backbone(pretrained_foundation, foundation_model_path)
        self.output_dim = args.features_dim
        if args.use_selaVPR_loss or self.args.use_reranking == 'selaVPR':
            self.local_adapt = LocalAdapt(args.features_dim)
        self.reranker = RerankingModule(args)

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
        out = self.shared_backbone(x, return_attention=True)
        if modality == 'rgb':
            agg_layer = self.rgb_aggregation
        elif modality == 'thermal':
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
        if self.args.use_cls_for_vpr:
            global_desc = out["x_norm_clstoken"]
        else:
            global_desc = agg_layer(x_feat) # [B, D]
            
        cls_attn_map = out["cls_attention"].sum(dim=1)
        
        sela_local_feature = None
        if self.args.use_selaVPR_loss or (not self.training and self.args.use_reranking == 'selaVPR'):
            x0 = patch_tokens.view(-1,H_feat,W_feat,self.output_dim).permute(0, 3, 1, 2)
            x0 = self.local_adapt(x0)
            x0 = x0.permute(0, 2, 3, 1)
            sela_local_feature = torch.nn.functional.normalize(x0, p=2, dim=-1) # [B, 61, 61, 128] / 224x224 기준
        
        return global_desc, patch_tokens, cls_attn_map, out["penultimate_norm_patchtokens"], sela_local_feature

    def forward(self, x, flags, paired_rgb=None, return_mask=False, return_masked_patch=False):
        is_rgb = torch.tensor([f == 'rgb' for f in flags], device=x.device)
        final_emb = torch.zeros((x.size(0), self.output_dim), device=x.device)
        patch_count = x.shape[2] // 14 * x.shape[3] // 14
        patch_emb = torch.zeros((x.size(0), patch_count, self.output_dim), device=x.device)
        cls_attn_map = torch.zeros((x.size(0), patch_count), device=x.device)
        penultimate_patch_emb = torch.zeros((x.size(0), patch_count, self.output_dim), device=x.device)
        sela_local_emb = None
        if self.args.use_selaVPR_loss or self.args.use_reranking == 'selaVPR':
            sela_local_emb = torch.zeros((x.size(0), 61, 61, 128), device=x.device)
        
        if is_rgb.any(): 
            final_emb[is_rgb], patch_emb[is_rgb], cls_attn_map[is_rgb], penultimate_patch_emb[is_rgb], sela_local_feature = self.forward_model(x[is_rgb], 'rgb')
            if self.args.use_selaVPR_loss or (not self.training and self.args.use_reranking == 'selaVPR'):
                sela_local_emb[is_rgb] = sela_local_feature
            
        if (~is_rgb).any():
            final_emb[~is_rgb], patch_emb[~is_rgb], cls_attn_map[~is_rgb], penultimate_patch_emb[~is_rgb], sela_local_feature = self.forward_model(x[~is_rgb], 'thermal')
            if self.args.use_selaVPR_loss or (not self.training and self.args.use_reranking == 'selaVPR'):
                sela_local_emb[~is_rgb] = sela_local_feature
        
        return [final_emb, patch_emb, None, None, cls_attn_map, penultimate_patch_emb, sela_local_emb]

def get_backbone(pretrained_foundation, foundation_model_path):
    model_path = Path(foundation_model_path)
    if 'reg4' in model_path.parts[-1]:
        backbone = vit_base(patch_size=14,img_size=518,init_values=1,block_chunks=0, num_register_tokens=4)
    else:
        backbone = vit_base(patch_size=14,img_size=518,init_values=1,block_chunks=0)
        
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