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
from scaling_on_scales.s2model import S2Wrapper

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

class RerankingModule(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.features_dim = args.features_dim
        self.num_classes = 2
        self.decoder_embed_dim = 32
        self.r2_decoder_norm = nn.LayerNorm(self.decoder_embed_dim)
        
        self.local_head_rgb = nn.Linear(self.args.features_dim*2, 128, bias=True)
        self.local_head_thermal = nn.Linear(self.args.features_dim*2, 128, bias=True)
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
        rgb_idx = rgb_order.unsqueeze(2).expand(-1, -1, self.args.features_dim*2)
        thermal_idx = thermal_order.unsqueeze(2).expand(-1, -1, self.args.features_dim*2)
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
        
class CrossModalVPR_Net(nn.Module):
    def __init__(self, args, pretrained_foundation=False, foundation_model_path=None):
        # NOTE: 그냥 args를 넘기는게 편하다는건 알지만, 이미 늦어버렸습니다...
        super().__init__()

        # 1. 두 개의 독립적인 Backbone 생성 (Weights Unshared)
        # Cross-modal에서는 모달리티 간 특성이 다르므로 가중치를 공유하지 않는 것이 일반적입니다.
        self.args = args
        self.rgb_backbone = get_backbone(pretrained_foundation, foundation_model_path)
        self.thermal_backbone = get_backbone(pretrained_foundation, foundation_model_path)
        self.rgb_s2wrapper = S2Wrapper(vit_model=self.rgb_backbone , target_size=(476, 644))
        self.thermal_s2wrapper = S2Wrapper(vit_model=self.thermal_backbone , target_size=(476, 644))

        self.output_dim = args.features_dim
        self.use_masked_inference = False
        self.use_only_cross_decdoer = args.use_only_cross_decoder
        self.recon_loss_type = args.recon_loss_type
               
        # Croco settings
        dec_depth = args.num_decoder_depth
        dec_num_heads = 16
        self.decoder_thermal_blocks = nn.ModuleList([
            CroCoDecoderBlock(self.args.features_dim*2, dec_num_heads) 
            for _ in range(dec_depth)
        ])
        self.decoder_rgb_blocks = nn.ModuleList([
            CroCoDecoderBlock(self.args.features_dim*2, dec_num_heads) 
            for _ in range(dec_depth)
        ])
        
        self.decoder_norm = nn.LayerNorm(self.args.features_dim*2)
        self.mask_token = None
        self.patch_count = int(args.resize[0]/14)*int(args.resize[1]/14)
        self._set_mask_token(self.output_dim)
        self._set_decode_positional_embedding(self.args.features_dim*2)
        self._set_mask_generator(self.patch_count, args.croco_mask_ratio)
        self._set_prediction_head(self.args.features_dim*2, 14)
        if args.use_r2former:
            self.reranker = RerankingModule(args)
        
        self.reconstruction_criterion = MaskedMSE(
            args,
            norm_pix_loss=False,
            masked=True,
            loss_type=self.recon_loss_type
        )

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
    
    def _set_mask_token(self, dec_embed_dim):
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_embed_dim))
        nn.init.normal_(self.mask_token, std=.02)
        
    def _set_decode_positional_embedding(self, dec_embed_dim):
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.patch_count, dec_embed_dim))
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
    
    def _set_mask_generator(self, num_patches, mask_ratio):
        """Random masking generator 초기화"""
        self.mask_generator = RandomMask(num_patches, mask_ratio)

    def _set_prediction_head(self, dec_embed_dim, patch_size):
        # FIXME: 이것도 같은거 써도됨...?
        self.prediction_head = nn.Sequential(
            nn.Linear(dec_embed_dim, patch_size**2 * 3), # 768 → 588
        )
        nn.init.normal_(self.prediction_head[0].weight, std=0.02)
        nn.init.zeros_(self.prediction_head[0].bias)
        
    def patchify(self, imgs):
        """
        imgs: (B, 3, H, W)
        x: (B, L, patch_size**2 *3)
        """
        try:
            p = 14
            assert imgs.shape[2] % p == 0 and imgs.shape[3] % p == 0

            h = imgs.shape[2] // p
            w = imgs.shape[3] // p
            x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
            x = torch.einsum('nchpwq->nhwpqc', x)
            x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        except Exception as e:
            print("Error: ", e)
            breakpoint()
        return x

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
    
    def croco_like_encoder(self, x, modality='thermal'):
        """
        [Optimized] Two-Track CroCo Encoder
        - Optimization 1: Batch Track 1 & 2 together (5x Batch)
        - Optimization 2: Pre-calculated Pos Embed (if possible, here dynamic but batched)
        """
        if modality == 'thermal':
            current_backbone = self.thermal_backbone
            target_h, target_w = self.args.resize[0], self.args.resize[1]
        elif modality == 'rgb':
            current_backbone = self.rgb_backbone
            target_h, target_w = self.args.resize[0], self.args.resize[1]
        else:
            raise ValueError(f"Wrong Modality: {modality}")

        # 0. Resize
        if x.shape[2] != target_h or x.shape[3] != target_w:
            x = F.interpolate(x, size=(target_h, target_w), mode='bicubic', align_corners=False)
        
        B, C, H, W = x.shape
        half_H, half_W = H // 2, W // 2
        
        # Grid Info
        h_grid_half = half_H // 14
        w_grid_half = half_W // 14
        N_half = h_grid_half * w_grid_half 

        # ============================================================
        # [Fast Step 1] Prepare All Inputs (Batch Size * 5)
        # ============================================================
        
        # Track 1: Resize [B, 3, H/2, W/2]
        x_track1 = F.interpolate(x, size=(half_H, half_W), mode='bicubic', align_corners=False)
        
        # Track 2: Crops [B, 3, H/2, W/2] x 4
        x_tl = x[:, :, :half_H, :half_W]
        x_tr = x[:, :, :half_H, half_W:]
        x_bl = x[:, :, half_H:, :half_W]
        x_br = x[:, :, half_H:, half_W:]
        
        # [Optimization] 5개 텐서를 한번에 Concat -> [5B, 3, H/2, W/2]
        # 순서: [Track1, TL, TR, BL, BR]
        x_all = torch.cat([x_track1, x_tl, x_tr, x_bl, x_br], dim=0)

        # ============================================================
        # [Fast Step 2] Prepare All Masks (Batch Size * 5)
        # ============================================================
        
        # 1. Base Mask Generation
        mask_ratio = self.mask_generator.mask_ratio
        len_keep = int(N_half * (1 - mask_ratio))
        
        noise = torch.rand(B, N_half, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        
        mask_base = torch.ones(B, N_half, dtype=torch.bool, device=x.device)
        mask_base = mask_base.scatter(1, ids_shuffle[:, :len_keep], False)
        
        # 2. Expand for Track 2 (Stitching Logic)
        mask_map = mask_base.reshape(B, h_grid_half, w_grid_half)
        mask_map_full = mask_map.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2) # [B, 34, 46]
        
        m_tl = mask_map_full[:, :h_grid_half, :w_grid_half].flatten(1)
        m_tr = mask_map_full[:, :h_grid_half, w_grid_half:].flatten(1)
        m_bl = mask_map_full[:, h_grid_half:, :w_grid_half].flatten(1)
        m_br = mask_map_full[:, h_grid_half:, w_grid_half:].flatten(1)
        
        # [Optimization] Mask도 5B로 합치기
        # Track 1은 Base Mask 그대로 사용
        mask_all = torch.cat([mask_base, m_tl, m_tr, m_bl, m_br], dim=0) # [5B, N_half]

        # ============================================================
        # [Fast Step 3] Single Pass Forward (GPU Saturation)
        # ============================================================
        
        # 3-1. Patch Embed
        patches = current_backbone.patch_embed(x_all) # [5B, N_half, D]
        _5B, _N, _D = patches.shape
        
        # 3-2. Pos Embed Interpolation (한번만 계산해서 Broadcasting)
        # (매번 interpolate하는게 느리다면, init에서 self.pos_embed_cache로 저장해두는게 좋음)
        pos_tokens = current_backbone.pos_embed[:, 1:, :]
        pos_embed_grid = pos_tokens.reshape(1, 37, 37, _D).permute(0, 3, 1, 2)
        pos_embed_resized = F.interpolate(
            pos_embed_grid, size=(h_grid_half, w_grid_half), 
            mode='bicubic', align_corners=False
        )
        pos_embed_final = pos_embed_resized.permute(0, 2, 3, 1).flatten(1, 2)
        
        patches = patches + pos_embed_final # Broadcasting [1, N, D] + [5B, N, D]

        # 3-3. CLS Token & Masking
        cls_token = current_backbone.cls_token.expand(_5B, -1, -1) + current_backbone.pos_embed[:, :1, :]
        patches_with_cls = torch.cat([cls_token, patches], dim=1)
        
        cls_mask = torch.zeros(_5B, 1, dtype=torch.bool, device=x.device)
        full_mask = torch.cat([cls_mask, mask_all], dim=1)
        
        # Visible Only Processing
        patches_visible = patches_with_cls[~full_mask].reshape(_5B, -1, _D)
        
        # 3-4. Transformer Layers (The Heavy Part)
        for blk in current_backbone.blocks:
            patches_visible = blk(patches_visible)
        patches_visible = current_backbone.norm(patches_visible)
        
        # ============================================================
        # [Fast Step 4] Reconstruct & Split
        # ============================================================
        
        # Expand Mask Tokens (Filling gaps)
        full_features = self.mask_token.expand(_5B, _N, -1).clone()
        full_features[~mask_all] = patches_visible[:, 1:, :].flatten(0, 1)
        
        all_cls = patches_visible[:, 0:1, :]
        
        # Split back into Track 1 and Track 2
        # [5B] -> [B] (Track1) + [4B] (Track2)
        feat_track1 = full_features[:B]
        cls_track1 = all_cls[:B]
        
        feat_track2_batch = full_features[B:]
        cls_track2_batch = all_cls[B:]

        # ============================================================
        # [Fast Step 5] Stitching & Merging
        # ============================================================
        
        # Track 1 Upsample
        feat_track1 = feat_track1.permute(0, 2, 1).reshape(B, -1, h_grid_half, w_grid_half)
        feat_track1_up = F.interpolate(feat_track1, scale_factor=2, mode='nearest')
        feat_track1_final = feat_track1_up.flatten(2).transpose(1, 2)
        
        # Track 2 Stitch
        f_tl, f_tr, f_bl, f_br = torch.chunk(feat_track2_batch, 4, dim=0)
        f_tl = f_tl.reshape(B, h_grid_half, w_grid_half, -1)
        f_tr = f_tr.reshape(B, h_grid_half, w_grid_half, -1)
        f_bl = f_bl.reshape(B, h_grid_half, w_grid_half, -1)
        f_br = f_br.reshape(B, h_grid_half, w_grid_half, -1)
        
        top_row = torch.cat([f_tl, f_tr], dim=2)
        bot_row = torch.cat([f_bl, f_br], dim=2)
        feat_track2_final = torch.cat([top_row, bot_row], dim=1).flatten(1, 2)
        
        # Final Concat
        final_patches = torch.cat([feat_track1_final, feat_track2_final], dim=2)
        
        # CLS Averaging
        cls_track2_avg = torch.mean(cls_track2_batch.reshape(4, B, 1, -1), dim=0)
        final_cls = torch.cat([cls_track1, cls_track2_avg], dim=2)
        
        final_mask = mask_map_full.reshape(B, -1)

        return final_patches, final_mask, B, (h_grid_half*2)*(w_grid_half*2), final_patches.shape[2], final_cls
            
    def croco_encoded_mask_expension(self, thermal_visible, mask, patch_B, patch_N, patch_D):
        # CROCO로 masking된 부분 mask token으로 채워넣기
        thermal_full = self.mask_token.expand(patch_B, patch_N, -1).clone()  # [B, 256, 768]
        thermal_full[~mask] = thermal_visible.flatten(0, 1)  # 이제 shape 맞음
        thermal_full = thermal_full.view(patch_B, patch_N, patch_D)
        return thermal_full

    
    def forward_model(self, x, paired_rgb=None, modality='rgb', return_masked_patch=False):
        """단일 모달리티에 대한 Forward"""
        # self.use_masked_inference: rerank를 위해, decoder에 들어가기 바로 전 단계를 뱉는다
        recon_loss = None
        global_desc = None
        cls_attn_map = None
        mask_thermal = None
        penultimate_patch = None
        masked_patch_thermal = None
        if modality == 'rgb':
            out = self.rgb_s2wrapper(x, return_attention=True)
            cls_attn_map = out["cls_attention"].sum(dim=0)
            agg_layer = self.rgb_aggregation
        elif modality == 'thermal':
            if self.training:
                # 1-4. masked thermal encoder
                thermal_full, mask_thermal, patch_B, patch_N, patch_D, thermal_cls = self.croco_like_encoder(x, modality='thermal')
                rgb_full, mask_rgb, patch_B_rgb, patch_N_rgb, patch_D_rgb, rgb_cls = self.croco_like_encoder(paired_rgb, modality='rgb')
                
                # 5. paired RGB도 feature tokens 추출하기
                paired_thermal = self.thermal_s2wrapper(x, return_attention=True)
                paired_thermal_cls_attn_single_head = paired_thermal["cls_attention"].squeeze(0) # [B, MHA, 256]
                paired_thermal_full = paired_thermal["x_norm_patchtokens"]
                cls_attn_map = paired_thermal_cls_attn_single_head
                
                paired_rgb_emb = self.rgb_s2wrapper(paired_rgb, return_attention=True)
                paired_rgb_full = paired_rgb_emb["x_norm_patchtokens"]
                # paired_rgb_cls_attn = paired_rgb_emb["cls_attention"][:, :, 1:]
                # paired_rgb_cls_attn_single_head = paired_rgb_cls_attn.sum(dim=1)
            
                # 6. Mask token expansion
                # breakpoint()
                # thermal_full = self.croco_encoded_mask_expension(thermal_visible, mask_thermal, patch_B, patch_N, patch_D)
                # rgb_full = self.croco_encoded_mask_expension(rgb_visible, mask_rgb, patch_B_rgb, patch_N_rgb, patch_D_rgb)
                
                # out을 미리 저장
                out = paired_thermal
                if return_masked_patch:
                    masked_patch_thermal = thermal_full # 이후로 안건들여서 clone안해도 됨
                
                # 7. Decoder Positional Encoding
                thermal_full_dec = thermal_full + self.decoder_pos_embed  # [B, 256, 768]
                rgb_full_dec = rgb_full + self.decoder_pos_embed  # [B, 256, 768]
                paired_thermal_dec = paired_thermal_full + self.decoder_pos_embed  # [B, 256, 768]
                paired_rgb_dec = paired_rgb_full + self.decoder_pos_embed  # [B, 256, 768]
                
                # 8. decoder 통과시키기
                for blk in self.decoder_thermal_blocks:
                    thermal_full_dec = blk(thermal_full_dec, paired_rgb_dec)
                thermal_full_dec = self.decoder_norm(thermal_full_dec)

                for blk in self.decoder_rgb_blocks:
                    rgb_full_dec = blk(rgb_full_dec, paired_thermal_dec)
                rgb_full_dec = self.decoder_norm(rgb_full_dec)

                recon_loss_fn = self.calculate_recon_loss
                
                # 9. Prediction Head
                reconstructed_thermal_patches = self.prediction_head(thermal_full_dec)
                reconstructed_rgb_patches = self.prediction_head(rgb_full_dec)
                target_thermal_patches = self.patchify(x)
                target_rgb_patches = self.patchify(paired_rgb)

                # 10. Reconstruction loss 계산
                recon_loss_thermal = recon_loss_fn(reconstructed_thermal_patches, mask_thermal, target_thermal_patches)
                recon_loss_rgb = recon_loss_fn(reconstructed_rgb_patches, mask_rgb, target_rgb_patches)
                recon_loss = (recon_loss_thermal + recon_loss_rgb) / 2

            else:
                # when inference
                out = self.thermal_s2wrapper(x,return_attention=True)
                cls_attn_map = out["cls_attention"].squeeze(1)
            agg_layer = self.thermal_aggregation
        else:
            raise ValueError("Modality must be 'rgb' or 'thermal'")
            
        # Backbone 출력 처리 (ViT 기준)
        # x['x_norm_patchtokens']: (B, num_patchs, D)
        patch_tokens = out["x_norm_patchtokens"]
    
        # attnetion_dict_keys(['x_norm_clstoken', 'x_norm_patchtokens', 'x_prenorm', 'masks'])
        B, N, D = patch_tokens.shape # B, # N # D
        
        # 224,224 정방 이미지 입력 가정(patch 2D 복원)
        H_feat = int(x.shape[2]/14)
        W_feat = int(x.shape[3]/14)
        x_feat = patch_tokens.permute(0, 2, 1).view(B, D, H_feat, W_feat)
        # Aggregation -> Descriptor
        if self.args.use_cls_for_vpr:
            global_desc = out["x_norm_clstoken"]
        else:
            global_desc = agg_layer(x_feat) # [B, D]
            # (Pdb) torch.Size([11, 768, 34, 46])
        
        return global_desc, patch_tokens, recon_loss, mask_thermal, \
            masked_patch_thermal, cls_attn_map, penultimate_patch

    def forward(self, x, flags, paired_rgb=None, return_mask=False, return_masked_patch=False):
        if not isinstance(flags, torch.Tensor):
            flags = torch.tensor(flags, device=x.device)
        
        # [수정] flags가 다른 디바이스에 있을 경우를 대비해 device 맞춤
        if flags.device != x.device:
            flags = flags.to(x.device)

        is_rgb = (flags == 1)
        final_emb = torch.zeros((x.size(0), self.output_dim*2), device=x.device)
        patch_emb = torch.zeros((x.size(0), self.patch_count, self.output_dim*2), device=x.device)
        penultimate_patch_emb = torch.zeros((x.size(0), self.patch_count, self.output_dim*2), device=x.device)
        masks = torch.zeros((x.size(0), self.patch_count), dtype=torch.bool, device=x.device)
        cls_attn_map = torch.zeros((x.size(0), self.patch_count), device=x.device)
        recon_loss = None
        masked_patch_emb = None
        if is_rgb.any():
            global_emb, patch_rgb, _, _, _, cls_rgb_attn_map, penultimate_patch_rgb = self.forward_model(x[is_rgb], modality='rgb')
            if global_emb is not None:
                final_emb[is_rgb] = global_emb
            if patch_rgb is not None:
                patch_emb[is_rgb] = patch_rgb
            if cls_rgb_attn_map is not None:
                cls_attn_map[is_rgb] = cls_rgb_attn_map
            if penultimate_patch_rgb is not None:
                penultimate_patch_emb[is_rgb] = penultimate_patch_rgb
        if (~is_rgb).any():
            global_emb, patch_thermal, recon_loss, mask, masked_patch_thermal, cls_thermal_attn_map, penultimate_patch_thermal = self.forward_model(x[~is_rgb], modality='thermal', paired_rgb=paired_rgb, return_masked_patch=return_masked_patch)
            if global_emb is not None:
                final_emb[~is_rgb] = global_emb
            if patch_thermal is not None:
                patch_emb[~is_rgb] = patch_thermal
            if return_mask:
                masks[~is_rgb] = mask
            if return_masked_patch:
                masked_patch_emb = masked_patch_thermal # torch.Size([4, 256, 768])
            if cls_attn_map is not None:
                cls_attn_map[~is_rgb] = cls_thermal_attn_map
            if penultimate_patch_thermal is not None:
                penultimate_patch_emb[~is_rgb] = penultimate_patch_thermal

        if return_masked_patch: # 무조건 masked_patch_emb가 제일 뒤에 오게
            return final_emb, patch_emb, recon_loss, masks, cls_attn_map, penultimate_patch_emb, masked_patch_emb
        else:
            return final_emb, patch_emb, recon_loss, masks, cls_attn_map, penultimate_patch_emb

    def calculate_recon_loss(self, pred, mask, target, confidence_map=None):
        recon_loss = self.reconstruction_criterion(
            pred=pred,        # [B, 256, 768]
            mask=mask,        # [B, 256]
            target=target,    # [B, 3, 256, 768]
        )
        return recon_loss


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