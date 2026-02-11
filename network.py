# network.py
import math
import torch
import wandb
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
from swin_transformer import *
from local_matching import LocalFeatureLoss
from diff_loss import DiffLoss
from backbone.dinov2.decoder import DINOv2Decoder
     
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
    def __init__(self, dim=768, bottleneck=192):
        super().__init__()
        self.gem = nn.Sequential(L2Norm(), GeM(), Flatten())
        self.mlp = nn.Sequential(
            nn.Linear(dim, bottleneck),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck, dim)
        )
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

        # x_xy (좌표) 계산
        rgb_col = (rgb_order % grid_W) * patch_size + (patch_size // 2)
        rgb_row = (rgb_order // grid_W) * patch_size + (patch_size // 2)
        x_xy_rgb = torch.stack([rgb_col / float(HW), rgb_row / float(HW)], dim=2)

        thermal_col = (thermal_order % grid_W) * patch_size + (patch_size // 2)
        thermal_row = (thermal_order // grid_W) * patch_size + (patch_size // 2)
        x_xy_thermal = torch.stack([thermal_col / float(HW), thermal_row / float(HW)], dim=2)

        # x_attention (중요도)
        rgb_att_val = torch.gather(current_target_cls_attn, axis=1, index=rgb_order)
        rgb_att_norm = rgb_att_val / torch.max(rgb_att_val, dim=1, keepdim=True)[0]
        rgb_att_norm = rgb_att_norm.unsqueeze(2)
        
        thermal_att_val = torch.gather(paired_thermal_cls_attn, axis=1, index=thermal_order)
        thermal_att_norm = thermal_att_val / torch.max(thermal_att_val, dim=1, keepdim=True)[0]
        thermal_att_norm = thermal_att_norm.unsqueeze(2)

        # 4. Final Concatenation
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
            # 기존 cosine similarity 방식
            correlation = torch.matmul(rgb_rerank_token, thermal_rerank_token.permute((0, 2, 1)))
        
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
        super().__init__()

        self.args = args
        self.shared_backbone = get_backbone(pretrained_foundation, foundation_model_path, args=args)
        
        if args.use_sela_local_loss:
            self.MNNLocalFeatureLoss = LocalFeatureLoss().to(args.device)
        if args.use_diff_loss:
            self.DiffGeMLoss = DiffLoss(args).to(args.device)
    
        self.output_dim = args.features_dim
        self.use_masked_inference = False
        self.use_only_cross_decdoer = args.use_only_cross_decoder
        self.recon_loss_type = args.recon_loss_type
        if args.use_sela_local_loss or args.use_reranking in ['selaVPR', 'reconSelaVPR']:
            self.local_adapt = LocalAdapt(self.output_dim)
               
        # Decoder settings
        dec_depth = args.num_decoder_depth
        dec_num_heads = 16
        self.patch_count = 256

        # Compute feature map size from patch count (assuming square images)
        img_size = int(math.sqrt(self.patch_count))  # e.g., 16 for 256 patches

        # Check which decoder type to use
        use_swin = getattr(args, 'use_swin_decoder', False)
        use_dino_decoder = getattr(args, 'use_dino_decoder', False)

        # Store decoder type for forward pass
        self.use_dino_decoder = use_dino_decoder

        if use_dino_decoder:
            self._set_recursive_head(self.output_dim, 128)
            # DINOv2 Decoder: uses frozen DINO blocks 3-9 with trainable cross-attention adapters
            dino_layer_start = getattr(args, 'dino_decoder_layer_start', 3)
            dino_layer_end = getattr(args, 'dino_decoder_layer_end', 9)  # exclusive, so 10 means layers 3-9
            drop_path_rate = getattr(args, 'drop_path_rate', 0.1)

            self.dino_decoder = DINOv2Decoder.from_encoder(
                encoder=self.shared_backbone,
                layer_range=(dino_layer_start, dino_layer_end),
                num_heads=self.shared_backbone.num_heads,
                drop_path_rate=drop_path_rate,
            )

            # Conditionally freeze DINO blocks
            if not getattr(args, 'unfreeze_dino_decoder', False):
                self.dino_decoder.freeze_dino_blocks()
                freeze_status = "frozen"
            else:
                freeze_status = "trainable"

            # Store layer indices for intermediate feature extraction
            self.dino_decoder_layers = list(range(dino_layer_start, dino_layer_end))

            num_trainable = self.dino_decoder.get_num_trainable_params()
            print(f"Using DINOv2 Decoder: layers {dino_layer_start}-{dino_layer_end-1}, "
                  f"DINO blocks: {freeze_status}, trainable params: {num_trainable:,}")

            # Create placeholder for compatibility (decoder_blocks not used with dino_decoder)
            self.decoder_blocks = None
        elif use_swin:
            window_size = getattr(args, 'swin_window_size', 4)
            drop_path_rate = getattr(args, 'drop_path_rate', 0.1)

            # Linearly increasing drop_path rate
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, dec_depth)]

            self.decoder_blocks = nn.ModuleList([
                SwinDecoderBlock(
                    dim=self.output_dim,
                    num_heads=dec_num_heads,
                    drop_path=dpr[i],
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    img_size=img_size,
                )
                for i in range(dec_depth)
            ])
            print(f"Using Swin Decoder: window_size={window_size}, drop_path_rate={drop_path_rate}")
        else:
            self.decoder_blocks = nn.ModuleList([
                CroCoDecoderBlock(self.output_dim, dec_num_heads)
                for _ in range(dec_depth)
            ])

        self.decoder_norm = nn.LayerNorm(self.output_dim)
        self.mask_token = None
        self.patch_count = int(args.resize[0]/14)*int(args.resize[1]/14)
        self._set_mask_token(self.output_dim)
        self._set_decode_positional_embedding(self.output_dim)
        self._set_mask_generator(self.patch_count, args.croco_mask_ratio)
        self._set_prediction_head(self.output_dim, args.resize[0], args.resize[1])
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

    def _set_prediction_head(self, dec_embed_dim, image_H, image_W):
        # 1. 차원 설정 (ViT Standard: 4x Expansion)
        hidden_dim = dec_embed_dim * 4  
        output_dim = 14 * 14 * 3        
        
        # # 2. Thermal Head 설계 
        # self.prediction_thermal_head = nn.Sequential(
        #     nn.Linear(dec_embed_dim, output_dim),
        # )

        # # 3. RGB Head 설계 
        # self.prediction_rgb_head = nn.Sequential(
        #     nn.Linear(dec_embed_dim, output_dim),
        # )
        
        # # 4. 가중치 초기화하기
        # nn.init.normal_(self.prediction_thermal_head[0].weight, std=0.02)
        # nn.init.zeros_(self.prediction_thermal_head[0].bias)
        # nn.init.normal_(self.prediction_rgb_head[0].weight, std=0.02)
        # nn.init.zeros_(self.prediction_rgb_head[0].bias)

        # 2. Thermal Head 설계
        self.prediction_thermal_head = nn.Sequential(
            nn.Linear(dec_embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

        # 3. RGB Head 설계 
        self.prediction_rgb_head = nn.Sequential(
            nn.Linear(dec_embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # 4. 가중치 초기화하기
        nn.init.normal_(self.prediction_thermal_head[0].weight, std=0.02)
        nn.init.zeros_(self.prediction_thermal_head[0].bias)
        nn.init.normal_(self.prediction_rgb_head[0].weight, std=0.02)
        nn.init.zeros_(self.prediction_rgb_head[0].bias)

        nn.init.normal_(self.prediction_thermal_head[2].weight, std=0.02)
        nn.init.zeros_(self.prediction_thermal_head[2].bias)
        nn.init.normal_(self.prediction_rgb_head[2].weight, std=0.02)
        nn.init.zeros_(self.prediction_rgb_head[2].bias)

    def _set_recursive_head(self, dec_embed_dim, hidden_dim):
        self.recursive_thermal_head = nn.Sequential(
            nn.Linear(dec_embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dec_embed_dim)
        )
        self.recursive_rgb_head = nn.Sequential(
            nn.Linear(dec_embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dec_embed_dim)
        )
        nn.init.normal_(self.recursive_thermal_head[0].weight, std=0.02)
        nn.init.zeros_(self.recursive_thermal_head[0].bias)
        nn.init.normal_(self.recursive_thermal_head[2].weight, std=0.02)
        nn.init.zeros_(self.recursive_thermal_head[2].bias)

        nn.init.normal_(self.recursive_rgb_head[0].weight, std=0.02)
        nn.init.zeros_(self.recursive_rgb_head[0].bias)
        nn.init.normal_(self.recursive_rgb_head[2].weight, std=0.02)
        nn.init.zeros_(self.recursive_rgb_head[2].bias)
        
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
    
    
    def croco_like_encoder(self, x, modality='thermal', collect_layers=None):
        """
        Masked encoder following CroCo style.

        Args:
            x: Input image [B, C, H, W]
            modality: 'thermal' or 'rgb'
            collect_layers: Optional list of layer indices to collect intermediate features
                           e.g., [3, 4, 5, 6, 7, 8, 9] for DINO decoder

        Returns:
            If collect_layers is None:
                patch_only_visible, mask, patch_B, patch_N, patch_D, cls_visible
            If collect_layers is provided:
                patch_only_visible, mask, patch_B, patch_N, patch_D, cls_visible, intermediate_features
                where intermediate_features is dict {layer_idx: features} (visible patches only)
        """
        current_backbone = self.shared_backbone

        image_patch = current_backbone.patch_embed(x)
        patch_B, patch_N, patch_D = image_patch.shape  # N=256, D=768

        cls_token = current_backbone.cls_token.expand(patch_B, -1, -1)  # [B, 1, 768]

        # Positional embedding
        pos_tokens = current_backbone.pos_embed[:, 1:, :]
        pos_embed_grid = pos_tokens.reshape(1, 37, 37, patch_D).permute(0, 3, 1, 2)
        pos_embed_resized = F.interpolate(pos_embed_grid, size=(int(x.shape[2]/14), int(x.shape[3]/14)), mode='bicubic', align_corners=False)
        pos_embed_final = pos_embed_resized.permute(0, 2, 3, 1).flatten(1, 2)
        image_patch = image_patch + pos_embed_final  # [B, 256, 768]

        # CLS positional embedding 추가
        cls_pos_embed = current_backbone.pos_embed[:, :1, :]  # [1, 1, 768]
        cls_token = cls_token + cls_pos_embed  # [B, 1, 768]

        image_with_cls = torch.cat([cls_token, image_patch], dim=1)  # [B, 257, 768]

        # Add register tokens if they exist (DINOv2 with registers)
        # Register tokens are added AFTER positional encoding (they don't have pos embed)
        num_register_tokens = current_backbone.num_register_tokens
        if current_backbone.register_tokens is not None:
            register_tokens = current_backbone.register_tokens.expand(patch_B, -1, -1)
            image_with_cls = torch.cat([
                image_with_cls[:, :1],      # CLS
                register_tokens,             # register tokens
                image_with_cls[:, 1:]        # patches
            ], dim=1)  # [B, 1 + num_register_tokens + 256, 768]

        # Masking (CLS and register tokens are always visible)
        mask = self.mask_generator(image_patch)  # [B, 256]
        cls_reg_mask = torch.zeros(patch_B, 1 + num_register_tokens, dtype=torch.bool, device=mask.device)
        full_mask = torch.cat([cls_reg_mask, mask], dim=1)  # [B, 1 + num_reg + 256]

        patch_visible = image_with_cls[~full_mask].reshape(patch_B, -1, patch_D)  # [B, ~(1+num_reg+visible_patches), 768]

        # Encoder forward with optional intermediate collection
        intermediate_features = {} if collect_layers else None
        for i, blk in enumerate(current_backbone.blocks):
            patch_visible = blk(patch_visible)
            if collect_layers and i in collect_layers:
                # Store intermediate features (visible patches only, excluding CLS and registers)
                intermediate_features[i] = patch_visible[:, 1 + num_register_tokens:, :].clone()

        patch_visible = current_backbone.norm(patch_visible)

        # ========== CLS 분리 (skip register tokens) ==========
        cls_visible = patch_visible[:, 0:1, :]  # [B, 1, 768]
        # Skip CLS and register tokens to get only patch tokens
        patch_only_visible = patch_visible[:, 1 + num_register_tokens:, :]  # [B, ~visible_patches, 768]

        if collect_layers:
            return patch_only_visible, mask, patch_B, patch_N, patch_D, cls_visible, intermediate_features
        return patch_only_visible, mask, patch_B, patch_N, patch_D, cls_visible


    def croco_encoded_mask_expension(self, thermal_visible, mask, patch_B, patch_N, patch_D):
        # CROCO로 masking된 부분 mask token으로 채워넣기
        thermal_full = self.mask_token.expand(patch_B, patch_N, -1).clone()  # [B, 256, 768]
        thermal_full[~mask] = thermal_visible.flatten(0, 1)  # 이제 shape 맞음
        thermal_full = thermal_full.view(patch_B, patch_N, patch_D)
        return thermal_full

    def expand_intermediate_features(self, intermediate_visible, mask, patch_B, patch_N, patch_D):
        """
        Expand intermediate features (visible patches only) to full sequence with mask tokens.

        Args:
            intermediate_visible: Dict {layer_idx: visible_features [B, num_visible, D]}
            mask: Boolean mask [B, 256] where True = masked position
            patch_B, patch_N, patch_D: batch size, num patches, feature dim

        Returns:
            Dict {layer_idx: full_features [B, 256, D]} with mask tokens at masked positions
        """
        expanded = {}
        for layer_idx, visible_feat in intermediate_visible.items():
            # Create full tensor filled with mask tokens
            full_feat = self.mask_token.expand(patch_B, patch_N, -1).clone()  # [B, 256, D]
            # Fill visible positions with actual features
            full_feat[~mask] = visible_feat.flatten(0, 1)
            full_feat = full_feat.view(patch_B, patch_N, patch_D)
            expanded[layer_idx] = full_feat
        return expanded

    def forward_model(self, x, paired_rgb=None, modality='rgb', return_masked_patch=False):
        """단일 모달리티에 대한 Forward"""
        # self.use_masked_inference: rerank를 위해, decoder에 들어가기 바로 전 단계를 뱉는다
        
        recon_loss_thermal = None
        recon_loss_rgb = None
        global_desc = None
        mask_thermal = None
        penultimate_patch = None
        masked_patch_thermal = None
        cls_attn_map = None
        if modality == 'rgb':
            if self.use_masked_inference:
                rgb_visible, mask_rgb, patch_B, patch_N, patch_D, rgb_cls = self.croco_like_encoder(x, modality='rgb')
                rgb_full = self.croco_encoded_mask_expension(rgb_visible, mask_rgb, patch_B, patch_N, patch_D)
                rgb_full_dec = rgb_full + self.decoder_pos_embed  # [B, 256, 768]
                out = {
                    "x_norm_patchtokens": rgb_full_dec,
                    "x_norm_clstoken": rgb_cls,
                }
            else:
                out = self.shared_backbone(x, return_attention=True)
                cls_attn_map = out["cls_attention"].sum(dim=1)
                penultimate_patch = out["penultimate_norm_patchtokens"]
                
            agg_layer = self.rgb_aggregation
        elif modality == 'thermal':
            if self.training:
                # if self.use_dino_decoder:
                #     # ========== DINO Decoder Path ==========
                #     # Use masked encoder with intermediate feature collection

                #     # 1. Masked encoder for TARGET modalities (collect intermediate features)
                #     thermal_visible, mask_thermal, patch_B, patch_N, patch_D, thermal_cls, thermal_inter_visible = \
                #         self.croco_like_encoder(x, modality='thermal', collect_layers=self.dino_decoder_layers)
                #     rgb_visible, mask_rgb, patch_B_rgb, patch_N_rgb, patch_D_rgb, rgb_cls, rgb_inter_visible = \
                #         self.croco_like_encoder(paired_rgb, modality='rgb', collect_layers=self.dino_decoder_layers)

                #     # 2. Expand masked intermediate features with mask tokens
                #     thermal_masked_intermediates = self.expand_intermediate_features(
                #         thermal_inter_visible, mask_thermal, patch_B, patch_N, patch_D
                #     )
                #     rgb_masked_intermediates = self.expand_intermediate_features(
                #         rgb_inter_visible, mask_rgb, patch_B_rgb, patch_N_rgb, patch_D_rgb
                #     )

                #     # 3. Get FULL features for REFERENCE (cross-attention needs full context)
                #     thermal_full_out = self.shared_backbone.forward_with_intermediate(
                #         x, layers=self.dino_decoder_layers, return_attention=True
                #     )
                #     rgb_full_out = self.shared_backbone.forward_with_intermediate(
                #         paired_rgb, layers=self.dino_decoder_layers, return_attention=True
                #     )

                #     # Extract full reference features (for cross-attention)
                #     num_special_tokens = 1 + self.shared_backbone.num_register_tokens
                #     thermal_full_features = {
                #         layer: thermal_full_out[layer][:, num_special_tokens:, :]
                #         for layer in self.dino_decoder_layers
                #     }
                #     rgb_full_features = {
                #         layer: rgb_full_out[layer][:, num_special_tokens:, :]
                #         for layer in self.dino_decoder_layers
                #     }

                #     # For global descriptor & attention (use full encoder output)
                #     paired_thermal_cls_attn_single_head = thermal_full_out["cls_attention"].sum(dim=1)
                #     penultimate_patch = thermal_full_out["patch_tokens"]
                #     paired_thermal_full = thermal_full_out["patch_tokens"]
                #     paired_rgb_full = rgb_full_out["patch_tokens"]

                #     paired_thermal = {
                #         "x_norm_patchtokens": paired_thermal_full,
                #         "x_norm_clstoken": thermal_full_out["cls_token"],
                #         "cls_attention": thermal_full_out.get("attention", None),
                #     }
                #     cls_attn_map = paired_thermal_cls_attn_single_head

                #     # 4. Mask token expansion for final features (for compatibility)
                #     thermal_full = self.croco_encoded_mask_expension(thermal_visible, mask_thermal, patch_B, patch_N, patch_D)
                #     rgb_full = self.croco_encoded_mask_expension(rgb_visible, mask_rgb, patch_B_rgb, patch_N_rgb, patch_D_rgb)

                #     out = paired_thermal
                #     if return_masked_patch:
                #         masked_patch_thermal = thermal_full

                #     # 5. Decoder forward
                #     # Target: MASKED intermediate features (true masked - visible never saw masked)
                #     # Reference: FULL intermediate features (for cross-attention context)
                #     start_layer = self.dino_decoder_layers[0]
                #     thermal_input = thermal_masked_intermediates[start_layer]
                #     rgb_input = rgb_masked_intermediates[start_layer]

                #     # Bidirectional decoding:
                #     # - thermal (masked) decoded with RGB (full) cross-attention
                #     # - RGB (masked) decoded with thermal (full) cross-attention
                #     thermal_decoded, rgb_decoded, _, _ = self.dino_decoder.forward_bidirectional(
                #         x_target=thermal_input,                      # TRUE MASKED thermal
                #         x_ref=rgb_input,                             # TRUE MASKED RGB
                #         target_intermediates=thermal_full_features,  # FULL thermal (for RGB->thermal cross-attn)
                #         ref_intermediates=rgb_full_features,         # FULL RGB (for thermal->RGB cross-attn)
                #     )

                #     thermal_reconed_dec = thermal_decoded
                #     rgb_full_dec = rgb_decoded

                if self.use_dino_decoder:
                    # 1-4. masked encoder
                    thermal_visible, mask_thermal, patch_B, patch_N, patch_D, thermal_cls = self.croco_like_encoder(x, modality='thermal')
                    rgb_visible, mask_rgb, patch_B_rgb, patch_N_rgb, patch_D_rgb, rgb_cls = self.croco_like_encoder(paired_rgb, modality='rgb')

                    # 5. Get full features for global descriptor
                    paired_thermal = self.shared_backbone(x, return_attention=True)
                    paired_thermal_cls_attn_single_head = paired_thermal["cls_attention"].sum(dim=1)
                    penultimate_patch = paired_thermal["penultimate_norm_patchtokens"]
                    paired_thermal_full = paired_thermal["x_norm_patchtokens"]

                    paired_rgb_emb = self.shared_backbone(paired_rgb, return_attention=True)
                    paired_rgb_full = paired_rgb_emb["x_norm_patchtokens"]

                    cls_attn_map = paired_thermal_cls_attn_single_head
                    
                    # Pass Through Recursive MLP
                    recur_paired_thermal_full = self.recursive_thermal_head(paired_thermal_full)
                    recur_thermal_visible = self.recursive_thermal_head(thermal_visible)
                    recur_paired_rgb_full = self.recursive_rgb_head(paired_rgb_full)
                    recur_rgb_visible = self.recursive_rgb_head(rgb_visible)

                    # 6. Mask token expansion (after recursive MLP)
                    thermal_full = self.croco_encoded_mask_expension(recur_thermal_visible, mask_thermal, patch_B, patch_N, patch_D)
                    rgb_full = self.croco_encoded_mask_expension(recur_rgb_visible, mask_rgb, patch_B_rgb, patch_N_rgb, patch_D_rgb)

                    out = paired_thermal
                    if return_masked_patch:
                        masked_patch_thermal = thermal_full

                    # 7. Prepare reference features for cross-attention
                    # For each decoder layer, use the full recursive MLP output as reference
                    thermal_ref_features = {layer: recur_paired_thermal_full for layer in self.dino_decoder_layers}
                    rgb_ref_features = {layer: recur_paired_rgb_full for layer in self.dino_decoder_layers}

                    # 8. DINO Decoder forward (separate pathway from encoder)
                    # Thermal: masked thermal decoded with full RGB reference
                    # RGB: masked RGB decoded with full thermal reference
                    thermal_decoded, _ = self.dino_decoder.forward(
                        x=thermal_full + self.decoder_pos_embed,
                        ref_features=rgb_ref_features,
                    )
                    rgb_decoded, _ = self.dino_decoder.forward(
                        x=rgb_full + self.decoder_pos_embed,
                        ref_features=thermal_ref_features,
                    )

                    thermal_reconed_dec = thermal_decoded
                    rgb_full_dec = rgb_decoded

                else:
                    # ========== Original CroCo/Swin Decoder Path ==========
                    # 1-4. masked encoder
                    thermal_visible, mask_thermal, patch_B, patch_N, patch_D, thermal_cls = self.croco_like_encoder(x, modality='thermal')
                    rgb_visible, mask_rgb, patch_B_rgb, patch_N_rgb, patch_D_rgb, rgb_cls = self.croco_like_encoder(paired_rgb, modality='rgb')

                    # 5. Get full features for global descriptor
                    paired_thermal = self.shared_backbone(x, return_attention=True)
                    paired_thermal_cls_attn_single_head = paired_thermal["cls_attention"].sum(dim=1)
                    penultimate_patch = paired_thermal["penultimate_norm_patchtokens"]
                    paired_thermal_full = paired_thermal["x_norm_patchtokens"]

                    paired_rgb_emb = self.shared_backbone(paired_rgb, return_attention=True)
                    paired_rgb_full = paired_rgb_emb["x_norm_patchtokens"]

                    cls_attn_map = paired_thermal_cls_attn_single_head

                    # 6. Mask token expansion
                    thermal_full = self.croco_encoded_mask_expension(thermal_visible, mask_thermal, patch_B, patch_N, patch_D)
                    rgb_full = self.croco_encoded_mask_expension(rgb_visible, mask_rgb, patch_B_rgb, patch_N_rgb, patch_D_rgb)

                    out = paired_thermal
                    if return_masked_patch:
                        masked_patch_thermal = thermal_full

                    # 7. CroCo Decoder forward
                    thermal_full_dec = thermal_full + self.decoder_pos_embed
                    rgb_full_dec = rgb_full + self.decoder_pos_embed
                    paired_thermal_dec = paired_thermal_full + self.decoder_pos_embed
                    paired_rgb_dec = paired_rgb_full + self.decoder_pos_embed

                    target_full_dec = torch.cat([thermal_full_dec, rgb_full_dec], dim=0)
                    ref_full_dec = torch.cat([paired_rgb_dec, paired_thermal_dec], dim=0)

                    for blk in self.decoder_blocks:
                        target_full_dec = blk(target_full_dec, ref_full_dec)
                    target_full_dec = self.decoder_norm(target_full_dec)

                    thermal_reconed_dec = target_full_dec[:thermal_full_dec.shape[0], :, :]
                    rgb_full_dec = target_full_dec[thermal_full_dec.shape[0]:, :, :]

                recon_loss_fn = self.calculate_recon_loss

                # 9. Prediction Head
                reconstructed_thermal_patches = self.prediction_thermal_head(thermal_reconed_dec)
                reconstructed_rgb_patches = self.prediction_rgb_head(rgb_full_dec)
                target_thermal_patches = self.patchify(x)
                target_rgb_patches = self.patchify(paired_rgb)

                # 10. Reconstruction loss 계산
                recon_loss_thermal = recon_loss_fn(reconstructed_thermal_patches, mask_thermal, target_thermal_patches)
                recon_loss_rgb = recon_loss_fn(reconstructed_rgb_patches, mask_rgb, target_rgb_patches)    
            else:
                # when inference
                # NOTE: 부르는 곳에 no_grad 호출하기
                if self.use_masked_inference:
                    # 1-4. masked thermal encoder
                    thermal_visible, mask_thermal, patch_B, patch_N, patch_D, thermal_cls = self.croco_like_encoder(x, modality='thermal')
                    
                    # 5. Mask token expansion
                    thermal_full = self.croco_encoded_mask_expension(thermal_visible, mask_thermal, patch_B, patch_N, patch_D)
                    
                    # 6. Decoder Positional Encoding
                    thermal_full_dec = thermal_full + self.decoder_pos_embed  # [B, 256, 768]
                    out = {
                        "x_norm_patchtokens": thermal_full_dec,
                        "x_norm_clstoken": thermal_cls,
                    }
                else:
                    out = self.shared_backbone(x,return_attention=True)
                    cls_attn_map = out["cls_attention"].sum(dim=1)
                    penultimate_patch = out["penultimate_norm_patchtokens"]
            agg_layer = self.thermal_aggregation
        else:
            raise ValueError("Modality must be 'rgb' or 'thermal'")
            
        # Backbone 출력 처리 (ViT 기준)
        # x['x_norm_patchtokens']: (B, num_patchs, D)
        patch_tokens = out["x_norm_patchtokens"] # torch.Size([64, 260, 768])

        sela_local_feature = None
        gem_attn_map = None
        if not self.use_masked_inference:
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

            # Compute GeM attention map: dot product between global descriptor and patch tokens
            # global_desc: [B, D], patch_tokens: [B, N, D] -> gem_attn_map: [B, N]
            gem_attn_map = torch.einsum('bd,bnd->bn', global_desc, patch_tokens)

            # selaVPR local feature computation using interpolation
            if self.args.use_sela_local_loss or (not self.training and (self.args.use_reranking in ['selaVPR', 'reconSelaVPR'])):
                x0 = patch_tokens.view(-1, H_feat, W_feat, self.output_dim).permute(0, 3, 1, 2)
                x0 = self.local_adapt(x0)
                x0 = x0.permute(0, 2, 3, 1)
                sela_local_feature = torch.nn.functional.normalize(x0, p=2, dim=-1)

        return global_desc, patch_tokens, [recon_loss_thermal, recon_loss_rgb], \
                mask_thermal, masked_patch_thermal, cls_attn_map, \
                penultimate_patch, sela_local_feature, gem_attn_map

    def forward_model_basic(self, x, modality='rgb'):
        """단일 모달리티에 대한 Forward"""
        out = self.shared_backbone(x)
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
        H_feat = W_feat = int(math.sqrt(N)) 
        x_feat = patch_tokens.permute(0, 2, 1).view(B, D, H_feat, W_feat)
        
        # Aggregation -> Descriptor
        global_desc = agg_layer(x_feat) # [B, D]

        return global_desc
    
    def forward_recon_sela_decode(self, thermal_feat, rgb_feat, return_layer=-1):
        """
        Bidirectional CroCo decoding for reconSelaVPR reranking.

        Args:
            thermal_feat: [B, 256, 768] encoder patch features from thermal
            rgb_feat: [B, 256, 768] encoder patch features from RGB
        Returns:
            thermal_decoded_local: [B, 61, 61, 768] thermal decoded with RGB context
            rgb_decoded_local: [B, 61, 61, 768] RGB decoded with thermal context
        """
        # Add positional embedding
        thermal_dec = thermal_feat + self.decoder_pos_embed
        rgb_dec = rgb_feat + self.decoder_pos_embed
        target_dec = torch.cat([thermal_dec, rgb_dec], dim=0)
        ref_dec = torch.cat([rgb_dec, thermal_dec], dim=0)

        RETURN_LAYER_NUM = return_layer
        # Direction 1: Thermal decoded with RGB as context
        layer_target_decoded = None
        target_decoded = target_dec.clone()
        for idx, blk in enumerate(self.decoder_blocks):
            target_decoded = blk(target_decoded, ref_dec)
            if idx == RETURN_LAYER_NUM - 1:
                layer_target_decoded = target_decoded.clone()
                break
        target_decoded = layer_target_decoded
        target_decoded = self.decoder_norm(target_decoded)

        # Reshape to spatial format: [B, N, C] -> [B, C, H, W]
        B = target_decoded.shape[0]
        H = W = int(math.sqrt(target_decoded.shape[1]))  # 16

        target_spatial = target_decoded.permute(0, 2, 1).view(B, -1, H, W)  # [B, 768, 16, 16]

        # Bilinear interpolation: [B, 768, 16, 16] -> [B, 768, 61, 61]
        # thermal_local = F.interpolate(thermal_spatial, size=(61, 61), mode='bilinear', align_corners=False)
        # rgb_local = F.interpolate(rgb_spatial, size=(61, 61), mode='bilinear', align_corners=False)
        target_local = self.local_adapt(target_spatial)
        target_local = target_local.permute(0, 2, 3, 1)  # [B, 61, 61, 768]
                
        # Permute to [B, H, W, C] and L2 normalize
        target_decoded_local = F.normalize(target_local, p=2, dim=-1)
        return target_decoded_local[:B//2,:,:,:], target_decoded_local[B//2:,:,:,:]

    def forward_recon_diff_decode(self, thermal_feat, rgb_feat, return_layer=-1):
        """
        Bidirectional CroCo decoding for reconSelaVPR reranking.

        Args:
            thermal_feat: [B, 256, 384] encoder patch features from thermal
            rgb_feat: [B, 256, 384] encoder patch features from RGB
        Returns:
            thermal_decoded_local: [B, 61, 61, 384] thermal decoded with RGB context
            rgb_decoded_local: [B, 61, 61, 384] RGB decoded with thermal context
        """
        # Add positional embedding
        # feature / positional 따로 interpolation
        # FIXME: 일단 size up 안하고 실험해보기
        thermal_dec = thermal_feat + self.decoder_pos_embed
        rgb_dec = rgb_feat + self.decoder_pos_embed
        target_dec = torch.cat([thermal_dec, rgb_dec], dim=0)
        ref_dec = torch.cat([rgb_dec, thermal_dec], dim=0)

        RETURN_LAYER_NUM = return_layer
        # Direction 1: Thermal decoded with RGB as context
        layer_target_decoded = None
        target_decoded = target_dec.clone()
        for idx, blk in enumerate(self.decoder_blocks):
            target_decoded = blk(target_decoded, ref_dec)
            if idx == RETURN_LAYER_NUM - 1:
                layer_target_decoded = target_decoded.clone()
                break
        target_decoded = layer_target_decoded
        target_decoded = self.decoder_norm(target_decoded)

        # Reshape to spatial format: [B, N, C] -> [B, C, H, W]
        B = target_decoded.shape[0]
        H = W = int(math.sqrt(target_decoded.shape[1]))  # 16

        target_spatial = target_decoded.permute(0, 1, 2)
        target_decoded = F.normalize(target_spatial, p=2, dim=-1)
        return target_decoded[:B//2,:,:], target_decoded[B//2:,:,:]

    def forward(self, x, flags, paired_rgb=None, return_mask=False, return_masked_patch=False):
        if not isinstance(flags, torch.Tensor):
            flags = torch.tensor(flags, device=x.device)

        if flags.device != x.device:
            flags = flags.to(x.device)

        is_rgb = (flags == 1)
        final_emb = torch.zeros((x.size(0), self.output_dim), device=x.device)
        patch_emb = torch.zeros((x.size(0), self.patch_count, self.output_dim), device=x.device)
        penultimate_patch_emb = torch.zeros((x.size(0), self.patch_count, self.output_dim), device=x.device)
        masks = torch.zeros((x.size(0), self.patch_count), dtype=torch.bool, device=x.device)
        cls_attn_map = torch.zeros((x.size(0), self.patch_count), device=x.device)
        gem_attn_map = torch.zeros((x.size(0), self.patch_count), device=x.device)
        recon_losses = None
        masked_patch_emb = None

        # selaVPR local embedding initialization (768-dim with interpolation)
        sela_local_emb = None
        if self.args.use_sela_local_loss or self.args.use_reranking in ['selaVPR', 'reconSelaVPR']:
            sela_local_emb = torch.zeros((x.size(0), 61, 61, 128), device=x.device)

        if is_rgb.any():
            global_emb, patch_rgb, _, _, _, cls_rgb_attn_map, penultimate_patch_rgb, sela_local_rgb, gem_rgb_attn_map = self.forward_model(x[is_rgb], modality='rgb')
            if global_emb is not None:
                final_emb[is_rgb] = global_emb
            if patch_rgb is not None:
                patch_emb[is_rgb] = patch_rgb
            if cls_rgb_attn_map is not None:
                cls_attn_map[is_rgb] = cls_rgb_attn_map
            if gem_rgb_attn_map is not None:
                gem_attn_map[is_rgb] = gem_rgb_attn_map
            if penultimate_patch_rgb is not None:
                penultimate_patch_emb[is_rgb] = penultimate_patch_rgb
            if sela_local_rgb is not None and sela_local_emb is not None:
                sela_local_emb[is_rgb] = sela_local_rgb

        if (~is_rgb).any():
            global_emb, patch_thermal, recon_losses, mask, masked_patch_thermal, cls_thermal_attn_map, penultimate_patch_thermal, sela_local_thermal, gem_thermal_attn_map = self.forward_model(x[~is_rgb], modality='thermal', paired_rgb=paired_rgb, return_masked_patch=return_masked_patch)
            if global_emb is not None:
                final_emb[~is_rgb] = global_emb
            if patch_thermal is not None:
                patch_emb[~is_rgb] = patch_thermal
            if return_mask:
                masks[~is_rgb] = mask
            if return_masked_patch:
                masked_patch_emb = masked_patch_thermal # torch.Size([4, 256, 768])
            if cls_thermal_attn_map is not None:
                cls_attn_map[~is_rgb] = cls_thermal_attn_map
            if gem_thermal_attn_map is not None:
                gem_attn_map[~is_rgb] = gem_thermal_attn_map
            if penultimate_patch_thermal is not None:
                penultimate_patch_emb[~is_rgb] = penultimate_patch_thermal
            if sela_local_thermal is not None and sela_local_emb is not None:
                sela_local_emb[~is_rgb] = sela_local_thermal

        if return_masked_patch:
            return final_emb, patch_emb, recon_losses, masks, cls_attn_map, penultimate_patch_emb, sela_local_emb, gem_attn_map, masked_patch_emb
        else:
            return final_emb, patch_emb, recon_losses, masks, cls_attn_map, penultimate_patch_emb, sela_local_emb, gem_attn_map

    def calculate_recon_loss(self, pred, mask, target, confidence_map=None):
        recon_loss = self.reconstruction_criterion(
            pred=pred,        # [B, 256, 768]
            mask=mask,        # [B, 256]
            target=target,    # [B, 3, 256, 768]
        )
        return recon_loss

def get_backbone(pretrained_foundation, foundation_model_path, args=None):
    model_path = Path(foundation_model_path)
    model_name = model_path.parts[-1].lower()

    # Determine model size (vit_small or vit_base)
    use_vit_small = 'vits' in model_name
    use_register = 'reg4' in model_name
    num_register_tokens = 4 if use_register else 0

    if use_register:
        print("=" * 40)
        print("- Using REGISTER DINOv2 -")
        print("=" * 40)

    if use_vit_small:
        print("=" * 40)
        print("- Using ViT-Small (embed_dim=384) -")
        print("=" * 40)
        backbone = vit_small(patch_size=14, img_size=518, init_values=1, block_chunks=0, num_register_tokens=num_register_tokens)
        if args is not None:
            args.features_dim = 384
    else:
        print("=" * 40)
        print("- Using ViT-Base (embed_dim=768) -")
        print("=" * 40)
        backbone = vit_base(patch_size=14, img_size=518, init_values=1, block_chunks=0, num_register_tokens=num_register_tokens)
        if args is not None:
            args.features_dim = 768

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