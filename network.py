# network.py
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from backbone.vision_transformer import vit_small, vit_base, vit_large, vit_giant2
from timm.models.vision_transformer import VisionTransformer, _cfg, PatchEmbed, Block
import math
import numpy as np
from sklearn.neighbors import NearestNeighbors
import torchvision.models as models
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
        self.rgb_backbone = get_backbone(pretrained_foundation, foundation_model_path)
        self.thermal_backbone = get_backbone(pretrained_foundation, foundation_model_path)
        self.output_dim = 768
        self.use_masked_inference = False
        self.use_only_cross_decdoer = args.use_only_cross_decoder
        self.recon_loss_type = args.recon_loss_type
               
        # Croco settings
        dec_depth = args.num_decoder_depth
        dec_num_heads = 16
        self.decoder_thermal_blocks = nn.ModuleList([
            CroCoDecoderBlock(self.output_dim, dec_num_heads) 
            for _ in range(dec_depth)
        ])
        self.decoder_rgb_blocks = nn.ModuleList([
            CroCoDecoderBlock(self.output_dim, dec_num_heads) 
            for _ in range(dec_depth)
        ])
        
        self.decoder_embed_dim = 384
        self.decoder_norm = nn.LayerNorm(self.output_dim)
        self.r2_decoder_norm = nn.LayerNorm(self.decoder_embed_dim)
        
        self.mask_token = None
        self._set_mask_token(self.output_dim)
        self._set_decode_positional_embedding(self.output_dim)
        self._set_mask_generator(16*16, args.croco_mask_ratio)
        self._set_prediction_head(self.output_dim, 14)
        self.local_head_rgb = nn.Linear(768, 128, bias=True)
        self.local_head_thermal = nn.Linear(768, 128, bias=True)
        self.local_head_rgb.weight.data.normal_(mean=0.0, std=0.01)
        self.local_head_thermal.weight.data.normal_(mean=0.0, std=0.01)
        self.local_head_rgb.bias.data.zero_()
        self.local_head_thermal.bias.data.zero_()
        
        self.pair_head = nn.Linear(7, self.decoder_embed_dim, bias=True)
        self.pair_head_2 = nn.Linear(self.decoder_embed_dim, self.decoder_embed_dim, bias=True)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.decoder_embed_dim))
        self.cls_token_2 = nn.Parameter(torch.zeros(1, 1, self.decoder_embed_dim))
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_token_2, std=.02)
        
        decoder_num_heads = 6
        decoder_mlp_ratio = 4.
        decoder_norm_layer = nn.LayerNorm
        decoder_depth = 4
        self.blocks = nn.ModuleList([
            Block(self.decoder_embed_dim, decoder_num_heads, decoder_mlp_ratio, qkv_bias=True, norm_layer=nn.LayerNorm)
            for i in range(decoder_depth)])

        self.blocks_2 = nn.ModuleList([
            Block(self.decoder_embed_dim, decoder_num_heads, decoder_mlp_ratio, qkv_bias=True, norm_layer=nn.LayerNorm)
            for i in range(2)])
        
        self.reconstruction_criterion = MaskedMSE(
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
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, 256, dec_embed_dim))
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
        p = 14
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        
        return x

    def unpatchify(self, x, channels=3):
        """
        x: (N, L, patch_size**2 *channels)
        imgs: (N, 3, H, W)
        """
        patch_size = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, patch_size, patch_size, channels))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], channels, h * patch_size, h * patch_size))
        return imgs
    
    def croco_like_encoder(self, x, modality='thermal'):
        # thermal       : torch.Size([4, 3, 224, 224])
        # paired_rgb   : torch.Size([4, 3, 224, 224])
        
        # 1. 먼저 thermal의 patch_embedding한 걸 가져온다
        if modality == 'thermal':
            current_backbone = self.thermal_backbone
        elif modality == 'rgb':
            current_backbone = self.rgb_backbone
        else:
            raise ValueError(f"Wrong Modality: {modality} in function::croco_like_encoder")
        
        image_patch = current_backbone.patch_embed(x)
        patch_B, patch_N, patch_D = image_patch.shape  # N=256, D=768
        
        # 2. positional embedding도 구해서 더해준다.
        pos_tokens = current_backbone.pos_embed[:, 1:, :]
        pos_embed_grid = pos_tokens.reshape(1, 37, 37, 768).permute(0, 3, 1, 2) # (BHWC) => (BCHW)
        pos_embed_resized = F.interpolate(pos_embed_grid, size=(16, 16), mode='bicubic', align_corners=False)
        pos_embed_final = pos_embed_resized.permute(0, 2, 3, 1).flatten(1, 2) # (BCHW) => (BNC)
        image_patch = image_patch + pos_embed_final  # [B, 256, 768]
        
        # 3. patch embedding을 masking 해준다.
        mask = self.mask_generator(image_patch)
        patch_visible = image_patch[~mask].reshape(patch_B, -1, patch_D)
        
        # 4. unmasked된 patch들만 DINOv2 통과시키기
        for blk in current_backbone.blocks:
            patch_visible = blk(patch_visible)
        patch_visible = current_backbone.norm(patch_visible)
        # attn_map.shape -> [4, 12, 52, 52]

        return patch_visible, mask, patch_B, patch_N, patch_D

    def croco_encoded_mask_expension(self, thermal_visible, mask, patch_B, patch_N, patch_D):
        # CROCO로 masking된 부분 mask token으로 채워넣기
        thermal_full = self.mask_token.expand(patch_B, patch_N, -1).clone()
        thermal_full[~mask] = thermal_visible.flatten(0, 1)
        thermal_full = thermal_full.view(patch_B, patch_N, patch_D)
        return thermal_full
    
    def forward_model(self, x, paired_rgb=None, modality='rgb', return_masked_patch=False):
        """단일 모달리티에 대한 Forward"""
        # self.use_masked_inference: rerank를 위해, decoder에 들어가기 바로 전 단계를 뱉는다
        
        global_desc = None
        recon_loss = None
        mask_thermal = None
        masked_patch_thermal = None
        if modality == 'rgb':
            if self.use_masked_inference:
                # rgb_full = self.rgb_backbone(x)
                # rgb_full = rgb_full["x_norm_patchtokens"] 
                # rgb_full = rgb_full + self.decoder_pos_embed  # [B, 256, 768]
                # out = {"x_norm_patchtokens": rgb_full}

                rgb_visible, mask_rgb, patch_B, patch_N, patch_D = self.croco_like_encoder(x, modality='rgb')
                rgb_full = self.croco_encoded_mask_expension(rgb_visible, mask_rgb, patch_B, patch_N, patch_D)
                rgb_full_dec = rgb_full + self.decoder_pos_embed  # [B, 256, 768]
                out = {"x_norm_patchtokens": rgb_full_dec}
            else:
                out = self.rgb_backbone(x)
            agg_layer = self.rgb_aggregation
        elif modality == 'thermal':
            if self.training:
                # 1-4. masked thermal encoder
                # FIXME: 같은 곳을 masking해야하나? 일단 성능 잘 나오니...
                thermal_visible, mask_thermal, patch_B, patch_N, patch_D = self.croco_like_encoder(x, modality='thermal')
                rgb_visible, mask_rgb, patch_B_rgb, patch_N_rgb, patch_D_rgb = self.croco_like_encoder(paired_rgb, modality='rgb')
                
                # 5. paired RGB도 feature tokens 추출하기
                paired_thermal = self.thermal_backbone(x, return_attention=True)
                paired_thermal_full = paired_thermal["x_norm_patchtokens"]
                paired_thermal_attn = paired_thermal["attention"][:,:,1:,1:] # [B, MHA, 256, 256] # TODO: 이렇게 제거하는게 맞는지도 한번 체크해보기
                paired_thermal_cls_attn = paired_thermal["cls_attention"][:, :, 1:] # [B, MHA, 256]
                paired_thermal_cls_attn_single_head = paired_thermal_cls_attn.sum(dim=1) # [B, 256]
                
                paired_rgb = self.rgb_backbone(paired_rgb, return_attention=True)
                paired_rgb_full = paired_rgb["x_norm_patchtokens"]
                paired_rgb_attn = paired_rgb["attention"][:,:,1:,1:] # CLS Token 제거
                paired_rgb_cls_attn = paired_rgb["cls_attention"][:, :, 1:]
                paired_rgb_cls_attn_single_head = paired_rgb_cls_attn.sum(dim=1)
                
                if True:
                    # R2Former Reranking module 학습 구현부
                    # 1. rgb/thermal에서 중요한 feature tokens(100개)만 남기기
                    TOP_PATCH_COUNT = 100
                    thermal_order = torch.argsort(paired_thermal_cls_attn_single_head, dim=1, descending=True) # thermal_order: [B, 256]
                    thermal_order = thermal_order[:, :TOP_PATCH_COUNT]
                    rgb_order = torch.argsort(paired_rgb_cls_attn_single_head, dim=1, descending=True) # thermal_order: [B, 256]
                    rgb_order = rgb_order[:, :TOP_PATCH_COUNT] # rgb_order: [4, 100]
                    rgb_idx = rgb_order.unsqueeze(2).expand(-1, -1, 768)
                    thermal_idx = thermal_order.unsqueeze(2).expand(-1, -1, 768)
                    selected_rgb_patches = torch.gather(paired_rgb_full, axis=1, index=rgb_idx)
                    selected_thermal_patches = torch.gather(paired_thermal_full, axis=1, index=thermal_idx)
                    
                    # 2. 선택된 patch를 각각 linear을 태워서 768 -> 128 dimension
                    ## local_features 만들기
                    local_rgb_features = self.local_head_rgb(selected_rgb_patches)
                    local_thermal_features = self.local_head_thermal(selected_thermal_patches)
                    
                    # 3. linear에 추가정보 넣어서 128 -> 131 차원 만들어주기 (positional embedding은 넣어야할지 말지 고민중)
                    B_sz, _, W, H = x.shape # x: [4, 3, 224, 224] 가정
                    patch_size = 16
                    grid_W = int(np.ceil(W / patch_size)) # 14
                    HW = max(H, W) # 정규화를 위한 분모 (224)

                    # --- [RGB] x_xy (좌표) 계산 ---
                    # Col (X좌표): index % 14
                    rgb_col = (rgb_order % grid_W) * patch_size + (patch_size // 2)
                    # Row (Y좌표): index // 14
                    rgb_row = (rgb_order // grid_W) * patch_size + (patch_size // 2)
                    
                    # 정규화 후 합치기: [B, 100, 2]
                    x_xy_rgb = torch.stack([rgb_col / float(HW), rgb_row / float(HW)], dim=2) # x_xy_rgb: [B, 100, 2]

                    # --- [Thermal] x_xy (좌표) 계산 ---
                    thermal_col = (thermal_order % grid_W) * patch_size + (patch_size // 2)
                    thermal_row = (thermal_order // grid_W) * patch_size + (patch_size // 2)
                    
                    # 정규화 후 합치기: [B, 100, 2]
                    x_xy_thermal = torch.stack([thermal_col / float(HW), thermal_row / float(HW)], dim=2) # x_xy_thermal: [B, 100, 2]

                    # --- [RGB] x_attention (중요도) 계산 ---
                    # 선택된 패치의 attention score 가져오기 (Gather)
                    rgb_att_val = torch.gather(paired_rgb_cls_attn_single_head, axis=1, index=rgb_order) # [B, 100]
                    rgb_att_norm = rgb_att_val / torch.max(rgb_att_val, dim=1, keepdim=True)[0]
                    rgb_att_norm = rgb_att_norm.unsqueeze(2) # [B, 100, 1]

                    # --- [Thermal] x_attention (중요도) 계산 ---
                    thermal_att_val = torch.gather(paired_thermal_cls_attn_single_head, axis=1, index=thermal_order)
                    thermal_att_norm = thermal_att_val / torch.max(thermal_att_val, dim=1, keepdim=True)[0]
                    thermal_att_norm = thermal_att_norm.unsqueeze(2) # [B, 100, 1]

                    # ---------------------------------------------------------
                    # 4. Final Concatenation (Feature + Coord + Score)
                    # ---------------------------------------------------------
                    # 결과 Shape: [B, 100, 2 + 1 + 128] = [B, 100, 131]
                    # R2Former에 들어갈 최종 입력 (x_rerank, y_rerank)
                    rgb_rerank_input = torch.cat([x_xy_rgb, rgb_att_norm, local_rgb_features], dim=2)
                    thermal_rerank_input = torch.cat([x_xy_thermal, thermal_att_norm, local_thermal_features], dim=2)
                    ####################################################################
                    
                    # 4. correlation matrix 만들기 (100x100x7)
                    '''
                    paired_rgb_attn.shape: [4, 12, 256, 256]
                    local_rgb_features.shape: [4, 100, 128]
                    local_thermal_features.shape: [4, 100, 128]
                    global_score: [4, 128]
                    '''
                    B = rgb_rerank_input.shape[0]
                    N = rgb_rerank_input.shape[1]
                    self.num_corr = 5
                    rgb_rerank_token = F.normalize(rgb_rerank_input[:, :, 3:], p=2, dim=2)
                    thermal_rerank_token = F.normalize(thermal_rerank_input[:, :, 3:], p=2, dim=2)
                    rgb_coordinate = rgb_rerank_token[:, :, :3].detach().clamp(min=0, max=1)
                    thermal_coordinate = thermal_rerank_token[:, :, :3].detach().clamp(min=0, max=1)
                    correlation = torch.matmul(rgb_rerank_token, thermal_rerank_token.permute((0, 2, 1)))
                    xy_matrix = torch.cat(
                        [rgb_coordinate.unsqueeze(2).repeat(1, 1, thermal_rerank_token.shape[1], 1),
                        thermal_coordinate.unsqueeze(1).repeat(1, rgb_rerank_token.shape[1], 1, 1), correlation.unsqueeze(3)],
                        dim=3)
                    
                    ###########
                    order_q = torch.argsort(correlation.unsqueeze(3), dim=2, descending=True).repeat(1, 1, 1, 7)
                    order_k = torch.argsort(correlation.unsqueeze(3), dim=1, descending=True).repeat(1, 1, 1, 7)
                    select_q = torch.gather(input=xy_matrix, index=order_q[:, :, :self.num_corr, :], dim=2)
                    select_k = torch.gather(input=xy_matrix, index=order_k[:, :self.num_corr, :, :], dim=1)
                    select_k_copy = select_k.clone()
                    select_k_copy[:,:,:,:6] = torch.flip(select_k[:,:,:,:6].reshape(select_k.shape[0], select_k.shape[1],select_k.shape[2],2,3),dims=(3,)).reshape(select_k.shape[0], select_k.shape[1],select_k.shape[2],6)
                    select = torch.cat([select_q, select_k.permute((0, 2, 1, 3))], dim=1)
                    select_copy = torch.cat([select_q, select_k.permute((0, 2, 1, 3))], dim=1)
                    N_select = select.shape[1]

                    ###########
                    # Linear1
                    pair_matrix = self.pair_head(select.reshape(B * N_select * self.num_corr, 7)).reshape(B * N_select, self.num_corr, self.decoder_embed_dim)
                    pair_matrix += get_2d_sincos_pos_embed_from_grid(self.decoder_embed_dim, select_copy.reshape(B * N_select, self.num_corr, 7)[:,:,3:5])
                    x = torch.cat([self.cls_token_2.repeat(B*N_select, 1, 1), pair_matrix], dim=1)
                    # Transformer1
                    for blk in self.blocks_2:
                        x = blk(x)
                    x = self.r2_decoder_norm(x)

                    # Linear2
                    x = self.pair_head_2(x[:,0,:].reshape(B*N_select, self.decoder_embed_dim)).reshape(B, N_select, self.decoder_embed_dim)
                    x = x.reshape(B, N_select, self.decoder_embed_dim) + get_2d_sincos_pos_embed_from_grid(self.decoder_embed_dim, select_copy[:,:,0,0:2])
                    x = torch.cat([self.cls_token.repeat(B, 1, 1), x], dim=1)

                    # Transformer2
                    for blk in self.blocks:
                        x = blk(x)
                    x = self.r2_decoder_norm(x)

                    # 4-1. cosine similarity 구하기
                    # 4-2. attention value 구하기
                    # 4-3. positional embedding값 구하기
                    # 4-4. 자기와 대응되는 좌표값 넣어주기
                    # 5. top5를 골라 두개로 나눠주기
                
                # 6. Mask token expansion
                thermal_full = self.croco_encoded_mask_expension(thermal_visible, mask_thermal, patch_B, patch_N, patch_D)
                rgb_full = self.croco_encoded_mask_expension(rgb_visible, mask_rgb, patch_B_rgb, patch_N_rgb, patch_D_rgb)
                
                # out을 미리 저장
                out = {"x_norm_patchtokens": paired_thermal_full.clone()}
                if return_masked_patch: masked_patch_thermal = thermal_full # 이후로 안건들여서 clone안해도 됨
                
                # 7. Decoder Positional Encoding
                thermal_full_dec = thermal_full + self.decoder_pos_embed  # [B, 256, 768]
                rgb_full_dec = rgb_full + self.decoder_pos_embed  # [B, 256, 768]
                paired_thermal_full = paired_thermal_full + self.decoder_pos_embed  # [B, 256, 768]
                paired_rgb_full = paired_rgb_full + self.decoder_pos_embed  # [B, 256, 768]
                
                # 8. decoder 통과시키기
                for blk in self.decoder_thermal_blocks:
                    thermal_full_dec = blk(thermal_full_dec, paired_rgb_full)
                thermal_full_dec = self.decoder_norm(thermal_full_dec)

                for blk in self.decoder_rgb_blocks:
                    rgb_full_dec = blk(rgb_full_dec, paired_thermal_full)
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
                # NOTE: 부르는 곳에 no_grad 호출하기
                if self.use_masked_inference:
                    # 1-4. masked thermal encoder
                    thermal_visible, mask_thermal, patch_B, patch_N, patch_D = self.croco_like_encoder(x, modality='thermal')
                    
                    # 5. Mask token expansion
                    thermal_full = self.croco_encoded_mask_expension(thermal_visible, mask_thermal, patch_B, patch_N, patch_D)
                    
                    # 6. Decoder Positional Encoding
                    thermal_full_dec = thermal_full + self.decoder_pos_embed  # [B, 256, 768]
                    out = {"x_norm_patchtokens": thermal_full_dec}
                else:
                    out = self.thermal_backbone(x)
            agg_layer = self.thermal_aggregation
        else:
            raise ValueError("Modality must be 'rgb' or 'thermal'")
            
        # Backbone 출력 처리 (ViT 기준)
        # x['x_norm_patchtokens']: (B, num_patchs, D)
        patch_tokens = out["x_norm_patchtokens"]
        
        if not self.use_masked_inference:
            # attnetion_dict_keys(['x_norm_clstoken', 'x_norm_patchtokens', 'x_prenorm', 'masks'])
            B, N, D = patch_tokens.shape
            
            # 224,224 정방 이미지 입력 가정(patch 2D 복원)
            H_feat = W_feat = int(math.sqrt(N)) 
            x_feat = patch_tokens.permute(0, 2, 1).view(B, D, H_feat, W_feat)
            
            # Aggregation -> Descriptor
            global_desc = agg_layer(x_feat) # [B, D]
        
        return global_desc, patch_tokens, recon_loss, mask_thermal, masked_patch_thermal

    def forward(self, x, flags, paired_rgb=None, return_mask=False, return_masked_patch=False):
        is_rgb = torch.tensor([f == 'rgb' for f in flags], device=x.device)
        final_emb = torch.zeros((x.size(0), self.output_dim), device=x.device)
        patch_emb = torch.zeros((x.size(0), 256, self.output_dim), device=x.device)
        masks = torch.zeros((x.size(0), 256), dtype=torch.bool, device=x.device)
        
        recon_loss = None
        masked_patch_emb = None
        if is_rgb.any():
            global_emb, patch_rgb, _, _, _ = self.forward_model(x[is_rgb], modality='rgb')
            if global_emb is not None: final_emb[is_rgb] = global_emb
            patch_emb[is_rgb] = patch_rgb
        if (~is_rgb).any():
            global_emb, patch_thermal, recon_loss, mask, masked_patch_thermal = self.forward_model(x[~is_rgb], modality='thermal', paired_rgb=paired_rgb, return_masked_patch=return_masked_patch)
            if global_emb is not None: final_emb[~is_rgb] = global_emb
            patch_emb[~is_rgb] = patch_thermal
            if return_mask: masks[~is_rgb] = mask
            if return_masked_patch:
                masked_patch_emb = masked_patch_thermal # torch.Size([4, 256, 768])

        
        if return_masked_patch:
            return final_emb, patch_emb, recon_loss, masks, masked_patch_emb
        else:
            return final_emb, patch_emb, recon_loss, masks

    def calculate_recon_loss(self, pred, mask, target, confidence_map=None):
        recon_loss = self.reconstruction_criterion(
            pred=pred,        # [B, 256, 768]
            mask=mask,        # [B, 256]
            target=target,    # [B, 3, 256, 768]
        )
        return recon_loss

def get_backbone(pretrained_foundation, foundation_model_path):
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