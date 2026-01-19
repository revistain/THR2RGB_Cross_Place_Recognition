# network.py
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from backbone.vision_transformer import vit_small, vit_base, vit_large, vit_giant2
import math
import numpy as np
from sklearn.neighbors import NearestNeighbors
import torchvision.models as models

from croco.models.masking import RandomMask
from croco.models.criterion import MaskedMSE

class DistanceBasedRerankLoss(nn.Module):
    def __init__(self, midpoint=25.0, steepness=0.15):
        super().__init__()
        self.midpoint = midpoint
        self.steepness = steepness
    
    def distance_to_label(self, distance):
        """Sigmoid-based soft label"""
        return 1.0 / (1.0 + torch.exp(
            self.steepness * (distance - self.midpoint)
        ))
    
    def forward(self, logits, distances):
        """
        Args:
            logits: [B] or [B*K] - model predictions
            distances: [B] or [B*K] - GPS distances in meters
        """
        soft_labels = self.distance_to_label(distances)
        loss = F.binary_cross_entropy_with_logits(logits, soft_labels)
        return loss
    
class CroCoDecoderBlock(nn.Module):
    """
    CroCo 원본 Decoder Block
    
    구조:
    1. Self-Attention: decoder 내부 token들 간 정보 혼합
    2. Cross-Attention: RGB encoder output 참조
    3. MLP: Position-wise feed-forward
    
    모두 Pre-LayerNorm + Residual connection 사용
    """
    def __init__(self, dim=768, num_heads=12, mlp_ratio=4.0):
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
        
    def forward(self, x, y):
        """
        Args:
            x: [B, N, D] - decoder input (thermal + mask tokens)
            encoder_output: [B, M, D] - RGB encoder output (참조할 정보)
        Returns:
            x: [B, N, D] - updated decoder features
        """
        # Step 1: Self-Attention
        x_norm = self.norm1(x)
        x = x + self.self_attn(x_norm, x_norm, x_norm)[0]
        
        # Step 2: Cross-Attention
        x_norm = self.norm2(x)
        encoder_norm = self.norm_cross(y)
        x = x + self.cross_attn(
            query=x_norm,
            key=encoder_norm,
            value=encoder_norm
        )[0]
        
        # Step 3: MLP
        x = x + self.mlp(self.norm3(x))
        
        return x

class CroCoOnlyCrossDecoderBlock(nn.Module):
    """
    CroCo 원본 Decoder Block
    
    구조:
    1. Self-Attention: decoder 내부 token들 간 정보 혼합
    2. Cross-Attention: RGB encoder output 참조
    3. MLP: Position-wise feed-forward
    
    모두 Pre-LayerNorm + Residual connection 사용
    """
    def __init__(self, dim=768, num_heads=12, mlp_ratio=4.0):
        super().__init__()
        
        # Self-Attention components
        self.norm1 = nn.LayerNorm(dim)
        self.norm_cross = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        # MLP components
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )
        
    def forward(self, x, y):
        """
        Args:
            x: [B, N, D] - decoder input (thermal + mask tokens)
            encoder_output: [B, M, D] - RGB encoder output (참조할 정보)
        Returns:
            x: [B, N, D] - updated decoder features
        """
        # Step 1: Cross-Attention
        x_norm = self.norm1(x)
        encoder_norm = self.norm_cross(y)
        x = x + self.cross_attn(
            query=x_norm,
            key=encoder_norm,
            value=encoder_norm
        )[0]
        
        # Step 2: MLP
        x = x + self.mlp(self.norm2(x))
        
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
        self.use_single_pass = args.use_single_pass
        self.use_masked_inference = False
        self.use_feature_level_recon_loss = args.use_feature_level_recon_loss
        self.use_confidence_map = args.use_confidence_map
        self.use_only_cross_decdoer = args.use_only_cross_decoder
        self.use_contrastive_recon_loss = args.use_contrastive_recon_loss
        self.recon_loss_type = args.recon_loss_type
        self.rerank_weight = args.rerank_weight

        # decoder에 들어갈 CLS 토큰 (학습 가능)
        self.rerank_cls_token = nn.Parameter(torch.zeros(1, 1, self.output_dim))
        nn.init.normal_(self.rerank_cls_token, std=.02)

        # CLS 토큰을 점수(0~1)로 변환할 Head (BCEWithLogitsLoss 사용 예정이므로 Sigmoid 생략 권장)
        self.criterion = DistanceBasedRerankLoss(midpoint=25, steepness=0.15)
        self.rerank_head = nn.Linear(self.output_dim, 1)
        nn.init.xavier_uniform_(self.rerank_head.weight)
        nn.init.zeros_(self.rerank_head.bias)
               
        # Croco settings
        dec_depth = args.num_decoder_depth
        dec_num_heads = 16
        if args.use_only_cross_decoder:
            self.decoder_blocks = nn.ModuleList([
                CroCoOnlyCrossDecoderBlock(self.output_dim, dec_num_heads) 
                for _ in range(dec_depth)
            ])
        else:
            self.decoder_blocks = nn.ModuleList([
                CroCoDecoderBlock(self.output_dim, dec_num_heads) 
                for _ in range(dec_depth)
            ])
        self.decoder_norm = nn.LayerNorm(self.output_dim)
        
        self.mask_token = None
        self._set_mask_token(self.output_dim)
        self._set_decode_positional_embedding(self.output_dim)
        self._set_mask_generator(16*16, args.croco_mask_ratio)
        self._set_prediction_head(self.output_dim, 14)
        self._set_confidence_head(self.output_dim, 16*16)
        
        if args.use_confidence_map:
            # FIXME: 왜 이따구로 짰었지, 걍 forward에서 None들어오는지로 판단하게 바꾸기(언젠가))
            self.reconstruction_criterion = MaskedMSE(
                norm_pix_loss=False,
                masked=True,
                loss_type=self.recon_loss_type
            )
        else:
            self.reconstruction_criterion = MaskedMSE(
                norm_pix_loss=False,
                masked=True,
                loss_type=self.recon_loss_type
            )

        # 2. Aggregation Layer (각각 따로 두는 것을 추천)
        # GeM의 파라미터 p가 모달리티별로 다르게 학습될 수 있도록 분리합니다.
        if args.use_GeMAdditionalLayer:
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
    
    def _set_mask_token(self, dec_embed_dim):
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_embed_dim))
        nn.init.normal_(self.mask_token, std=.02)

    def _set_confidence_head(self, dec_embed_dim, hidden_dim=256):
        self.confidence_head = nn.Sequential(
            nn.Linear(dec_embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        nn.init.xavier_uniform_(self.confidence_head[0].weight)
        nn.init.zeros_(self.confidence_head[0].bias)
        
    def _set_decode_positional_embedding(self, dec_embed_dim):
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, 256, dec_embed_dim))
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
    
    def _set_mask_generator(self, num_patches, mask_ratio):
        """Random masking generator 초기화"""
        self.mask_generator = RandomMask(num_patches, mask_ratio)

    def _set_prediction_head(self, dec_embed_dim, patch_size):
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
    
    def through_thermal_decoder(thermal_patch, rgb_patch):
        ...

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
    
    def croco_like_encoder(self, x):
        # thermal       : torch.Size([4, 3, 224, 224])
        # aligned_rgb   : torch.Size([4, 3, 224, 224])
        
        # 1. 먼저 thermal의 patch_embedding한 걸 가져온다
        thermal_patch = self.thermal_backbone.patch_embed(x)
        patch_B, patch_N, patch_D = thermal_patch.shape  # N=256, D=768
        
        # 2. positional embedding도 구해서 더해준다.
        pos_tokens = self.thermal_backbone.pos_embed[:, 1:, :]
        pos_embed_grid = pos_tokens.reshape(1, 37, 37, 768).permute(0, 3, 1, 2) # (BHWC) => (BCHW)
        pos_embed_resized = F.interpolate(pos_embed_grid, size=(16, 16), mode='bicubic', align_corners=False)
        pos_embed_final = pos_embed_resized.permute(0, 2, 3, 1).flatten(1, 2) # (BCHW) => (BNC)
        thermal_patch = thermal_patch + pos_embed_final  # [B, 256, 768]
        
        # 3. patch embedding을 masking 해준다.
        mask = self.mask_generator(thermal_patch)
        thermal_visible = thermal_patch[~mask].reshape(patch_B, -1, patch_D)
        
        # 4. unmasked된 patch들만 DINOv2 통과시키기
        for blk in self.thermal_backbone.blocks:
            thermal_visible = blk(thermal_visible)
        thermal_visible = self.thermal_backbone.norm(thermal_visible)

        return thermal_visible, mask, patch_B, patch_N, patch_D

    def croco_encoded_mask_expension(self, thermal_visible, mask, patch_B, patch_N, patch_D):
        # CROCO로 masking된 부분 mask token으로 채워넣기
        try:
            mask_tokens = self.mask_token.expand(patch_B, patch_N, -1) # [B, 256, 768]
            thermal_full = mask_tokens.clone()
            for i in range(patch_B):
                thermal_full[i, ~mask[i]] = thermal_visible[i]
            thermal_full = thermal_full.reshape(patch_B, -1, patch_D)
        except Exception as e:
            print("Error: ", e)
            breakpoint()
        return thermal_full
    
    def forward_model(self, x, aligned_rgb=None, modality='rgb', return_masked_patch=False,rgbs=None,dist_pos=None,dist_neg=None):
        """단일 모달리티에 대한 Forward"""
        # self.use_masked_inference: rerank를 위해, decoder에 들어가기 바로 전 단계를 뱉는다
        
        global_desc = None
        recon_loss = None
        mask = None
        masked_patch = None
        if modality == 'rgb':
            if self.use_masked_inference:
                rgb_full = self.rgb_backbone(x)
                rgb_full = rgb_full["x_norm_patchtokens"] 
                rgb_full = rgb_full + self.decoder_pos_embed  # [B, 256, 768]
                out = {"x_norm_patchtokens": rgb_full}
            else:
                out = self.rgb_backbone(x)
            agg_layer = self.rgb_aggregation
        elif modality == 'thermal':
            if self.training:
                # 1-4. masked thermal encoder
                thermal_visible, mask, patch_B, patch_N, patch_D = self.croco_like_encoder(x)
                
                # 5. aligned RGB feature tokens 추출
                rgb_full = self.rgb_backbone(aligned_rgb)
                rgb_full = rgb_full["x_norm_patchtokens"] 
                
                # 6. Mask token expansion
                thermal_full = self.croco_encoded_mask_expension(thermal_visible, mask, patch_B, patch_N, patch_D)
            
                # 7. Decoder Positional Encoding
                thermal_full_dec = thermal_full + self.decoder_pos_embed
                rgb_full = rgb_full + self.decoder_pos_embed
                
                # 8. Positive pair decoder (thermal + aligned_rgb)
                B = thermal_full_dec.shape[0]
                cls_token_pos = self.rerank_cls_token.expand(B, -1, -1)
                thermal_full_dec_cls = torch.cat((cls_token_pos, thermal_full_dec), dim=1)

                for blk in self.decoder_blocks:
                    thermal_full_dec_cls = blk(thermal_full_dec_cls, rgb_full)
                thermal_full_dec_cls = self.decoder_norm(thermal_full_dec_cls)
                
                # 9. Reconstruction loss
                thermal_full_cls = thermal_full_dec_cls[:, 0, :]
                thermal_full_patch = thermal_full_dec_cls[:, 1:, :]
                reconstructed_patches = self.prediction_head(thermal_full_patch)
                target_patches = self.patchify(x)
                recon_loss = self.reconstruction_criterion(
                    pred=reconstructed_patches,
                    mask=mask,
                    target=target_patches
                )
                
                # 10. Negative pairs decoder (thermal + random_rgbs)
                NEG_COUNT = 3
                # rgbs에서 필요한 negative만 추출 (B개씩 NEG_COUNT묶음)
                rgb_neg_samples = []
                dist_neg_samples = []
                for i in range(B):
                    start_idx = i * 11 + 1  # positive 건너뛰기
                    neg_indices = list(range(start_idx, start_idx + NEG_COUNT))
                    rgb_neg_samples.append(rgbs[neg_indices])

                    neg_indices_neg = list(range(0, 0 + NEG_COUNT))
                    dist_neg_samples.append(dist_neg[i, neg_indices_neg])
                rgb_neg_batch = torch.cat(rgb_neg_samples, dim=0)  # [B*NEG_COUNT, 3, H, W]
                dist_neg_batch = torch.cat(dist_neg_samples, dim=0)  # [B*NEG_COUNT, 3, H, W]
                
                # Negative RGB encoder
                rgb_negs = self.rgb_backbone(rgb_neg_batch)["x_norm_patchtokens"]
                rgb_negs = rgb_negs
                rgb_negs = rgb_negs + self.decoder_pos_embed  # [B*NEG_COUNT, 256, 768]
                
                # Thermal 복제 및 decoder
                thermal_expanded = torch.repeat_interleave(
                    thermal_full_dec,
                    repeats=NEG_COUNT,
                    dim=0
                )
                cls_tokens_neg = self.rerank_cls_token.expand(B * NEG_COUNT, -1, -1)
                decoder_input_neg = torch.cat((cls_tokens_neg, thermal_expanded), dim=1)
                
                for blk in self.decoder_blocks:
                    decoder_input_neg = blk(decoder_input_neg, rgb_negs)
                decoder_output_neg = self.decoder_norm(decoder_input_neg)
                
                # 11. Rerank loss
                cls_neg = decoder_output_neg[:, 0, :]
                logits_pos = self.rerank_head(thermal_full_cls)  # [B, 1]
                logits_neg = self.rerank_head(cls_neg)  # [B*NEG_COUNT, 1]
                
                if None:
                    loss_pos = self.criterion(logits_pos, dist_pos.unsqueeze(1))
                    loss_neg = self.criterion(logits_neg, dist_neg_batch.unsqueeze(1))
    
                    rerank_loss = loss_pos + loss_neg / NEG_COUNT
                    # recon_loss = recon_loss + self.rerank_weight * rerank_loss  # weight 조절 필요
                    recon_loss = recon_loss + self.rerank_weight * rerank_loss  # weight 조절 필요
                else:
                    # 1. Sigmoid로 확신도(Confidence) 계산 (0.0 ~ 1.0)
                    # 모델이 "이 쌍은 같은 장소다"라고 생각하는 확률
                    confidence_pos = torch.sigmoid(logits_pos)

                    # 2. Rerank Loss 계산 (CLS 토큰의 정답 학습용)
                    # CLS가 GPS 거리(dist_pos) 기반으로 정답을 맞추도록 유도
                    loss_pos = self.criterion(logits_pos, dist_pos.unsqueeze(1))
                    loss_neg = self.criterion(logits_neg, dist_neg_batch.unsqueeze(1))
                    rerank_loss = loss_pos + (loss_neg / NEG_COUNT)

                    # 3. Weighted Reconstruction Loss (핵심 로직)
                    # "CLS가 확신할수록(Confidence가 높을수록) 복원 Loss를 중요하게 다룬다."
                    # CLS가 0.1(오답)을 내뱉으면 복원 Loss도 0.1배만 반영되어 무시됨
                    # CLS가 0.9(정답)를 내뱉으면 복원 Loss가 0.9배 반영되어 강하게 학습됨
                    weighted_recon_loss = recon_loss * confidence_pos.mean()

                    # 4. 최종 Loss 합산
                    # 기존 recon_loss 변수를 덮어씌워 리턴
                    recon_loss = weighted_recon_loss # + (self.rerank_weight * rerank_loss)

                # VPR용 patch tokens
                if return_masked_patch:
                    masked_patch = thermal_full
                
                out = self.thermal_backbone(x)
            else:
                # when inference
                # NOTE: 부르는 곳에 no_grad 호출하기
                if self.use_masked_inference:
                    # 1-4. masked thermal encoder
                    thermal_visible, mask, patch_B, patch_N, patch_D = self.croco_like_encoder(x)
                    
                    # 5. Mask token expansion
                    thermal_full = self.croco_encoded_mask_expension(thermal_visible, mask, patch_B, patch_N, patch_D)
                    
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
        
        return global_desc, patch_tokens, recon_loss, mask, masked_patch

    def forward(self, x, flags, aligned_rgb=None, return_mask=False, return_masked_patch=False,dist_pos=None, dist_neg=None):
        is_rgb = torch.tensor([f == 'rgb' for f in flags], device=x.device)
        final_emb = torch.zeros((x.size(0), self.output_dim), device=x.device)
        patch_emb = torch.zeros((x.size(0), 256, self.output_dim), device=x.device)
        masks = torch.zeros((x.size(0), 256), dtype=torch.bool, device=x.device)
        
        # thermal_count = len([_ for _ in flags if _ == 'thermal'])
        # masked_patch_emb = torch.zeros((thermal_count, 256, self.output_dim), device=x.device)
        masked_patch_emb = None
        
        recon_loss = None
        try:
            if is_rgb.any():
                global_emb, patch_rgb, _, _, _ = self.forward_model(x[is_rgb], modality='rgb')
                if global_emb is not None: final_emb[is_rgb] = global_emb
                patch_emb[is_rgb] = patch_rgb
            if (~is_rgb).any():
                global_emb, patch_thermal, recon_loss, mask, masked_patch_thermal = self.forward_model(x[~is_rgb], modality='thermal',
                            aligned_rgb=aligned_rgb, return_masked_patch=return_masked_patch,rgbs=x[is_rgb],
                            dist_pos=dist_pos, dist_neg=dist_neg)
                if global_emb is not None: final_emb[~is_rgb] = global_emb
                patch_emb[~is_rgb] = patch_thermal
                if return_mask: masks[~is_rgb] = mask
                if return_masked_patch:
                    masked_patch_emb = masked_patch_thermal # torch.Size([4, 256, 768])
        except Exception as e:
            import traceback
            print(f"ERROR caught: {e}")
            traceback.print_exc()  # 전체 stack trace 출력
            breakpoint()
        
        if return_masked_patch:
            return final_emb, patch_emb, recon_loss, masks, masked_patch_emb
        else:
            return final_emb, patch_emb, recon_loss, masks

def get_backbone(pretrained_foundation, foundation_model_path):
    backbone = vit_base(patch_size=14,img_size=518,init_values=1,block_chunks=0)
    if pretrained_foundation:
        assert foundation_model_path is not None, "Please specify foundation model path."
        model_dict = backbone.state_dict()
        state_dict = torch.load(foundation_model_path)
        model_dict.update(state_dict.items())
        backbone.load_state_dict(model_dict)
    return backbone