# network.py
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from backbone.vision_transformer import vit_small, vit_base
from pathlib import Path

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
        self.work_with_tokens = work_with_tokens

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps, work_with_tokens=self.work_with_tokens)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


def gem(x, p=3, eps=1e-6, work_with_tokens=False):
    if work_with_tokens:
        x = x.permute(0, 2, 1)
        return F.avg_pool1d(x.clamp(min=eps).pow(p), (x.size(-1))).pow(1./p).unsqueeze(3)
    else:
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        assert x.shape[2] == x.shape[3] == 1; return x[:,:,0,0]


class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim)

from timm.models.layers import trunc_normal_
class DistanceModule(nn.Module):
    def __init__(self, decoder_dim):
        super().__init__()
        self.decoder_dim = decoder_dim
        
        # 1. 거리 예측 전용 CLS 토큰 (학습 가능한 파라미터)
        self.decoder_dist_cls_token = nn.Parameter(torch.zeros(1, 1, self.decoder_dim))
        trunc_normal_(self.decoder_dist_cls_token, std=0.02)
        
        # 2. 거리를 예측하는 MLP Head
        self.distance_head = nn.Sequential(
            nn.Linear(self.decoder_dim, self.decoder_dim // 2),
            nn.GELU(),
            nn.Linear(self.decoder_dim // 2, 1)
        )
        
        # 가중치 초기화 (안정적인 학습을 위해 필수)
        nn.init.normal_(self.distance_head[0].weight, std=0.02)
        nn.init.zeros_(self.distance_head[0].bias)
        nn.init.normal_(self.distance_head[2].weight, std=0.02)
        nn.init.zeros_(self.distance_head[2].bias)
        
    @staticmethod
    def distance_to_score(distance, tau=20.0):
        return torch.exp(-distance / tau)
        
    def prepend_cls_token(self, patch_features):
        """
        [디코더 입력 전 사용] 
        배치 사이즈에 맞춰 CLS 토큰을 복제하고 패치들 맨 앞에 붙여줍니다.
        patch_features: [B, N, D]
        return: [B, N+1, D]
        """
        B = patch_features.shape[0]
        cls_tokens = self.decoder_dist_cls_token.expand(B, -1, -1)
        return torch.cat([cls_tokens, patch_features], dim=1)
        
    def forward(self, decoder_output, distance_gt=None, tau=20.0):
        """
        [디코더 출력 후 사용]
        decoder_output: 디코더를 통과한 결과 [B, N+1, D]
        distance_gt: 정답 거리(m) [B] (Training 시 제공, Inference 시 None)
        """
        # 1. 맨 앞의 CLS 토큰만 추출
        cls_output = decoder_output[:, 0, :] # [B, D]
        
        # 2. MLP 통과 및 Sigmoid로 0~1 사이 점수화
        logits = self.distance_head(cls_output).squeeze(-1) # [B]
        pred_scores = torch.sigmoid(logits)
        
        # 3. 정답이 주어지면 Loss까지 계산해서 반환
        if distance_gt is not None:
            target_scores = self.distance_to_score(distance_gt, tau=tau)
            loss = F.binary_cross_entropy(pred_scores, target_scores)
            return loss, pred_scores
            
        return None, pred_scores

class CrossModalVPR_Net(nn.Module):
    """
    Cross-modal VPR network with BiReconstruction CroCo decoder.

    Architecture:
    - Shared DINOv2 backbone for both RGB and Thermal
    - CroCo-style masked encoder for reconstruction
    - Bidirectional cross-attention decoder (thermal→RGB, RGB→thermal)
    - GeM aggregation for global descriptors
    """
    def __init__(self, args, pair_sampler, pretrained_foundation=False, foundation_model_path=None):
        super().__init__()

        self.args = args
        self.shared_backbone = get_backbone(pretrained_foundation, foundation_model_path, args=args)
        self.output_dim = args.features_dim
        self.recon_loss_type = args.recon_loss_type
        self.sampler = pair_sampler

        # Decoder settings
        dec_depth = args.num_decoder_depth
        dec_num_heads = 16
        self.patch_count = int(args.resize[0]/14) * int(args.resize[1]/14)

        # CroCo Decoder blocks
        self.decoder_blocks = nn.ModuleList([
            CroCoDecoderBlock(self.output_dim, dec_num_heads)
            for _ in range(dec_depth)
        ])
        self.decoder_norm = nn.LayerNorm(self.output_dim)

        # Mask and positional embeddings
        self._set_mask_token(self.output_dim)
        self._set_decode_positional_embedding(self.output_dim)
        self._set_mask_generator(self.patch_count, args.croco_mask_ratio, args.masking_method)
        self._set_prediction_head(self.output_dim)
        self.masking_method = args.masking_method

        # Reconstruction criterion
        self.reconstruction_criterion = MaskedMSE(
            args,
            norm_pix_loss=False,
            masked=True,
            loss_type=self.recon_loss_type,
        )

        # Aggregation layers
        self.aggregation = nn.Sequential(
            L2Norm(),
            GeM(work_with_tokens=None),
            Flatten(),
        )
        
        # Distance Module
        self.distance_module = DistanceModule(self.output_dim)
        self.distance_tau = getattr(args, 'distance_tau', 20.0)

    def _set_mask_token(self, dec_embed_dim):
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_embed_dim))
        nn.init.normal_(self.mask_token, std=.02)

    def _set_decode_positional_embedding(self, dec_embed_dim):
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.patch_count, dec_embed_dim))
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)

    def _set_mask_generator(self, num_patches, mask_ratio, masking_method='random'):
        self.mask_generator = RandomMask(num_patches, mask_ratio, masking_method=masking_method)

    def _set_prediction_head(self, dec_embed_dim):
        hidden_dim = dec_embed_dim * 4
        output_dim = 14 * 14 * 3

        self.prediction_thermal_head = nn.Sequential(
            nn.Linear(dec_embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.prediction_rgb_head = nn.Sequential(
            nn.Linear(dec_embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
        # self.prediction_thermal_head = nn.Linear(dec_embed_dim, output_dim)
        # self.prediction_rgb_head = nn.Linear(dec_embed_dim, output_dim)

        # Initialize weights
        for head in [self.prediction_thermal_head, self.prediction_rgb_head]:
            nn.init.normal_(head[0].weight, std=0.02)
            nn.init.zeros_(head[0].bias)
            nn.init.normal_(head[2].weight, std=0.02)
            nn.init.zeros_(head[2].bias)

    def patchify(self, imgs):
        """
        imgs: (B, 3, H, W)
        x: (B, L, patch_size**2 *3)
        """
        p = 14
        assert imgs.shape[2] % p == 0 and imgs.shape[3] % p == 0

        h = imgs.shape[2] // p
        w = imgs.shape[3] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def croco_like_encoder(self, x, modality='thermal', cls_score=None):
        """Masked encoder following CroCo style."""
        current_backbone = self.shared_backbone

        image_patch = current_backbone.patch_embed(x)
        patch_B, patch_N, patch_D = image_patch.shape

        cls_token = current_backbone.cls_token.expand(patch_B, -1, -1)

        # Positional embedding
        # - DINO의 positional embedding을 224x224(16x16)에 맞게 interpolate해서 넣기
        pos_tokens = current_backbone.pos_embed[:, 1:, :]
        pos_embed_grid = pos_tokens.reshape(1, 37, 37, patch_D).permute(0, 3, 1, 2)
        pos_embed_resized = F.interpolate(pos_embed_grid, size=(int(x.shape[2]/14), int(x.shape[3]/14)), mode='bicubic', align_corners=False)
        pos_embed_final = pos_embed_resized.permute(0, 2, 3, 1).flatten(1, 2)
        image_patch = image_patch + pos_embed_final

        # CLS positional embedding
        cls_pos_embed = current_backbone.pos_embed[:, :1, :]
        cls_token = cls_token + cls_pos_embed

        image_with_cls = torch.cat([cls_token, image_patch], dim=1)

        # Register tokens (DINOv2 with registers)
        # - register backbone 호환용
        num_register_tokens = current_backbone.num_register_tokens
        if current_backbone.register_tokens is not None:
            register_tokens = current_backbone.register_tokens.expand(patch_B, -1, -1)
            image_with_cls = torch.cat([
                image_with_cls[:, :1],
                register_tokens,
                image_with_cls[:, 1:]
            ], dim=1)

        # Masking
        mask = self.mask_generator(image_patch, cls_score=cls_score)
        cls_reg_mask = torch.zeros(patch_B, 1 + num_register_tokens, dtype=torch.bool, device=mask.device)
        full_mask = torch.cat([cls_reg_mask, mask], dim=1)

        # Masking되고 살아남은 patch token들
        patch_visible = image_with_cls[~full_mask].reshape(patch_B, -1, patch_D)

        # Encoder forward
        for blk in current_backbone.blocks:
            patch_visible = blk(patch_visible)
        patch_visible = current_backbone.norm(patch_visible)

        # Separate CLS token
        # - CLS token 분리
        cls_visible = patch_visible[:, 0:1, :]
        patch_only_visible = patch_visible[:, 1 + num_register_tokens:, :]

        '''
        return
        - [0] masking되고 남은 token [B, N*(1-mask_ratio), C]
        - [1] mask: boolean mask [B, N]
        - [2,3,4] patch BNC
        - [5] 분리된 CLS token
        '''
        return patch_only_visible, mask, patch_B, patch_N, patch_D, cls_visible

    def croco_encoded_mask_expension(self, visible_tokens, mask, patch_B, patch_N, patch_D):
        """Fill masked positions with mask tokens."""
        full_tokens = self.mask_token.expand(patch_B, patch_N, -1).clone()
        full_tokens[~mask] = visible_tokens.flatten(0, 1)
        full_tokens = full_tokens.view(patch_B, patch_N, patch_D)
        
        '''
        return
        - [0]: visible(masking안된) token들에 mask_token을 붙여서 [B, N]으로 만든거
        '''
        return full_tokens

    def forward_model(self, x, paired_rgb=None, recon_pairs=None, distances=None, modality='rgb'):
        """Forward pass for a single modality.

        Args:
            x: input images (thermal for thermal modality)
            paired_rgb: aligned RGB images (기존 방식)
            recon_pairs: dict with reconstruction pair images:
                - 'pos_thermal': [B, 3, H, W] - 유사한 thermal (intra-modal)
                - 'similar_rgb': [B, 3, H, W] - 유사한 RGB (inter-modal)
                - 'rgb_similar_rgb': [B, 3, H, W] - RGB와 유사한 RGB (intra-modal)
            distances: [B, 1+negs] distance GT for distance prediction
            modality: 'rgb' or 'thermal'
        """
        recon_losses_dict = None
        mask_thermal = None

        if modality == 'rgb':
            out = self.shared_backbone(x)
        elif modality == 'thermal':
            if self.training:
                # recon_pairs가 제공되면 4가지 reconstruction pairs 사용
                if recon_pairs is not None and self._has_valid_recon_pairs(recon_pairs):
                    recon_losses_dict, mask_thermal, out = self._forward_with_recon_pairs(
                        x, recon_pairs, paired_rgb, distances=distances
                    )
                else:
                    # reconstruction 없이 forward
                    out = self.shared_backbone(x)
            else:
                # inference 때
                out = self.shared_backbone(x)

        # Process backbone output
        patch_tokens = out["x_norm_patchtokens"]
        B, N, D = patch_tokens.shape
        H_feat = int(x.shape[2]/14)
        W_feat = int(x.shape[3]/14)
        x_feat = patch_tokens.permute(0, 2, 1).view(B, D, H_feat, W_feat)

        # Aggregation -> Descriptor
        # - feature embedding -> GeM -> Descriptor
        global_desc = self.aggregation(x_feat)

        return global_desc, patch_tokens, recon_losses_dict, mask_thermal

    def _has_valid_recon_pairs(self, recon_pairs):
        """recon_pairs에 유효한 데이터가 있는지 확인"""
        if recon_pairs is None:
            return False
        # 최소 하나 이상의 유효한 pair가 있어야 함
        return any(recon_pairs.get(key) is not None for key in ['pos_thermal', 'similar_rgb', 'rgb_similar_rgb'])

    def _get_cls_attn_map(self, backbone_out):
        """Extract CLS attention score from backbone output (MHA average)"""
        if "cls_attention" in backbone_out:
            # cls_attention: [B, num_heads, N] → [B, N]
            return backbone_out["cls_attention"].mean(dim=1)
        return None

    def _forward_with_recon_pairs(self, query_thermal, recon_pairs, paired_rgb=None, distances=None):
        """
        4가지 Reconstruction Pairs를 사용한 forward + Distance Prediction

        Pairs:
        1. Query Thermal (masked) ← Pos Thermal (ref) : intra-modal thermal (uni)
        2. Similar RGB (masked) ← RGB-similar RGB (ref) : intra-modal RGB (uni)
        3. Query Thermal (masked) ← Similar RGB (ref) : inter-modal (bi)
        4. Similar RGB (masked) ← Query Thermal (ref) : inter-modal (bi)

        Distance Prediction:
        - Inter-modal decoder에 distance CLS token 추가
        - Bidirectional distance prediction (T→R, R→T)
        """
        pos_thermal = recon_pairs.get('pos_thermal')
        similar_rgb = recon_pairs.get('similar_rgb')
        rgb_similar_rgb = recon_pairs.get('rgb_similar_rgb')

        use_cls_masking = (self.masking_method in ['CLSDown', 'CLSTop'])

        # Query thermal의 full feature (global descriptor용 및 inter-modal reference용)
        query_thermal_out = self.shared_backbone(query_thermal, return_attention=use_cls_masking)
        query_thermal_full = query_thermal_out["x_norm_patchtokens"]
        query_cls_score = self._get_cls_attn_map(query_thermal_out) if use_cls_masking else None

        # Loss dictionary 초기화
        recon_losses_dict = {
            'intra_thermal': None,  # Pair 1: Query Thermal ← Pos Thermal
            'intra_rgb': None,      # Pair 2: Similar RGB ← RGB-similar RGB
            'inter_t2r': None,      # Pair 3: Query Thermal ← Similar RGB
            'inter_r2t': None,      # Pair 4: Similar RGB ← Query Thermal
            'distance_intra_thermal': None,  # Intra-Thermal distance loss
            'distance_intra_rgb': None,      # Intra-RGB distance loss
            'distance_inter': None,          # Inter-modal distance loss
        }

        mask_thermal = None
        use_cls_attn_weight = self.args.recon_attn_scale
        # ===== Pair 1: Query Thermal ← Pos Thermal (intra-modal thermal, uni) =====
        if pos_thermal is not None:
            pos_thermal_out = self.shared_backbone(pos_thermal)
            pos_thermal_full = pos_thermal_out["x_norm_patchtokens"]

            query_visible, mask_query, B_q, N_q, D_q, _ = self.croco_like_encoder(
                query_thermal, modality='thermal', cls_score=query_cls_score
            )
            query_full = self.croco_encoded_mask_expension(query_visible, mask_query, B_q, N_q, D_q)

            query_dec = query_full + self.decoder_pos_embed
            pos_ref = pos_thermal_full + self.decoder_pos_embed

            # Prepend distance CLS token
            query_dec_with_cls = self.distance_module.prepend_cls_token(query_dec)

            for blk in self.decoder_blocks:
                query_dec_with_cls = blk(query_dec_with_cls, pos_ref)
            query_dec_with_cls = self.decoder_norm(query_dec_with_cls)

            # Remove CLS token for reconstruction
            query_dec = query_dec_with_cls[:, 1:, :]
            query_pred = self.prediction_thermal_head(query_dec)
            query_target = self.patchify(query_thermal)

            cls_attn_map = query_cls_score if use_cls_attn_weight else None
            recon_losses_dict['intra_thermal'] = self.calculate_recon_loss(query_pred, mask_query, query_target, cls_attn_map=cls_attn_map)
            mask_thermal = mask_query

            # Distance prediction for Intra-Thermal
            if distances is not None and 'intra_thermal' in distances:
                dist_gt = distances['intra_thermal'].to(query_thermal.device)
                valid_mask = dist_gt >= 0
                if valid_mask.any():
                    loss_dist, _ = self.distance_module(
                        query_dec_with_cls[valid_mask], dist_gt[valid_mask], tau=self.distance_tau
                    )
                    recon_losses_dict['distance_intra_thermal'] = loss_dist

        # ===== Pair 2: Similar RGB ← RGB-similar RGB (intra-modal RGB, uni) =====
        if similar_rgb is not None and rgb_similar_rgb is not None:
            rgb_sim_out = self.shared_backbone(rgb_similar_rgb)
            rgb_sim_full = rgb_sim_out["x_norm_patchtokens"]

            # Similar RGB의 cls_score 필요
            sim_rgb_out_attn = self.shared_backbone(similar_rgb, return_attention=use_cls_masking)
            sim_rgb_cls_score = self._get_cls_attn_map(sim_rgb_out_attn) if use_cls_masking else None

            sim_rgb_visible, mask_sim_rgb, B_sr, N_sr, D_sr, _ = self.croco_like_encoder(
                similar_rgb, modality='rgb', cls_score=sim_rgb_cls_score
            )
            sim_rgb_full = self.croco_encoded_mask_expension(sim_rgb_visible, mask_sim_rgb, B_sr, N_sr, D_sr)

            sim_rgb_dec = sim_rgb_full + self.decoder_pos_embed
            rgb_sim_ref = rgb_sim_full + self.decoder_pos_embed

            # Prepend distance CLS token
            sim_rgb_dec_with_cls = self.distance_module.prepend_cls_token(sim_rgb_dec)

            for blk in self.decoder_blocks:
                sim_rgb_dec_with_cls = blk(sim_rgb_dec_with_cls, rgb_sim_ref)
            sim_rgb_dec_with_cls = self.decoder_norm(sim_rgb_dec_with_cls)

            # Remove CLS token for reconstruction
            sim_rgb_dec = sim_rgb_dec_with_cls[:, 1:, :]
            sim_rgb_pred = self.prediction_rgb_head(sim_rgb_dec)
            sim_rgb_target = self.patchify(similar_rgb)

            cls_attn_map = sim_rgb_cls_score if use_cls_attn_weight else None
            recon_losses_dict['intra_rgb'] = self.calculate_recon_loss(sim_rgb_pred, mask_sim_rgb, sim_rgb_target, cls_attn_map=cls_attn_map)

            # Distance prediction for Intra-RGB
            if distances is not None and 'intra_rgb' in distances:
                dist_gt = distances['intra_rgb'].to(similar_rgb.device)
                valid_mask = dist_gt >= 0
                if valid_mask.any():
                    loss_dist, _ = self.distance_module(
                        sim_rgb_dec_with_cls[valid_mask], dist_gt[valid_mask], tau=self.distance_tau
                    )
                    recon_losses_dict['distance_intra_rgb'] = loss_dist

        # ===== Pair 3 & 4: Inter-modal bi-directional + Distance Prediction =====
        if similar_rgb is not None:
            sim_rgb_out = self.shared_backbone(similar_rgb, return_attention=use_cls_masking)
            sim_rgb_full = sim_rgb_out["x_norm_patchtokens"]
            sim_rgb_cls_score = self._get_cls_attn_map(sim_rgb_out) if use_cls_masking else None

            # Masked encoding for both directions
            query_visible_inter, mask_query_inter, B_qi, N_qi, D_qi, _ = self.croco_like_encoder(
                query_thermal, modality='thermal', cls_score=query_cls_score
            )
            rgb_visible_inter, mask_rgb_inter, B_ri, N_ri, D_ri, _ = self.croco_like_encoder(
                similar_rgb, modality='rgb', cls_score=sim_rgb_cls_score
            )

            query_inter_full = self.croco_encoded_mask_expension(query_visible_inter, mask_query_inter, B_qi, N_qi, D_qi)
            rgb_inter_full = self.croco_encoded_mask_expension(rgb_visible_inter, mask_rgb_inter, B_ri, N_ri, D_ri)

            # Decoder input + positional embedding
            query_inter_dec = query_inter_full + self.decoder_pos_embed
            rgb_inter_dec = rgb_inter_full + self.decoder_pos_embed
            query_ref = query_thermal_full + self.decoder_pos_embed
            sim_rgb_ref = sim_rgb_full + self.decoder_pos_embed

            # Prepend distance CLS token for distance prediction
            query_inter_dec_with_cls = self.distance_module.prepend_cls_token(query_inter_dec)
            rgb_inter_dec_with_cls = self.distance_module.prepend_cls_token(rgb_inter_dec)

            # Batch: [thermal←rgb, rgb←thermal]
            target_batch = torch.cat([query_inter_dec_with_cls, rgb_inter_dec_with_cls], dim=0)
            ref_batch = torch.cat([sim_rgb_ref, query_ref], dim=0)

            for blk in self.decoder_blocks:
                target_batch = blk(target_batch, ref_batch)
            target_batch = self.decoder_norm(target_batch)

            # Split batch back and remove CLS token for reconstruction
            thermal_decoded = target_batch[:B_qi, :, :]
            rgb_decoded = target_batch[B_qi:, :, :]

            thermal_recon = thermal_decoded[:, 1:, :]  # Remove CLS token
            rgb_recon = rgb_decoded[:, 1:, :]  # Remove CLS token

            # Pair 3: Query Thermal ← Similar RGB
            thermal_pred = self.prediction_thermal_head(thermal_recon)
            thermal_target = self.patchify(query_thermal)
            cls_attn_map = query_cls_score if use_cls_attn_weight else None
            recon_losses_dict['inter_t2r'] = self.calculate_recon_loss(thermal_pred, mask_query_inter, thermal_target, cls_attn_map=cls_attn_map)

            # Pair 4: Similar RGB ← Query Thermal
            rgb_pred = self.prediction_rgb_head(rgb_recon)
            rgb_target = self.patchify(similar_rgb)
            cls_attn_map = sim_rgb_cls_score if use_cls_attn_weight else None
            recon_losses_dict['inter_r2t'] = self.calculate_recon_loss(rgb_pred, mask_rgb_inter, rgb_target, cls_attn_map=cls_attn_map)

            # ===== Distance Prediction (Bidirectional) =====
            if distances is not None and 'inter' in distances:
                # distances['inter']: [B] - similar_rgb와 query_thermal 간 거리 (-1은 invalid)
                distance_gt = distances['inter'].to(query_thermal.device)

                # valid distance만 사용 (distance >= 0)
                valid_mask = distance_gt >= 0
                if valid_mask.any():
                    valid_thermal = thermal_decoded[valid_mask]
                    valid_rgb = rgb_decoded[valid_mask]
                    valid_distance = distance_gt[valid_mask]

                    # T→R direction: thermal_decoded의 CLS token 사용
                    loss_t2r, _ = self.distance_module(valid_thermal, valid_distance, tau=self.distance_tau)

                    # R→T direction: rgb_decoded의 CLS token 사용
                    loss_r2t, _ = self.distance_module(valid_rgb, valid_distance, tau=self.distance_tau)

                    # Bidirectional distance loss 평균
                    recon_losses_dict['distance_inter'] = (loss_t2r + loss_r2t) / 2

            if mask_thermal is None:
                mask_thermal = mask_query_inter

        return recon_losses_dict, mask_thermal, query_thermal_out

    def forward(self, x, flags, paired_rgb=None, recon_pairs=None, distances=None, return_mask=False):
        if not isinstance(flags, torch.Tensor):
            flags = torch.tensor(flags, device=x.device)
        if flags.device != x.device:
            flags = flags.to(x.device)

        is_rgb = (flags == 1)
        final_emb = torch.zeros((x.size(0), self.output_dim), device=x.device) # GeM된 descriptor 모음
        patch_emb = torch.zeros((x.size(0), self.patch_count, self.output_dim), device=x.device) # feature embedding들 모음
        masks = torch.zeros((x.size(0), self.patch_count), dtype=torch.bool, device=x.device) # masking 모음
        recon_losses = None

        if is_rgb.any():
            global_emb, patch_rgb, _, _ = self.forward_model(x[is_rgb], modality='rgb')
            if global_emb is not None:
                final_emb[is_rgb] = global_emb
            if patch_rgb is not None:
                patch_emb[is_rgb] = patch_rgb

        if (~is_rgb).any():
            global_emb, patch_thermal, recon_losses, mask = self.forward_model(
                x[~is_rgb], modality='thermal', paired_rgb=paired_rgb, recon_pairs=recon_pairs, distances=distances)
            if global_emb is not None:
                final_emb[~is_rgb] = global_emb
            if patch_thermal is not None:
                patch_emb[~is_rgb] = patch_thermal
            if return_mask and mask is not None:
                masks[~is_rgb] = mask

        return final_emb, patch_emb, recon_losses, masks

    def calculate_recon_loss(self, pred, mask, target, cls_attn_map=None):
        exclude_ratio = getattr(self.args, 'recon_exclude_bottom_ratio', 0.0)
        return self.reconstruction_criterion(pred=pred, mask=mask, target=target, cls_attn_map=cls_attn_map, exclude_ratio=exclude_ratio)

    @torch.no_grad()
    def visualize_reconstruction(self, query_thermal, recon_pairs, batch_idx=0):
        """
        Reconstruction 결과를 시각화용으로 반환

        Returns:
            dict: 각 reconstruction pair에 대한 시각화 데이터
                - 'intra_thermal': (input, masked, recon, ref)
                - 'intra_rgb': (input, masked, recon, ref)
                - 'inter_t2r': (input, masked, recon, ref)
                - 'inter_r2t': (input, masked, recon, ref)
        """
        vis_data = {}
        pos_thermal = recon_pairs.get('pos_thermal')
        similar_rgb = recon_pairs.get('similar_rgb')
        rgb_similar_rgb = recon_pairs.get('rgb_similar_rgb')

        use_cls_masking = (self.masking_method in ['CLSDown', 'CLSTop'])

        # 배치에서 하나만 추출
        query_t = query_thermal[batch_idx:batch_idx+1]
        H, W = query_t.shape[2], query_t.shape[3]
        h_feat, w_feat = H // 14, W // 14

        # Query thermal의 full feature + cls_score
        query_thermal_out = self.shared_backbone(query_t, return_attention=use_cls_masking)
        query_thermal_full = query_thermal_out["x_norm_patchtokens"]
        query_cls_score = self._get_cls_attn_map(query_thermal_out) if use_cls_masking else None

        # Pair 1: Intra Thermal
        if pos_thermal is not None:
            pos_t = pos_thermal[batch_idx:batch_idx+1]
            pos_thermal_full = self.shared_backbone(pos_t)["x_norm_patchtokens"]

            query_visible, mask, B, N, D, _ = self.croco_like_encoder(
                query_t, modality='thermal', cls_score=query_cls_score
            )
            query_full = self.croco_encoded_mask_expension(query_visible, mask, B, N, D)

            query_dec = query_full + self.decoder_pos_embed
            pos_ref = pos_thermal_full + self.decoder_pos_embed

            for blk in self.decoder_blocks:
                query_dec = blk(query_dec, pos_ref)
            query_dec = self.decoder_norm(query_dec)

            pred = self.prediction_thermal_head(query_dec)
            recon_img = self.unpatchify(pred, h_feat, w_feat)
            masked_img = self._apply_mask_to_image(query_t, mask, h_feat, w_feat)

            vis_data['intra_thermal'] = {
                'input': query_t[0].cpu(),
                'masked': masked_img[0].cpu(),
                'recon': recon_img[0].cpu(),
                'ref': pos_t[0].cpu(),
            }

        # Pair 2: Intra RGB
        if similar_rgb is not None and rgb_similar_rgb is not None:
            sim_rgb = similar_rgb[batch_idx:batch_idx+1]
            rgb_sim = rgb_similar_rgb[batch_idx:batch_idx+1]
            rgb_sim_full = self.shared_backbone(rgb_sim)["x_norm_patchtokens"]

            # Similar RGB의 cls_score
            sim_rgb_out = self.shared_backbone(sim_rgb, return_attention=use_cls_masking)
            sim_rgb_cls_score = self._get_cls_attn_map(sim_rgb_out) if use_cls_masking else None

            sim_visible, mask, B, N, D, _ = self.croco_like_encoder(
                sim_rgb, modality='rgb', cls_score=sim_rgb_cls_score
            )
            sim_full = self.croco_encoded_mask_expension(sim_visible, mask, B, N, D)

            sim_dec = sim_full + self.decoder_pos_embed
            rgb_ref = rgb_sim_full + self.decoder_pos_embed

            for blk in self.decoder_blocks:
                sim_dec = blk(sim_dec, rgb_ref)
            sim_dec = self.decoder_norm(sim_dec)

            pred = self.prediction_rgb_head(sim_dec)
            recon_img = self.unpatchify(pred, h_feat, w_feat)
            masked_img = self._apply_mask_to_image(sim_rgb, mask, h_feat, w_feat)

            vis_data['intra_rgb'] = {
                'input': sim_rgb[0].cpu(),
                'masked': masked_img[0].cpu(),
                'recon': recon_img[0].cpu(),
                'ref': rgb_sim[0].cpu(),
            }

        # Pair 3 & 4: Inter-modal
        if similar_rgb is not None:
            sim_rgb = similar_rgb[batch_idx:batch_idx+1]
            sim_rgb_out = self.shared_backbone(sim_rgb, return_attention=use_cls_masking)
            sim_rgb_full = sim_rgb_out["x_norm_patchtokens"]
            sim_rgb_cls_score = self._get_cls_attn_map(sim_rgb_out) if use_cls_masking else None

            query_visible, mask_t, B_t, N_t, D_t, _ = self.croco_like_encoder(
                query_t, modality='thermal', cls_score=query_cls_score
            )
            rgb_visible, mask_r, B_r, N_r, D_r, _ = self.croco_like_encoder(
                sim_rgb, modality='rgb', cls_score=sim_rgb_cls_score
            )

            query_full = self.croco_encoded_mask_expension(query_visible, mask_t, B_t, N_t, D_t)
            rgb_full = self.croco_encoded_mask_expension(rgb_visible, mask_r, B_r, N_r, D_r)

            query_dec = query_full + self.decoder_pos_embed
            rgb_dec = rgb_full + self.decoder_pos_embed
            query_ref = query_thermal_full + self.decoder_pos_embed
            sim_rgb_ref = sim_rgb_full + self.decoder_pos_embed

            target_batch = torch.cat([query_dec, rgb_dec], dim=0)
            ref_batch = torch.cat([sim_rgb_ref, query_ref], dim=0)

            for blk in self.decoder_blocks:
                target_batch = blk(target_batch, ref_batch)
            target_batch = self.decoder_norm(target_batch)

            thermal_recon = target_batch[:B_t]
            rgb_recon = target_batch[B_t:]

            # Inter T2R
            pred_t = self.prediction_thermal_head(thermal_recon)
            recon_t = self.unpatchify(pred_t, h_feat, w_feat)
            masked_t = self._apply_mask_to_image(query_t, mask_t, h_feat, w_feat)

            vis_data['inter_t2r'] = {
                'input': query_t[0].cpu(),
                'masked': masked_t[0].cpu(),
                'recon': recon_t[0].cpu(),
                'ref': sim_rgb[0].cpu(),
            }

            # Inter R2T
            pred_r = self.prediction_rgb_head(rgb_recon)
            recon_r = self.unpatchify(pred_r, h_feat, w_feat)
            masked_r = self._apply_mask_to_image(sim_rgb, mask_r, h_feat, w_feat)

            vis_data['inter_r2t'] = {
                'input': sim_rgb[0].cpu(),
                'masked': masked_r[0].cpu(),
                'recon': recon_r[0].cpu(),
                'ref': query_t[0].cpu(),
            }

        return vis_data

    def unpatchify(self, x, h_feat, w_feat):
        """Patchified tensor를 이미지로 복원"""
        patch_size = 14
        B, N, C = x.shape
        x = x.reshape(B, h_feat, w_feat, patch_size, patch_size, 3)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.reshape(B, 3, h_feat * patch_size, w_feat * patch_size)
        return x

    def _apply_mask_to_image(self, img, mask, h_feat, w_feat):
        """Mask를 이미지에 적용 (masked 영역을 회색으로)"""
        B, C, H, W = img.shape
        patch_size = 14

        # mask: [B, N] -> [B, 1, h_feat, w_feat]
        mask_2d = mask.view(B, h_feat, w_feat).unsqueeze(1).float()
        # upscale to image size
        mask_img = F.interpolate(mask_2d, size=(H, W), mode='nearest')
        # masked area to gray
        masked_img = img.clone()
        masked_img = masked_img * (1 - mask_img) + 0.5 * mask_img
        return masked_img

def get_backbone(pretrained_foundation, foundation_model_path, args=None):
    model_path = Path(foundation_model_path)
    model_name = model_path.parts[-1].lower()

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
