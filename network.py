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
        self._set_mask_generator(self.patch_count, args.croco_mask_ratio)
        self._set_prediction_head(self.output_dim)

        # Reconstruction criterion
        self.reconstruction_criterion = MaskedMSE(
            args,
            norm_pix_loss=False,
            masked=True,
            loss_type=self.recon_loss_type
        )

        # Aggregation layers
        self.aggregation = nn.Sequential(
            L2Norm(),
            GeM(work_with_tokens=None),
            Flatten(),
        )

    def _set_mask_token(self, dec_embed_dim):
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_embed_dim))
        nn.init.normal_(self.mask_token, std=.02)

    def _set_decode_positional_embedding(self, dec_embed_dim):
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.patch_count, dec_embed_dim))
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)

    def _set_mask_generator(self, num_patches, mask_ratio):
        self.mask_generator = RandomMask(num_patches, mask_ratio)

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

    def forward_model(self, x, paired_rgb=None, recon_pairs=None, modality='rgb'):
        """Forward pass for a single modality.

        Args:
            x: input images (thermal for thermal modality)
            paired_rgb: aligned RGB images (기존 방식)
            recon_pairs: dict with reconstruction pair images:
                - 'pos_thermal': [B, 3, H, W] - 유사한 thermal (intra-modal)
                - 'similar_rgb': [B, 3, H, W] - 유사한 RGB (inter-modal)
                - 'rgb_similar_rgb': [B, 3, H, W] - RGB와 유사한 RGB (intra-modal)
            modality: 'rgb' or 'thermal'
        """
        recon_loss_thermal = None
        recon_loss_rgb = None
        mask_thermal = None

        if modality == 'rgb':
            out = self.shared_backbone(x)
        elif modality == 'thermal':
            if self.training:
                # recon_pairs가 제공되면 4가지 reconstruction pairs 사용
                if recon_pairs is not None and self._has_valid_recon_pairs(recon_pairs):
                    recon_loss_thermal, recon_loss_rgb, mask_thermal, out = self._forward_with_recon_pairs(
                        x, recon_pairs, paired_rgb
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

        return global_desc, patch_tokens, [recon_loss_thermal, recon_loss_rgb], mask_thermal

    def _has_valid_recon_pairs(self, recon_pairs):
        """recon_pairs에 유효한 데이터가 있는지 확인"""
        if recon_pairs is None:
            return False
        # 최소 하나 이상의 유효한 pair가 있어야 함
        return any(recon_pairs.get(key) is not None for key in ['pos_thermal', 'similar_rgb', 'rgb_similar_rgb'])

    def _forward_with_recon_pairs(self, query_thermal, recon_pairs, paired_rgb=None):
        """
        4가지 Reconstruction Pairs를 사용한 forward

        Pairs:
        1. Query Thermal (masked) ← Pos Thermal (ref) : intra-modal thermal
        2. Pos Thermal (masked) ← Query Thermal (ref) : intra-modal thermal (vice versa)
        3. Similar RGB (masked) ← RGB-similar RGB (ref) : intra-modal RGB
        4. Query Thermal (masked) ← Similar RGB (ref) : inter-modal
        """
        pos_thermal = recon_pairs.get('pos_thermal')
        similar_rgb = recon_pairs.get('similar_rgb')
        rgb_similar_rgb = recon_pairs.get('rgb_similar_rgb')

        # Query thermal의 full feature (global descriptor용)
        query_thermal_out = self.shared_backbone(query_thermal)
        query_thermal_full = query_thermal_out["x_norm_patchtokens"]

        all_recon_losses = []

        # ===== Pair 1 & 2: Thermal ↔ Thermal (intra-modal) =====
        if pos_thermal is not None:
            # Full features for reference
            pos_thermal_out = self.shared_backbone(pos_thermal)
            pos_thermal_full = pos_thermal_out["x_norm_patchtokens"]

            # Masked encoding
            query_visible, mask_query, B_q, N_q, D_q, _ = self.croco_like_encoder(query_thermal, modality='thermal')
            pos_visible, mask_pos, B_p, N_p, D_p, _ = self.croco_like_encoder(pos_thermal, modality='thermal')

            # Mask token expansion
            query_full = self.croco_encoded_mask_expension(query_visible, mask_query, B_q, N_q, D_q)
            pos_full = self.croco_encoded_mask_expension(pos_visible, mask_pos, B_p, N_p, D_p)

            # Decoder forward
            query_dec = query_full + self.decoder_pos_embed
            pos_dec = pos_full + self.decoder_pos_embed
            query_ref = query_thermal_full + self.decoder_pos_embed
            pos_ref = pos_thermal_full + self.decoder_pos_embed

            # Batch: [query←pos, pos←query]
            target_batch = torch.cat([query_dec, pos_dec], dim=0)
            ref_batch = torch.cat([pos_ref, query_ref], dim=0)

            for blk in self.decoder_blocks:
                target_batch = blk(target_batch, ref_batch)
            target_batch = self.decoder_norm(target_batch)

            query_recon = target_batch[:B_q, :, :]
            pos_recon = target_batch[B_q:, :, :]

            # Prediction & Loss
            query_pred = self.prediction_thermal_head(query_recon)
            pos_pred = self.prediction_thermal_head(pos_recon)

            query_target = self.patchify(query_thermal)
            pos_target = self.patchify(pos_thermal)

            loss_1 = self.calculate_recon_loss(query_pred, mask_query, query_target)
            loss_2 = self.calculate_recon_loss(pos_pred, mask_pos, pos_target)
            all_recon_losses.extend([loss_1, loss_2])

        # ===== Pair 3: Similar RGB ← RGB-similar RGB (intra-modal RGB) =====
        if similar_rgb is not None and rgb_similar_rgb is not None:
            rgb_sim_out = self.shared_backbone(rgb_similar_rgb)
            rgb_sim_full = rgb_sim_out["x_norm_patchtokens"]

            sim_rgb_visible, mask_sim_rgb, B_sr, N_sr, D_sr, _ = self.croco_like_encoder(similar_rgb, modality='rgb')
            sim_rgb_full = self.croco_encoded_mask_expension(sim_rgb_visible, mask_sim_rgb, B_sr, N_sr, D_sr)

            sim_rgb_dec = sim_rgb_full + self.decoder_pos_embed
            rgb_sim_ref = rgb_sim_full + self.decoder_pos_embed

            for blk in self.decoder_blocks:
                sim_rgb_dec = blk(sim_rgb_dec, rgb_sim_ref)
            sim_rgb_dec = self.decoder_norm(sim_rgb_dec)

            sim_rgb_pred = self.prediction_rgb_head(sim_rgb_dec)
            sim_rgb_target = self.patchify(similar_rgb)

            loss_3 = self.calculate_recon_loss(sim_rgb_pred, mask_sim_rgb, sim_rgb_target)
            all_recon_losses.append(loss_3)

        # ===== Pair 4: Query Thermal ← Similar RGB (inter-modal) =====
        if similar_rgb is not None:
            sim_rgb_out = self.shared_backbone(similar_rgb)
            sim_rgb_full = sim_rgb_out["x_norm_patchtokens"]

            query_visible_inter, mask_query_inter, B_qi, N_qi, D_qi, _ = self.croco_like_encoder(query_thermal, modality='thermal')
            query_inter_full = self.croco_encoded_mask_expension(query_visible_inter, mask_query_inter, B_qi, N_qi, D_qi)

            query_inter_dec = query_inter_full + self.decoder_pos_embed
            sim_rgb_ref = sim_rgb_full + self.decoder_pos_embed

            for blk in self.decoder_blocks:
                query_inter_dec = blk(query_inter_dec, sim_rgb_ref)
            query_inter_dec = self.decoder_norm(query_inter_dec)

            query_inter_pred = self.prediction_thermal_head(query_inter_dec)
            query_inter_target = self.patchify(query_thermal)

            loss_4 = self.calculate_recon_loss(query_inter_pred, mask_query_inter, query_inter_target)
            all_recon_losses.append(loss_4)

        # Loss 합산
        if len(all_recon_losses) > 0:
            # thermal loss (pairs 1, 2, 4)와 rgb loss (pair 3) 분리
            thermal_losses = []
            rgb_losses = []

            if pos_thermal is not None:
                thermal_losses.extend([all_recon_losses[0], all_recon_losses[1]])

            if similar_rgb is not None and rgb_similar_rgb is not None:
                idx = 2 if pos_thermal is not None else 0
                rgb_losses.append(all_recon_losses[idx])

            if similar_rgb is not None:
                idx = 3 if (pos_thermal is not None and rgb_similar_rgb is not None) else \
                      2 if pos_thermal is not None else \
                      1 if rgb_similar_rgb is not None else 0
                thermal_losses.append(all_recon_losses[idx])

            recon_loss_thermal = torch.stack(thermal_losses).mean() if thermal_losses else None
            recon_loss_rgb = torch.stack(rgb_losses).mean() if rgb_losses else None
        else:
            recon_loss_thermal = None
            recon_loss_rgb = None

        # mask는 첫 번째 query thermal의 mask 사용
        mask_thermal = mask_query if pos_thermal is not None else \
                       mask_query_inter if similar_rgb is not None else None

        return recon_loss_thermal, recon_loss_rgb, mask_thermal, query_thermal_out

    def forward(self, x, flags, paired_rgb=None, recon_pairs=None, return_mask=False):
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
                x[~is_rgb], modality='thermal', paired_rgb=paired_rgb, recon_pairs=recon_pairs)
            if global_emb is not None:
                final_emb[~is_rgb] = global_emb
            if patch_thermal is not None:
                patch_emb[~is_rgb] = patch_thermal
            if return_mask and mask is not None:
                masks[~is_rgb] = mask

        return final_emb, patch_emb, recon_losses, masks

    def calculate_recon_loss(self, pred, mask, target):
        return self.reconstruction_criterion(pred=pred, mask=mask, target=target)

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
