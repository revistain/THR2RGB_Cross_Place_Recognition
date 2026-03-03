# network.py
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from backbone.vision_transformer import vit_small, vit_base
from pathlib import Path

from croco.models.masking import RandomMask
from croco.models.criterion import MaskedMSE


# ============== 2-Stage GMRW Components ==============

class LabelWarping(nn.Module):
    """
    Random affine augmentation + label warping for shortcut prevention
    Based on GMRW (Self-Supervised Any-Point Tracking by Contrastive Random Walks, ECCV 2024)
    """
    def __init__(self, num_patches=256, H=16, W=16):
        super().__init__()
        self.num_patches = num_patches
        self.H = H
        self.W = W

    def generate_random_affine(self, B, device):
        """Random affine transformation matrix"""
        # Random rotation: -15 to 15 degrees
        angle = (torch.rand(B, device=device) * 30 - 15) * (3.14159 / 180)
        # Random scale: 0.85 to 1.15
        scale = torch.rand(B, device=device) * 0.3 + 0.85
        # Random translation: -0.1 to 0.1
        tx = torch.rand(B, device=device) * 0.2 - 0.1
        ty = torch.rand(B, device=device) * 0.2 - 0.1

        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)

        # Affine matrix [2, 3]
        affine = torch.zeros(B, 2, 3, device=device)
        affine[:, 0, 0] = scale * cos_a
        affine[:, 0, 1] = -scale * sin_a
        affine[:, 0, 2] = tx
        affine[:, 1, 0] = scale * sin_a
        affine[:, 1, 1] = scale * cos_a
        affine[:, 1, 2] = ty

        return affine

    def apply_affine_to_features(self, features, affine):
        """
        Apply affine transformation to patch features
        features: (B, N, C) where N = H * W
        affine: (B, 2, 3)
        """
        B, N, C = features.shape
        H, W = self.H, self.W

        # Reshape to spatial
        feat_spatial = features.view(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)

        # Generate sampling grid
        grid = F.affine_grid(affine, (B, C, H, W), align_corners=False)

        # Sample
        feat_warped = F.grid_sample(feat_spatial, grid, align_corners=False, mode='bilinear', padding_mode='zeros')

        # Reshape back
        return feat_warped.permute(0, 2, 3, 1).reshape(B, N, C)

    def compute_warped_label(self, affine_t, affine_r):
        """
        Compute warped identity label: T_t @ T_r^(-1)
        Returns: (B, N, N) warped one-hot labels, (B, N) valid mask
        """
        B = affine_t.size(0)
        H, W = self.H, self.W
        N = H * W
        device = affine_t.device

        # Initial coordinate grid: (B, H, W, 2) normalized [-1, 1]
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing='ij'
        )
        init_grid = torch.stack([grid_x, grid_y], dim=-1)  # (H, W, 2)
        init_grid = init_grid.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, W, 2)

        # Apply T_t to grid
        grid_t = self._apply_affine_to_grid(init_grid, affine_t)

        # Apply T_r^(-1) to grid_t
        affine_r_inv = self._invert_affine(affine_r)
        grid_final = self._apply_affine_to_grid(grid_t, affine_r_inv)

        # Convert to indices
        grid_final_idx = grid_final.reshape(B, N, 2)
        # Normalize to [0, H-1] and [0, W-1]
        idx_x = ((grid_final_idx[..., 0] + 1) / 2 * (W - 1)).round().long()
        idx_y = ((grid_final_idx[..., 1] + 1) / 2 * (H - 1)).round().long()

        # Valid mask: within bounds
        valid = (idx_x >= 0) & (idx_x < W) & (idx_y >= 0) & (idx_y < H)

        # Convert to 1D index
        idx_1d = idx_y * W + idx_x  # (B, N)
        idx_1d = idx_1d.clamp(0, N - 1)  # Safety clamp

        # Create one-hot label
        label = F.one_hot(idx_1d, num_classes=N).float()  # (B, N, N)

        # Mask invalid positions
        label = label * valid.unsqueeze(-1).float()

        return label, valid.float()

    def _apply_affine_to_grid(self, grid, affine):
        """Apply affine transform to coordinate grid"""
        B, H, W, _ = grid.shape
        grid_flat = grid.reshape(B, -1, 2)  # (B, H*W, 2)

        # Homogeneous coordinates
        ones = torch.ones(B, H*W, 1, device=grid.device)
        grid_homo = torch.cat([grid_flat, ones], dim=-1)  # (B, H*W, 3)

        # Apply affine: (B, 2, 3) @ (B, 3, H*W) -> (B, 2, H*W)
        grid_transformed = torch.bmm(affine, grid_homo.transpose(1, 2))
        grid_transformed = grid_transformed.transpose(1, 2).reshape(B, H, W, 2)

        return grid_transformed

    def _invert_affine(self, affine):
        """Invert 2x3 affine matrix"""
        B = affine.size(0)
        # Extract rotation/scale (2x2) and translation (2x1)
        A = affine[:, :, :2]  # (B, 2, 2)
        t = affine[:, :, 2:]  # (B, 2, 1)

        # Invert A
        A_inv = torch.inverse(A)

        # New translation: -A_inv @ t
        t_inv = -torch.bmm(A_inv, t)

        # Combine
        affine_inv = torch.cat([A_inv, t_inv], dim=-1)
        return affine_inv

    def forward(self, thermal_feat, rgb_feat):
        """
        Apply augmentation and compute warped labels

        Args:
            thermal_feat: (B, N, C) thermal patch features
            rgb_feat: (B, N, C) RGB patch features

        Returns:
            thermal_aug: (B, N, C) augmented thermal features
            rgb_aug: (B, N, C) augmented RGB features
            warped_label: (B, N, N) target label
            valid_mask: (B, N) valid positions
        """
        B = thermal_feat.size(0)
        device = thermal_feat.device

        # Generate random affines
        affine_t = self.generate_random_affine(B, device)
        affine_r = self.generate_random_affine(B, device)

        # Apply to features
        thermal_aug = self.apply_affine_to_features(thermal_feat, affine_t)
        rgb_aug = self.apply_affine_to_features(rgb_feat, affine_r)

        # Compute warped label
        warped_label, valid_mask = self.compute_warped_label(affine_t, affine_r)

        return thermal_aug, rgb_aug, warped_label, valid_mask


class GMRWLoss(nn.Module):
    """
    GMRW Loss with Label Warping
    Based on GMRW (Self-Supervised Any-Point Tracking by Contrastive Random Walks, ECCV 2024)
    """
    def __init__(self, use_smoothness=False, smoothness_weight=0.1, edge_constant=150.0):
        super().__init__()
        self.use_smoothness = use_smoothness
        self.smoothness_weight = smoothness_weight
        self.edge_constant = edge_constant

    def forward(self, cycle, warped_label, valid_mask, flow=None, image=None):
        """
        Args:
            cycle: (B, N, N) cycle consistency matrix
            warped_label: (B, N, N) warped target label
            valid_mask: (B, N) valid positions
            flow: (B, 2, H, W) optical flow (for smoothness)
            image: (B, 3, H, W) image (for edge-aware smoothness)
        """
        B, N, _ = cycle.shape

        # Cycle consistency loss with warped label
        # For each position i, find probability of landing on warped target
        # Sum probability mass on correct targets
        correct_prob = (cycle * warped_label).sum(dim=-1)  # (B, N)

        # Mask invalid positions
        correct_prob = correct_prob * valid_mask

        # -log(probability)
        cycle_loss = -torch.log(correct_prob.clamp(min=1e-8))
        cycle_loss = cycle_loss.sum() / (valid_mask.sum() + 1e-8)

        total_loss = cycle_loss

        # Smoothness loss (optional)
        if self.use_smoothness and flow is not None and image is not None:
            smooth_loss = self.smoothness_loss(flow, image)
            total_loss = total_loss + self.smoothness_weight * smooth_loss
        else:
            smooth_loss = torch.tensor(0.0, device=cycle.device)

        return total_loss, cycle_loss, smooth_loss

    def smoothness_loss(self, flow, image):
        """Edge-aware smoothness loss"""
        dx = flow[:, :, :, 1:] - flow[:, :, :, :-1]
        dy = flow[:, :, 1:, :] - flow[:, :, :-1, :]

        img_dx = (image[:, :, :, 1:] - image[:, :, :, :-1]).abs().mean(dim=1, keepdim=True)
        img_dy = (image[:, :, 1:, :] - image[:, :, :-1, :]).abs().mean(dim=1, keepdim=True)

        wx = torch.exp(-self.edge_constant * img_dx)
        wy = torch.exp(-self.edge_constant * img_dy)

        loss_x = (wx * torch.sqrt(dx**2 + 1e-6)).mean()
        loss_y = (wy * torch.sqrt(dy**2 + 1e-6)).mean()

        return loss_x + loss_y


# ============== End 2-Stage GMRW Components ==============


class AffinityAggregator(nn.Module):
    """
    Patch tokens에서 affinity matrix를 계산하고 descriptor로 변환

    N→N mixing으로 spatial structure 유지 후 pooling

    Input: (B, N, C) patch tokens
    Output: (B, output_dim) affinity descriptor
    """
    def __init__(self, num_patches=256, embed_dim=768, output_dim=384):
        super().__init__()
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.output_dim = output_dim

        # N→N spatial mixing (preserves structure)
        self.row_mixer = nn.Linear(num_patches, num_patches)
        self.col_mixer = nn.Linear(num_patches, num_patches)

        # Final projection: N → output_dim
        self.proj = nn.Linear(num_patches, output_dim)

        # GeM pooling parameter
        self.gem_p = nn.Parameter(torch.ones(1) * 3.0)

    def forward(self, patch_tokens):
        # patch_tokens: (B, N, C)

        # L2 normalize patch tokens before affinity
        tokens_norm = F.normalize(patch_tokens, p=2, dim=-1)

        # Self-affinity matrix
        affinity = tokens_norm @ tokens_norm.transpose(-1, -2)  # (B, N, N)

        # Adaptive pooling to fixed size (handles variable N)
        affinity = F.adaptive_avg_pool2d(
            affinity.unsqueeze(1),
            (self.num_patches, self.num_patches)
        ).squeeze(1)  # (B, num_patches, num_patches)

        # N→N mixing (spatial context exchange)
        x = self.row_mixer(affinity)  # (B, N, N) - each row mixes with other rows
        x = self.col_mixer(x.transpose(-1, -2)).transpose(-1, -2)  # (B, N, N) - each col mixes

        # GeM pooling over columns → (B, N)
        x = x.clamp(min=1e-6).pow(self.gem_p)
        x = x.mean(dim=-1)  # (B, N)
        x = x.pow(1.0 / self.gem_p)

        # Project to output dim
        x = self.proj(x)  # (B, output_dim)

        return x



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
    def __init__(self, args, pretrained_foundation=False, foundation_model_path=None):
        super().__init__()

        self.args = args
        self.shared_backbone = get_backbone(pretrained_foundation, foundation_model_path, args=args)
        self.output_dim = args.features_dim
        self.recon_loss_type = args.recon_loss_type

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

        # Aggregation layers (without L2Norm - will normalize individually after)
        self.rgb_aggregation = nn.Sequential(
            GeM(work_with_tokens=None),
            Flatten(),
        )
        self.thermal_aggregation = nn.Sequential(
            GeM(work_with_tokens=None),
            Flatten()
        )

        # Affinity aggregator with N→N mixing
        self.affinity_dim = args.affinity_dim
        self.affinity_aggregator = AffinityAggregator(
            num_patches=self.patch_count,  # (224/14)^2 = 256
            embed_dim=self.output_dim,
            output_dim=self.affinity_dim
        )

        # Descriptor mixer: concat(GeM, Affinity) → final descriptor
        # Input: output_dim + affinity_dim
        self.descriptor_mixer = nn.Linear(self.output_dim + self.affinity_dim, self.affinity_dim)

        # ============== 2-Stage GMRW Components ==============
        H_feat = int(args.resize[0] / 14)
        W_feat = int(args.resize[1] / 14)
        self.label_warping = LabelWarping(
            num_patches=self.patch_count,
            H=H_feat,
            W=W_feat
        )
        self.gmrw_loss_fn = GMRWLoss(
            use_smoothness=getattr(args, 'use_smoothness_loss', False),
            smoothness_weight=getattr(args, 'smoothness_weight', 0.1)
        )
        self.gmrw_temperature = 0.07

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

    def unpatchify(self, x, H, W):
        """
        Reverse of patchify: (B, L, p**2 * 3) → (B, 3, H, W)
        """
        p = 14
        h, w = H // p, W // p
        B = x.shape[0]
        x = x.reshape(B, h, w, p, p, 3)
        x = torch.einsum('nhwpqc->nchpwq', x)
        return x.reshape(B, 3, H, W)

    @torch.no_grad()
    def forward_for_viz(self, thermal_img, rgb_img):
        """
        Visualization-only forward pass for a single pair (B=1).
        Must be called in model.eval() state.

        Args:
            thermal_img: (1, 3, H, W)
            rgb_img:     (1, 3, H, W)
        Returns:
            dict with all visualization tensors (CPU)
        """
        H, W = thermal_img.shape[2], thermal_img.shape[3]

        # ── 1. Full backbone pass + attention ────────────────────────────────
        out_t = self.shared_backbone(thermal_img, return_attention=True)
        out_r = self.shared_backbone(rgb_img,     return_attention=True)

        patch_t = out_t['x_norm_patchtokens']   # (1, N, C)
        patch_r = out_r['x_norm_patchtokens']   # (1, N, C)

        # cls_attention: (1, num_heads, N)
        backbone_attn_t = out_t.get('cls_attention')
        backbone_attn_r = out_r.get('cls_attention')

        # ── 2. Masked encoder (random masks generated each call) ─────────────
        t_vis, mask_t, pB,  pN,  pD,  _ = self.croco_like_encoder(thermal_img)
        r_vis, mask_r, pBr, pNr, pDr, _ = self.croco_like_encoder(rgb_img)

        t_full = self.croco_encoded_mask_expension(t_vis, mask_t, pB,  pN,  pD)
        r_full = self.croco_encoded_mask_expension(r_vis, mask_r, pBr, pNr, pDr)

        # ── 3. Decoder: thermal (target=thermal, ref=rgb_full) ───────────────
        t_dec = t_full + self.decoder_pos_embed
        r_ref = patch_r + self.decoder_pos_embed  # full RGB as reference

        x = t_dec
        for blk in self.decoder_blocks:
            x = blk(x, r_ref, return_attention=True)
        x = self.decoder_norm(x)
        # clone before RGB decoder overwrites the weights
        decoder_cross_attn_t = self.decoder_blocks[-1].cross_attn_weights.clone()

        recon_t_patches = self.prediction_thermal_head(x)
        recon_t_img     = self.unpatchify(recon_t_patches, H, W)

        # ── 4. Decoder: RGB (target=rgb, ref=thermal_full) ───────────────────
        r_dec = r_full + self.decoder_pos_embed
        t_ref = patch_t + self.decoder_pos_embed  # full thermal as reference

        y = r_dec
        for blk in self.decoder_blocks:
            y = blk(y, t_ref, return_attention=True)
        y = self.decoder_norm(y)
        decoder_cross_attn_r = self.decoder_blocks[-1].cross_attn_weights.clone()

        recon_r_patches = self.prediction_rgb_head(y)
        recon_r_img     = self.unpatchify(recon_r_patches, H, W)

        # ── 5. Affinity matrices (self-similarity) ───────────────────────────
        pt_norm = F.normalize(patch_t, p=2, dim=-1)
        pr_norm = F.normalize(patch_r, p=2, dim=-1)
        aff_t = (pt_norm @ pt_norm.transpose(-1, -2)).squeeze(0)  # (N, N)
        aff_r = (pr_norm @ pr_norm.transpose(-1, -2)).squeeze(0)  # (N, N)

        # ── 6. Transition matrices A_T2R, A_R2T, Cycle ───────────────────────
        temp = 0.07  # standard temperature for contrastive learning
        A_T2R = F.softmax(torch.bmm(pt_norm, pr_norm.transpose(-1, -2)) / temp, dim=-1).squeeze(0)
        A_R2T = F.softmax(torch.bmm(pr_norm, pt_norm.transpose(-1, -2)) / temp, dim=-1).squeeze(0)
        cycle = torch.mm(A_T2R, A_R2T)  # (N, N)

        return {
            'thermal_orig':         thermal_img,           # (1, 3, H, W)
            'rgb_orig':             rgb_img,               # (1, 3, H, W)
            'recon_thermal':        recon_t_img,           # (1, 3, H, W)
            'recon_rgb':            recon_r_img,           # (1, 3, H, W)
            'mask_thermal':         mask_t,                # (1, N) bool
            'mask_rgb':             mask_r,                # (1, N) bool
            'backbone_attn_t':      backbone_attn_t,       # (1, num_heads, N) or None
            'backbone_attn_r':      backbone_attn_r,       # (1, num_heads, N) or None
            'decoder_cross_attn_t': decoder_cross_attn_t, # (1, N_q, N_ref)
            'decoder_cross_attn_r': decoder_cross_attn_r, # (1, N_q, N_ref)
            'affinity_t':           aff_t,                 # (N, N)
            'affinity_r':           aff_r,                 # (N, N)
            'A_T2R':                A_T2R,                 # (N, N)
            'cycle':                cycle,                 # (N, N)
        }

    def croco_like_encoder(self, x, modality='thermal'):
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
        mask = self.mask_generator(image_patch)
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

    def forward_model(self, x, paired_rgb=None, modality='rgb'):
        """Forward pass for a single modality."""
        recon_loss_thermal = None
        recon_loss_rgb = None
        mask_thermal = None

        if modality == 'rgb':
            agg_layer = self.rgb_aggregation # RGB용 GeM
            out = self.shared_backbone(x)
        elif modality == 'thermal':
            agg_layer = self.thermal_aggregation # Thermal용 GeM
            if self.training:
                # Masked encoder for both modalities
                # - 이미지 masking하고 DINOv2 통과하고 남은 token들 return
                thermal_visible, mask_thermal, patch_B, patch_N, patch_D, _ = self.croco_like_encoder(x, modality='thermal')
                rgb_visible, mask_rgb, patch_B_rgb, patch_N_rgb, patch_D_rgb, _ = self.croco_like_encoder(paired_rgb, modality='rgb')

                # Full features for global descriptor
                # - pair 이미지들도 DINOv2 통과후 feature embedding 추출
                paired_thermal = self.shared_backbone(x)
                paired_thermal_full = paired_thermal["x_norm_patchtokens"]

                paired_rgb_emb = self.shared_backbone(paired_rgb)
                paired_rgb_full = paired_rgb_emb["x_norm_patchtokens"]

                # Mask token expansion
                # - masking되고 encoding된 token들을 mask_token이랑 합쳐서 feature embedding 만들기
                thermal_full = self.croco_encoded_mask_expension(thermal_visible, mask_thermal, patch_B, patch_N, patch_D)
                rgb_full = self.croco_encoded_mask_expension(rgb_visible, mask_rgb, patch_B_rgb, patch_N_rgb, patch_D_rgb)

                # CroCo 태우기전 결과저장 (paried_thermal 뒤에서 안써서 clone 안했음)
                out = paired_thermal

                # CroCo Decoder forward (bidirectional)
                # - mask_token 추가된 feature embedding들에 decoder의 positional embedding 추가
                thermal_full_dec = thermal_full + self.decoder_pos_embed
                rgb_full_dec = rgb_full + self.decoder_pos_embed
                paired_thermal_dec = paired_thermal_full + self.decoder_pos_embed
                paired_rgb_dec = paired_rgb_full + self.decoder_pos_embed

                # 연산 효율 위해 decoder 태우기전 batch로 구성
                target_full_dec = torch.cat([thermal_full_dec, rgb_full_dec], dim=0)
                ref_full_dec = torch.cat([paired_rgb_dec, paired_thermal_dec], dim=0)

                # decoder 통과
                for blk in self.decoder_blocks:
                    target_full_dec = blk(target_full_dec, ref_full_dec)
                target_full_dec = self.decoder_norm(target_full_dec)

                # batch 다시 분리
                thermal_reconed_dec = target_full_dec[:thermal_full_dec.shape[0], :, :]
                rgb_reconed_dec = target_full_dec[thermal_full_dec.shape[0]:, :, :]

                # Prediction Head
                # - MLP로 decoded feature embedding -> pixel level
                reconstructed_thermal_patches = self.prediction_thermal_head(thermal_reconed_dec)
                reconstructed_rgb_patches = self.prediction_rgb_head(rgb_reconed_dec)
                target_thermal_patches = self.patchify(x)
                target_rgb_patches = self.patchify(paired_rgb)

                # Reconstruction loss
                # - masked된 부분에 대해 loss 계산
                recon_loss_thermal = self.calculate_recon_loss(reconstructed_thermal_patches, mask_thermal, target_thermal_patches)
                recon_loss_rgb = self.calculate_recon_loss(reconstructed_rgb_patches, mask_rgb, target_rgb_patches)
            else:
                # inference 때
                out = self.shared_backbone(x)

        # Process backbone output
        patch_tokens = out["x_norm_patchtokens"]
        B, N, D = patch_tokens.shape
        H_feat = int(x.shape[2]/14)
        W_feat = int(x.shape[3]/14)
        x_feat = patch_tokens.permute(0, 2, 1).view(B, D, H_feat, W_feat)

        # GeM descriptor
        gem_desc = agg_layer(x_feat)  # (B, C)

        # Affinity descriptor (from full patch tokens for VPR)
        affinity_desc = self.affinity_aggregator(patch_tokens)  # (B, C)

        # Individual L2 normalization
        gem_desc = F.normalize(gem_desc, p=2, dim=-1)
        affinity_desc = F.normalize(affinity_desc, p=2, dim=-1)

        # Concat and mix
        concat_desc = torch.cat([gem_desc, affinity_desc], dim=-1)  # (B, 2C)
        global_desc = self.descriptor_mixer(concat_desc)  # (B, affinity_dim)

        return global_desc, patch_tokens, [recon_loss_thermal, recon_loss_rgb], mask_thermal

    def forward(self, x, flags, paired_rgb=None, return_mask=False):
        if not isinstance(flags, torch.Tensor):
            flags = torch.tensor(flags, device=x.device)
        if flags.device != x.device:
            flags = flags.to(x.device)

        is_rgb = (flags == 1)
        final_emb = torch.zeros((x.size(0), self.affinity_dim), device=x.device) # mixed descriptor 모음
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
                x[~is_rgb], modality='thermal', paired_rgb=paired_rgb)
            if global_emb is not None:
                final_emb[~is_rgb] = global_emb
            if patch_thermal is not None:
                patch_emb[~is_rgb] = patch_thermal
            if return_mask and mask is not None:
                masks[~is_rgb] = mask

        return final_emb, patch_emb, recon_losses, masks

    def calculate_recon_loss(self, pred, mask, target):
        return self.reconstruction_criterion(pred=pred, mask=mask, target=target)

    # ============== 2-Stage GMRW Methods ==============

    def stage2_forward_gmrw(self, thermal_img, rgb_img, use_warp=True):
        """
        2-stage forward with GMRW loss

        Args:
            thermal_img: (B, 3, H, W)
            rgb_img: (B, 3, H, W)
            use_warp: whether to use label warping

        Returns:
            loss: total GMRW loss
            cycle: (B, N, N) cycle matrix
            A_T2R: (B, N, N) transition matrix T→R
            warped_label: (B, N, N) target label
        """
        # Encoder
        thermal_out = self.shared_backbone(thermal_img)
        rgb_out = self.shared_backbone(rgb_img)

        patch_T = thermal_out["x_norm_patchtokens"]  # (B, N, C)
        patch_R = rgb_out["x_norm_patchtokens"]

        # Label Warping (augmentation) - only during training
        if use_warp and self.training:
            patch_T_aug, patch_R_aug, warped_label, valid_mask = \
                self.label_warping(patch_T, patch_R)
        else:
            patch_T_aug = patch_T
            patch_R_aug = patch_R
            B, N, C = patch_T.shape
            warped_label = torch.eye(N, device=patch_T.device)
            warped_label = warped_label.unsqueeze(0).expand(B, -1, -1)
            valid_mask = torch.ones(B, N, device=patch_T.device)

        # Decoder (bidirectional cross-attention)
        T_dec = patch_T_aug + self.decoder_pos_embed
        R_dec = patch_R_aug + self.decoder_pos_embed

        for blk in self.decoder_blocks:
            T_new = blk(T_dec, R_dec)
            R_new = blk(R_dec, T_dec)
            T_dec, R_dec = T_new, R_new

        refined_T = self.decoder_norm(T_dec)
        refined_R = self.decoder_norm(R_dec)

        # Cross-modal affinity
        T_norm = F.normalize(refined_T, dim=-1)
        R_norm = F.normalize(refined_R, dim=-1)
        affinity = T_norm @ R_norm.transpose(-1, -2) / self.gmrw_temperature

        # Transition probabilities
        A_T2R = F.softmax(affinity, dim=-1)
        A_R2T = F.softmax(affinity, dim=-2)

        # Cycle matrix: T→R→T
        cycle = A_T2R @ A_R2T  # (B, N, N)

        # GMRW Loss
        total_loss, cycle_loss, smooth_loss = self.gmrw_loss_fn(
            cycle, warped_label, valid_mask
        )

        return total_loss, cycle, A_T2R, warped_label, cycle_loss, smooth_loss

    def stage2_compute_mnn_score(self, A_T2R, A_R2T):
        """
        Compute MNN (Mutual Nearest Neighbor) count score using hard matching.

        Args:
            A_T2R: (B, N, N) - Thermal→RGB transition matrix (softmax normalized)
            A_R2T: (B, N, N) - RGB→Thermal transition matrix (softmax normalized)

        Returns:
            score: (B,) - MNN count normalized by number of patches [0, 1]
        """
        if A_T2R.dim() == 2:
            A_T2R = A_T2R.unsqueeze(0)
            A_R2T = A_R2T.unsqueeze(0)

        B, N, _ = A_T2R.shape

        # Hard matching: argmax-based
        # For each thermal patch i, find nearest RGB patch
        nearest_rgb = torch.argmax(A_T2R, dim=-1)  # (B, N)

        # For each RGB patch j, find nearest thermal patch
        nearest_thermal = torch.argmax(A_R2T, dim=-2)  # (B, N)

        # Check mutual nearest neighbors
        mnn_count = torch.zeros(B, device=A_T2R.device)
        for b in range(B):
            for i in range(N):
                j = nearest_rgb[b, i]  # Nearest RGB to thermal i
                if nearest_thermal[b, j] == i:  # Check if thermal i is nearest to RGB j
                    mnn_count[b] += 1

        # Normalize by number of patches
        score = mnn_count / N

        return score

    def stage2_compute_score(self, cycle, method='trace', A_T2R=None, A_R2T=None):
        """
        Compute matching score from cycle matrix (for inference)

        Args:
            cycle: (B, N, N) cycle matrix
            method: 'trace', 'entropy', or 'mnn'
            A_T2R: (B, N, N) - required for 'mnn' method
            A_R2T: (B, N, N) - required for 'mnn' method

        Returns:
            score: (B,) matching score
        """
        if cycle.dim() == 2:
            cycle = cycle.unsqueeze(0)

        if method == 'trace':
            # Higher diagonal sum = better cycle consistency
            score = torch.diagonal(cycle, dim1=-2, dim2=-1).mean(dim=-1)
        elif method == 'entropy':
            # Lower entropy = more confident matching
            entropy = -(cycle * torch.log(cycle.clamp(min=1e-8))).sum(dim=(-1, -2))
            score = -entropy / cycle.size(-1) ** 2  # Normalize and negate
        elif method == 'mnn':
            if A_T2R is None or A_R2T is None:
                raise ValueError("MNN method requires A_T2R and A_R2T matrices")
            score = self.stage2_compute_mnn_score(A_T2R, A_R2T)
        else:
            raise ValueError(f"Unknown score method: {method}")

        return score

    def stage2_inference(self, thermal_img, rgb_img, score_method='trace'):
        """
        2-stage inference: compute matching score between thermal and RGB

        Args:
            thermal_img: (B, 3, H, W) or (1, 3, H, W)
            rgb_img: (B, 3, H, W) or (1, 3, H, W)
            score_method: 'trace', 'entropy', or 'mnn'

        Returns:
            score: (B,) matching score
            cycle: (B, N, N) cycle matrix
            A_T2R: (B, N, N) transition matrix T→R
            A_R2T: (B, N, N) transition matrix R→T
        """
        with torch.no_grad():
            # Encoder
            thermal_out = self.shared_backbone(thermal_img)
            rgb_out = self.shared_backbone(rgb_img)

            patch_T = thermal_out["x_norm_patchtokens"]
            patch_R = rgb_out["x_norm_patchtokens"]

            # Decoder (bidirectional)
            T_dec = patch_T + self.decoder_pos_embed
            R_dec = patch_R + self.decoder_pos_embed

            for blk in self.decoder_blocks:
                T_new = blk(T_dec, R_dec)
                R_new = blk(R_dec, T_dec)
                T_dec, R_dec = T_new, R_new

            refined_T = self.decoder_norm(T_dec)
            refined_R = self.decoder_norm(R_dec)

            # Cross-modal affinity
            T_norm = F.normalize(refined_T, dim=-1)
            R_norm = F.normalize(refined_R, dim=-1)
            affinity = T_norm @ R_norm.transpose(-1, -2) / self.gmrw_temperature

            # Transition probabilities
            A_T2R = F.softmax(affinity, dim=-1)
            A_R2T = F.softmax(affinity, dim=-2)

            # Cycle matrix
            cycle = A_T2R @ A_R2T

            # Compute score (pass A_T2R, A_R2T for MNN)
            score = self.stage2_compute_score(cycle, method=score_method,
                                             A_T2R=A_T2R, A_R2T=A_R2T)

        return score, cycle, A_T2R, A_R2T

    # ============== End 2-Stage GMRW Methods ==============


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
