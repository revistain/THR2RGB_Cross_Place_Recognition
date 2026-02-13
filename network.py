# network.py
# BiReconstruction CroCo 방식 네트워크
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from backbone.vision_transformer import vit_small, vit_base
from pathlib import Path

from croco.models.masking import RandomMask
from croco.models.criterion import MaskedMSE
from croco.models.blocks import CroCoDecoderBlock


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
    def __init__(self): super().__init__()
    def forward(self, x): assert x.shape[2] == x.shape[3] == 1; return x[:,:,0,0]


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

        # Aggregation layers
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

    def croco_like_encoder(self, x, modality='thermal'):
        """Masked encoder following CroCo style."""
        current_backbone = self.shared_backbone

        image_patch = current_backbone.patch_embed(x)
        patch_B, patch_N, patch_D = image_patch.shape

        cls_token = current_backbone.cls_token.expand(patch_B, -1, -1)

        # Positional embedding
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

        patch_visible = image_with_cls[~full_mask].reshape(patch_B, -1, patch_D)

        # Encoder forward
        for blk in current_backbone.blocks:
            patch_visible = blk(patch_visible)
        patch_visible = current_backbone.norm(patch_visible)

        # Separate CLS token
        cls_visible = patch_visible[:, 0:1, :]
        patch_only_visible = patch_visible[:, 1 + num_register_tokens:, :]

        return patch_only_visible, mask, patch_B, patch_N, patch_D, cls_visible

    def croco_encoded_mask_expension(self, visible_tokens, mask, patch_B, patch_N, patch_D):
        """Fill masked positions with mask tokens."""
        full_tokens = self.mask_token.expand(patch_B, patch_N, -1).clone()
        full_tokens[~mask] = visible_tokens.flatten(0, 1)
        full_tokens = full_tokens.view(patch_B, patch_N, patch_D)
        return full_tokens

    def forward_model(self, x, paired_rgb=None, modality='rgb'):
        """Forward pass for a single modality."""
        recon_loss_thermal = None
        recon_loss_rgb = None
        mask_thermal = None

        if modality == 'rgb':
            out = self.shared_backbone(x)
            agg_layer = self.rgb_aggregation

        elif modality == 'thermal':
            if self.training:
                # Masked encoder for both modalities
                thermal_visible, mask_thermal, patch_B, patch_N, patch_D, _ = self.croco_like_encoder(x, modality='thermal')
                rgb_visible, mask_rgb, patch_B_rgb, patch_N_rgb, patch_D_rgb, _ = self.croco_like_encoder(paired_rgb, modality='rgb')

                # Full features for global descriptor
                paired_thermal = self.shared_backbone(x)
                paired_thermal_full = paired_thermal["x_norm_patchtokens"]

                paired_rgb_emb = self.shared_backbone(paired_rgb)
                paired_rgb_full = paired_rgb_emb["x_norm_patchtokens"]

                # Mask token expansion
                thermal_full = self.croco_encoded_mask_expension(thermal_visible, mask_thermal, patch_B, patch_N, patch_D)
                rgb_full = self.croco_encoded_mask_expension(rgb_visible, mask_rgb, patch_B_rgb, patch_N_rgb, patch_D_rgb)

                out = paired_thermal

                # CroCo Decoder forward (bidirectional)
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
                rgb_reconed_dec = target_full_dec[thermal_full_dec.shape[0]:, :, :]

                # Prediction Head
                reconstructed_thermal_patches = self.prediction_thermal_head(thermal_reconed_dec)
                reconstructed_rgb_patches = self.prediction_rgb_head(rgb_reconed_dec)
                target_thermal_patches = self.patchify(x)
                target_rgb_patches = self.patchify(paired_rgb)

                # Reconstruction loss
                recon_loss_thermal = self.calculate_recon_loss(reconstructed_thermal_patches, mask_thermal, target_thermal_patches)
                recon_loss_rgb = self.calculate_recon_loss(reconstructed_rgb_patches, mask_rgb, target_rgb_patches)
            else:
                out = self.shared_backbone(x)

            agg_layer = self.thermal_aggregation
        else:
            raise ValueError("Modality must be 'rgb' or 'thermal'")

        # Process backbone output
        patch_tokens = out["x_norm_patchtokens"]
        B, N, D = patch_tokens.shape
        H_feat = int(x.shape[2]/14)
        W_feat = int(x.shape[3]/14)
        x_feat = patch_tokens.permute(0, 2, 1).view(B, D, H_feat, W_feat)

        # Aggregation -> Descriptor
        global_desc = agg_layer(x_feat)

        return global_desc, patch_tokens, [recon_loss_thermal, recon_loss_rgb], mask_thermal

    def forward(self, x, flags, paired_rgb=None, return_mask=False):
        if not isinstance(flags, torch.Tensor):
            flags = torch.tensor(flags, device=x.device)
        if flags.device != x.device:
            flags = flags.to(x.device)

        is_rgb = (flags == 1)
        final_emb = torch.zeros((x.size(0), self.output_dim), device=x.device)
        patch_emb = torch.zeros((x.size(0), self.patch_count, self.output_dim), device=x.device)
        masks = torch.zeros((x.size(0), self.patch_count), dtype=torch.bool, device=x.device)
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
