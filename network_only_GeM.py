# network_only_GeM.py
# GeM 방식 (no decoder) 베이스라인 네트워크
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from backbone.vision_transformer import vit_small, vit_base
from pathlib import Path


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
    Baseline cross-modal VPR network with GeM aggregation only.

    Architecture:
    - Shared DINOv2 backbone for both RGB and Thermal
    - GeM aggregation for global descriptors
    - No reconstruction loss, no decoder
    """
    def __init__(self, args, pretrained_foundation=False, foundation_model_path=None):
        super().__init__()

        self.args = args
        self.shared_backbone = get_backbone(pretrained_foundation, foundation_model_path, args=args)
        self.output_dim = args.features_dim

        # Aggregation layers
        self.aggregation = nn.Sequential(
            L2Norm(),
            GeM(work_with_tokens=None),
            Flatten(),
        )

    def forward_model(self, x, modality='rgb'):
        """Forward pass for a single modality."""
        out = self.shared_backbone(x)

        agg_layer = self.aggregation

        # Process backbone output
        patch_tokens = out["x_norm_patchtokens"]
        B, N, D = patch_tokens.shape
        H_feat = int(self.args.resize[0] / 14)
        W_feat = int(self.args.resize[1] / 14)
        x_feat = patch_tokens.permute(0, 2, 1).view(B, D, H_feat, W_feat)

        # Aggregation -> Descriptor
        global_desc = agg_layer(x_feat)

        return global_desc, patch_tokens

    def forward(self, x, flags, paired_rgb=None, return_mask=False):
        is_rgb = torch.tensor([f == 'rgb' for f in flags], device=x.device)
        patch_count = x.shape[2] // 14 * x.shape[3] // 14

        final_emb = torch.zeros((x.size(0), self.output_dim), device=x.device)
        patch_emb = torch.zeros((x.size(0), patch_count, self.output_dim), device=x.device)

        if is_rgb.any():
            final_emb[is_rgb], patch_emb[is_rgb] = self.forward_model(x[is_rgb], 'rgb')

        if (~is_rgb).any():
            final_emb[~is_rgb], patch_emb[~is_rgb] = self.forward_model(x[~is_rgb], 'thermal')

        # Return format compatible with network.py
        return final_emb, patch_emb, None, None


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
