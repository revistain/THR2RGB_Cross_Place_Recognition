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
    """
    Residual + Bottleneck 구조
    - Residual: 초반 학습 안정성 (GeM을 거친 feature를 유지)
    - Bottleneck: Overfitting 방지 (파라미터 수 감소)
    """
    def __init__(self, dim=768, bottleneck=192):
        super().__init__()
        self.gem = nn.Sequential(L2Norm(), GeM(), Flatten())
        self.mlp = nn.Sequential(
            nn.Linear(dim, bottleneck),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck, dim)
        )
        # MLP 출력을 0으로 초기화 -> residual connection 초반에는 identity
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
    
    def forward(self, x):
        x = self.gem(x)
        return x + self.mlp(x)

# ============= CroCo 원본 구조 추가 =============
class CroCoDecoderBlock(nn.Module):
    """
    CroCo 원본 Decoder Block
    
    구조:
    1. Self-Attention: decoder 내부 token들 간 정보 혼합
    2. Cross-Attention: RGB encoder output 참조 (CroCo의 핵심!)
    3. MLP: Position-wise feed-forward
    
    모두 Pre-LayerNorm + Residual connection 사용
    """
    def __init__(self, dim=768, num_heads=12, mlp_ratio=4.0):
        super().__init__()
        
        # Self-Attention components
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        
        # Cross-Attention components (CroCo의 핵심!)
        # Thermal(query)이 RGB(key, value)를 참조
        self.norm2 = nn.LayerNorm(dim)
        self.norm_cross = nn.LayerNorm(dim)  # encoder output용 norm
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        
        # MLP components
        self.norm3 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )
        
    def forward(self, x, encoder_output):
        """
        Args:
            x: [B, N, D] - decoder input (thermal + mask tokens)
            encoder_output: [B, M, D] - RGB encoder output (참조할 정보)
        Returns:
            x: [B, N, D] - updated decoder features
        """
        # Step 1: Self-Attention (decoder 내부 정보 혼합)
        x_norm = self.norm1(x)
        x = x + self.self_attn(x_norm, x_norm, x_norm)[0]
        
        # Step 2: Cross-Attention (RGB 정보 참조!)
        # Query: thermal (복원하려는 대상)
        # Key, Value: RGB (참조할 정보)
        x_norm = self.norm2(x)
        encoder_norm = self.norm_cross(encoder_output)
        x = x + self.cross_attn(
            query=x_norm,           # Thermal
            key=encoder_norm,       # RGB
            value=encoder_norm      # RGB
        )[0]
        
        # Step 3: MLP (position-wise transformation)
        x = x + self.mlp(self.norm3(x))
        
        return x

class ThermalDecoder(nn.Module):
    """
    CroCo-style Cross-modal Decoder
    
    목적:
    - Thermal의 masked patches를 RGB 정보를 참조하여 복원
    - Cross-attention으로 RGB의 semantic 정보 활용
    
    입력:
    1. Thermal encoder output (FULL 257 tokens)
    2. RGB encoder output (full patches)
    3. Mask positions
    
    출력:
    - 복원된 thermal patch features [B, 256, 768]
    """
    def __init__(self, embed_dim=768, num_patches=256, decoder_depth=4, num_heads=12):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        
        # Masked position용 learnable token
        # 각 masked patch는 이 token으로 초기화됨
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        
        # Decoder용 positional encoding (encoder와 별도)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.normal_(self.decoder_pos_embed, std=0.02)
        
        # CroCo Decoder blocks (cross-attention 포함!)
        self.decoder_blocks = nn.ModuleList([
            CroCoDecoderBlock(embed_dim, num_heads) 
            for _ in range(decoder_depth)
        ])
        
        self.decoder_norm = nn.LayerNorm(embed_dim)
        
        # Prediction head: decoder output → 원본 patch feature space
        self.decoder_pred = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x_thermal_full, masks, x_rgb_full):
        """
        ============ 수정: Full 257 tokens 받도록 변경 ============
        Args:
            x_thermal_full: [B, 257, D] - thermal encoder output (CLS + FULL 256 patches)
            masks: [B, 256] - boolean mask (True = masked position)
            x_rgb_full: [B, 256, D] - RGB full patch features
        Returns:
            x_rec: [B, 256, D] - 복원된 thermal patch features
        """
        B, _, D = x_thermal_full.shape
        
        # Step 1: CLS token 분리
        cls_token = x_thermal_full[:, :1, :]  # [B, 1, D]
        x_patches_full = x_thermal_full[:, 1:, :]  # [B, 256, D] - Full patches!
        
        # Step 2: Decoder input 생성
        # Visible positions: encoder output 사용
        # Masked positions: mask token 사용
        mask_tokens = self.mask_token.repeat(B, self.num_patches, 1)  # [B, 256, D]
        x_decoder_input = x_patches_full.clone()
        
        # Masked positions만 mask token으로 교체
        for i in range(B):
            x_decoder_input[i, masks[i]] = mask_tokens[i, masks[i]]
        
        # Step 3: CLS token prepend
        x_full = torch.cat([cls_token, x_decoder_input], dim=1)  # [B, 257, D]
        
        # Step 4: Decoder positional encoding 추가
        x_full = x_full + self.decoder_pos_embed
        
        # Step 5: Decoder blocks 통과 (RGB 참조하며 복원!)
        for blk in self.decoder_blocks:
            x_full = blk(x_full, encoder_output=x_rgb_full)
        
        x_full = self.decoder_norm(x_full)
        
        # Step 6: CLS token 제거
        x_patches = x_full[:, 1:, :]  # [B, 256, D]
        
        # Step 7: Prediction head로 최종 복원
        x_rec = self.decoder_pred(x_patches)  # [B, 256, D]
        
        return x_rec

class CrossModalVPR_Net(nn.Module):
    def __init__(self, pretrained_foundation=False, foundation_model_path=None,
                 use_alignment_proj=False, use_GeMAdditionalLayer=False,
                 use_rgb_adapter=True, use_thermal_adapter=True, 
                 mask_ratio=0.75, decoder_depth=4, recon_loss_weight=0.1):
        super().__init__()

        print("="*30)
        print("- use_rgb_adapter: \t", use_rgb_adapter)
        print("- use_thermal_adapter: \t", use_thermal_adapter)
        print("- thermal mask_ratio: \t", mask_ratio)
        print("- decoder_depth: \t", decoder_depth)
        print("- recon_loss_weight: \t", recon_loss_weight)
        print("="*30)

        # 1. Backbone 생성 (RGB/Thermal 독립적)
        self.rgb_backbone = get_backbone(pretrained_foundation, foundation_model_path, use_adapter=use_rgb_adapter)
        self.thermal_backbone = get_backbone(pretrained_foundation, foundation_model_path, use_adapter=use_thermal_adapter)
        self.output_dim = 768

        # 2. Aggregation Layer (모달리티별 독립적)
        if use_GeMAdditionalLayer:
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

        # 3. Thermal Decoder (CroCo-style cross-modal reconstruction)
        self.thermal_decoder = ThermalDecoder(
            embed_dim=768, 
            num_patches=256, 
            decoder_depth=decoder_depth,
            num_heads=12
        )
        
        # 4. Masking & Loss 설정
        self.mask_ratio = mask_ratio
        self.recon_loss_weight = recon_loss_weight
        self._set_mask_generator(16*16, mask_ratio)
        
        # 5. Reconstruction loss 모니터링용
        self.last_recon_loss = None
        self.vis_data = None

    def _set_mask_generator(self, num_patches, mask_ratio):
        """Random masking generator 초기화"""
        self.mask_generator = RandomMask(num_patches, mask_ratio)

    def compute_reconstruction_loss(self, pred_features, target_features, masks):
        """
        CroCo-style reconstruction loss (Smooth L1)
        
        Smooth L1 Loss:
        - 작은 오차 (|x| < 1): 0.5 * x^2 (L2처럼, 정확도 중시)
        - 큰 오차 (|x| >= 1): |x| - 0.5 (L1처럼, outlier robust)
        - 방향 + 크기 모두 고려
        
        Args:
            pred_features: [B, 256, 768] - decoder가 예측한 features
            target_features: [B, 256, 768] - 원본 patch features
            masks: [B, 256] - masked positions (True = 복원 대상)
        Returns:
            loss: scalar - masked positions에서만 계산한 평균 loss
        """
        # Smooth L1 loss (element-wise)
        # reduction='none'으로 각 element별 loss 계산
        loss = F.smooth_l1_loss(pred_features, target_features, reduction='none', beta=1.0)
        # Output: [B, 256, 768]
        
        # Feature dimension (768)에 대해 sum → patch별 loss
        loss_per_patch = loss.sum(dim=-1)  # [B, 256]
        
        # Safety check: masked positions가 없으면 0 반환
        if masks.sum() == 0:
            return torch.tensor(0.0, device=pred_features.device, requires_grad=True)
        
        # Masked positions만 loss 계산
        masked_loss = loss_per_patch[masks].mean()
        
        return masked_loss
    
    def forward_model(self, x, modality='rgb', rgb_reference=None, save_rgb_img=None):
        """
        단일 모달리티 Forward Pass
        
        Args:
            x: [B, 3, 224, 224] - 입력 이미지
            modality: 'rgb' or 'thermal'
            rgb_reference: [B, 256, 768] - RGB patch features (thermal 복원 시 참조)
        """
        
        if modality == 'rgb':
            # === RGB: 일반 DINOv2 forward ===
            out = self.rgb_backbone(x, return_attention=True)
            agg_layer = self.rgb_aggregation
            recon_loss = None
            
        elif modality == 'thermal':
            # === Thermal: MAE-style masking + CroCo cross-modal reconstruction ===
            agg_layer = self.thermal_aggregation
            
            # Step 1: Patch Embedding
            x_patch = self.thermal_backbone.patch_embed(x)
            B, N, D = x_patch.shape  # N=256, D=768
            
            # Step 2: Positional Embedding 리사이징 (518x518 → 224x224)
            pos_tokens = self.thermal_backbone.pos_embed[:, 1:, :]
            pos_embed_grid = pos_tokens.reshape(1, 37, 37, 768).permute(0, 3, 1, 2)
            
            pos_embed_resized = F.interpolate(
                pos_embed_grid, 
                size=(16, 16), 
                mode='bicubic', 
                align_corners=False
            )
            
            pos_embed_final = pos_embed_resized.permute(0, 2, 3, 1).flatten(1, 2)
            x_patch = x_patch + pos_embed_final  # [B, 256, 768]
            
            # === Training vs Inference 분기 ===
            if self.training:
                # ============ Target 계산: Full thermal을 Transformer 통과 (no_grad) ============
                with torch.no_grad():
                    cls_full = self.thermal_backbone.cls_token.expand(B, -1, -1)
                    cls_full = cls_full + self.thermal_backbone.pos_embed[:, :1, :]
                    x_full = torch.cat([cls_full, x_patch], dim=1)  # [B, 257, 768]
                    
                    for blk in self.thermal_backbone.blocks:
                        x_full = blk(x_full)
                    
                    x_full_norm = self.thermal_backbone.norm(x_full)
                    target_features = x_full_norm[:, 1:, :]  # [B, 256, 768]
                
                # ============ Masking 적용 - 시퀀스 길이 유지! ============
                masks = self.mask_generator(x_patch)  # [B, 256]
                
                # Masked positions를 0으로 채움 (시퀀스 길이 유지)
                x_patch_masked = x_patch.clone()
                for i in range(B):
                    x_patch_masked[i, masks[i]] = 0  # Masked positions를 0으로
                
                # CLS token 추가 - 항상 257 tokens!
                cls_token = self.thermal_backbone.cls_token.expand(B, -1, -1)
                cls_pos = self.thermal_backbone.pos_embed[:, :1, :]
                cls_token = cls_token + cls_pos
                x_with_cls = torch.cat([cls_token, x_patch_masked], dim=1)  # [B, 257, 768]
                
                # Encoder 통과 - Full 257 tokens
                for blk in self.thermal_backbone.blocks:
                    x_with_cls = blk(x_with_cls)
                x_norm = self.thermal_backbone.norm(x_with_cls)
                
                # ============ Decoder로 복원 (RGB 참조) ============
                if rgb_reference is not None:
                    # Decoder에 Full 257 tokens 전달!
                    pred_features = self.thermal_decoder(
                        x_norm,  # [B, 257, 768] - Full sequence!
                        masks, 
                        rgb_reference
                    )
                                    
                    recon_loss = self.compute_reconstruction_loss(
                        pred_features,      # [B, 256, 768]
                        target_features,    # [B, 256, 768]
                        masks
                    )
                    
                    self.last_recon_loss = recon_loss.item()

                    if x.size(0) > 0:
                        self.vis_data = {
                            'thermal_input': x[0].detach().cpu(),
                            'target_features': target_features[0].detach().cpu(),
                            'pred_features': pred_features[0].detach().cpu(),
                            'masks': masks[0].detach().cpu(),
                            'rgb_reference_img': save_rgb_img[0].detach().cpu() if save_rgb_img is not None else None,
                        }
                else:
                    recon_loss = None
                
                # ============ Aggregation에 Encoder output 직접 사용 ============
                # Gradient가 aggregation까지 흐름!
                cls_token_final = x_full_norm[:, 0]  # Target CLS
                full_patches = target_features  # Target features (깨끗한!)
                    
            else:
                # Inference: Masking 없이 전체 패치 사용
                recon_loss = None
                
                cls_token = self.thermal_backbone.cls_token.expand(B, -1, -1)
                cls_pos = self.thermal_backbone.pos_embed[:, :1, :]
                cls_token = cls_token + cls_pos
                x_with_cls = torch.cat([cls_token, x_patch], dim=1)  # [B, 257, 768]
                
                for blk in self.thermal_backbone.blocks:
                    x_with_cls = blk(x_with_cls)
                x_norm = self.thermal_backbone.norm(x_with_cls)
                
                cls_token_final = x_norm[:, 0]
                full_patches = x_norm[:, 1:]  # [B, 256, 768]
            
            out = {
                'x_norm_clstoken': cls_token_final,
                'x_norm_patchtokens': full_patches,
                'cls_attention': None
            }
            
        else:
            raise ValueError("Modality must be 'rgb' or 'thermal'")
        
        # === 공통 처리: Aggregation ===
        patch_tokens = out["x_norm_patchtokens"]
        cls_token = out["x_norm_clstoken"]
        cls_attn_map = out.get("cls_attention", None)
        
        B, N, D = patch_tokens.shape
        H_feat = W_feat = int(math.sqrt(N))
        x_feat = patch_tokens.permute(0, 2, 1).view(B, D, H_feat, W_feat)
        
        global_desc = agg_layer(x_feat)
        
        return global_desc, patch_tokens, cls_token, cls_attn_map, recon_loss

    def forward(self, x, aligned_x, flags):
        """
        Main forward pass
        
        CroCo 수정 사항:
        1. RGB를 먼저 처리해서 patch features 저장
        2. Thermal 처리 시 RGB features를 decoder에 전달
        """
        is_rgb = torch.tensor([f == 'rgb' for f in flags], device=x.device)
        final_emb = torch.zeros((x.size(0), self.output_dim), device=x.device)
        total_recon_loss = 0.0
        
        # 전체 batch에 대한 patch embeddings 저장
        patch_emb = torch.zeros((x.size(0), 256, 768), device=x.device)
        
        # Step 1: RGB 먼저 처리
        if is_rgb.any():
            final_emb[is_rgb], rgb_patches, cls_token, cls_attn_map, recon_loss = \
                self.forward_model(x[is_rgb], 'rgb')
            patch_emb[is_rgb] = rgb_patches  # 전체 batch 기준으로 저장
        
        # Step 2: Thermal 처리
        if aligned_x is None:
            if (~is_rgb).any():
                final_emb[~is_rgb], thermal_patches, cls_token, cls_attn_map, recon_loss = \
                    self.forward_model(x[~is_rgb], 'thermal')
                patch_emb[~is_rgb] = thermal_patches
        else:
            # aligned RGB image의 embedding 추출
            with torch.no_grad():
                _, aligned_x_embed, _, _, _ = self.forward_model(aligned_x, 'rgb')

            if (~is_rgb).any():
                final_emb[~is_rgb], thermal_patches, cls_token, cls_attn_map, recon_loss = \
                    self.forward_model(
                        x[~is_rgb], 
                        'thermal', 
                        rgb_reference=aligned_x_embed,
                        save_rgb_img=aligned_x,
                    )
                patch_emb[~is_rgb] = thermal_patches
                    
                if recon_loss is not None:
                    total_recon_loss = recon_loss
        
        # Training vs Inference 구분
        if self.training:
            return final_emb, total_recon_loss
        else:
            return final_emb

def get_backbone(pretrained_foundation, foundation_model_path, use_adapter=False):
    """DINOv2 ViT-B/14 backbone 로드"""
    backbone = vit_base(patch_size=14, img_size=518, init_values=1, block_chunks=0, use_adapter=use_adapter)
    if pretrained_foundation:
        assert foundation_model_path is not None, "Please specify foundation model path."
        model_dict = backbone.state_dict()
        state_dict = torch.load(foundation_model_path)
        model_dict.update(state_dict.items())
        backbone.load_state_dict(model_dict)
    return backbone