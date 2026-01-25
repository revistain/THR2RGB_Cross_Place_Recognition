# S2Wrapper.py (MIM 제거 버전)
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class S2Wrapper(nn.Module):
    """
    Scaling on Scales (MIM 없는 순수 버전)
    
    - Track 1: 1/4 downsample → encoder → upsample
    - Track 2: 4-split → encoder each → spatial concat
    - Channel concat → [B, N, 2D]
    """
    def __init__(self, vit_model, target_size=(476, 644)):
        super().__init__()
        self.vit = vit_model
        self.target_size = target_size  # (H, W)
        
        # ViT 설정
        self.patch_size = self._get_patch_size()
        self.embed_dim = vit_model.embed_dim
        
        # Patch grid 계산
        self.h_patches = target_size[0] // self.patch_size  # 34
        self.w_patches = target_size[1] // self.patch_size  # 46
        self.num_patches = self.h_patches * self.w_patches  # 1564
        
        # 1/4 downsample 기준
        self.h_patches_quarter = self.h_patches // 2  # 17
        self.w_patches_quarter = self.w_patches // 2  # 23
        self.num_patches_quarter = self.h_patches_quarter * self.w_patches_quarter  # 391
        
        # 검증
        assert target_size[0] % self.patch_size == 0
        assert target_size[1] % self.patch_size == 0
    
    def _get_patch_size(self):
        """Patch size 추출"""
        ps = self.vit.patch_embed.patch_size
        return ps[0] if isinstance(ps, (tuple, list)) else ps
    
    def interpolate_pos_encoding(self, pos_embed, h, w):
        """
        Positional embedding을 (h, w) grid로 interpolate
        
        Args:
            pos_embed: [1, N_old, D]
            h, w: target grid size
        Returns:
            [1, h*w, D]
        """
        N_old = pos_embed.shape[1]
        if N_old == h * w:
            return pos_embed
        
        D = pos_embed.shape[-1]
        h_old = w_old = int(math.sqrt(N_old))
        
        # Spatial reshape
        pos_embed_spatial = pos_embed.reshape(1, h_old, w_old, D).permute(0, 3, 1, 2)
        
        # Bicubic interpolation
        pos_embed_new = F.interpolate(
            pos_embed_spatial, 
            size=(h, w), 
            mode='bicubic', 
            align_corners=False
        )
        
        pos_embed_new = pos_embed_new.permute(0, 2, 3, 1).reshape(1, h * w, D)
        return pos_embed_new
    
    def extract_crop_pos_embed(self, pos_embed_full, crop_idx):
        """
        전체 pos embed에서 crop 부분 추출
        
        Args:
            pos_embed_full: [1, h_full*w_full, D]
            crop_idx: 0=TL, 1=TR, 2=BL, 3=BR
        Returns:
            [1, h_half*w_half, D]
        """
        h_full, w_full = self.h_patches, self.w_patches
        h_half, w_half = h_full // 2, w_full // 2
        D = pos_embed_full.shape[-1]
        
        pos_spatial = pos_embed_full.reshape(1, h_full, w_full, D)
        
        if crop_idx == 0:  # top-left
            crop_pos = pos_spatial[:, :h_half, :w_half, :]
        elif crop_idx == 1:  # top-right
            crop_pos = pos_spatial[:, :h_half, w_half:, :]
        elif crop_idx == 2:  # bottom-left
            crop_pos = pos_spatial[:, h_half:, :w_half, :]
        else:  # bottom-right
            crop_pos = pos_spatial[:, h_half:, w_half:, :]
        
        return crop_pos.reshape(1, h_half * w_half, D)
    
    def forward_encoder(self, patches, return_attention=False):
        """
        CLS token 추가 → Encoder 통과
        
        Args:
            patches: [B, N, D]
            return_attention: bool
        Returns:
            features: [B, N, D]
            cls_attn: [B, num_heads, N+1, N+1] or None
            penultimate: [B, N, D] or None
        """
        B, N, D = patches.shape
        
        # CLS token 추가
        cls_token = self.vit.cls_token.expand(B, -1, -1)
        if hasattr(self.vit, 'pos_embed'):
            cls_pos = self.vit.pos_embed[:, :1, :]
            cls_token = cls_token + cls_pos
        
        x = torch.cat([cls_token, patches], dim=1)  # [B, N+1, D]
        
        # Transformer blocks
        penultimate = None
        last_attn = None
        
        for i, blk in enumerate(self.vit.blocks):
            if i == len(self.vit.blocks) - 1:
                penultimate = x.clone()
            
            x = blk(x)
            
            # 마지막 block attention 추출
            if i == len(self.vit.blocks) - 1 and return_attention:
                if hasattr(blk, 'attn'):
                    B_, N_, C = x.shape
                    qkv = blk.attn.qkv(blk.norm1(x))
                    qkv = qkv.reshape(B_, N_, 3, blk.attn.num_heads, C // blk.attn.num_heads).permute(2, 0, 3, 1, 4)
                    q, k, v = qkv[0], qkv[1], qkv[2]
                    
                    attn = (q @ k.transpose(-2, -1)) * (C // blk.attn.num_heads) ** -0.5
                    attn = attn.softmax(dim=-1)
                    last_attn = attn
        
        # Norm
        x = self.vit.norm(x)
        if penultimate is not None:
            penultimate = self.vit.norm(penultimate)
        
        # CLS 제거
        features = x[:, 1:, :]
        penultimate_patches = penultimate[:, 1:, :] if penultimate is not None else None
        
        return features, last_attn, penultimate_patches
    
    def forward_track1(self, x, return_attention=False):
        """
        Track 1: 1/4 downsample → encoder
        
        Returns:
            features: [B, N_quarter, D]
            attention: [B, H, N_quarter+1, N_quarter+1] or None
            penultimate: [B, N_quarter, D] or None
        """
        B, _, H, W = x.shape
        
        # 1/4 downsample
        x_quarter = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        
        # Patch embedding
        patches = self.vit.patch_embed(x_quarter)  # [B, N_quarter, D]
        
        # Positional embedding
        if hasattr(self.vit, 'pos_embed'):
            pos_embed = self.vit.pos_embed[:, 1:, :]
            pos_embed = self.interpolate_pos_encoding(
                pos_embed, 
                self.h_patches_quarter, 
                self.w_patches_quarter
            )
            patches = patches + pos_embed
        
        # Encoder 통과
        features, attn, penultimate = self.forward_encoder(patches, return_attention)
        
        return features, attn, penultimate
    
    def forward_track2(self, x, return_attention=False):
        """
        Track 2: 4-split → encoder each
        
        Returns:
            list of 4 tuples: (features, attention, penultimate)
        """
        B, _, H, W = x.shape
        h_half, w_half = H // 2, W // 2
        
        # 4-split
        crops = [
            x[:, :, :h_half, :w_half],       # 0: TL
            x[:, :, :h_half, w_half:],       # 1: TR
            x[:, :, h_half:, :w_half],       # 2: BL
            x[:, :, h_half:, w_half:]        # 3: BR
        ]
        
        # Full resolution pos embed 준비
        if hasattr(self.vit, 'pos_embed'):
            pos_embed_full = self.vit.pos_embed[:, 1:, :]
            pos_embed_full = self.interpolate_pos_encoding(
                pos_embed_full,
                self.h_patches,
                self.w_patches
            )
        
        # 각 crop 처리
        crop_results = []
        for i, crop in enumerate(crops):
            # Patch embedding
            patches = self.vit.patch_embed(crop)
            
            # Crop pos embed
            if hasattr(self.vit, 'pos_embed'):
                pos_embed_crop = self.extract_crop_pos_embed(pos_embed_full, i)
                patches = patches + pos_embed_crop
            
            # Encoder 통과
            feat, attn, penult = self.forward_encoder(patches, return_attention)
            crop_results.append((feat, attn, penult))
        
        return crop_results
    
    def spatial_reassemble_crops(self, crop_feats):
        """
        4개 crop features를 spatial 재조립
        
        Args:
            crop_feats: list of 4 x [B, N_crop, D]
        Returns:
            [B, D, h_full, w_full]
        """
        B = crop_feats[0].shape[0]
        D = crop_feats[0].shape[-1]
        h_crop = self.h_patches // 2
        w_crop = self.w_patches // 2
        
        # [B, N, D] → [B, D, h, w]
        crop_spatial = []
        for feat in crop_feats:
            feat_2d = feat.reshape(B, h_crop, w_crop, D).permute(0, 3, 1, 2)
            crop_spatial.append(feat_2d)
        
        # Spatial concat
        top = torch.cat([crop_spatial[0], crop_spatial[1]], dim=3)
        bottom = torch.cat([crop_spatial[2], crop_spatial[3]], dim=3)
        assembled = torch.cat([top, bottom], dim=2)  # [B, D, h_full, w_full]
        
        return assembled
    
    def merge_attention_maps(self, attn_t1, attn_t2_list):
        """
        Track2 CLS attention 병합
        
        Returns:
            [B, N_full] or None
        """
        if attn_t2_list[0] is None:
            return None
        
        B = attn_t2_list[0].shape[0]
        
        # 각 crop의 CLS attention
        crop_cls_attns = []
        for attn_crop in attn_t2_list:
            cls_to_patches = attn_crop[:, :, 0, 1:]  # [B, H, N_crop]
            crop_cls_attns.append(cls_to_patches)
        
        # Spatial reassemble
        h_crop, w_crop = self.h_patches // 2, self.w_patches // 2
        crop_cls_spatial = []
        for cls_attn in crop_cls_attns:
            cls_spatial = cls_attn.reshape(B, -1, h_crop, w_crop)
            crop_cls_spatial.append(cls_spatial)
        
        top = torch.cat([crop_cls_spatial[0], crop_cls_spatial[1]], dim=3)
        bottom = torch.cat([crop_cls_spatial[2], crop_cls_spatial[3]], dim=3)
        assembled = torch.cat([top, bottom], dim=2)  # [B, H, h_full, w_full]
        
        # Average over heads, flatten
        cls_attn = assembled.mean(dim=1).flatten(1)  # [B, N_full]
        
        return cls_attn
    
    def forward(self, x, return_attention=False):
        """
        Main forward
        """
        B = x.shape[0]
        
        # Track 1, 2 처리
        feat_t1, attn_t1, penult_t1 = self.forward_track1(x, return_attention)
        crop_results = self.forward_track2(x, return_attention)
        
        crop_feats = [cr[0] for cr in crop_results]
        crop_attns = [cr[1] for cr in crop_results]
        crop_penults = [cr[2] for cr in crop_results]
        
        feat_t2 = self.spatial_reassemble_crops(crop_feats)
        
        feat_t1_spatial = feat_t1.reshape(
            B, self.h_patches_quarter, self.w_patches_quarter, -1
        ).permute(0, 3, 1, 2)
        
        feat_t1_up = F.interpolate(
            feat_t1_spatial,
            size=(self.h_patches, self.w_patches),
            mode='bilinear',
            align_corners=False
        )
        
        features_concat = torch.cat([feat_t1_up, feat_t2], dim=1)
        features_tokens = features_concat.flatten(2).permute(0, 2, 1)
        
        output = {
            'x_norm_patchtokens': features_tokens,
            'x_norm_clstoken': None,
        }
        
        # ========== [수정] Attention 반환 형식 변경 ==========
        if return_attention:
            cls_attn = self.merge_attention_maps(attn_t1, crop_attns)
            if cls_attn is not None:
                # 기존: [B, 1, 1, N_full]
                # 수정: [B, num_heads, 1, N_full] - 기존 ViT 형식과 유사하게
                # 하지만 실제로는 이미 head-averaged되어 있으므로
                # [B, 1, N_full]로 반환하고 network.py에서 처리
                output['cls_attention'] = cls_attn.unsqueeze(1)  # [B, 1, N_full]
            
            # Penultimate
            if penult_t1 is not None and crop_penults[0] is not None:
                penult_t2 = self.spatial_reassemble_crops(crop_penults)
                
                penult_t1_spatial = penult_t1.reshape(
                    B, self.h_patches_quarter, self.w_patches_quarter, -1
                ).permute(0, 3, 1, 2)
                
                penult_t1_up = F.interpolate(
                    penult_t1_spatial,
                    size=(self.h_patches, self.w_patches),
                    mode='bilinear',
                    align_corners=False
                )
                
                penult_concat = torch.cat([penult_t1_up, penult_t2], dim=1)
                penult_tokens = penult_concat.flatten(2).permute(0, 2, 1)
                output['penultimate_norm_patchtokens'] = penult_tokens
        # ====================================================
        
        return output
    