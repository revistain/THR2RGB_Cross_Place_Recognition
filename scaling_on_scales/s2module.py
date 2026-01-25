# S2WrapperMIM.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class S2WrapperMIM(nn.Module):
    """
    Scaling on Scales + MIM support for ViT-S
    
    Normal mode:
        - Track 1: 1/4 downsample → full encoder
        - Track 2: 4-split → full encoder each → spatial concat
        - Channel concat → [B, N, 2D]
    
    MIM mode:
        - Track 1 (1/4) 기준으로 mask 생성
        - Track 2 (4-split)의 동일 spatial location masking
        - 각각 full encoder 통과 (masked patches는 mask token)
        - Channel concat → [B, N, 2D]
    """
    def __init__(self, vit_model, target_size=(476, 644), use_mim=False, mask_ratio=0.75):
        super().__init__()
        self.vit = vit_model
        self.target_size = target_size  # (H, W)
        self.use_mim = use_mim
        self.mask_ratio = mask_ratio
        
        # ViT 설정
        self.patch_size = self._get_patch_size()
        self.embed_dim = vit_model.embed_dim
        
        # Patch grid (원본 크기 기준)
        self.h_patches = target_size[0] // self.patch_size  # 34
        self.w_patches = target_size[1] // self.patch_size  # 46
        self.num_patches = self.h_patches * self.w_patches  # 1564
        
        # 1/4 downsample 기준
        self.h_patches_quarter = self.h_patches // 2  # 17
        self.w_patches_quarter = self.w_patches // 2  # 23
        self.num_patches_quarter = self.h_patches_quarter * self.w_patches_quarter  # 391
        
        # 검증
        assert target_size[0] % self.patch_size == 0, \
            f"Height {target_size[0]} not divisible by patch_size {self.patch_size}"
        assert target_size[1] % self.patch_size == 0, \
            f"Width {target_size[1]} not divisible by patch_size {self.patch_size}"
        
        # MIM용 mask token
        if use_mim:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            torch.nn.init.normal_(self.mask_token, std=0.02)
    
    def _get_patch_size(self):
        """Patch size 추출"""
        ps = self.vit.patch_embed.patch_size
        return ps[0] if isinstance(ps, (tuple, list)) else ps
    
    def generate_mask_hierarchy(self, B, device):
        """
        Track 1 (1/4 크기) 기준으로 mask 생성
        → Track 2 (원본 크기 4분할)의 동일 spatial location masking
        
        Returns:
            mask_quarter: [B, N_quarter] - Track 1용 (391)
            mask_full: [B, N_full] - Track 2용 (1564, 4배 확장)
            ids_shuffle: [B, N_quarter]
        """
        N_quarter = self.num_patches_quarter
        
        # Random masking
        len_keep = int(N_quarter * (1 - self.mask_ratio))
        noise = torch.rand(B, N_quarter, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        
        # Binary mask (True=masked)
        mask_quarter = torch.ones(B, N_quarter, dtype=torch.bool, device=device)
        mask_quarter = mask_quarter.scatter(1, ids_shuffle[:, :len_keep], False)
        
        # Full resolution으로 spatial 확장 (1→4)
        # [B, 391] → [B, 17, 23] → [B, 34, 46] → [B, 1564]
        mask_quarter_spatial = mask_quarter.reshape(B, self.h_patches_quarter, self.w_patches_quarter)
        mask_full_spatial = mask_quarter_spatial.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)
        mask_full = mask_full_spatial.reshape(B, -1)
        
        return mask_quarter, mask_full, ids_shuffle
    
    def interpolate_pos_encoding(self, pos_embed, h, w):
        """
        Positional embedding을 (h, w) grid로 interpolate
        
        Args:
            pos_embed: [1, N_old, D] (CLS token 제외된 상태)
            h, w: target grid size
        Returns:
            [1, h*w, D]
        """
        N_old = pos_embed.shape[1]
        if N_old == h * w:
            return pos_embed
        
        D = pos_embed.shape[-1]
        
        # ViT-S DINOv2 원본: 518x518 → 37x37 patches
        h_old = w_old = int(math.sqrt(N_old))
        assert h_old * w_old == N_old, f"Original pos_embed not square: {N_old}"
        
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
        전체 positional embedding에서 crop에 해당하는 부분 추출
        
        Args:
            pos_embed_full: [1, h_full*w_full, D]
            crop_idx: 0=TL, 1=TR, 2=BL, 3=BR
        Returns:
            [1, h_half*w_half, D]
        """
        h_full, w_full = self.h_patches, self.w_patches
        h_half, w_half = h_full // 2, w_full // 2
        D = pos_embed_full.shape[-1]
        
        # Spatial로 reshape
        pos_spatial = pos_embed_full.reshape(1, h_full, w_full, D)
        
        # Crop slicing
        if crop_idx == 0:  # top-left
            crop_pos = pos_spatial[:, :h_half, :w_half, :]
        elif crop_idx == 1:  # top-right
            crop_pos = pos_spatial[:, :h_half, w_half:, :]
        elif crop_idx == 2:  # bottom-left
            crop_pos = pos_spatial[:, h_half:, :w_half, :]
        elif crop_idx == 3:  # bottom-right
            crop_pos = pos_spatial[:, h_half:, w_half:, :]
        else:
            raise ValueError(f"Invalid crop_idx: {crop_idx}")
        
        return crop_pos.reshape(1, h_half * w_half, D)
    
    def apply_mask_to_patches(self, patches, mask):
        """
        Patch embeddings에 mask 적용
        
        Args:
            patches: [B, N, D]
            mask: [B, N] boolean (True=masked)
        Returns:
            patches_masked: [B, N, D] - encoder에 넣을 버전 (mask token 삽입)
        """
        B, N, D = patches.shape
        
        if not mask.any():
            return patches
        
        # Masked positions를 mask token으로 교체
        patches_masked = patches.clone()
        mask_expanded = mask.unsqueeze(-1).expand_as(patches)
        mask_token_expanded = self.mask_token.expand(B, N, D)
        patches_masked = torch.where(mask_expanded, mask_token_expanded, patches)
        
        return patches_masked
    
    def forward_encoder_with_cls(self, patches, extract_attention=False):
        """
        CLS token 추가 → Full Encoder 통과
        
        Args:
            patches: [B, N, D] - positional embedding 이미 추가된 상태
            extract_attention: attention map 추출 여부
        Returns:
            features: [B, N, D] - CLS 제거된 patch features
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
            # 마지막 block 전에 저장
            if i == len(self.vit.blocks) - 1:
                penultimate = x.clone()
            
            x = blk(x)
            
            # 마지막 block의 attention 추출
            if i == len(self.vit.blocks) - 1 and extract_attention:
                if hasattr(blk, 'attn'):
                    B_, N_, C = x.shape
                    qkv = blk.attn.qkv(blk.norm1(x))
                    qkv = qkv.reshape(B_, N_, 3, blk.attn.num_heads, C // blk.attn.num_heads).permute(2, 0, 3, 1, 4)
                    q, k, v = qkv[0], qkv[1], qkv[2]
                    
                    attn = (q @ k.transpose(-2, -1)) * (C // blk.attn.num_heads) ** -0.5
                    attn = attn.softmax(dim=-1)
                    last_attn = attn  # [B, num_heads, N+1, N+1]
        
        # Norm
        x = self.vit.norm(x)
        if penultimate is not None:
            penultimate = self.vit.norm(penultimate)
        
        # CLS 제거
        features = x[:, 1:, :]  # [B, N, D]
        penultimate_patches = penultimate[:, 1:, :] if penultimate is not None else None
        
        return features, last_attn, penultimate_patches
    
    def forward_track1(self, x, mask=None, return_attention=False):
        """
        Track 1: 1/4 downsample → full encoder
        
        Args:
            x: [B, 3, H, W]
            mask: [B, N_quarter] boolean or None
            return_attention: attention 추출 여부
        Returns:
            features: [B, N_quarter, D]
            attention: [B, num_heads, N_quarter+1, N_quarter+1] or None
            penultimate: [B, N_quarter, D] or None
        """
        B, _, H, W = x.shape
        
        # 1/4 downsample (scale_factor=0.5)
        x_quarter = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        
        # Patch embedding
        patches = self.vit.patch_embed(x_quarter)  # [B, N_quarter, D]
        
        # Positional embedding
        if hasattr(self.vit, 'pos_embed'):
            pos_embed = self.vit.pos_embed[:, 1:, :]  # CLS 제외
            pos_embed = self.interpolate_pos_encoding(
                pos_embed, 
                self.h_patches_quarter, 
                self.w_patches_quarter
            )
            patches = patches + pos_embed
        
        # Masking (MIM mode)
        if self.use_mim and mask is not None:
            patches = self.apply_mask_to_patches(patches, mask)
        
        # Full encoder 통과
        features, attn, penultimate = self.forward_encoder_with_cls(patches, return_attention)
        
        return features, attn, penultimate
    
    def forward_track2(self, x, mask_full=None, return_attention=False):
        """
        Track 2: 4-split → full encoder each → spatial reassemble
        
        Args:
            x: [B, 3, H, W]
            mask_full: [B, N_full] boolean or None
            return_attention: attention 추출 여부
        Returns:
            list of 4 tuples: (features, attention, penultimate)
        """
        B, _, H, W = x.shape
        h_half, w_half = H // 2, W // 2
        
        # 4-split
        crops = [
            x[:, :, :h_half, :w_half],       # 0: top-left
            x[:, :, :h_half, w_half:],       # 1: top-right
            x[:, :, h_half:, :w_half],       # 2: bottom-left
            x[:, :, h_half:, w_half:]        # 3: bottom-right
        ]
        
        # Full resolution positional embedding 준비
        if hasattr(self.vit, 'pos_embed'):
            pos_embed_full = self.vit.pos_embed[:, 1:, :]
            pos_embed_full = self.interpolate_pos_encoding(
                pos_embed_full,
                self.h_patches,
                self.w_patches
            )
        
        # Mask를 4개 영역으로 split
        crop_masks = [None, None, None, None]
        if self.use_mim and mask_full is not None:
            mask_spatial = mask_full.reshape(B, self.h_patches, self.w_patches)
            h_half_p, w_half_p = self.h_patches // 2, self.w_patches // 2
            
            crop_masks = [
                mask_spatial[:, :h_half_p, :w_half_p].reshape(B, -1),      # 0: TL
                mask_spatial[:, :h_half_p, w_half_p:].reshape(B, -1),      # 1: TR
                mask_spatial[:, h_half_p:, :w_half_p].reshape(B, -1),      # 2: BL
                mask_spatial[:, h_half_p:, w_half_p:].reshape(B, -1)       # 3: BR
            ]
        
        # 각 crop 처리
        crop_results = []
        for i, (crop, crop_mask) in enumerate(zip(crops, crop_masks)):
            # Patch embedding
            patches = self.vit.patch_embed(crop)
            
            # Crop-specific positional embedding
            if hasattr(self.vit, 'pos_embed'):
                pos_embed_crop = self.extract_crop_pos_embed(pos_embed_full, i)
                patches = patches + pos_embed_crop
            
            # Masking (MIM mode)
            if self.use_mim and crop_mask is not None:
                patches = self.apply_mask_to_patches(patches, crop_mask)
            
            # Full encoder 통과
            feat, attn, penult = self.forward_encoder_with_cls(patches, return_attention)
            crop_results.append((feat, attn, penult))
        
        return crop_results
    
    def spatial_reassemble_crops(self, crop_feats):
        """
        4개 crop features를 spatial하게 재조립
        
        Args:
            crop_feats: list of 4 x [B, N_crop, D]
        Returns:
            [B, D, h_full, w_full]
        """
        B = crop_feats[0].shape[0]
        D = crop_feats[0].shape[-1]
        h_crop = self.h_patches // 2
        w_crop = self.w_patches // 2
        
        # [B, N_crop, D] → [B, D, h_crop, w_crop]
        crop_spatial = []
        for feat in crop_feats:
            feat_2d = feat.reshape(B, h_crop, w_crop, D).permute(0, 3, 1, 2)
            crop_spatial.append(feat_2d)
        
        # Spatial concatenation
        top = torch.cat([crop_spatial[0], crop_spatial[1]], dim=3)     # horizontal
        bottom = torch.cat([crop_spatial[2], crop_spatial[3]], dim=3)  # horizontal
        assembled = torch.cat([top, bottom], dim=2)  # vertical → [B, D, h_full, w_full]
        
        return assembled
    
    def merge_attention_maps(self, attn_t1, attn_t2_list):
        """
        Track 1과 Track 2의 CLS attention 병합
        Track 2 우선 사용 (더 고해상도)
        
        Args:
            attn_t1: [B, H, N_quarter+1, N_quarter+1] or None
            attn_t2_list: list of 4 x [B, H, N_crop+1, N_crop+1] or None
        Returns:
            merged_cls_attn: [B, N_full] - CLS attention to all patches
        """
        if attn_t2_list[0] is None:
            return None
        
        B = attn_t2_list[0].shape[0]
        
        # 각 crop의 CLS attention to patches
        crop_cls_attns = []
        for attn_crop in attn_t2_list:
            # [B, H, N_crop+1, N_crop+1] → CLS(0) to patches(1:)
            cls_to_patches = attn_crop[:, :, 0, 1:]  # [B, H, N_crop]
            crop_cls_attns.append(cls_to_patches)
        
        # Spatial reassemble
        h_crop, w_crop = self.h_patches // 2, self.w_patches // 2
        crop_cls_spatial = []
        for cls_attn in crop_cls_attns:
            # [B, H, N_crop] → [B, H, h_crop, w_crop]
            cls_spatial = cls_attn.reshape(B, -1, h_crop, w_crop)
            crop_cls_spatial.append(cls_spatial)
        
        # Concat
        top = torch.cat([crop_cls_spatial[0], crop_cls_spatial[1]], dim=3)
        bottom = torch.cat([crop_cls_spatial[2], crop_cls_spatial[3]], dim=3)
        assembled = torch.cat([top, bottom], dim=2)  # [B, H, h_full, w_full]
        
        # Average over heads, flatten
        cls_attn = assembled.mean(dim=1).flatten(1)  # [B, N_full]
        
        return cls_attn
    
    def forward(self, x, return_attention=False):
        """
        Main forward
        
        Args:
            x: [B, 3, H, W]
            return_attention: attention map 반환 여부
        
        Returns:
            dict with keys:
                'x_norm_patchtokens': [B, N_full, 2D] - channel concat features
                'x_norm_clstoken': None (S2는 CLS 없음)
                'cls_attention': [B, 1, 1, N_full] or None
                'penultimate_norm_patchtokens': [B, N_full, 2D] or None
                'mask_quarter': [B, N_quarter] (MIM only)
                'mask_full': [B, N_full] (MIM only)
        """
        B = x.shape[0]
        
        # Masking (MIM mode)
        mask_quarter, mask_full = None, None
        if self.use_mim:
            mask_quarter, mask_full, ids_shuffle = self.generate_mask_hierarchy(B, x.device)
        
        # Track 1: 1/4 downsample → encoder
        feat_t1, attn_t1, penult_t1 = self.forward_track1(
            x, mask_quarter, return_attention
        )
        
        # Track 2: 4-split → encoder each
        crop_results = self.forward_track2(
            x, mask_full, return_attention
        )
        
        # Unpack Track 2
        crop_feats = [cr[0] for cr in crop_results]
        crop_attns = [cr[1] for cr in crop_results]
        crop_penults = [cr[2] for cr in crop_results]
        
        # Spatial reassemble Track 2
        feat_t2 = self.spatial_reassemble_crops(crop_feats)  # [B, D, h, w]
        
        # Upsample Track 1 to match Track 2 (patch grid level)
        feat_t1_spatial = feat_t1.reshape(
            B, self.h_patches_quarter, self.w_patches_quarter, -1
        ).permute(0, 3, 1, 2)
        
        feat_t1_up = F.interpolate(
            feat_t1_spatial,
            size=(self.h_patches, self.w_patches),  # Patch grid size!
            mode='bilinear',
            align_corners=False
        )
        
        # Channel concat
        features_concat = torch.cat([feat_t1_up, feat_t2], dim=1)  # [B, 2D, h, w]
        
        # Flatten to tokens
        features_tokens = features_concat.flatten(2).permute(0, 2, 1)  # [B, N_full, 2D]
        
        # Output dict
        output = {
            'x_norm_patchtokens': features_tokens,
            'x_norm_clstoken': None,  # S2는 CLS 없음
        }
        
        # MIM 정보
        if self.use_mim:
            output['mask_quarter'] = mask_quarter
            output['mask_full'] = mask_full
        
        # Attention 처리
        if return_attention:
            cls_attn = self.merge_attention_maps(attn_t1, crop_attns)
            if cls_attn is not None:
                # [B, 1, 1, N_full] 형태로 (기존 코드 호환)
                output['cls_attention'] = cls_attn.unsqueeze(1).unsqueeze(1)
            
            # Penultimate features
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
            else:
                output['penultimate_norm_patchtokens'] = features_tokens
        
        return output