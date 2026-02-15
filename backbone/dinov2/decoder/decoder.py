# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
DINOv2 Decoder with Cross-Attention Adapters

This module implements a decoder that uses frozen DINOv2 components (self-attention + MLP)
with trainable cross-attention adapters inserted between them.

Structure of each decoder block:
    1. Self-Attention (frozen, from pretrained DINO)
    2. Cross-Attention (trainable adapter) with RoPE
    3. MLP (frozen, from pretrained DINO)
"""

import copy
import math
from functools import partial
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from timm.models.layers import DropPath


class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) for 2D vision transformers.

    RoPE encodes position by rotating query and key vectors, enabling better
    extrapolation to unseen positions and capturing relative position information.

    Args:
        dim: Dimension of the embedding (must be divisible by 2)
        max_seq_len: Maximum sequence length (for precomputing frequencies)
        base: Base for the frequency computation (default: 10000)
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 1024,
        base: float = 10000.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

        # Cache for cos and sin values
        self._cos_cache = None
        self._sin_cache = None
        self._cache_seq_len = 0

    def _update_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """Update the cos/sin cache if needed."""
        if seq_len <= self._cache_seq_len and self._cos_cache is not None:
            return

        self._cache_seq_len = max(seq_len, self._cache_seq_len)

        # Create position indices
        t = torch.arange(self._cache_seq_len, device=device, dtype=dtype)

        # Compute frequencies: [seq_len, dim/2]
        freqs = torch.einsum('i,j->ij', t, self.inv_freq.to(device=device, dtype=dtype))

        # Duplicate for full dimension: [seq_len, dim]
        emb = torch.cat([freqs, freqs], dim=-1)

        self._cos_cache = emb.cos()[None, None, :, :]  # [1, 1, seq_len, dim]
        self._sin_cache = emb.sin()[None, None, :, :]

    def _rotate_half(self, x: Tensor) -> Tensor:
        """Rotate half the hidden dims of the input."""
        x1 = x[..., : self.dim // 2]
        x2 = x[..., self.dim // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        seq_len: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Apply rotary position embedding to query and key tensors.

        Args:
            q: Query tensor [B, num_heads, N, head_dim]
            k: Key tensor [B, num_heads, M, head_dim]
            seq_len: Optional sequence length (defaults to max of q and k lengths)

        Returns:
            Tuple of rotated (q, k) tensors
        """
        q_len = q.shape[2]
        k_len = k.shape[2]
        max_len = max(q_len, k_len) if seq_len is None else seq_len

        self._update_cache(max_len, q.device, q.dtype)

        # Get relevant cos/sin values
        cos_q = self._cos_cache[:, :, :q_len, :self.dim].to(q.dtype)
        sin_q = self._sin_cache[:, :, :q_len, :self.dim].to(q.dtype)
        cos_k = self._cos_cache[:, :, :k_len, :self.dim].to(k.dtype)
        sin_k = self._sin_cache[:, :, :k_len, :self.dim].to(k.dtype)

        # Apply rotation
        q_rot = (q * cos_q) + (self._rotate_half(q) * sin_q)
        k_rot = (k * cos_k) + (self._rotate_half(k) * sin_k)

        return q_rot, k_rot


class RotaryPositionEmbedding2D(nn.Module):
    """
    2D Rotary Position Embedding for vision transformers.

    Applies separate RoPE for height and width dimensions, better suited for
    2D image patches than 1D RoPE.

    Args:
        dim: Dimension of the embedding (must be divisible by 4)
        max_h: Maximum height in patches
        max_w: Maximum width in patches
        base: Base for the frequency computation
    """

    def __init__(
        self,
        dim: int,
        max_h: int = 32,
        max_w: int = 32,
        base: float = 10000.0,
    ) -> None:
        super().__init__()
        assert dim % 4 == 0, "dim must be divisible by 4 for 2D RoPE"

        self.dim = dim
        self.max_h = max_h
        self.max_w = max_w
        self.base = base

        # Half the dimensions for each spatial direction
        half_dim = dim // 2

        # Separate inverse frequencies for h and w
        inv_freq = 1.0 / (base ** (torch.arange(0, half_dim, 2).float() / half_dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

        # Cache
        self._cos_cache = None
        self._sin_cache = None
        self._cached_h = 0
        self._cached_w = 0

    def _update_cache(self, h: int, w: int, device: torch.device, dtype: torch.dtype):
        """Update 2D cos/sin cache."""
        if h <= self._cached_h and w <= self._cached_w and self._cos_cache is not None:
            return

        self._cached_h = max(h, self._cached_h)
        self._cached_w = max(w, self._cached_w)

        # Create position grids
        pos_h = torch.arange(self._cached_h, device=device, dtype=dtype)
        pos_w = torch.arange(self._cached_w, device=device, dtype=dtype)

        inv_freq = self.inv_freq.to(device=device, dtype=dtype)

        # Compute frequencies for each dimension
        freqs_h = torch.einsum('i,j->ij', pos_h, inv_freq)  # [H, dim/4]
        freqs_w = torch.einsum('i,j->ij', pos_w, inv_freq)  # [W, dim/4]

        # Expand to all positions: [H, W, dim/4]
        freqs_h = freqs_h[:, None, :].expand(-1, self._cached_w, -1)
        freqs_w = freqs_w[None, :, :].expand(self._cached_h, -1, -1)

        # Flatten spatial dimensions: [H*W, dim/4]
        freqs_h = freqs_h.reshape(-1, freqs_h.shape[-1])
        freqs_w = freqs_w.reshape(-1, freqs_w.shape[-1])

        # Combine: [H*W, dim] with pattern [h_cos, h_sin, w_cos, w_sin]
        emb = torch.cat([freqs_h, freqs_h, freqs_w, freqs_w], dim=-1)

        self._cos_cache = emb.cos()[None, None, :, :]  # [1, 1, H*W, dim]
        self._sin_cache = emb.sin()[None, None, :, :]

    def _rotate_half(self, x: Tensor) -> Tensor:
        """Rotate for 2D: separate rotations for h and w dimensions."""
        quarter = self.dim // 4
        x1 = x[..., :quarter]
        x2 = x[..., quarter:quarter*2]
        x3 = x[..., quarter*2:quarter*3]
        x4 = x[..., quarter*3:]
        return torch.cat([-x2, x1, -x4, x3], dim=-1)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        h: int,
        w: int,
    ) -> Tuple[Tensor, Tensor]:
        """
        Apply 2D rotary position embedding.

        Args:
            q: Query tensor [B, num_heads, N, head_dim]
            k: Key tensor [B, num_heads, M, head_dim]
            h: Height in patches
            w: Width in patches

        Returns:
            Tuple of rotated (q, k) tensors
        """
        seq_len_q = q.shape[2]
        seq_len_k = k.shape[2]

        self._update_cache(h, w, q.device, q.dtype)

        # Get relevant cos/sin values
        cos_q = self._cos_cache[:, :, :seq_len_q, :self.dim].to(q.dtype)
        sin_q = self._sin_cache[:, :, :seq_len_q, :self.dim].to(q.dtype)
        cos_k = self._cos_cache[:, :, :seq_len_k, :self.dim].to(k.dtype)
        sin_k = self._sin_cache[:, :, :seq_len_k, :self.dim].to(k.dtype)

        # Apply rotation
        q_rot = (q * cos_q) + (self._rotate_half(q) * sin_q)
        k_rot = (k * cos_k) + (self._rotate_half(k) * sin_k)

        return q_rot, k_rot


class VanillaAdapter(nn.Module):
    """
    Trainable adapter module (bottleneck MLP).
    Same as the one used in DINO encoder blocks.
    """
    def __init__(
        self,
        fc_in_channels: int,
        in_channels: int,
        skip_connect: bool = False,
    ) -> None:
        super().__init__()
        self.skip_connect = skip_connect
        self.D_fc1 = nn.Linear(fc_in_channels, in_channels)
        self.D_fc2 = nn.Linear(in_channels, fc_in_channels)

    def forward(self, x: Tensor) -> Tensor:
        x0 = self.D_fc1(x)
        x0 = F.relu(x0, inplace=True)
        outputs = self.D_fc2(x0)
        if self.skip_connect:
            outputs = outputs + x
        return outputs


class CrossAttentionAdapter(nn.Module):
    """
    Trainable cross-attention adapter with Rotary Position Embedding (RoPE).

    This adapter enables cross-modal conditioning by attending to features
    from a reference modality (e.g., RGB features when decoding thermal).
    Uses RoPE instead of additive positional embeddings for better
    position encoding.

    Args:
        dim: Feature dimension
        num_heads: Number of attention heads
        drop_path: Drop path rate for regularization
        use_rope: Whether to use RoPE (default: True)
        rope_2d: Whether to use 2D RoPE (default: True for vision)
        max_seq_len: Maximum sequence length for RoPE cache
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 12,
        drop_path: float = 0.0,
        use_rope: bool = True,
        rope_2d: bool = True,
        max_seq_len: int = 1024,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_rope = use_rope
        self.rope_2d = rope_2d

        # Layer norms for query and key/value
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

        # Separate projections for Q, K, V (needed for RoPE)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        # RoPE
        if use_rope:
            if rope_2d:
                # For 2D, we need head_dim divisible by 4
                assert self.head_dim % 4 == 0, f"head_dim={self.head_dim} must be divisible by 4 for 2D RoPE"
                self.rope = RotaryPositionEmbedding2D(
                    dim=self.head_dim,
                    max_h=32,  # Supports up to 32x32 patches
                    max_w=32,
                )
            else:
                self.rope = RotaryPositionEmbedding(
                    dim=self.head_dim,
                    max_seq_len=max_seq_len,
                )
        else:
            self.rope = None

        # Drop path for regularization
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self._reset_parameters()

    def _reset_parameters(self):
        # Initialize projections with xavier uniform
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.q_proj.bias)
        nn.init.zeros_(self.k_proj.bias)
        nn.init.zeros_(self.v_proj.bias)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        x: Tensor,
        ref: Tensor,
        return_attention: bool = False,
        spatial_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass of cross-attention adapter with RoPE.

        Args:
            x: Query features from target modality [B, N, D]
            ref: Key/Value features from reference modality [B, M, D]
            return_attention: Whether to return attention weights
            spatial_size: Optional (H, W) tuple for 2D RoPE. If None, assumes square.

        Returns:
            Tuple of (output features, optional attention weights)
        """
        B, N, _ = x.shape
        M = ref.shape[1]

        # Normalize inputs
        q_in = self.norm_q(x)
        kv_in = self.norm_kv(ref)

        # Project to Q, K, V
        q = self.q_proj(q_in)
        k = self.k_proj(kv_in)
        v = self.v_proj(kv_in)

        # Reshape for multi-head attention: [B, num_heads, seq_len, head_dim]
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, M, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K
        if self.rope is not None:
            if self.rope_2d:
                # Determine spatial size
                if spatial_size is not None:
                    h, w = spatial_size
                else:
                    # Assume square spatial arrangement (excluding CLS token if present)
                    h = w = int(math.sqrt(N))
                q, k = self.rope(q, k, h, w)
            else:
                q, k = self.rope(q, k)

        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Apply attention to values
        attn_out = torch.matmul(attn_weights, v)

        # Reshape back: [B, N, D]
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N, self.dim)

        # Output projection
        attn_out = self.out_proj(attn_out)

        # Residual connection with drop path
        out = x + self.drop_path(attn_out)

        if return_attention:
            # Average attention weights across heads
            avg_attn = attn_weights.mean(dim=1)
            return out, avg_attn
        return out, None


class DinoDecoderBlock(nn.Module):
    """
    Decoder block using pretrained DINOv2 components with trainable cross-attention.

    Structure:
        x = x + self_attn(norm1(x))      # Frozen, from DINO
        x = x + cross_attn(x, ref)        # Trainable adapter with RoPE
        x = x + mlp(norm2(x))             # Frozen, from DINO

    Args:
        dino_block: Pre-trained DINOv2 block (will extract self-attn and MLP)
        dim: Feature dimension
        num_heads: Number of attention heads for cross-attention
        drop_path: Drop path rate for cross-attention adapter
        use_rope: Whether to use RoPE in cross-attention
        rope_2d: Whether to use 2D RoPE
    """

    def __init__(
        self,
        dino_block: nn.Module,
        dim: int,
        num_heads: int = 12,
        drop_path: float = 0.0,
        use_rope: bool = True,
        rope_2d: bool = True,
    ) -> None:
        super().__init__()

        # ============ Self-Attention components (frozen, from DINO) ============
        self.norm1 = dino_block.norm1
        self.attn = dino_block.attn
        self.ls1 = dino_block.ls1
        self.drop_path1 = dino_block.drop_path1

        # ============ Cross-Attention (trainable adapter with RoPE) ============
        self.cross_attn_adapter = CrossAttentionAdapter(
            dim=dim,
            num_heads=num_heads,
            drop_path=drop_path,
            use_rope=use_rope,
            rope_2d=rope_2d,
        )

        # ============ MLP components (frozen, from DINO) ============
        self.norm2 = dino_block.norm2
        self.mlp = dino_block.mlp
        self.ls2 = dino_block.ls2
        self.drop_path2 = dino_block.drop_path2

        # ============ Vanilla Adapter (trainable, like in encoder) ============
        # Bottleneck MLP adapter added to MLP output
        self.adapter = VanillaAdapter(dim, dim // 2)
        self.adapter_drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # Store layer index for debugging
        self.layer_idx = None

    def freeze_dino_components(self):
        """Freeze all DINO components (self-attention and MLP).

        Note: VanillaAdapter and CrossAttentionAdapter remain trainable.
        """
        # Freeze self-attention
        for param in self.norm1.parameters():
            param.requires_grad = False
        for param in self.attn.parameters():
            param.requires_grad = False
        if hasattr(self.ls1, 'parameters'):
            for param in self.ls1.parameters():
                param.requires_grad = False

        # Freeze MLP
        for param in self.norm2.parameters():
            param.requires_grad = False
        for param in self.mlp.parameters():
            param.requires_grad = False
        if hasattr(self.ls2, 'parameters'):
            for param in self.ls2.parameters():
                param.requires_grad = False

        # Note: adapter (VanillaAdapter) remains trainable - NOT frozen

    def unfreeze_dino_components(self):
        """Unfreeze all DINO components."""
        for param in self.norm1.parameters():
            param.requires_grad = True
        for param in self.attn.parameters():
            param.requires_grad = True
        if hasattr(self.ls1, 'parameters'):
            for param in self.ls1.parameters():
                param.requires_grad = True
        for param in self.norm2.parameters():
            param.requires_grad = True
        for param in self.mlp.parameters():
            param.requires_grad = True
        if hasattr(self.ls2, 'parameters'):
            for param in self.ls2.parameters():
                param.requires_grad = True
        if self.adapter is not None:
            for param in self.adapter.parameters():
                param.requires_grad = True

    def forward(
        self,
        x: Tensor,
        ref: Tensor,
        return_attention: bool = False,
        spatial_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass: self-attn → cross-attn → MLP

        Args:
            x: Input features [B, N, D]
            ref: Reference features for cross-attention [B, M, D]
            return_attention: Whether to return cross-attention weights
            spatial_size: Optional (H, W) tuple for 2D RoPE

        Returns:
            Tuple of (output features, optional cross-attention weights)
        """
        # 1. Self-Attention (frozen, from DINO)
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))

        # 2. Cross-Attention (trainable adapter with RoPE)
        x, attn_weights = self.cross_attn_adapter(x, ref, return_attention, spatial_size)

        # 3. MLP (frozen, from DINO)
        mlp_out = self.mlp(self.norm2(x))
        if self.adapter is not None and self.adapter_drop_path is not None:
            # Include original adapter if it exists
            mlp_out = mlp_out + self.adapter_drop_path(0.2 * self.adapter(self.norm2(x)))
        x = x + self.drop_path2(self.ls2(mlp_out))

        return x, attn_weights


class DINOv2Decoder(nn.Module):
    """
    Full decoder using DINOv2 blocks (layers 3-9) with cross-attention adapters.

    Each block has the structure:
        1. Self-Attention (frozen, pretrained DINO weights)
        2. Cross-Attention (trainable adapter with RoPE)
        3. MLP (frozen, pretrained DINO weights)

    Args:
        dim: Feature dimension (768 for ViT-Base, 384 for ViT-Small)
        num_heads: Number of attention heads
        layer_indices: Which DINO layers to use (default: [3,4,5,6,7,8,9])
        drop_path_rate: Drop path rate for cross-attention adapters
        use_rope: Whether to use RoPE in cross-attention (default: True)
        rope_2d: Whether to use 2D RoPE for vision (default: True)
    """

    def __init__(
        self,
        dim: int = 768,
        num_heads: int = 12,
        layer_indices: List[int] = None,
        drop_path_rate: float = 0.0,
        use_rope: bool = True,
        rope_2d: bool = True,
    ) -> None:
        super().__init__()

        if layer_indices is None:
            layer_indices = [3, 4, 5, 6, 7, 8, 9]

        self.dim = dim
        self.num_heads = num_heads
        self.layer_indices = layer_indices
        self.num_blocks = len(layer_indices)
        self.use_rope = use_rope
        self.rope_2d = rope_2d

        # Placeholder for decoder blocks (will be populated by from_encoder)
        self.blocks = nn.ModuleList()

        # Final layer norm
        self.norm = nn.LayerNorm(dim)

        # Drop path rates for each block
        if drop_path_rate > 0:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.num_blocks)]
        else:
            dpr = [0.0] * self.num_blocks
        self.drop_path_rates = dpr

    @classmethod
    def from_encoder(
        cls,
        encoder: nn.Module,
        layer_range: Tuple[int, int] = (3, 10),
        num_heads: int = None,
        drop_path_rate: float = 0.0,
        use_rope: bool = True,
        rope_2d: bool = True,
    ) -> 'DINOv2Decoder':
        """
        Create decoder by extracting DINO block components from encoder.

        Args:
            encoder: DINOv2 encoder (DinoVisionTransformer)
            layer_range: Range of layers to use (start, end), e.g., (3, 10) for layers 3-9
            num_heads: Number of attention heads (defaults to encoder's num_heads)
            drop_path_rate: Drop path rate for cross-attention adapters
            use_rope: Whether to use RoPE in cross-attention (default: True)
            rope_2d: Whether to use 2D RoPE for vision (default: True)

        Returns:
            DINOv2Decoder instance with frozen DINO components + trainable cross-attention with RoPE
        """
        start_layer, end_layer = layer_range
        layer_indices = list(range(start_layer, end_layer))

        # Get dimension and num_heads from encoder
        dim = encoder.embed_dim
        if num_heads is None:
            num_heads = encoder.num_heads

        # Create decoder instance
        decoder = cls(
            dim=dim,
            num_heads=num_heads,
            layer_indices=layer_indices,
            drop_path_rate=drop_path_rate,
            use_rope=use_rope,
            rope_2d=rope_2d,
        )

        # Create decoder blocks from encoder blocks
        for i, layer_idx in enumerate(layer_indices):
            # Deep copy the DINO block to extract its components
            dino_block = copy.deepcopy(encoder.blocks[layer_idx])

            # Create decoder block with DINO components + cross-attention with RoPE
            decoder_block = DinoDecoderBlock(
                dino_block=dino_block,
                dim=dim,
                num_heads=num_heads,
                drop_path=decoder.drop_path_rates[i],
                use_rope=use_rope,
                rope_2d=rope_2d,
            )
            decoder_block.layer_idx = layer_idx

            decoder.blocks.append(decoder_block)

        # Freeze DINO components by default
        decoder.freeze_dino_blocks()

        return decoder

    def freeze_dino_blocks(self):
        """Freeze all DINO components (only cross-attention remains trainable)."""
        for block in self.blocks:
            block.freeze_dino_components()

    def unfreeze_dino_blocks(self):
        """Unfreeze all DINO components."""
        for block in self.blocks:
            block.unfreeze_dino_components()

    def get_trainable_params(self) -> List[nn.Parameter]:
        """Get list of trainable parameters (cross-attention + vanilla adapters)."""
        params = []
        for block in self.blocks:
            params.extend(block.cross_attn_adapter.parameters())
            if block.adapter is not None:
                params.extend(block.adapter.parameters())
        params.extend(self.norm.parameters())
        return params

    def get_num_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.get_trainable_params())

    def get_num_frozen_params(self) -> int:
        """Get number of frozen parameters (DINO components)."""
        total = 0
        for block in self.blocks:
            # Self-attention components
            total += sum(p.numel() for p in block.norm1.parameters())
            total += sum(p.numel() for p in block.attn.parameters())
            if hasattr(block.ls1, 'parameters'):
                total += sum(p.numel() for p in block.ls1.parameters())
            # MLP components
            total += sum(p.numel() for p in block.norm2.parameters())
            total += sum(p.numel() for p in block.mlp.parameters())
            if hasattr(block.ls2, 'parameters'):
                total += sum(p.numel() for p in block.ls2.parameters())
            # Adapter if exists
            if block.adapter is not None:
                total += sum(p.numel() for p in block.adapter.parameters())
        return total

    def forward(
        self,
        x: Tensor,
        ref_features: Dict[int, Tensor],
        return_attention: bool = False,
        spatial_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[Tensor, Optional[Dict[int, Tensor]]]:
        """
        Forward pass through decoder.

        Args:
            x: Input features from layer 3 of target modality [B, N, D]
               If N includes CLS token, spatial_size should be computed from N-1
            ref_features: Dict mapping layer index to reference features
                         e.g., {3: [B,N,D], 4: [B,N,D], ..., 9: [B,N,D]}
            return_attention: Whether to return attention weights
            spatial_size: Optional (H, W) tuple for 2D RoPE. If None, auto-inferred.

        Returns:
            Tuple of:
                - Output features [B, N, D]
                - Optional dict of attention weights per layer
        """
        attn_weights_dict = {} if return_attention else None

        # Infer spatial size if not provided
        if spatial_size is None and self.rope_2d:
            # x might have CLS token, so we check if N-1 is a perfect square
            N = x.shape[1]
            # Try without CLS first
            sqrt_n = int(math.sqrt(N))
            if sqrt_n * sqrt_n == N:
                spatial_size = (sqrt_n, sqrt_n)
            else:
                # Try with CLS token (N-1 patches)
                sqrt_n = int(math.sqrt(N - 1))
                if sqrt_n * sqrt_n == (N - 1):
                    spatial_size = (sqrt_n, sqrt_n)

        for block in self.blocks:
            layer_idx = block.layer_idx

            # Get reference features for this layer
            if layer_idx in ref_features:
                ref = ref_features[layer_idx]
            else:
                raise KeyError(f"Reference features for layer {layer_idx} not found. "
                              f"Available layers: {list(ref_features.keys())}")

            # Forward through block: self-attn → cross-attn (with RoPE) → MLP
            x, attn = block(x, ref, return_attention, spatial_size)

            if return_attention and attn is not None:
                attn_weights_dict[layer_idx] = attn

        # Final normalization
        x = self.norm(x)

        return x, attn_weights_dict

    def forward_bidirectional(
        self,
        x_target: Tensor,
        x_ref: Tensor,
        target_intermediates: Dict[int, Tensor],
        ref_intermediates: Dict[int, Tensor],
        return_attention: bool = False,
        spatial_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[Tensor, Tensor, Optional[Dict], Optional[Dict]]:
        """
        Bidirectional forward pass (both target→ref and ref→target).

        Useful for training with bidirectional cross-modal learning.

        Args:
            x_target: Layer 3 features from target modality
            x_ref: Layer 3 features from reference modality
            target_intermediates: All intermediate layers from target
            ref_intermediates: All intermediate layers from reference
            return_attention: Whether to return attention weights
            spatial_size: Optional (H, W) tuple for 2D RoPE

        Returns:
            Tuple of:
                - Decoded target features
                - Decoded reference features
                - Optional target attention weights
                - Optional reference attention weights
        """
        # Target decoding with reference cross-attention
        target_out, target_attn = self.forward(
            x_target, ref_intermediates, return_attention, spatial_size
        )

        # Reference decoding with target cross-attention
        ref_out, ref_attn = self.forward(
            x_ref, target_intermediates, return_attention, spatial_size
        )

        return target_out, ref_out, target_attn, ref_attn
