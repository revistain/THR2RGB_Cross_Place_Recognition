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
    2. Cross-Attention (trainable adapter)
    3. MLP (frozen, from pretrained DINO)
"""

import copy
from functools import partial
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from timm.models.layers import DropPath


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
    Trainable cross-attention adapter inserted between frozen self-attention and MLP.

    This adapter enables cross-modal conditioning by attending to features
    from a reference modality (e.g., RGB features when decoding thermal).

    Args:
        dim: Feature dimension
        num_heads: Number of attention heads
        drop_path: Drop path rate for regularization
        init_scale: Initial scale value (0 for stable training start)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 12,
        drop_path: float = 0.0,
        init_scale: float = 0.0,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads

        # Layer norms for query and key/value
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

        # Cross-attention layer
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
        )

        # Learnable scale factor (initialized to init_scale for stable training)
        # When init_scale=0, decoder starts as pure DINO, then gradually learns cross-modal
        self.scale = nn.Parameter(torch.ones(1) * init_scale)

        # Drop path for regularization
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self._reset_parameters()

    def _reset_parameters(self):
        # Initialize cross-attention with small weights for stability
        nn.init.xavier_uniform_(self.cross_attn.in_proj_weight)
        nn.init.xavier_uniform_(self.cross_attn.out_proj.weight)
        nn.init.zeros_(self.cross_attn.in_proj_bias)
        nn.init.zeros_(self.cross_attn.out_proj.bias)

    def forward(
        self,
        x: Tensor,
        ref: Tensor,
        return_attention: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass of cross-attention adapter.

        Args:
            x: Query features from target modality [B, N, D]
            ref: Key/Value features from reference modality [B, M, D]
            return_attention: Whether to return attention weights

        Returns:
            Tuple of (output features, optional attention weights)
        """
        # Normalize inputs
        q = self.norm_q(x)
        kv = self.norm_kv(ref)

        # Cross-attention
        attn_out, attn_weights = self.cross_attn(
            query=q,
            key=kv,
            value=kv,
            need_weights=return_attention,
            average_attn_weights=True,
        )

        # Scale and residual connection with drop path
        out = x + self.drop_path(self.scale * attn_out)

        if return_attention:
            return out, attn_weights
        return out, None


class DinoDecoderBlock(nn.Module):
    """
    Decoder block using pretrained DINOv2 components with trainable cross-attention.

    Structure:
        x = x + self_attn(norm1(x))      # Frozen, from DINO
        x = x + cross_attn(x, ref)        # Trainable adapter
        x = x + mlp(norm2(x))             # Frozen, from DINO

    Args:
        dino_block: Pre-trained DINOv2 block (will extract self-attn and MLP)
        dim: Feature dimension
        num_heads: Number of attention heads for cross-attention
        drop_path: Drop path rate for cross-attention adapter
    """

    def __init__(
        self,
        dino_block: nn.Module,
        dim: int,
        num_heads: int = 12,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()

        # ============ Self-Attention components (frozen, from DINO) ============
        self.norm1 = dino_block.norm1
        self.attn = dino_block.attn
        self.ls1 = dino_block.ls1
        self.drop_path1 = dino_block.drop_path1

        # ============ Cross-Attention (trainable adapter) ============
        self.cross_attn_adapter = CrossAttentionAdapter(
            dim=dim,
            num_heads=num_heads,
            drop_path=drop_path,
            init_scale=0.0,  # Start with no cross-modal influence
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
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass: self-attn → cross-attn → MLP

        Args:
            x: Input features [B, N, D]
            ref: Reference features for cross-attention [B, M, D]
            return_attention: Whether to return cross-attention weights

        Returns:
            Tuple of (output features, optional cross-attention weights)
        """
        # 1. Self-Attention (frozen, from DINO)
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))

        # 2. Cross-Attention (trainable adapter)
        x, attn_weights = self.cross_attn_adapter(x, ref, return_attention)

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
        2. Cross-Attention (trainable adapter)
        3. MLP (frozen, pretrained DINO weights)

    Args:
        dim: Feature dimension (768 for ViT-Base, 384 for ViT-Small)
        num_heads: Number of attention heads
        layer_indices: Which DINO layers to use (default: [3,4,5,6,7,8,9])
        drop_path_rate: Drop path rate for cross-attention adapters
    """

    def __init__(
        self,
        dim: int = 768,
        num_heads: int = 12,
        layer_indices: List[int] = None,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()

        if layer_indices is None:
            layer_indices = [3, 4, 5, 6, 7, 8, 9]

        self.dim = dim
        self.num_heads = num_heads
        self.layer_indices = layer_indices
        self.num_blocks = len(layer_indices)

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
    ) -> 'DINOv2Decoder':
        """
        Create decoder by extracting DINO block components from encoder.

        Args:
            encoder: DINOv2 encoder (DinoVisionTransformer)
            layer_range: Range of layers to use (start, end), e.g., (3, 10) for layers 3-9
            num_heads: Number of attention heads (defaults to encoder's num_heads)
            drop_path_rate: Drop path rate for cross-attention adapters

        Returns:
            DINOv2Decoder instance with frozen DINO components + trainable cross-attention
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
        )

        # Create decoder blocks from encoder blocks
        for i, layer_idx in enumerate(layer_indices):
            # Deep copy the DINO block to extract its components
            dino_block = copy.deepcopy(encoder.blocks[layer_idx])

            # Create decoder block with DINO components + cross-attention
            decoder_block = DinoDecoderBlock(
                dino_block=dino_block,
                dim=dim,
                num_heads=num_heads,
                drop_path=decoder.drop_path_rates[i],
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
    ) -> Tuple[Tensor, Optional[Dict[int, Tensor]]]:
        """
        Forward pass through decoder.

        Args:
            x: Input features from layer 3 of target modality [B, N, D]
            ref_features: Dict mapping layer index to reference features
                         e.g., {3: [B,N,D], 4: [B,N,D], ..., 9: [B,N,D]}
            return_attention: Whether to return attention weights

        Returns:
            Tuple of:
                - Output features [B, N, D]
                - Optional dict of attention weights per layer
        """
        attn_weights_dict = {} if return_attention else None

        for block in self.blocks:
            layer_idx = block.layer_idx

            # Get reference features for this layer
            if layer_idx in ref_features:
                ref = ref_features[layer_idx]
            else:
                raise KeyError(f"Reference features for layer {layer_idx} not found. "
                              f"Available layers: {list(ref_features.keys())}")

            # Forward through block: self-attn → cross-attn → MLP
            x, attn = block(x, ref, return_attention)

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

        Returns:
            Tuple of:
                - Decoded target features
                - Decoded reference features
                - Optional target attention weights
                - Optional reference attention weights
        """
        # Target decoding with reference cross-attention
        target_out, target_attn = self.forward(
            x_target, ref_intermediates, return_attention
        )

        # Reference decoding with target cross-attention
        ref_out, ref_attn = self.forward(
            x_ref, target_intermediates, return_attention
        )

        return target_out, ref_out, target_attn, ref_attn
