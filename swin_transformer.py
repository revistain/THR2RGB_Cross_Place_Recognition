import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath
from typing import Tuple, Optional


class RotaryPositionEmbedding2D(nn.Module):
    """
    2D Rotary Position Embedding (RoPE) for vision transformers.

    Applies separate rotations for height and width dimensions, better suited
    for 2D image patches than 1D RoPE.

    Args:
        dim: Dimension per head (must be divisible by 4 for 2D)
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
        assert dim % 4 == 0, f"dim={dim} must be divisible by 4 for 2D RoPE"

        self.dim = dim
        self.max_h = max_h
        self.max_w = max_w
        self.base = base

        # Half dimensions for each spatial direction (h and w)
        half_dim = dim // 2

        # Inverse frequencies for position encoding
        inv_freq = 1.0 / (base ** (torch.arange(0, half_dim, 2).float() / half_dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

        # Cache for precomputed cos/sin values
        self._cos_cache = None
        self._sin_cache = None
        self._cached_h = 0
        self._cached_w = 0

    def _build_cache(self, h: int, w: int, device: torch.device, dtype: torch.dtype):
        """Build cos/sin cache for given spatial dimensions."""
        if h <= self._cached_h and w <= self._cached_w and self._cos_cache is not None:
            return

        self._cached_h = max(h, self._cached_h)
        self._cached_w = max(w, self._cached_w)

        # Position indices for height and width
        pos_h = torch.arange(self._cached_h, device=device, dtype=dtype)
        pos_w = torch.arange(self._cached_w, device=device, dtype=dtype)

        inv_freq = self.inv_freq.to(device=device, dtype=dtype)

        # Compute frequencies: [H, dim/4] and [W, dim/4]
        freqs_h = torch.einsum('i,j->ij', pos_h, inv_freq)
        freqs_w = torch.einsum('i,j->ij', pos_w, inv_freq)

        # Expand to grid: [H, W, dim/4]
        freqs_h = freqs_h[:, None, :].expand(-1, self._cached_w, -1)
        freqs_w = freqs_w[None, :, :].expand(self._cached_h, -1, -1)

        # Flatten: [H*W, dim/4]
        freqs_h = freqs_h.reshape(-1, freqs_h.shape[-1])
        freqs_w = freqs_w.reshape(-1, freqs_w.shape[-1])

        # Combine: [H*W, dim] = [h_freq, h_freq, w_freq, w_freq]
        emb = torch.cat([freqs_h, freqs_h, freqs_w, freqs_w], dim=-1)

        # Cache as [1, 1, H*W, dim] for broadcasting
        self._cos_cache = emb.cos()[None, None, :, :]
        self._sin_cache = emb.sin()[None, None, :, :]

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Apply rotation for 2D RoPE (separate h and w rotations)."""
        quarter = self.dim // 4
        x1 = x[..., :quarter]
        x2 = x[..., quarter:quarter*2]
        x3 = x[..., quarter*2:quarter*3]
        x4 = x[..., quarter*3:]
        return torch.cat([-x2, x1, -x4, x3], dim=-1)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        h: int,
        w: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply 2D rotary position embedding to Q and K.

        Args:
            q: Query tensor [B, num_heads, N, head_dim]
            k: Key tensor [B, num_heads, M, head_dim]
            h: Height in patches
            w: Width in patches

        Returns:
            Rotated (q, k) tensors
        """
        seq_len_q = q.shape[2]
        seq_len_k = k.shape[2]

        self._build_cache(h, w, q.device, q.dtype)

        # Get cos/sin for Q and K sequence lengths
        cos_q = self._cos_cache[:, :, :seq_len_q, :self.dim].to(q.dtype)
        sin_q = self._sin_cache[:, :, :seq_len_q, :self.dim].to(q.dtype)
        cos_k = self._cos_cache[:, :, :seq_len_k, :self.dim].to(k.dtype)
        sin_k = self._sin_cache[:, :, :seq_len_k, :self.dim].to(k.dtype)

        # Apply rotation: x * cos + rotate_half(x) * sin
        q_rot = (q * cos_q) + (self._rotate_half(q) * sin_q)
        k_rot = (k * cos_k) + (self._rotate_half(k) * sin_k)

        return q_rot, k_rot


class MultiheadAttentionWithRoPE(nn.Module):
    """
    Multi-head attention with 2D Rotary Position Embedding.

    Replaces nn.MultiheadAttention with RoPE support for vision transformers.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        use_rope: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_rope = use_rope

        # Separate Q, K, V projections (required for RoPE)
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim)

        # 2D RoPE
        if use_rope:
            self.rope = RotaryPositionEmbedding2D(
                dim=self.head_dim,
                max_h=32,
                max_w=32,
            )
        else:
            self.rope = None

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.q_proj.bias is not None:
            nn.init.zeros_(self.q_proj.bias)
            nn.init.zeros_(self.k_proj.bias)
            nn.init.zeros_(self.v_proj.bias)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        spatial_size: Optional[Tuple[int, int]] = None,
        need_weights: bool = False,
        average_attn_weights: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            query: [B, N, D]
            key: [B, M, D]
            value: [B, M, D]
            spatial_size: (H, W) for 2D RoPE. If None, inferred from sequence length.
            need_weights: Whether to return attention weights
            average_attn_weights: Whether to average attention weights across heads

        Returns:
            output: [B, N, D]
            attn_weights: [B, N, M] if need_weights else None
        """
        B, N, _ = query.shape
        M = key.shape[1]

        # Project Q, K, V
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Reshape to multi-head: [B, num_heads, seq_len, head_dim]
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, M, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        if self.rope is not None:
            if spatial_size is not None:
                h, w = spatial_size
            else:
                # Infer square spatial size
                h = w = int(math.sqrt(N))
            q, k = self.rope(q, k, h, w)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        # Apply attention to values
        out = attn @ v

        # Reshape back: [B, N, D]
        out = out.transpose(1, 2).contiguous().view(B, N, self.dim)
        out = self.out_proj(out)

        if need_weights:
            if average_attn_weights:
                attn_weights = attn.mean(dim=1)  # [B, N, M]
            else:
                attn_weights = attn  # [B, num_heads, N, M]
            return out, attn_weights

        return out, None

def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Partition feature map into non-overlapping windows.

    Args:
        x: [B, H, W, C] tensor
        window_size: window size
    Returns:
        windows: [num_windows*B, window_size, window_size, C]
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """
    Merge windows back to feature map.

    Args:
        windows: [num_windows*B, window_size, window_size, C]
        window_size: window size
        H, W: height and width of image
    Returns:
        x: [B, H, W, C]
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class SwinWindowAttention(nn.Module):
    """
    Swin Transformer V2 Window Attention with:
    - Cosine attention with learnable logit scale
    - Log-spaced continuous position bias (CPB-MLP)
    """
    def __init__(self, dim: int, window_size: int, num_heads: int, qkv_bias: bool = True):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads

        # QKV projection
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        # Swin V2: learnable logit scale (tau) - one per head
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))

        # Swin V2: Continuous Position Bias MLP (CPB-MLP)
        # Input: relative coordinate (2D), Output: bias per head
        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_heads, bias=False)
        )

        # Create relative position index for windows
        # Get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2

        # Normalize to [-1, 1] and apply log-spacing
        relative_coords_normalized = relative_coords.float()
        relative_coords_normalized[:, :, 0] /= (self.window_size - 1)
        relative_coords_normalized[:, :, 1] /= (self.window_size - 1)
        # Log-spaced: sign(x) * log2(1 + |x| * 7) / log2(8)
        relative_coords_log = torch.sign(relative_coords_normalized) * torch.log2(
            1 + relative_coords_normalized.abs() * 7
        ) / math.log2(8)

        self.register_buffer("relative_coords_table", relative_coords_log)  # Wh*Ww, Wh*Ww, 2

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.trunc_normal_(self.qkv.weight, std=.02)
        if self.qkv.bias is not None:
            nn.init.constant_(self.qkv.bias, 0)
        nn.init.trunc_normal_(self.proj.weight, std=.02)
        nn.init.constant_(self.proj.bias, 0)
        # Initialize CPB-MLP
        nn.init.trunc_normal_(self.cpb_mlp[0].weight, std=.02)
        nn.init.constant_(self.cpb_mlp[0].bias, 0)
        nn.init.trunc_normal_(self.cpb_mlp[2].weight, std=.02)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [num_windows*B, N, C] where N = window_size * window_size
            mask: [num_windows, N, N] or None for attention masking (shifted windows)
        Returns:
            output: [num_windows*B, N, C]
            attn: [num_windows*B, num_heads, N, N] attention weights
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # [B_, num_heads, N, head_dim]

        # Swin V2: Cosine attention with learnable scale
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # Clamp logit_scale to avoid extreme values
        logit_scale = torch.clamp(self.logit_scale, max=math.log(100.0)).exp()

        attn = (q @ k.transpose(-2, -1)) * logit_scale  # [B_, num_heads, N, N]

        # Add continuous position bias
        # relative_coords_table: [N, N, 2]
        relative_position_bias = self.cpb_mlp(self.relative_coords_table).permute(2, 0, 1)  # [num_heads, N, N]
        # Apply 16x sigmoid as in Swin V2
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        # Apply mask for shifted window attention
        if mask is not None:
            num_windows = mask.shape[0]
            attn = attn.view(B_ // num_windows, num_windows, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)  # [1, num_win, 1, N, N]
            attn = attn.view(-1, self.num_heads, N, N)

        attn_weights = attn.softmax(dim=-1)

        out = (attn_weights @ v).transpose(1, 2).reshape(B_, N, C)
        out = self.proj(out)

        return out, attn_weights


class SwinDecoderBlock(nn.Module):
    """
    Swin Transformer V2 Decoder Block with:
    - Windowed Self-Attention (W-MSA / SW-MSA)
    - Global Cross-Attention (for cross-modal features)
    - MLP with DropPath
    """
    def __init__(
        self,
        dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        window_size: int = 4,
        shift_size: int = 0,
        img_size: int = 16,  # Feature map size (e.g., 16x16 for 256 patches)
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.img_size = img_size

        # Ensure shift_size is valid
        if self.shift_size > 0:
            assert 0 < self.shift_size < self.window_size, "shift_size must be in (0, window_size)"

        # 1. Windowed Self-Attention
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = SwinWindowAttention(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=True
        )

        # 2. Global Cross-Attention (unchanged from CroCo)
        self.norm2 = nn.LayerNorm(dim)
        self.norm_cross = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        # 3. MLP
        self.norm3 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )

        # DropPath for stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Attention storage for visualization
        self.self_attn_weights = None
        self.cross_attn_weights = None

        # Create attention mask for shifted window attention
        if self.shift_size > 0:
            self._create_attn_mask()
        else:
            self.register_buffer("attn_mask", None)

    def _create_attn_mask(self):
        """Create attention mask for SW-MSA (shifted window attention)"""
        H, W = self.img_size, self.img_size
        img_mask = torch.zeros((1, H, W, 1))

        # Slice indices for shifted windows
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))

        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # [num_win, ws, ws, 1]
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x: torch.Tensor, y: torch.Tensor, return_attention: bool = False) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] - decoder input (e.g., thermal)
            y: [B, M, D] - encoder output (e.g., RGB reference)
            return_attention: bool - whether to store attention weights
        Returns:
            x: [B, N, D] - updated decoder features
        """
        B, N, D = x.shape
        H = W = self.img_size
        assert N == H * W, f"Input token count {N} doesn't match img_size {H}x{W}"

        # ========== Step 1: Windowed Self-Attention ==========
        shortcut = x
        x_norm = self.norm1(x)

        # Reshape to 2D: [B, N, D] -> [B, H, W, D]
        x_2d = x_norm.view(B, H, W, D)

        # Apply cyclic shift for SW-MSA
        if self.shift_size > 0:
            shifted_x = torch.roll(x_2d, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x_2d

        # Partition windows: [B, H, W, D] -> [num_win*B, ws*ws, D]
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, D)

        # Window attention
        attn_out, self_attn_weights = self.self_attn(x_windows, mask=self.attn_mask)
        if return_attention:
            self.self_attn_weights = self_attn_weights

        # Merge windows back: [num_win*B, ws*ws, D] -> [B, H, W, D]
        attn_out = attn_out.view(-1, self.window_size, self.window_size, D)
        shifted_x = window_reverse(attn_out, self.window_size, H, W)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x_2d = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x_2d = shifted_x

        # Reshape back: [B, H, W, D] -> [B, N, D]
        self_out = x_2d.view(B, N, D)
        x = shortcut + self.drop_path(self_out)

        # ========== Step 2: Global Cross-Attention ==========
        x_norm = self.norm2(x)
        encoder_norm = self.norm_cross(y)

        if return_attention:
            cross_out, cross_attn_weights = self.cross_attn(
                query=x_norm,
                key=encoder_norm,
                value=encoder_norm,
                need_weights=True,
                average_attn_weights=True
            )
            self.cross_attn_weights = cross_attn_weights
        else:
            cross_out = self.cross_attn(
                query=x_norm,
                key=encoder_norm,
                value=encoder_norm
            )[0]

        x = x + self.drop_path(cross_out)

        # ========== Step 3: MLP ==========
        x = x + self.drop_path(self.mlp(self.norm3(x)))

        return x


class CroCoDecoderBlock(nn.Module):
    def __init__(self, dim=768, num_heads=12, mlp_ratio=4.0, drop_path=0.0, use_rope=False):
        super().__init__()

        self.use_rope = use_rope

        # Self-Attention components
        self.norm1 = nn.LayerNorm(dim)
        if use_rope:
            self.self_attn = MultiheadAttentionWithRoPE(dim, num_heads, use_rope=True)
        else:
            self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        # Cross-Attention components
        self.norm2 = nn.LayerNorm(dim)
        self.norm_cross = nn.LayerNorm(dim)
        if use_rope:
            self.cross_attn = MultiheadAttentionWithRoPE(dim, num_heads, use_rope=True)
        else:
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

    def forward(self, x, y, return_attention=False, spatial_size=None):
        """
        Args:
            x: [B, N, D] - decoder input (RGB masked)
            y: [B, M, D] - encoder output (Thermal reference)
            return_attention: bool - attention map 반환 여부
            spatial_size: (H, W) for RoPE. If None, inferred from sequence length.
        Returns:
            x: [B, N, D] - updated decoder features
        """
        # Step 1: Self-Attention
        x_norm = self.norm1(x)
        if self.use_rope:
            if return_attention:
                self_out, self_attn_weights = self.self_attn(
                    x_norm, x_norm, x_norm,
                    spatial_size=spatial_size,
                    need_weights=True,
                    average_attn_weights=True
                )
                self.self_attn_weights = self_attn_weights
            else:
                self_out, _ = self.self_attn(x_norm, x_norm, x_norm, spatial_size=spatial_size)
        else:
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
        if self.use_rope:
            if return_attention:
                cross_out, cross_attn_weights = self.cross_attn(
                    query=x_norm,
                    key=encoder_norm,
                    value=encoder_norm,
                    spatial_size=spatial_size,
                    need_weights=True,
                    average_attn_weights=True
                )
                self.cross_attn_weights = cross_attn_weights
            else:
                cross_out, _ = self.cross_attn(
                    query=x_norm,
                    key=encoder_norm,
                    value=encoder_norm,
                    spatial_size=spatial_size
                )
        else:
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


class CroCoDecoderBlockSelfAttn(nn.Module):
    """CroCoDecoderBlock with only Self-Attention (no Cross-Attention)"""
    def __init__(self, dim=768, num_heads=12, mlp_ratio=4.0, drop_path=0.0):
        super().__init__()

        # Self-Attention components
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        # MLP components
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )

        # DropPath for stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Attention storage
        self.self_attn_weights = None

    def forward(self, x, y=None, return_attention=False):
        """
        Args:
            x: [B, N, D] - decoder input
            y: [B, M, D] - encoder output (ignored, kept for API compatibility)
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
                average_attn_weights=True
            )
            self.self_attn_weights = self_attn_weights
        else:
            self_out = self.self_attn(x_norm, x_norm, x_norm)[0]
        x = x + self.drop_path(self_out)

        # Step 2: MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class CroCoDecoderBlockCrossAttn(nn.Module):
    """CroCoDecoderBlock with only Cross-Attention (no Self-Attention)"""
    def __init__(self, dim=768, num_heads=12, mlp_ratio=4.0, drop_path=0.0):
        super().__init__()

        # Cross-Attention components
        self.norm1 = nn.LayerNorm(dim)
        self.norm_cross = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        # MLP components
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )

        # DropPath for stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Attention storage
        self.cross_attn_weights = None

    def forward(self, x, y, return_attention=False):
        """
        Args:
            x: [B, N, D] - decoder input
            y: [B, M, D] - encoder output (reference)
            return_attention: bool - attention map 반환 여부
        Returns:
            x: [B, N, D] - updated decoder features
        """
        # Step 1: Cross-Attention
        x_norm = self.norm1(x)
        encoder_norm = self.norm_cross(y)
        if return_attention:
            cross_out, cross_attn_weights = self.cross_attn(
                query=x_norm,
                key=encoder_norm,
                value=encoder_norm,
                need_weights=True,
                average_attn_weights=True
            )
            self.cross_attn_weights = cross_attn_weights
        else:
            cross_out = self.cross_attn(
                query=x_norm,
                key=encoder_norm,
                value=encoder_norm
            )[0]
        x = x + self.drop_path(cross_out)

        # Step 2: MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x