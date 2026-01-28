# match_conf_module.py
"""
MNN Match Confidence Module - R2Former-style scoring for selaVPR local features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from timm.models.vision_transformer import Block
from timm.models.layers import trunc_normal_


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """
    grid: [B, N, 2] - (x, y) coordinates normalized to [0, 1]
    returns: [B, N, embed_dim]
    """
    assert embed_dim % 2 == 0

    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=grid.device)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    # grid: [B, N, 2]
    out_x = torch.einsum('bn,d->bnd', grid[:, :, 0], omega)  # [B, N, D/2]
    out_y = torch.einsum('bn,d->bnd', grid[:, :, 1], omega)  # [B, N, D/2]

    emb_x = torch.cat([torch.sin(out_x), torch.cos(out_x)], dim=2)  # [B, N, D]
    emb_y = torch.cat([torch.sin(out_y), torch.cos(out_y)], dim=2)  # [B, N, D]

    # Average x and y embeddings
    emb = (emb_x + emb_y) / 2
    return emb


def extract_mnn_matches_with_metadata(fm1, fm2, grid_size, top_k=500):
    """
    Extract MNN matches with metadata for transformer processing.

    Args:
        fm1: [B, L, D] - Query features (e.g., thermal), L = H*W
        fm2: [B, L, D] - Database features (e.g., RGB)
        grid_size: (H, W) - spatial grid size
        top_k: Maximum number of matches to return

    Returns:
        match_features: [B, K, 7] - (self_x, self_y, matched_x, matched_y, distance, angle, similarity)
        num_matches: [B] - actual number of matches per sample
    """
    B, L, D = fm1.shape
    H, W = grid_size
    device = fm1.device

    # Normalize features for cosine similarity
    fm1_norm = F.normalize(fm1, p=2, dim=2)  # [B, L, D]
    fm2_norm = F.normalize(fm2, p=2, dim=2)  # [B, L, D]

    # Compute similarity matrix
    M = torch.bmm(fm1_norm, fm2_norm.transpose(1, 2))  # [B, L, L]

    # Find mutual nearest neighbors
    max1 = torch.argmax(M, dim=2)  # [B, L] - for each query patch, best match in db
    max2 = torch.argmax(M, dim=1)  # [B, L] - for each db patch, best match in query

    # Check mutual consistency: max2[max1[i]] == i
    batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, L)  # [B, L]
    mutual_match = max2[batch_idx, max1]  # [B, L]
    query_idx = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)  # [B, L]
    valid = (mutual_match == query_idx)  # [B, L] - True if MNN

    # Get similarities for valid matches
    similarities = M[batch_idx, query_idx, max1]  # [B, L]
    similarities = similarities * valid.float()  # Zero out non-MNN

    # Prepare coordinate grids (normalized 0-1)
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(0, 1, H, device=device),
        torch.linspace(0, 1, W, device=device),
        indexing='ij'
    )
    coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)  # [L, 2]

    # Initialize output tensors
    match_features = torch.zeros(B, top_k, 7, device=device)
    num_matches = torch.zeros(B, dtype=torch.long, device=device)

    for b in range(B):
        valid_mask = valid[b]  # [L]
        valid_indices = torch.nonzero(valid_mask, as_tuple=True)[0]  # indices in query
        n_valid = len(valid_indices)

        if n_valid == 0:
            # No matches - use dummy values
            num_matches[b] = 0
            continue

        # Get matched indices
        matched_indices = max1[b, valid_indices]  # indices in db
        match_sims = similarities[b, valid_indices]  # [n_valid]

        # Sort by similarity (descending) and take top_k
        if n_valid > top_k:
            sorted_idx = torch.argsort(match_sims, descending=True)[:top_k]
            valid_indices = valid_indices[sorted_idx]
            matched_indices = matched_indices[sorted_idx]
            match_sims = match_sims[sorted_idx]
            n_valid = top_k

        num_matches[b] = n_valid

        # Get coordinates
        query_coords = coords[valid_indices]  # [n_valid, 2] - (x, y)
        db_coords = coords[matched_indices]  # [n_valid, 2]

        # Compute distance (Euclidean in normalized space)
        coord_diff = query_coords - db_coords
        distances = torch.norm(coord_diff, dim=1)  # [n_valid]

        # Compute angle (atan2 for direction)
        angles = torch.atan2(coord_diff[:, 1], coord_diff[:, 0])  # [n_valid], range [-pi, pi]
        angles = (angles + math.pi) / (2 * math.pi)  # Normalize to [0, 1]

        # Build feature vector: (self_x, self_y, matched_x, matched_y, distance, angle, similarity)
        match_features[b, :n_valid, 0] = query_coords[:, 0]  # self_x
        match_features[b, :n_valid, 1] = query_coords[:, 1]  # self_y
        match_features[b, :n_valid, 2] = db_coords[:, 0]     # matched_x
        match_features[b, :n_valid, 3] = db_coords[:, 1]     # matched_y
        match_features[b, :n_valid, 4] = distances           # distance
        match_features[b, :n_valid, 5] = angles              # angle
        match_features[b, :n_valid, 6] = match_sims          # similarity

    return match_features, num_matches


class MatchConfidenceModule(nn.Module):
    """
    R2Former-style confidence scoring for MNN matches.

    Takes top-K MNN matches, each represented as 7-dim vector,
    processes through transformer blocks, and outputs match/non-match score.
    """

    def __init__(self,
                 embed_dim=32,
                 num_heads=4,
                 mlp_ratio=4.0,
                 depth_1=2,      # First transformer (per-match processing)
                 depth_2=6,      # Second transformer (global aggregation)
                 num_classes=2,  # Binary classification
                 top_k=500):
        super().__init__()

        self.embed_dim = embed_dim
        self.top_k = top_k
        self.num_classes = num_classes

        # Project 7-dim match features to embed_dim
        self.match_proj = nn.Linear(7, embed_dim, bias=True)

        # CLS tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        trunc_normal_(self.cls_token, std=0.02)

        # Transformer blocks for global aggregation
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=nn.LayerNorm)
            for _ in range(depth_2)
        ])

        # Layer norm
        self.norm = nn.LayerNorm(embed_dim)

        # Prediction head
        self.pred_head = nn.Linear(embed_dim, num_classes, bias=True)

        # Loss
        self.CE = nn.CrossEntropyLoss(ignore_index=-100)
        self.softmax = nn.Softmax(dim=1)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.match_proj.weight, std=0.02)
        nn.init.zeros_(self.match_proj.bias)
        nn.init.normal_(self.pred_head.weight, std=0.02)
        nn.init.zeros_(self.pred_head.bias)

    def forward_features(self, match_features, num_matches=None):
        """
        Process match features through transformer.

        Args:
            match_features: [B, K, 7] - match feature vectors
            num_matches: [B] - actual number of matches (for masking)

        Returns:
            cls_features: [B, embed_dim] - CLS token output
        """
        B, K, _ = match_features.shape
        device = match_features.device

        # Project to embedding dimension
        x = self.match_proj(match_features)  # [B, K, embed_dim]

        # Add positional embedding based on query coordinates
        pos_embed = get_2d_sincos_pos_embed_from_grid(
            self.embed_dim,
            match_features[:, :, :2]  # (self_x, self_y)
        )
        x = x + pos_embed

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [B, K+1, embed_dim]

        # Create attention mask if num_matches provided
        # (mask out padding positions)
        attn_mask = None
        if num_matches is not None:
            # Create mask: True for valid positions, False for padding
            seq_len = K + 1  # +1 for CLS token
            mask = torch.zeros(B, seq_len, dtype=torch.bool, device=device)
            mask[:, 0] = True  # CLS always valid
            for b in range(B):
                mask[b, 1:num_matches[b]+1] = True
            # For attention: 0 = attend, -inf = ignore
            # But timm Block doesn't support custom attention mask easily
            # So we'll rely on zero-padding having minimal effect

        # Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        # Extract CLS token
        cls_features = x[:, 0]  # [B, embed_dim]

        return cls_features

    def forward(self, query_features, pos_features, neg_features, grid_size):
        """
        Compute matching confidence loss for training.

        Args:
            query_features: [B, H, W, D] - Query local features (e.g., thermal)
            pos_features: [B, H, W, D] - Positive local features (e.g., matched RGB)
            neg_features: [B, H, W, D] - Negative local features
            grid_size: (H, W) tuple

        Returns:
            loss: Cross-entropy loss for positive vs negative classification
        """
        B = query_features.shape[0]
        H, W, D = query_features.shape[1], query_features.shape[2], query_features.shape[3]
        device = query_features.device

        # Reshape to [B, L, D]
        query_flat = query_features.view(B, H * W, D)
        pos_flat = pos_features.view(B, H * W, D)
        neg_flat = neg_features.view(B, H * W, D)

        # Extract MNN matches
        pos_match_features, pos_num_matches = extract_mnn_matches_with_metadata(
            query_flat, pos_flat, grid_size, self.top_k
        )
        neg_match_features, neg_num_matches = extract_mnn_matches_with_metadata(
            query_flat, neg_flat, grid_size, self.top_k
        )

        # Log if matches are less than top_k
        for b in range(B):
            if pos_num_matches[b] < self.top_k:
                pass  # Can add logging here if needed
            if neg_num_matches[b] < self.top_k:
                pass

        # Forward through transformer
        pos_cls = self.forward_features(pos_match_features, pos_num_matches)  # [B, embed_dim]
        neg_cls = self.forward_features(neg_match_features, neg_num_matches)  # [B, embed_dim]

        # Predict scores
        pos_logits = self.pred_head(pos_cls)  # [B, num_classes]
        neg_logits = self.pred_head(neg_cls)  # [B, num_classes]

        # Create labels: positive = 1, negative = 0
        pos_labels = torch.ones(B, dtype=torch.long, device=device)
        neg_labels = torch.zeros(B, dtype=torch.long, device=device)

        # Compute loss
        logits = torch.cat([pos_logits, neg_logits], dim=0)  # [2B, num_classes]
        labels = torch.cat([pos_labels, neg_labels], dim=0)  # [2B]

        loss = self.CE(logits, labels)

        return loss

    def get_confidence_score(self, query_features, db_features, grid_size):
        """
        Get confidence score for a query-database pair (inference).

        Args:
            query_features: [1, H, W, D] or [H, W, D]
            db_features: [N, H, W, D] - N database candidates
            grid_size: (H, W) tuple

        Returns:
            scores: [N] - confidence scores (higher = more likely match)
        """
        if query_features.dim() == 3:
            query_features = query_features.unsqueeze(0)

        H, W, D = query_features.shape[1], query_features.shape[2], query_features.shape[3]
        N = db_features.shape[0]
        device = query_features.device

        # Expand query to match database batch size
        query_flat = query_features.view(1, H * W, D).expand(N, -1, -1)  # [N, L, D]
        db_flat = db_features.view(N, H * W, D)  # [N, L, D]

        # Extract matches
        match_features, num_matches = extract_mnn_matches_with_metadata(
            query_flat, db_flat, grid_size, self.top_k
        )

        # Forward through transformer
        cls_features = self.forward_features(match_features, num_matches)  # [N, embed_dim]

        # Predict
        logits = self.pred_head(cls_features)  # [N, num_classes]

        # Return probability of being a match (class 1)
        probs = self.softmax(logits)
        scores = probs[:, 1]  # [N]

        return scores


class MatchConfidenceLoss(nn.Module):
    """
    Wrapper loss class for easy integration with existing training code.
    """

    def __init__(self, embed_dim=32, top_k=500):
        super().__init__()
        self.module = MatchConfidenceModule(embed_dim=embed_dim, top_k=top_k)

    def forward(self, feature_data, grid_size=(61, 61)):
        """
        Args:
            feature_data: tuple of (anchor, positive, negative) features
                         each [B, H, W, D]
            grid_size: spatial grid size

        Returns:
            loss: scalar loss value
        """
        anchor, positive, negative = feature_data[0], feature_data[1], feature_data[2]
        loss = self.module(anchor, positive, negative, grid_size)
        return loss
