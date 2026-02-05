import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DiffLoss(torch.nn.Module):
    def __init__(self, args, depth=1):
        super().__init__()
        self.input_dim = args.features_dim
        self.patch_count = int(args.resize[0]/14)*int(args.resize[1]/14)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.input_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.input_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.input_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.input_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.GeM_linear1 = nn.Linear(self.input_dim, self.input_dim)
        self.GeM_linear2 = nn.Linear(self.input_dim, self.input_dim)
        self.score_linear = nn.Sequential(
            nn.Linear(self.patch_count*2, self.patch_count),
            nn.ReLU(),
            nn.Linear(self.patch_count, 1),
        )        
        self.BCE = torch.nn.BCEWithLogitsLoss().cuda()
        
        self.depth = depth
        self.args = args
        
    def lambda_init_fn(self, depth):
        return 0.8 - 0.6 * math.exp(-0.3 * depth)

    def forward(self, model, queries_indexes, positives_indexes, negatives_indexes, \
                      global_features, patch_embedding):
        # GeM pooling
        query_GeM, pos_GeM, neg_GeM = \
                global_features[queries_indexes], \
                global_features[positives_indexes], \
                global_features[negatives_indexes]
        
        # Feature Embeddings
        query_patches, pos_patches, neg_patches = \
                patch_embedding[queries_indexes], \
                patch_embedding[positives_indexes], \
                patch_embedding[negatives_indexes]
        
        # Add Positional Embedding to Feature Embedding
        positional_embedding = model.decoder_pos_embed.detach()
        query_patches = query_patches + positional_embedding
        pos_patches = pos_patches + positional_embedding
        neg_patches = neg_patches + positional_embedding
        
        # Make it to batch for fast decode
        target_batch = torch.cat([query_patches, query_patches, pos_patches, neg_patches], dim=0)
        ref_batch = torch.cat([pos_patches, neg_patches, query_patches, query_patches], dim=0)

        # Pass it through decoder (no_grad to prevent gradients flowing to decoder)
        with torch.no_grad():
            for blk in model.decoder_blocks:
                target_batch = blk(target_batch, ref_batch)
            target_batch = model.decoder_norm(target_batch)

        # split the decoded result
        chunks = torch.chunk(target_batch, 4, dim=0)
        query_pos_recon = chunks[0]
        query_neg_recon = chunks[1]
        pos_query_recon = chunks[2]
        neg_query_recon = chunks[3]

        # GeM pass through linear layers
        query_GeM1 = self.GeM_linear1(query_GeM).unsqueeze(1)
        query_GeM2 = self.GeM_linear2(query_GeM).unsqueeze(1)
        pos_GeM1 = self.GeM_linear1(pos_GeM).unsqueeze(1)
        pos_GeM2 = self.GeM_linear2(pos_GeM).unsqueeze(1)
        neg_GeM1 = self.GeM_linear1(neg_GeM).unsqueeze(1)
        neg_GeM2 = self.GeM_linear2(neg_GeM).unsqueeze(1)
        
        # make attention maps(decoded result with GeM)
        sqrt_d = math.sqrt(self.input_dim)
        query_pos_GeM1_attn_map = F.softmax((query_GeM1 @ query_pos_recon.transpose(-2, -1)) / sqrt_d, dim=-1).squeeze(1)
        query_pos_GeM2_attn_map = F.softmax((query_GeM2 @ query_pos_recon.transpose(-2, -1)) / sqrt_d, dim=-1).squeeze(1)
        pos_query_GeM1_attn_map = F.softmax((pos_GeM1 @ pos_query_recon.transpose(-2, -1)) / sqrt_d, dim=-1).squeeze(1)
        pos_query_GeM2_attn_map = F.softmax((pos_GeM2 @ pos_query_recon.transpose(-2, -1)) / sqrt_d, dim=-1).squeeze(1)
        
        query_neg_GeM1_attn_map = F.softmax((query_GeM1 @ query_neg_recon.transpose(-2, -1)) / sqrt_d, dim=-1).squeeze(1)
        query_neg_GeM2_attn_map = F.softmax((query_GeM2 @ query_neg_recon.transpose(-2, -1)) / sqrt_d, dim=-1).squeeze(1)
        neg_query_GeM1_attn_map = F.softmax((neg_GeM1 @ neg_query_recon.transpose(-2, -1)) / sqrt_d, dim=-1).squeeze(1)
        neg_query_GeM2_attn_map = F.softmax((neg_GeM2 @ neg_query_recon.transpose(-2, -1)) / sqrt_d, dim=-1).squeeze(1)
        
        # calculate differential map
        lambda_init = self.lambda_init_fn(self.depth)
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float())
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float())
        lambda_ = lambda_1 - lambda_2 + lambda_init
        
        query_pos_diff_attn_map = query_pos_GeM1_attn_map - lambda_ * query_pos_GeM2_attn_map
        pos_query_diff_attn_map = pos_query_GeM1_attn_map - lambda_ * pos_query_GeM2_attn_map
        query_neg_diff_attn_map = query_neg_GeM1_attn_map - lambda_ * query_neg_GeM2_attn_map
        neg_query_diff_attn_map = neg_query_GeM1_attn_map - lambda_ * neg_query_GeM2_attn_map
        
        # 아래의 shape: [4, 256]
        query_pos_diff_attn_map_flat = query_pos_diff_attn_map.reshape(self.args.train_batch_size, -1)
        pos_query_diff_attn_map_flat = pos_query_diff_attn_map.reshape(self.args.train_batch_size, -1)
        query_neg_diff_attn_map_flat = query_neg_diff_attn_map.reshape(self.args.train_batch_size, -1)
        neg_query_diff_attn_map_flat = neg_query_diff_attn_map.reshape(self.args.train_batch_size, -1)
        
        # before fc layer
        query_pos_concated = torch.cat([query_pos_diff_attn_map_flat, pos_query_diff_attn_map_flat], dim=-1)
        neg_query_concated = torch.cat([query_neg_diff_attn_map_flat, neg_query_diff_attn_map_flat], dim=-1)
        
        # pass through fc layer (BC가 내부적으로 sigmoid 적용)
        pos_scores = self.score_linear(query_pos_concated)
        neg_scores = self.score_linear(neg_query_concated)
        
        # make loss using CE
        target = torch.zeros(pos_scores.shape[0] * 2, dtype=torch.float).cuda()
        target[:pos_scores.shape[0]] = 1
        rerank_loss = self.BCE(torch.cat([pos_scores, neg_scores], dim=0).squeeze(1), target)

        return rerank_loss

    def inference(self, model, query_patches, query_GeM, candidate_patches, candidate_GeMs):
        """
        Inference method for reranking.

        Args:
            model: the main model (for decoder access)
            query_patches: [1, N, D] query patch embeddings
            query_GeM: [1, D] query global descriptor
            candidate_patches: [K, N, D] candidate patch embeddings
            candidate_GeMs: [K, D] candidate global descriptors

        Returns:
            scores: [K] reranking scores (higher = more similar)
        """
        K = candidate_patches.shape[0]
        positional_embedding = model.decoder_pos_embed.detach()

        # Expand query to match candidates: [K, N, D]
        query_patches_exp = query_patches.expand(K, -1, -1)
        query_GeM_exp = query_GeM.expand(K, -1)

        # Add positional embedding
        query_patches_pos = query_patches_exp + positional_embedding
        candidate_patches_pos = candidate_patches + positional_embedding

        # Bidirectional decoding: query->candidate and candidate->query
        target_batch = torch.cat([query_patches_pos, candidate_patches_pos], dim=0)
        ref_batch = torch.cat([candidate_patches_pos, query_patches_pos], dim=0)

        # Pass through decoder
        with torch.no_grad():
            for blk in model.decoder_blocks:
                target_batch = blk(target_batch, ref_batch)
            target_batch = model.decoder_norm(target_batch)

        # Split: query_recon (query decoded with candidate context), candidate_recon (vice versa)
        query_recon = target_batch[:K]      # [K, N, D]
        candidate_recon = target_batch[K:]  # [K, N, D]

        # GeM through linear layers
        query_GeM1 = self.GeM_linear1(query_GeM_exp).unsqueeze(1)    # [K, 1, D]
        query_GeM2 = self.GeM_linear2(query_GeM_exp).unsqueeze(1)    # [K, 1, D]
        candidate_GeM1 = self.GeM_linear1(candidate_GeMs).unsqueeze(1)  # [K, 1, D]
        candidate_GeM2 = self.GeM_linear2(candidate_GeMs).unsqueeze(1)  # [K, 1, D]

        # Compute attention maps
        sqrt_d = math.sqrt(self.input_dim)

        # query GeM attending to query_recon (query decoded with candidate context)
        query_cand_GeM1_attn = F.softmax((query_GeM1 @ query_recon.transpose(-2, -1)) / sqrt_d, dim=-1).squeeze(1)
        query_cand_GeM2_attn = F.softmax((query_GeM2 @ query_recon.transpose(-2, -1)) / sqrt_d, dim=-1).squeeze(1)

        # candidate GeM attending to candidate_recon (candidate decoded with query context)
        cand_query_GeM1_attn = F.softmax((candidate_GeM1 @ candidate_recon.transpose(-2, -1)) / sqrt_d, dim=-1).squeeze(1)
        cand_query_GeM2_attn = F.softmax((candidate_GeM2 @ candidate_recon.transpose(-2, -1)) / sqrt_d, dim=-1).squeeze(1)

        # Differential attention
        lambda_init = self.lambda_init_fn(self.depth)
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float())
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float())
        lambda_ = lambda_1 - lambda_2 + lambda_init

        query_cand_diff_attn = query_cand_GeM1_attn - lambda_ * query_cand_GeM2_attn  # [K, N]
        cand_query_diff_attn = cand_query_GeM1_attn - lambda_ * cand_query_GeM2_attn  # [K, N]

        # Concatenate bidirectional attention maps
        concat_attn = torch.cat([query_cand_diff_attn, cand_query_diff_attn], dim=-1)  # [K, 2N]

        # Score prediction
        scores = self.score_linear(concat_attn).squeeze(-1)  # [K]

        # Apply sigmoid to get probability-like scores
        scores = torch.sigmoid(scores)

        return scores

    def visualize_attention_maps(self, model, query_patches, query_GeM, candidate_patches, candidate_GeMs,
                                  query_idx, save_dir, query_image=None, candidate_images=None):
        """
        Visualize GeM1, GeM2, and differential attention maps.

        Args:
            model: the main model (for decoder access)
            query_patches: [1, N, D] query patch embeddings
            query_GeM: [1, D] query global descriptor
            candidate_patches: [K, N, D] candidate patch embeddings
            candidate_GeMs: [K, D] candidate global descriptors
            query_idx: query index for saving
            save_dir: directory to save visualizations
            query_image: optional query image for overlay
            candidate_images: optional candidate images for overlay
        """
        import os
        import matplotlib.pyplot as plt
        import numpy as np

        os.makedirs(save_dir, exist_ok=True)

        K = candidate_patches.shape[0]
        N = candidate_patches.shape[1]
        H = W = int(math.sqrt(N))  # Assuming square grid

        positional_embedding = model.decoder_pos_embed.detach()

        # Expand query to match candidates
        query_patches_exp = query_patches.expand(K, -1, -1)
        query_GeM_exp = query_GeM.expand(K, -1)

        # Add positional embedding
        query_patches_pos = query_patches_exp + positional_embedding
        candidate_patches_pos = candidate_patches + positional_embedding

        # Bidirectional decoding
        target_batch = torch.cat([query_patches_pos, candidate_patches_pos], dim=0)
        ref_batch = torch.cat([candidate_patches_pos, query_patches_pos], dim=0)

        with torch.no_grad():
            for blk in model.decoder_blocks:
                target_batch = blk(target_batch, ref_batch)
            target_batch = model.decoder_norm(target_batch)

        query_recon = target_batch[:K]
        candidate_recon = target_batch[K:]

        # GeM through linear layers
        query_GeM1 = self.GeM_linear1(query_GeM_exp).unsqueeze(1)
        query_GeM2 = self.GeM_linear2(query_GeM_exp).unsqueeze(1)
        candidate_GeM1 = self.GeM_linear1(candidate_GeMs).unsqueeze(1)
        candidate_GeM2 = self.GeM_linear2(candidate_GeMs).unsqueeze(1)

        # Compute attention maps
        sqrt_d = math.sqrt(self.input_dim)

        # Query attending to decoded query (with candidate context)
        query_cand_GeM1_attn = F.softmax((query_GeM1 @ query_recon.transpose(-2, -1)) / sqrt_d, dim=-1).squeeze(1)
        query_cand_GeM2_attn = F.softmax((query_GeM2 @ query_recon.transpose(-2, -1)) / sqrt_d, dim=-1).squeeze(1)

        # Candidate attending to decoded candidate (with query context)
        cand_query_GeM1_attn = F.softmax((candidate_GeM1 @ candidate_recon.transpose(-2, -1)) / sqrt_d, dim=-1).squeeze(1)
        cand_query_GeM2_attn = F.softmax((candidate_GeM2 @ candidate_recon.transpose(-2, -1)) / sqrt_d, dim=-1).squeeze(1)

        # Differential attention
        lambda_init = self.lambda_init_fn(self.depth)
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float())
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float())
        lambda_ = lambda_1 - lambda_2 + lambda_init

        query_cand_diff_attn = query_cand_GeM1_attn - lambda_ * query_cand_GeM2_attn
        cand_query_diff_attn = cand_query_GeM1_attn - lambda_ * cand_query_GeM2_attn

        # Convert to numpy and reshape to 2D
        query_cand_GeM1_np = query_cand_GeM1_attn.cpu().numpy().reshape(K, H, W)
        query_cand_GeM2_np = query_cand_GeM2_attn.cpu().numpy().reshape(K, H, W)
        query_cand_diff_np = query_cand_diff_attn.cpu().numpy().reshape(K, H, W)

        cand_query_GeM1_np = cand_query_GeM1_attn.cpu().numpy().reshape(K, H, W)
        cand_query_GeM2_np = cand_query_GeM2_attn.cpu().numpy().reshape(K, H, W)
        cand_query_diff_np = cand_query_diff_attn.cpu().numpy().reshape(K, H, W)

        # Plot for each candidate
        for k in range(K):
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))

            # Row 1: Query attention maps
            axes[0, 0].set_title(f'Query Image')
            if query_image is not None:
                axes[0, 0].imshow(query_image)
            else:
                axes[0, 0].text(0.5, 0.5, f'Query {query_idx}', ha='center', va='center')
            axes[0, 0].axis('off')

            im1 = axes[0, 1].imshow(query_cand_GeM1_np[k], cmap='hot', interpolation='bilinear')
            axes[0, 1].set_title('Query GeM1 Attn')
            axes[0, 1].axis('off')
            plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

            im2 = axes[0, 2].imshow(query_cand_GeM2_np[k], cmap='hot', interpolation='bilinear')
            axes[0, 2].set_title('Query GeM2 Attn')
            axes[0, 2].axis('off')
            plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)

            im3 = axes[0, 3].imshow(query_cand_diff_np[k], cmap='RdBu_r', interpolation='bilinear')
            axes[0, 3].set_title(f'Query Diff Attn (λ={lambda_.item():.3f})')
            axes[0, 3].axis('off')
            plt.colorbar(im3, ax=axes[0, 3], fraction=0.046)

            # Row 2: Candidate attention maps
            axes[1, 0].set_title(f'Candidate {k}')
            if candidate_images is not None and k < len(candidate_images):
                axes[1, 0].imshow(candidate_images[k])
            else:
                axes[1, 0].text(0.5, 0.5, f'Candidate {k}', ha='center', va='center')
            axes[1, 0].axis('off')

            im4 = axes[1, 1].imshow(cand_query_GeM1_np[k], cmap='hot', interpolation='bilinear')
            axes[1, 1].set_title('Cand GeM1 Attn')
            axes[1, 1].axis('off')
            plt.colorbar(im4, ax=axes[1, 1], fraction=0.046)

            im5 = axes[1, 2].imshow(cand_query_GeM2_np[k], cmap='hot', interpolation='bilinear')
            axes[1, 2].set_title('Cand GeM2 Attn')
            axes[1, 2].axis('off')
            plt.colorbar(im5, ax=axes[1, 2], fraction=0.046)

            im6 = axes[1, 3].imshow(cand_query_diff_np[k], cmap='RdBu_r', interpolation='bilinear')
            axes[1, 3].set_title(f'Cand Diff Attn (λ={lambda_.item():.3f})')
            axes[1, 3].axis('off')
            plt.colorbar(im6, ax=axes[1, 3], fraction=0.046)

            plt.suptitle(f'Query {query_idx} - Candidate {k} Attention Maps', fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'query_{query_idx}_cand_{k}_attn.png'), dpi=150, bbox_inches='tight')
            plt.close()
