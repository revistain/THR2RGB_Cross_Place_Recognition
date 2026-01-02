# train_wandb.py 수정

def train_reranking_with_triplets(args, model, triplets_ds):
    """
    기존 triplet을 reranking decoder 학습에 재사용
    """
    # Encoder freeze
    model.module.rgb_backbone.requires_grad_(False)
    model.module.thermal_backbone.requires_grad_(False)
    
    # Decoder unfreeze
    model.module.decoder_blocks.requires_grad_(True)
    model.module.decoder_norm.requires_grad_(True)
    model.module.prediction_head.requires_grad_(True)
    
    # Optimizer (decoder만)
    decoder_params = [
        p for n, p in model.named_parameters() 
        if 'decoder' in n or 'prediction_head' in n
    ]
    optimizer = torch.optim.Adam(decoder_params, lr=1e-5)
    
    # Triplet 가져오기
    triplets = triplets_ds.triplets_global_indexes  # [N, 12]
    
    model.train()
    rerank_losses = []
    
    for triplet in tqdm(triplets[:100], desc="Reranking decoder"):  # 100개만 샘플
        query_idx, pos_idx, *neg_indices = triplet
        neg_indices = neg_indices[:5]  # Top-5 negatives만
        
        # Images 로드
        query_img = triplets_ds.get_thermal_img(
            triplets_ds.t_queries_paths[query_idx]
        )
        query_img = base_transform(query_img).unsqueeze(0).to(args.device)
        
        pos_img = triplets_ds.get_rgb_img(
            triplets_ds.rgb_database_paths[pos_idx]
        )
        pos_img = base_transform(pos_img).unsqueeze(0).to(args.device)
        
        neg_imgs = []
        for neg_idx in neg_indices:
            neg_img = triplets_ds.get_rgb_img(
                triplets_ds.rgb_database_paths[neg_idx]
            )
            neg_imgs.append(base_transform(neg_img))
        neg_imgs = torch.stack(neg_imgs).to(args.device)  # [5, 3, 224, 224]
        
        # Forward (encoder frozen)
        with torch.no_grad():
            # Query encoding
            query_visible, mask, B, N, D = model.module.croco_like_encoder(query_img)
            query_full = model.module.croco_encoded_mask_expension(
                query_visible, mask, B, N, D
            )
            
            # Positive encoding
            pos_feat = model.module.rgb_backbone(pos_img)["x_norm_patchtokens"]
            
            # Negative encoding
            neg_feats = []
            for neg_img in neg_imgs:
                neg_feat = model.module.rgb_backbone(neg_img.unsqueeze(0))
                neg_feats.append(neg_feat["x_norm_patchtokens"])
            neg_feats = torch.cat(neg_feats, dim=0)  # [5, 256, 768]
        
        # Decoder forward (learnable)
        # Positive reconstruction
        query_dec = query_full + model.module.decoder_pos_embed
        pos_dec = pos_feat + model.module.decoder_pos_embed
        
        thermal_dec_pos = query_dec
        for blk in model.module.decoder_blocks:
            thermal_dec_pos = blk(thermal_dec_pos, pos_dec)
        thermal_dec_pos = model.module.decoder_norm(thermal_dec_pos)
        
        pos_recon = model.module.prediction_head(thermal_dec_pos)
        target = model.module.patchify(query_img)
        
        # Positive loss
        pos_loss = F.mse_loss(
            pos_recon[mask], 
            target[mask], 
            reduction='mean'
        )
        
        # Negative reconstruction
        neg_losses = []
        for i, neg_feat in enumerate(neg_feats):
            neg_dec = neg_feat.unsqueeze(0) + model.module.decoder_pos_embed
            
            thermal_dec_neg = query_dec
            for blk in model.module.decoder_blocks:
                thermal_dec_neg = blk(thermal_dec_neg, neg_dec)
            thermal_dec_neg = model.module.decoder_norm(thermal_dec_neg)
            
            neg_recon = model.module.prediction_head(thermal_dec_neg)
            neg_loss = F.mse_loss(
                neg_recon[mask], 
                target[mask], 
                reduction='mean'
            )
            neg_losses.append(neg_loss)
        
        neg_losses = torch.stack(neg_losses)
        
        # Ranking loss: pos_loss < neg_loss + margin
        margin = 0.05
        ranking_loss = torch.relu(pos_loss - neg_losses + margin).mean()
        
        # Backward
        optimizer.zero_grad()
        ranking_loss.backward()
        optimizer.step()
        
        rerank_losses.append(ranking_loss.item())
    
    # Encoder 다시 unfreeze
    model.module.rgb_backbone.requires_grad_(True)
    model.module.thermal_backbone.requires_grad_(True)
    
    logging.info(f"Reranking decoder loss: {np.mean(rerank_losses):.4f}")


# Main training loop에 추가
for epoch_num in range(start_epoch_num, args.epochs_num):
    # ... 기존 training ...
    
    # 기존 triplet 계산
    triplets_ds.compute_triplets(args, model)
    
    # ... 기존 training loop ...
    
    # ===== Reranking decoder training (5 epoch마다) =====
    if epoch_num >= 10 and epoch_num % 5 == 0:
        logging.info("Training reranking decoder...")
        train_reranking_with_triplets(args, model, triplets_ds)
    
    # ... validation ...