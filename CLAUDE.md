# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Cross-spectral visual place recognition system matching thermal (query) images to RGB (database) images. Uses DINOv2 backbone with optional CroCo-style bidirectional reconstruction for robust place recognition.

## Architecture

### Two Training Modes

1. **GeM Only** (`network_only_GeM.py`): Baseline with GeM aggregation, no decoder
2. **BiReconstruction CroCo** (`network.py`): CroCo decoder with bidirectional reconstruction loss

### Core Model: `CrossModalVPR_Net`

```
Input: Thermal Query + RGB Database
         ↓
   Shared DINOv2 Backbone (ViT-B/14 or ViT-S/14)
         ↓
   ┌─────────────────────────────────────┐
   │ GeM Aggregation → Global Descriptor │
   └─────────────────────────────────────┘
         ↓
   (Optional: CroCo BiReconstruction)
   ┌─────────────────────────────────────┐
   │ Masked Encoder (75% masked)         │
   │ → Bidirectional Cross-Attention     │
   │ → Reconstruction Loss               │
   └─────────────────────────────────────┘
```

## Commands

### Training (GeM baseline)
```bash
python3 train_wandb.py \
    --train_batch_size 4 \
    --lr 5e-5 \
    --epochs_num 100 \
    --train_seq KAIST \
    --test_seq SNU Valley \
    --foundation_model_path /path/to/dinov2_vitb14_pretrain.pth \
    --comment "gem-baseline"
```

### Training (with BiReconstruction CroCo)
```bash
python3 train_wandb.py \
    --train_batch_size 4 \
    --lr 5e-5 \
    --epochs_num 100 \
    --train_seq KAIST \
    --test_seq SNU Valley \
    --foundation_model_path /path/to/dinov2_vitb14_pretrain.pth \
    --use_recon_loss \
    --recon_loss_type 'mse+ssim' \
    --croco_mask_ratio 0.75 \
    --recon_weight 2.5 \
    --comment "bireconstruction-croco"
```

### Evaluation
```bash
python3 eval.py \
    --resume "path/to/best_model.pth" \
    --test_seq SNU Valley \
    --img_time allday \
    --foundation_model_path /path/to/dinov2_vitb14_pretrain.pth
```

## Key Files

| File | Description |
|------|-------------|
| `network.py` | CroCo BiReconstruction model |
| `network_only_GeM.py` | GeM baseline model |
| `train_wandb.py` | Training script |
| `inference.py` | Evaluation/inference |
| `Parser.py` | Argument definitions |
| `datasets_T2R.py` | Dataset loaders |

## Key Arguments

| Argument | Description |
|----------|-------------|
| `--use_recon_loss` | Enable CroCo reconstruction loss |
| `--recon_loss_type` | mse, l1, ssim, or mse+ssim |
| `--croco_mask_ratio` | Masking ratio (default: 0.75) |
| `--recon_weight` | Weight for reconstruction loss |
| `--num_decoder_depth` | CroCo decoder depth (default: 8) |
| `--num_trainable_blocks_RGB` | Number of unfrozen backbone blocks |

## Dataset

SThReO dataset with 3 sequences: KAIST (train), SNU (test), Valley (test)

- `.mat` files in `Dataset/save_mat/`
- Temporal splits: allday, daytime, nighttime, latetime
- Query: thermal images, Database: RGB images

## Pretrained Weights

Download DINOv2 ViT-B/14 or ViT-S/14 from [Meta's DINOv2](https://github.com/facebookresearch/dinov2). Place in `backbone/dinov2/pretrained/`.
