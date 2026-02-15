# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Cross-spectral visual place recognition system matching thermal (query) images to RGB (database) images. Based on the IROS 2025 paper "RGB-Thermal Visual Place Recognition via Vision Foundation Model". Uses DINOv2 backbone with cross-modal fusion for robust place recognition in extreme conditions (poor illumination, smoke, fog).

## Commands

### Training

**Single-stage training with DINO decoder:**
```bash
python3 train_wandb.py \
    --train_batch_size 4 \
    --lr 5e-5 \
    --epochs_num 100 \
    --sequences KAIST \
    --foundation_model_path /path/to/dinov2_vits14_pretrain.pth \
    --use_recon_loss \
    --recon_loss_type 'mse+ssim' \
    --use_dino_decoder \
    --dino_decoder_layer_start 6 \
    --dino_decoder_layer_end 12 \
    --croco_mask_ratio 0.8 \
    --recon_weight 2.5 \
    --comment "experiment-description"
```

**Two-stage training (adds diff loss):**
```bash
python3 train_wandb_2stage.py \
    --resume "path/to/checkpoint.pth" \
    --use_diff_loss \
    --use_reranking 'diffGeM' \
    --lr 1e-4
```

### Evaluation

```bash
python3 eval.py \
    --resume "path/to/best_model.pth" \
    --sequences SNU Valley \
    --img_time allday \
    --foundation_model_path /path/to/dinov2_vits14_pretrain.pth
```

### Fast Inference with Visualization

```bash
python3 fast_inference.py \
    --resume "path/to/checkpoint.pth" \
    --use_reranking 'diffGeM' \
    --visualize_attention
```

## Architecture

### Core Model: `CrossModalVPR_Net` (network.py)

```
Input: Thermal Query + RGB Database
         ↓
   Shared DINOv2 Backbone (ViT-B/14 or ViT-S/14)
         ↓
   ┌─────────────────────────────────────┐
   │         Two Parallel Paths          │
   ├─────────────────────────────────────┤
   │ VPR Path:                           │
   │   GeM Aggregation → Global Descriptor│
   │                                     │
   │ Reconstruction Path (training):     │
   │   Masked Encoder (75-80% masked)    │
   │   → Recursive MLP                   │
   │   → DINO Decoder (cross-attention)  │
   │   → Reconstruction Loss             │
   └─────────────────────────────────────┘
         ↓
   Optional Reranking (R2Former/DiffGeM/SelaVPR)
```

### Key Components

- **`network.py`**: Main `CrossModalVPR_Net` with encoder, decoder, aggregation, reranking
- **`backbone/dinov2/decoder/decoder.py`**: `DINOv2Decoder` with `CrossAttentionAdapter` and `VanillaAdapter`
- **`datasets_T2R.py`**: SThReO dataset loader with `TripletsSTheReODual` for training
- **`diff_loss.py`**: `DiffLoss` for differential GeM attention reranking
- **`Parser.py`**: All training/inference arguments

### DINO Decoder Architecture (use_dino_decoder=True)

Separate reconstruction pathway that doesn't interfere with encoder:
1. Masked encoder produces visible patches
2. Recursive MLP transforms features
3. Mask expansion fills masked positions with learnable tokens
4. DINO decoder with cross-attention (using RoPE): thermal uses RGB as reference, RGB uses thermal
5. Prediction heads output reconstructed patches

### Positional Encoding: RoPE (Rotary Position Embedding)

The CroCo decoder uses RoPE (2D Rotary Position Embedding) instead of learnable positional embeddings:
- **2D RoPE**: Separate rotations for height/width dimensions, better suited for 2D image patches
- RoPE encodes position by rotating Q/K vectors in attention, enabling better extrapolation and relative position awareness
- When `--use_rope` is enabled (default: True), the decoder's self-attention and cross-attention use RoPE
- When disabled, falls back to learnable positional embeddings added before the decoder

### Reranking Options (`--use_reranking`)

- `none`: Global descriptor only
- `r2former`: Attention-based local patch matching
- `diffGeM`: Differential GeM attention loss
- `selaVPR`: MNN-based local feature matching
- `recon`: Reconstruction loss as reranking score

## Dataset

SThReO dataset with 3 sequences: KAIST (train), SNU (test), Valley (test)

- `.mat` files in `Dataset/save_mat/`
- Temporal splits: allday, daytime, nighttime, latetime
- Query: thermal images, Database: RGB images

## Key Arguments

| Argument | Description |
|----------|-------------|
| `--use_dino_decoder` | Use DINOv2 decoder instead of CroCo |
| `--dino_decoder_layer_start/end` | Decoder layer range (e.g., 6-12) |
| `--unfreeze_dino_decoder` | Train DINO blocks (not just adapters) |
| `--use_rope` / `--no_rope` | Enable/disable 2D RoPE in CroCo decoder (default: enabled) |
| `--paired_rgb_epochs` | Epochs to use aligned/paired RGB, then switch to positive RGB (default: 40) |
| `--use_recon_loss` | Enable reconstruction loss |
| `--recon_loss_type` | mse, l1, ssim, or mse+ssim |
| `--croco_mask_ratio` | Masking ratio (default: 0.8) |
| `--use_diff_loss` | Enable differential loss for reranking |
| `--num_trainable_blocks_RGB/THERMAL` | Number of unfrozen backbone blocks |

## Pretrained Weights

Download DINOv2 ViT-B/14 or ViT-S/14 from [Meta's DINOv2](https://github.com/facebookresearch/dinov2). Place in `backbone/dinov2/pretrained/`.
