#!/bin/bash
# Visualization Script for THR2RGB Cross-Place Recognition
#
# This script loads a pretrained checkpoint and visualizes:
# 1. Attention maps (last and penultimate layers)
# 2. MNN (Mutual Nearest Neighbor) matches (if selaVPR is enabled)
#
# Usage:
#   ./visualize.sh <checkpoint_path> [options]
#
# Examples:
#   ./visualize.sh ./logs/experiment/best_model.pth
#   ./visualize.sh ./logs/experiment/best_model.pth --num_samples 8
#   ./visualize.sh ./logs/experiment/best_model.pth --output_dir ./my_visualizations
#
# The config.yaml is automatically loaded from the checkpoint directory.

# Default values
CHECKPOINT_PATH=""
NUM_SAMPLES=4
OUTPUT_DIR=""
DATASET_FOLDER="./Dataset"
MAX_MATCHES=50
DEVICE="cuda"
SEED=42

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num_samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --dataset_folder)
            DATASET_FOLDER="$2"
            shift 2
            ;;
        --max_matches)
            MAX_MATCHES="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: ./visualize.sh <checkpoint_path> [options]"
            echo ""
            echo "Arguments:"
            echo "  checkpoint_path        Path to the checkpoint (.pth file)"
            echo ""
            echo "Options:"
            echo "  --num_samples N        Number of samples to visualize (default: 4)"
            echo "  --output_dir DIR       Output directory (default: checkpoint_dir/visualizations)"
            echo "  --dataset_folder DIR   Dataset folder path (default: ./Dataset)"
            echo "  --max_matches N        Maximum MNN lines to draw (default: 50)"
            echo "  --device DEVICE        Device to use: cuda or cpu (default: cuda)"
            echo "  --seed N               Random seed (default: 42)"
            echo "  --help, -h             Show this help message"
            exit 0
            ;;
        *)
            if [[ -z "$CHECKPOINT_PATH" ]]; then
                CHECKPOINT_PATH="$1"
            else
                echo "Unknown argument: $1"
                exit 1
            fi
            shift
            ;;
    esac
done

# Check if checkpoint path is provided
if [[ -z "$CHECKPOINT_PATH" ]]; then
    echo "Error: checkpoint_path is required"
    echo "Usage: ./visualize.sh <checkpoint_path> [options]"
    echo "Run ./visualize.sh --help for more information"
    exit 1
fi

# Check if checkpoint file exists
if [[ ! -f "$CHECKPOINT_PATH" ]]; then
    echo "Error: Checkpoint file not found: $CHECKPOINT_PATH"
    exit 1
fi

# Check if config.yaml exists in the checkpoint directory
CHECKPOINT_DIR=$(dirname "$CHECKPOINT_PATH")
CONFIG_PATH="$CHECKPOINT_DIR/config.yaml"
if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Error: config.yaml not found in checkpoint directory: $CONFIG_PATH"
    exit 1
fi

echo "============================================"
echo "  THR2RGB Visualization"
echo "============================================"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Config: $CONFIG_PATH"
echo "Num samples: $NUM_SAMPLES"
echo "Device: $DEVICE"
echo "============================================"

# Build command
CMD="python inference_vis.py \
    --checkpoint_path $CHECKPOINT_PATH \
    --num_samples $NUM_SAMPLES \
    --dataset_folder $DATASET_FOLDER \
    --max_matches $MAX_MATCHES \
    --device $DEVICE \
    --seed $SEED"

# Add optional output_dir if specified
if [[ -n "$OUTPUT_DIR" ]]; then
    CMD="$CMD --output_dir $OUTPUT_DIR"
fi

# Run the visualization
echo ""
echo "Running: $CMD"
echo ""
$CMD

'''
  - --num_samples N: Number of samples to visualize (default: 4)                                                                                                                                           
  - --output_dir DIR: Output directory (default: checkpoint_dir/visualizations)                                                                                                                            
  - --dataset_folder DIR: Dataset folder path (default: ./Dataset)                                                                                                                                         
  - --max_matches N: Maximum MNN lines to draw (default: 50)                                                                                                                                               
  - --device DEVICE: cuda or cpu (default: cuda)                                                                                                                                                           
  - --seed N: Random seed (default: 42)                                                                                                                                                                    
  - --help: Show help message

  visualize.sh ./logs/experiment/best_model.pth \
                --output_dir ./my_vis \
                --max_matches 100     
  '''

/home/jwkim/workspace/THR2RGB_Cross_Place_Recognition/logs/224x224-GeM-ViTs/260127_135254