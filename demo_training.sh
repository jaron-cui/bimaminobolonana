#!/bin/bash
# Demo training script for ACT policy
# This script shows you how to run ACT training with your dataset

set -e

echo "================================================"
echo "ACT Policy Training Demo"
echo "================================================"
echo ""

# Activate environment
echo "1. Activating environment..."
eval "$(conda shell.bash hook)"
mamba activate act_training
echo "✓ Environment activated: act_training"
echo ""

# Verify GPU
echo "2. Checking GPU setup..."
python -c "import torch; print(f'  PyTorch: {torch.__version__}'); print(f'  CUDA: {torch.cuda.is_available()}'); print(f'  GPUs: {torch.cuda.device_count()}')"
echo ""

# Check for dataset
echo "3. Dataset setup..."
if [ -z "$1" ]; then
    echo "  Usage: ./demo_training.sh <dataset_dir> [output_dir]"
    echo ""
    echo "  Example:"
    echo "    ./demo_training.sh /path/to/your/dataset runs/my_experiment"
    echo ""
    echo "  Dataset should contain:"
    echo "    - observations.npy"
    echo "    - actions.npy"
    echo "    - rollout_length.npy"
    echo ""
    exit 1
fi

DATASET_DIR=$1
OUTPUT_DIR=${2:-"runs/demo_$(date +%Y%m%d_%H%M%S)"}

if [ ! -d "$DATASET_DIR" ]; then
    echo "  ❌ Dataset directory not found: $DATASET_DIR"
    exit 1
fi

echo "  ✓ Dataset: $DATASET_DIR"
echo "  ✓ Output: $OUTPUT_DIR"
echo ""

# Verify dataset files
echo "4. Verifying dataset files..."
REQUIRED_FILES=("observations.npy" "actions.npy" "rollout_length.npy")
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$DATASET_DIR/$file" ]; then
        echo "  ✓ Found: $file"
    else
        echo "  ❌ Missing: $file"
        echo "  Please ensure your dataset contains all required files."
        exit 1
    fi
done
echo ""

# Show dataset info
echo "5. Dataset information..."
python << EOF
import numpy as np
try:
    actions = np.load('$DATASET_DIR/actions.npy')
    rollout_lengths = np.load('$DATASET_DIR/rollout_length.npy')

    print(f"  Timesteps: {len(actions)}")
    print(f"  Episodes: {len(rollout_lengths)}")
    print(f"  Action shape: {actions.shape}")
    print(f"  Avg episode length: {rollout_lengths.mean():.1f}")
except Exception as e:
    print(f"  ⚠ Could not load dataset info: {e}")
EOF
echo ""

# Show config
echo "6. Training configuration..."
echo "  Config file: configs/policy_act.yaml"
python << EOF
from omegaconf import OmegaConf
config = OmegaConf.load('configs/policy_act.yaml')
print(f"  Batch size: {config.batch_size}")
print(f"  Learning rate: {config.lr}")
print(f"  Epochs: {config.num_epochs}")
print(f"  Chunk size: {config.chunk_size}")
print(f"  Temporal context: {config.temporal_context}")
EOF
echo ""

# Confirm
echo "================================================"
echo "Ready to start training!"
echo "================================================"
echo ""
echo "This will:"
echo "  • Train ACT policy on your dataset"
echo "  • Use GPU acceleration (CUDA)"
echo "  • Save checkpoints to: $OUTPUT_DIR/checkpoint/"
echo "  • Save config to: $OUTPUT_DIR/config.yaml"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training cancelled."
    exit 0
fi
echo ""

# Start training
echo "7. Starting training..."
echo "================================================"
echo ""

python scripts/train_act.py \
    --config configs/policy_act.yaml \
    --dataset_dir "$DATASET_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --device cuda

echo ""
echo "================================================"
echo "Training complete!"
echo "================================================"
echo ""
echo "Checkpoints saved to: $OUTPUT_DIR/checkpoint/"
echo ""
echo "To resume training, run:"
echo "  python scripts/train_act.py \\"
echo "    --config configs/policy_act.yaml \\"
echo "    --dataset_dir $DATASET_DIR \\"
echo "    --output_dir $OUTPUT_DIR \\"
echo "    --device cuda \\"
echo "    --resume_from $OUTPUT_DIR/checkpoint/act-train-<epoch>.pth"
echo ""
