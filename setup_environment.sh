#!/bin/bash
# Setup script for ACT training environment with mamba

set -e  # Exit on error

echo "========================================"
echo "ACT Training Environment Setup"
echo "========================================"
echo ""

# Check if mamba is installed
if ! command -v mamba &> /dev/null; then
    echo "Error: mamba is not installed or not in PATH"
    echo "Please install mamba first:"
    echo "  wget https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh"
    echo "  bash Mambaforge-Linux-x86_64.sh"
    exit 1
fi

echo "✓ Mamba found: $(which mamba)"
echo ""

# Check CUDA availability
echo "Checking CUDA availability..."
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    echo ""
else
    echo "⚠ Warning: nvidia-smi not found. GPU may not be available."
    echo ""
fi

# Create environment from YAML
echo "Creating mamba environment from environment.yaml..."
echo "This may take several minutes..."
echo ""

mamba env create -f environment.yaml

echo ""
echo "========================================"
echo "Environment created successfully!"
echo "========================================"
echo ""
echo "To activate the environment, run:"
echo "  mamba activate act_training"
echo ""
echo "To verify installation:"
echo "  python -c 'import torch; print(f\"PyTorch version: {torch.__version__}\"); print(f\"CUDA available: {torch.cuda.is_available()}\"); print(f\"CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}\")'"
echo ""
echo "To run tests:"
echo "  pytest policy/tests/test_act.py -v"
echo ""
echo "To start training:"
echo "  python scripts/train_act.py --config configs/policy_act.yaml --dataset_dir bc-train-data-test --output_dir runs/act_experiment"
echo ""


