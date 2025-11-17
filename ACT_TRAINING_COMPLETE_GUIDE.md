# ACT Training Complete Guide

**Everything you need to train ACT policies on RTX 5090 GPUs**

---

## Table of Contents
1. [Environment Setup](#environment-setup)
2. [Dataset Preparation](#dataset-preparation)
3. [Training Pipeline](#training-pipeline)
4. [Monitoring with WandB](#monitoring-with-wandb)
5. [Configuration](#configuration)
6. [Troubleshooting](#troubleshooting)

---

## Environment Setup

### ✅ Current Status

Your environment is **fully configured** and ready for GPU training!

| Component | Status | Details |
|-----------|--------|---------|
| Environment | ✅ Ready | `act_training` |
| Python | ✅ 3.10.19 | |
| PyTorch | ✅ 2.8.0+cu128 | RTX 5090 support |
| CUDA | ✅ 12.8 | |
| GPUs | ✅ Working | 2x RTX 5090 (32GB each) |
| ACT Code | ✅ Ready | All imports working |

### Quick Verification

```bash
cd /home_shared/grail_andre/code/bimaminobolonana
mamba activate act_training

# Verify setup
python -c "import torch; from policy.act import build_act_policy; \
print(f'PyTorch: {torch.__version__}'); \
print(f'CUDA: {torch.cuda.is_available()}'); \
print(f'GPUs: {torch.cuda.device_count()}'); \
print('✅ Environment ready!')"
```

### GPU Information

```
GPU 0: NVIDIA GeForce RTX 5090 (31.36 GB)
GPU 1: NVIDIA GeForce RTX 5090 (31.37 GB)
Compute Capability: 12.0 (Blackwell)
Driver: 570.181
```

---

## Dataset Preparation

### Required Format

Your dataset directory must contain:

```
your_dataset/
├── observations.npy      # All observations
├── actions.npy          # All actions
└── rollout_length.npy   # Episode lengths
```

### Data Specifications

**Observations** (`observations.npy`):
- Array of dictionaries or dict of arrays
- Each observation contains:
  ```python
  {
      'image_left': np.ndarray,   # Shape: (H, W, 3), dtype: uint8
      'image_right': np.ndarray,  # Shape: (H, W, 3), dtype: uint8
      'state': np.ndarray         # Shape: (state_dim,), dtype: float32
  }
  ```

**Actions** (`actions.npy`):
- Shape: `(total_timesteps, 14)`
- First 7 values: left arm (6 joints + 1 gripper)
- Last 7 values: right arm (6 joints + 1 gripper)
- dtype: float32

**Rollout Lengths** (`rollout_length.npy`):
- Shape: `(num_episodes,)`
- Each value is the length of that episode
- Sum of all values should equal total_timesteps
- dtype: int64

### Verify Your Dataset

```python
import numpy as np

# Load data
obs = np.load('your_dataset/observations.npy', allow_pickle=True)
actions = np.load('your_dataset/actions.npy')
rollout_lengths = np.load('your_dataset/rollout_length.npy')

# Verify
print(f"Total timesteps: {len(actions)}")
print(f"Number of episodes: {len(rollout_lengths)}")
print(f"Action shape: {actions.shape}")
print(f"Sum of rollout lengths: {rollout_lengths.sum()}")
assert rollout_lengths.sum() == len(actions), "Mismatch!"
print("✅ Dataset valid!")
```

---

## Training Pipeline

### Basic Training

```bash
# Activate environment
mamba activate act_training

# Train
python scripts/train_act.py \
  --config configs/policy_act.yaml \
  --dataset_dir /path/to/your/dataset \
  --output_dir runs/my_experiment \
  --device cuda
```

### With WandB Monitoring

```bash
# Train with WandB logging
python scripts/train_act.py \
  --config configs/policy_act.yaml \
  --dataset_dir /path/to/your/dataset \
  --output_dir runs/my_experiment \
  --device cuda \
  --wandb_project "act-training" \
  --wandb_name "my-experiment"
```

### Resume Training

```bash
python scripts/train_act.py \
  --config configs/policy_act.yaml \
  --dataset_dir /path/to/your/dataset \
  --output_dir runs/my_experiment \
  --device cuda \
  --resume_from runs/my_experiment/checkpoint/act-train-100.pth
```

### Use Specific GPU

```bash
# GPU 0
CUDA_VISIBLE_DEVICES=0 python scripts/train_act.py \
  --config configs/policy_act.yaml \
  --dataset_dir /path/to/your/dataset \
  --output_dir runs/my_experiment \
  --device cuda

# GPU 1
CUDA_VISIBLE_DEVICES=1 python scripts/train_act.py \
  --config configs/policy_act.yaml \
  --dataset_dir /path/to/your/dataset \
  --output_dir runs/my_experiment \
  --device cuda
```

### Demo Script

```bash
# Automated training with interactive prompts
./demo_training.sh /path/to/your/dataset runs/my_experiment
```

---

## Monitoring with WandB

### Setup WandB

```bash
# Install wandb (already included)
pip install wandb

# Login to WandB
wandb login
```

### Track Training Metrics

WandB automatically logs:
- **Loss metrics**: total_loss, action_loss, kl_loss
- **Learning rate**: Current LR
- **Epoch/Step**: Training progress
- **Time**: Time per batch/epoch
- **GPU utilization**: Memory and compute
- **Model checkpoints**: Saved models

### View Training Dashboard

1. Run training with `--wandb_project` flag
2. Visit: https://wandb.ai/your-username/act-training
3. See real-time metrics, loss curves, and system stats

### Example WandB Run

```bash
python scripts/train_act.py \
  --config configs/policy_act.yaml \
  --dataset_dir /path/to/dataset \
  --output_dir runs/exp1 \
  --device cuda \
  --wandb_project "bimanual-manipulation" \
  --wandb_name "act-clip-baseline" \
  --wandb_tags "clip,baseline,rtx5090"
```

---

## Configuration

### Review Config

```bash
cat configs/policy_act.yaml
```

### Key Parameters

**Model Architecture:**
```yaml
chunk_size: 50              # Action prediction horizon
temporal_context: 3         # Past observations to use
hidden_dim: 512            # Transformer dimension
nheads: 8                  # Attention heads
num_encoder_layers: 4      # Encoder depth
num_decoder_layers: 7      # Decoder depth (ACT uses more)
latent_dim: 32            # CVAE latent dimension
```

**Training Hyperparameters:**
```yaml
batch_size: 8              # Adjust for GPU memory
lr: 1.0e-5                # Learning rate
num_epochs: 2000          # Total epochs
kl_weight: 10.0           # CVAE KL divergence weight
action_loss: l1           # 'l1' or 'l2'
```

**Visual Encoder:**
```yaml
encoder:
  name: clip_vit
  model_name: ViT-B-32     # CLIP model size
  pretrained: openai       # Use pretrained weights
  freeze: true             # Freeze during training
  fuse: mean              # Camera fusion: mean/concat_mlp/gated
```

**Checkpointing:**
```yaml
checkpoint_frequency: 10   # Save every N epochs
val_split: 0.1            # Validation fraction
```

### Adjust for Your Task

**For faster training:**
- Reduce `hidden_dim: 256`
- Reduce `num_encoder_layers: 2`
- Reduce `num_decoder_layers: 4`

**For longer horizon tasks:**
- Increase `chunk_size: 100`

**For more context:**
- Increase `temporal_context: 5`

**For memory constraints:**
- Reduce `batch_size: 4`
- Reduce `hidden_dim: 256`

---

## Training Monitoring

### Terminal Output

```
=== ACT Training ===
Config: configs/policy_act.yaml
Dataset: your_dataset
Output: runs/my_experiment

Building ACT policy...
Total parameters: 45,234,567
Trainable parameters: 42,123,456

Creating data loaders...
Training samples: 9500

Starting training...
Epoch 1/2000, Step 100: loss=2.345, action_loss=2.100, kl_loss=0.245
Epoch 1/2000, Step 200: loss=1.987, action_loss=1.765, kl_loss=0.222
...
Checkpoint saved: runs/my_experiment/checkpoint/act-train-10.pth
```

### GPU Monitoring

In another terminal:
```bash
watch -n 1 nvidia-smi
```

**Expected:**
- GPU Utilization: 80-100%
- Memory Usage: 8-20GB (depends on batch_size)
- Temperature: 60-80°C
- Both GPUs visible (training uses one by default)

### Performance Metrics

| Metric | Good Value | Notes |
|--------|-----------|-------|
| Action Loss | 0.01-0.1 | Lower is better |
| KL Loss | 0.01-0.1 | Should stabilize |
| Total Loss | 0.05-0.2 | Sum of above |
| Time/Batch | 2-5 sec | On RTX 5090 |
| GPU Util | 80-100% | Check with nvidia-smi |

### Output Directory Structure

```
runs/my_experiment/
├── config.yaml                      # Saved configuration
├── checkpoint/
│   ├── act-train-10.pth            # Epoch 10
│   ├── act-train-20.pth            # Epoch 20
│   └── ...
└── wandb/                           # WandB logs (if enabled)
```

---

## Evaluation

### Load Trained Policy

```python
import torch
from omegaconf import OmegaConf
from policy.act import build_act_policy

# Load config
config = OmegaConf.load('runs/my_experiment/config.yaml')

# Build and load policy
device = torch.device('cuda')
policy = build_act_policy(config).to(device)
policy.load_state_dict(
    torch.load('runs/my_experiment/checkpoint/act-train-1000.pth',
               map_location=device)
)
policy.eval()

print("✅ Policy loaded!")
```

### Test on Dataset

```python
from policy.act import create_temporal_dataloader

# Create test loader
test_loader = create_temporal_dataloader(
    dataset_path='your_dataset',
    temporal_context=config.temporal_context,
    chunk_size=config.chunk_size,
    batch_size=1,
    shuffle=False
)

# Evaluate
policy.eval()
total_error = 0
count = 0

with torch.no_grad():
    for batch in test_loader:
        obs, actions = batch
        obs = {k: v.to(device) for k, v in obs.items()}
        actions = actions.to(device)

        # Predict
        pred_actions, _ = policy(obs, actions)

        # Compute error
        error = (pred_actions - actions).abs().mean()
        total_error += error.item()
        count += 1

print(f"Mean Absolute Error: {total_error / count:.4f}")
```

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| **CUDA out of memory** | Reduce `batch_size` in config (try 4 or 2) |
| **Dataset not found** | Use absolute path to dataset directory |
| **No module named 'policy'** | Run from project root: `cd /home_shared/grail_andre/code/bimaminobolonana` |
| **Loss not decreasing** | Lower learning rate (try `lr: 5e-6`) or check data quality |
| **Training very slow** | Check GPU utilization with `nvidia-smi`, increase dataloader workers |
| **KL loss too high** | Reduce `kl_weight` (try 1.0 or 5.0) |
| **WandB not logging** | Run `wandb login` and check internet connection |

### GPU Errors

**"CUDA error: no kernel image"**
- Make sure you have PyTorch 2.8.0+cu128
- Check: `python -c "import torch; print(torch.__version__)"`
- Should output: `2.8.0+cu128`

**"GPU not detected"**
- Verify: `nvidia-smi`
- Check CUDA: `python -c "import torch; print(torch.cuda.is_available())"`

### Data Issues

**"Rollout length mismatch"**
- Verify: `rollout_lengths.sum() == len(actions)`
- Check for corrupted data

**"Invalid observation format"**
- Ensure observations are dicts with 'image_left', 'image_right', 'state'
- Check data loading code

### Training Issues

**Loss exploding:**
1. Lower learning rate: `lr: 1e-6`
2. Check for NaN in data
3. Reduce batch size

**Loss not decreasing:**
1. Verify data quality (visualize)
2. Check action normalization
3. Try lower `kl_weight`
4. Increase learning rate slightly

**Slow convergence:**
1. Increase learning rate: `lr: 5e-5`
2. Decrease `kl_weight: 5.0`
3. Check encoder is frozen: `encoder.freeze: true`

---

## Advanced Topics

### Multi-GPU Training

Current setup uses single GPU. For multi-GPU:
- Implement `DistributedDataParallel` in trainer
- Use `torch.distributed.launch`
- Scale learning rate with num_gpUs

### Temporal Ensembling

Enable for inference:
```yaml
temporal_ensemble: true
ensemble_window: 10
```

This averages predictions over multiple timesteps for smoother actions.

### Custom Encoders

Try different visual encoders:

**PRI3D (spatial features):**
```yaml
encoder:
  name: pri3d
  pretrained: true
  freeze: false  # Fine-tune if needed
```

**Random initialization:**
```yaml
encoder:
  pretrained: null
  freeze: false
```

### Hyperparameter Tuning

Use WandB sweeps:
```yaml
# sweep.yaml
program: scripts/train_act.py
method: bayes
metric:
  name: val_loss
  goal: minimize
parameters:
  lr:
    min: 1e-6
    max: 1e-4
  kl_weight:
    values: [1.0, 5.0, 10.0, 50.0]
  batch_size:
    values: [4, 8, 16]
```

Run sweep:
```bash
wandb sweep sweep.yaml
wandb agent <sweep-id>
```

---

## Quick Reference

### Essential Commands

```bash
# Activate environment
mamba activate act_training

# Basic training
python scripts/train_act.py \
  --config configs/policy_act.yaml \
  --dataset_dir /path/to/dataset \
  --output_dir runs/exp \
  --device cuda

# With WandB
python scripts/train_act.py \
  --config configs/policy_act.yaml \
  --dataset_dir /path/to/dataset \
  --output_dir runs/exp \
  --device cuda \
  --wandb_project "my-project"

# Monitor GPU
watch -n 1 nvidia-smi

# Check environment
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### Important Files

- `configs/policy_act.yaml` - Main config file
- `scripts/train_act.py` - Training script
- `policy/act/` - ACT implementation
- `train/act_trainer.py` - Training loop
- `demo_training.sh` - Automated training

### Useful Links

- Original ACT paper: https://arxiv.org/abs/2304.13705
- WandB docs: https://docs.wandb.ai/
- PyTorch docs: https://pytorch.org/docs/

---

## Tips for Success

1. **Start small**: Test with 100 episodes and 100 epochs first
2. **Monitor closely**: Watch loss curves and GPU utilization
3. **Tune gradually**: Change one hyperparameter at a time
4. **Save often**: Use frequent checkpointing
5. **Visualize**: Plot observations and actions to verify data
6. **Use WandB**: Track all experiments systematically
7. **Be patient**: Good policies take 1000-2000 epochs

---

## Summary

**You're ready to train!** 🚀

```bash
# Three simple steps:
cd /home_shared/grail_andre/code/bimaminobolonana
mamba activate act_training
python scripts/train_act.py \
  --config configs/policy_act.yaml \
  --dataset_dir /path/to/dataset \
  --output_dir runs/experiment \
  --device cuda \
  --wandb_project "my-training"
```

**Environment**: ✅ Ready
**GPUs**: ✅ 2x RTX 5090 working
**Code**: ✅ ACT implementation complete
**Monitoring**: ✅ WandB integrated

**Happy Training!** 🎉
