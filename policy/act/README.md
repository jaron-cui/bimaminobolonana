# ACT: Action Chunking with Transformers

This module contains the complete implementation of the Action Chunking Transformer (ACT) policy for bimanual manipulation.

## Paper
**"Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware"**
Tony Z. Zhao, Vikash Kumar, Sergey Levine, Chelsea Finn
[Project Page](https://tonyzhaozh.github.io/aloha/) | [GitHub](https://github.com/tonyzhaozh/act)

## Module Structure

```
policy/act/
├── __init__.py           # Module exports
├── policy.py             # ACTPolicy - Main policy with CVAE
├── trainer.py            # ACTTrainer - Training loop with KL loss
├── dataset.py            # TemporalBimanualDataset - Temporal context wrapper
├── detr/                 # Transformer architecture
│   ├── transformer.py    # Encoder-decoder transformer
│   └── position_encoding.py  # Positional embeddings
└── README.md             # This file
```

## Quick Start

### Import the Policy

```python
from policy.act import ACTPolicy, build_act_policy
from omegaconf import OmegaConf

# Load configuration
config = OmegaConf.load("configs/policy_act.yaml")

# Build policy
policy = build_act_policy(config)
```

### Training

```python
from policy.act import ACTTrainer, create_temporal_dataloader

# Create dataloader
train_loader = create_temporal_dataloader(
    dataset_path="data/demos",
    temporal_context=3,
    chunk_size=50,
    batch_size=8,
)

# Create trainer
trainer = ACTTrainer(
    dataloader=train_loader,
    kl_weight=10.0,
    lr=1e-5,
)

# Train
trainer.train(policy, num_epochs=2000)
```

### Or use the training script:

```bash
python scripts/train_act.py \
  --config configs/policy_act.yaml \
  --dataset_dir data/demos \
  --output_dir runs/experiment \
  --device cuda
```

## Key Features

### 1. Swappable Visual Encoders
Easily swap between different visual encoders via configuration:

**CLIP (frozen, pretrained):**
```yaml
encoder:
  name: clip_vit
  model_name: ViT-B-32
  pretrained: openai
  freeze: true
  fuse: mean
```

**Pri3D (trainable):**
```yaml
encoder:
  name: pri3d
  variant: resnet50
  freeze: false
  fuse: bilinear
```

### 2. Temporal Context
Uses past observations for better predictions:
- `temporal_context=3`: Use last 3 observations (default)
- Handles episode boundaries correctly
- Padding options: repeat or zero

### 3. Action Chunking
Predicts sequences of future actions:
- `chunk_size=50`: Predict 50 future actions (default)
- Smoother execution than single-step prediction
- Temporal ensembling for even smoother rollouts

### 4. CVAE for Diversity
Conditional VAE generates diverse behaviors:
- Latent variable captures action distribution
- KL divergence regularizes latent space
- Balance diversity vs accuracy with `kl_weight`

## Architecture

```
Input: Observations (visual + proprioception) × temporal_context
  ↓
Visual Encoder (CLIP/Pri3D)
  → Fused visual features [batch, 512]
  ↓
Proprioception Encoder
  → Joint states [batch, hidden_dim]
  ↓
Temporal Context Encoder
  → Aggregated features [batch, hidden_dim]
  ↓
CVAE
  ├─ Training: Posterior (obs + actions) → z
  └─ Inference: Prior (obs only) → z
  ↓
Latent Projection + Context
  ↓
Transformer Decoder
  ├─ Query Embeddings (learnable)
  └─ Cross-attention to context
  ↓
Action Head
  ↓
Output: Action chunks [batch, chunk_size, action_dim]
```

## Configuration

See `configs/policy_act.yaml` for full configuration options.

Key parameters:
- `chunk_size`: Number of actions to predict (default: 50)
- `temporal_context`: Past observations to use (default: 3)
- `hidden_dim`: Transformer dimension (default: 512)
- `kl_weight`: CVAE regularization (default: 10.0)
- `lr`: Learning rate (default: 1e-5)

## Components

### ACTPolicy (`policy.py`)
Main policy class with:
- `forward()`: Training forward pass
- `predict_action_chunk()`: Inference method
- `get_kl_loss()`: Compute KL divergence

### ACTTrainer (`trainer.py`)
Training loop with:
- Action prediction loss (L1/L2)
- KL divergence loss
- Validation support
- Temporal ensembling for evaluation

### TemporalBimanualDataset (`dataset.py`)
Dataset wrapper that:
- Stacks temporal observations
- Extracts action chunks
- Handles episode boundaries
- Custom collate function

### DETR Transformer (`detr/`)
Transformer components:
- Multi-head attention
- Encoder-decoder architecture
- Positional encodings
- Based on DETR from Facebook Research

## Dependencies

- PyTorch
- einops (tensor operations)
- open-clip-torch (for CLIP encoder)
- omegaconf (configuration)

No MuJoCo required for training!

## Examples

See:
- `scripts/train_act.py` - Training script
- `policy/tests/test_act.py` - Unit tests
- `../../ACT_INTEGRATION_GUIDE.md` - Detailed guide

## Citation

```bibtex
@article{zhao2023learning,
  title={Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware},
  author={Zhao, Tony Z and Kumar, Vikash and Levine, Sergey and Finn, Chelsea},
  journal={arXiv preprint arXiv:2304.13705},
  year={2023}
}
```
