"""
ACT (Action Chunking Transformer) Policy Implementation.

A complete implementation of the ACT policy from "Learning Fine-Grained Bimanual
Manipulation with Low-Cost Hardware" (Zhao et al., 2023).

Components:
- policy.py: Main ACT policy with CVAE and transformer
- trainer.py: ACT-specific trainer with KL divergence loss
- dataset.py: Temporal dataset wrapper for action chunking
- detr/: Transformer encoder-decoder architecture
"""

from .policy import ACTPolicy, build_act_policy
from .trainer import ACTTrainer
from .dataset import TemporalBimanualDataset, create_temporal_dataloader

__all__ = [
    'ACTPolicy',
    'build_act_policy',
    'ACTTrainer',
    'TemporalBimanualDataset',
    'create_temporal_dataloader',
]
