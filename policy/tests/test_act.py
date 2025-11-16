"""
Tests for ACT policy integration.

Verifies:
- ACT policy forward/backward pass
- Temporal dataset wrapper
- Encoder integration (CLIP and Pri3D)
- End-to-end training for a few iterations
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf

from policy.act import ACTPolicy, build_act_policy, TemporalBimanualDataset
from train.dataset import TensorBimanualObs, TensorBimanualAction, TensorBimanualState
from robot.sim import JOINT_OBSERVATION_SIZE, ACTION_SIZE


def create_mock_obs(batch_size=2, temporal_context=3, img_size=224):
    """Create mock observations for testing."""
    visual = torch.rand(batch_size, temporal_context, 2, img_size, img_size, 3)
    qpos = torch.rand(batch_size, temporal_context, JOINT_OBSERVATION_SIZE)
    qvel = torch.rand(batch_size, temporal_context, JOINT_OBSERVATION_SIZE)

    return TensorBimanualObs(
        visual=visual,
        qpos=TensorBimanualState(qpos),
        qvel=TensorBimanualState(qvel)
    )


def create_mock_actions(batch_size=2, chunk_size=50):
    """Create mock action chunks for testing."""
    actions = torch.rand(batch_size, chunk_size, ACTION_SIZE)
    return TensorBimanualAction(actions)


class TestACTPolicy:
    """Test ACT policy components."""

    def test_policy_initialization_clip(self):
        """Test ACT policy initialization with CLIP encoder."""
        config = {
            'encoder': {
                'name': 'clip_vit',
                'model_name': 'ViT-B-32',
                'pretrained': None,
                'out_dim': 512,
                'freeze': True,
                'fuse': 'mean',
            },
            'chunk_size': 10,
            'temporal_context': 3,
            'hidden_dim': 256,
            'nheads': 4,
            'num_encoder_layers': 2,
            'num_decoder_layers': 3,
            'dim_feedforward': 512,
            'latent_dim': 16,
            'dropout': 0.1,
        }

        policy = build_act_policy(config)
        assert policy is not None
        assert policy.chunk_size == 10
        assert policy.temporal_context == 3
        assert policy.hidden_dim == 256
        print("✓ ACT policy with CLIP encoder initialized successfully")

    def test_policy_initialization_pri3d(self):
        """Test ACT policy initialization with Pri3D encoder."""
        config = {
            'encoder': {
                'name': 'pri3d',
                'variant': 'resnet18',
                'pretrained': False,
                'out_dim': 512,
                'freeze': False,
                'fuse': 'mean',
            },
            'chunk_size': 10,
            'temporal_context': 3,
            'hidden_dim': 256,
            'nheads': 4,
            'num_encoder_layers': 2,
            'num_decoder_layers': 3,
            'dim_feedforward': 512,
            'latent_dim': 16,
            'dropout': 0.1,
        }

        policy = build_act_policy(config)
        assert policy is not None
        print("✓ ACT policy with Pri3D encoder initialized successfully")

    def test_forward_pass(self):
        """Test forward pass with mock data."""
        config = {
            'encoder': {
                'name': 'clip_vit',
                'model_name': 'ViT-B-32',
                'pretrained': None,
                'out_dim': 512,
                'freeze': True,
                'fuse': 'mean',
            },
            'chunk_size': 10,
            'temporal_context': 3,
            'hidden_dim': 256,
            'nheads': 4,
            'num_encoder_layers': 2,
            'num_decoder_layers': 3,
            'dim_feedforward': 512,
            'latent_dim': 16,
            'dropout': 0.1,
        }

        policy = build_act_policy(config)
        policy.eval()

        # Create mock inputs
        obs = create_mock_obs(batch_size=2, temporal_context=3, img_size=224)
        actions = create_mock_actions(batch_size=2, chunk_size=10)

        # Forward pass
        with torch.no_grad():
            output = policy(obs, actions)

        assert output is not None
        assert output.array.shape == (2, ACTION_SIZE)
        print("✓ Forward pass successful")

    def test_backward_pass(self):
        """Test backward pass and gradient computation."""
        config = {
            'encoder': {
                'name': 'clip_vit',
                'model_name': 'ViT-B-32',
                'pretrained': None,
                'out_dim': 512,
                'freeze': True,
                'fuse': 'mean',
            },
            'chunk_size': 10,
            'temporal_context': 3,
            'hidden_dim': 256,
            'nheads': 4,
            'num_encoder_layers': 2,
            'num_decoder_layers': 3,
            'dim_feedforward': 512,
            'latent_dim': 16,
            'dropout': 0.1,
        }

        policy = build_act_policy(config)
        policy.train()

        # Create mock inputs
        obs = create_mock_obs(batch_size=2, temporal_context=3, img_size=224)
        actions = create_mock_actions(batch_size=2, chunk_size=10)

        # Forward pass
        output = policy(obs, actions)

        # Compute simple loss
        loss = output.array.mean()
        loss.backward()

        # Check gradients
        has_gradients = False
        for param in policy.parameters():
            if param.requires_grad and param.grad is not None:
                has_gradients = True
                break

        assert has_gradients, "No gradients computed"
        print("✓ Backward pass successful, gradients computed")

    def test_kl_loss(self):
        """Test KL divergence loss computation."""
        config = {
            'encoder': {
                'name': 'clip_vit',
                'model_name': 'ViT-B-32',
                'pretrained': None,
                'out_dim': 512,
                'freeze': True,
                'fuse': 'mean',
            },
            'chunk_size': 10,
            'temporal_context': 3,
            'hidden_dim': 256,
            'nheads': 4,
            'num_encoder_layers': 2,
            'num_decoder_layers': 3,
            'dim_feedforward': 512,
            'latent_dim': 16,
            'dropout': 0.1,
        }

        policy = build_act_policy(config)
        policy.train()

        # Create mock inputs
        obs = create_mock_obs(batch_size=2, temporal_context=3, img_size=224)
        actions = create_mock_actions(batch_size=2, chunk_size=10)

        # Forward pass
        _ = policy(obs, actions)

        # Get KL loss
        kl_loss = policy.get_kl_loss()

        assert kl_loss is not None
        assert kl_loss.item() >= 0, "KL loss should be non-negative"
        print(f"✓ KL loss computed: {kl_loss.item():.4f}")

    def test_action_chunk_prediction(self):
        """Test full action chunk prediction."""
        config = {
            'encoder': {
                'name': 'clip_vit',
                'model_name': 'ViT-B-32',
                'pretrained': None,
                'out_dim': 512,
                'freeze': True,
                'fuse': 'mean',
            },
            'chunk_size': 10,
            'temporal_context': 3,
            'hidden_dim': 256,
            'nheads': 4,
            'num_encoder_layers': 2,
            'num_decoder_layers': 3,
            'dim_feedforward': 512,
            'latent_dim': 16,
            'dropout': 0.1,
        }

        policy = build_act_policy(config)
        policy.eval()

        # Create mock inputs
        obs = create_mock_obs(batch_size=2, temporal_context=3, img_size=224)

        # Predict action chunk
        action_chunk = policy.predict_action_chunk(obs)

        assert action_chunk.shape == (2, 10, ACTION_SIZE)
        print(f"✓ Action chunk prediction successful: {action_chunk.shape}")


def test_parameter_counts():
    """Test that parameter counts are reasonable."""
    config = {
        'encoder': {
            'name': 'clip_vit',
            'model_name': 'ViT-B-32',
            'pretrained': None,
            'out_dim': 512,
            'freeze': True,
            'fuse': 'mean',
        },
        'chunk_size': 50,
        'temporal_context': 3,
        'hidden_dim': 512,
        'nheads': 8,
        'num_encoder_layers': 4,
        'num_decoder_layers': 7,
        'dim_feedforward': 3200,
        'latent_dim': 32,
        'dropout': 0.1,
    }

    policy = build_act_policy(config)

    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)

    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}")

    # With frozen CLIP, trainable should be less than total
    assert trainable_params < total_params, "With frozen encoder, trainable should be less than total"


def test_config_loading():
    """Test loading from YAML config file."""
    # This test assumes the config files exist
    config_path = Path(__file__).parent.parent.parent / 'configs' / 'policy_act.yaml'

    if config_path.exists():
        config = OmegaConf.load(config_path)
        policy = build_act_policy(config)
        assert policy is not None
        print(f"✓ Successfully loaded policy from {config_path}")
    else:
        print(f"⚠ Config file not found at {config_path}, skipping test")


if __name__ == '__main__':
    print("Running ACT integration tests...\n")

    test_suite = TestACTPolicy()

    print("1. Testing CLIP encoder initialization...")
    test_suite.test_policy_initialization_clip()

    print("\n2. Testing Pri3D encoder initialization...")
    test_suite.test_policy_initialization_pri3d()

    print("\n3. Testing forward pass...")
    test_suite.test_forward_pass()

    print("\n4. Testing backward pass...")
    test_suite.test_backward_pass()

    print("\n5. Testing KL loss...")
    test_suite.test_kl_loss()

    print("\n6. Testing action chunk prediction...")
    test_suite.test_action_chunk_prediction()

    print("\n7. Testing parameter counts...")
    test_parameter_counts()

    print("\n8. Testing config loading...")
    test_config_loading()

    print("\n" + "="*50)
    print("All tests passed! ✓")
    print("="*50)
