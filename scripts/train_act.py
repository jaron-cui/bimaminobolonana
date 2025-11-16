"""
Training script for ACT (Action Chunking Transformer) policy.

Usage:
    python scripts/train_act.py --config configs/policy_act.yaml --dataset_dir data/bimanual_demo --output_dir runs/act_experiment

With WandB:
    python scripts/train_act.py --config configs/policy_act.yaml --dataset_dir data/bimanual_demo --output_dir runs/act_experiment \
        --wandb_project my-project --wandb_name my-run
"""

import argparse
import sys
from pathlib import Path
import yaml
import torch
from omegaconf import OmegaConf

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from policy.act import build_act_policy, ACTTrainer, create_temporal_dataloader
from train.train_utils import Logs

# WandB support
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with: pip install wandb")


def train_act(
    config_path: str,
    dataset_dir: str,
    output_dir: str,
    device: str = 'cuda',
    resume_from: str = None,
    wandb_project: str = None,
    wandb_name: str = None,
    wandb_tags: list = None,
):
    """
    Train ACT policy with the given configuration.

    Args:
        config_path: Path to ACT config YAML file
        dataset_dir: Path to BimanualDataset directory
        output_dir: Output directory for checkpoints and logs
        device: Device to train on ('cuda' or 'cpu')
        resume_from: Optional checkpoint path to resume from
        wandb_project: WandB project name (enables WandB logging)
        wandb_name: WandB run name
        wandb_tags: WandB tags for organizing runs
    """
    print(f"=== ACT Training ===")
    print(f"Config: {config_path}")
    print(f"Dataset: {dataset_dir}")
    print(f"Output: {output_dir}")
    print(f"Device: {device}")
    if wandb_project:
        print(f"WandB Project: {wandb_project}")
        print(f"WandB Run: {wandb_name or 'auto'}")
    print()

    # Load configuration
    config = OmegaConf.load(config_path)
    print("Configuration:")
    print(OmegaConf.to_yaml(config))
    print()

    # Set device
    if device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        device = 'cpu'
    device = torch.device(device)

    # Create output directory and job
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save config to output directory
    config_save_path = output_path / 'config.yaml'
    OmegaConf.save(config, config_save_path)
    print(f"Saved config to {config_save_path}")

    # Initialize WandB if requested
    use_wandb = wandb_project is not None and WANDB_AVAILABLE
    if use_wandb:
        # Convert OmegaConf to dict for WandB
        config_dict = OmegaConf.to_container(config, resolve=True)

        wandb.init(
            project=wandb_project,
            name=wandb_name,
            tags=wandb_tags or [],
            config=config_dict,
            dir=output_dir,
            resume='allow'
        )
        print(f"✓ WandB initialized: {wandb.run.url}")
        print()
    elif wandb_project and not WANDB_AVAILABLE:
        print("Warning: WandB requested but not installed. Continuing without WandB.")
        print()

    # Create job for checkpoint management
    from train.train_utils import Job
    from datetime import datetime
    job = Job(
        path=output_path,
        date=datetime.now(),
        tag='act'
    )

    # Build ACT policy
    print("Building ACT policy...")
    policy = build_act_policy(config)
    policy = policy.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print()

    # Log model info to WandB
    if use_wandb:
        wandb.config.update({
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
        })
        # Note: wandb.watch() removed to avoid logging gradients

    # Load checkpoint if resuming
    if resume_from is not None:
        print(f"Loading checkpoint from {resume_from}...")
        checkpoint = torch.load(resume_from, map_location=device)
        policy.load_state_dict(checkpoint)
        print("Checkpoint loaded successfully")
        print()

    # Create data loaders
    print("Creating data loaders...")
    train_loader = create_temporal_dataloader(
        dataset_path=dataset_dir,
        temporal_context=config.temporal_context,
        chunk_size=config.chunk_size,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for debugging, increase for faster loading
    )
    print(f"Training samples: {len(train_loader.dataset)}")

    # Create validation loader if specified
    val_loader = None
    if config.get('val_split', 0) > 0:
        # For simplicity, we'll use the same dataset for validation
        # In practice, you might want to split the dataset
        val_loader = create_temporal_dataloader(
            dataset_path=dataset_dir,
            temporal_context=config.temporal_context,
            chunk_size=config.chunk_size,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
        )
        print(f"Validation samples: {len(val_loader.dataset)}")
    print()

    # Create trainer
    print("Creating ACT trainer...")
    trainer = ACTTrainer(
        dataloader=train_loader,
        val_dataloader=val_loader,
        checkpoint_frequency=config.checkpoint_frequency,
        job=job,
        kl_weight=config.kl_weight,
        lr=config.lr,
        action_loss_type=config.get('action_loss', 'l1'),
        temporal_ensemble=config.get('temporal_ensemble', False),
        on_log_message=print,
        use_wandb=use_wandb,
    )
    print()

    # Train
    print("Starting training...")
    print(f"Training for {config.num_epochs} epochs")
    print(f"Checkpoints will be saved to {job.path / 'checkpoint'}")
    print()

    trainer.train(policy, num_epochs=config.num_epochs)

    # Save final model
    final_path = job.save_checkpoint(policy, 'act-train', config.num_epochs)
    print()
    print(f"Training complete! Final model saved to {final_path}")

    # Finish WandB run
    if use_wandb:
        wandb.finish()
        print("✓ WandB run finished")


def main():
    parser = argparse.ArgumentParser(description='Train ACT policy for bimanual manipulation')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to ACT configuration YAML file (e.g., configs/policy_act.yaml)'
    )
    parser.add_argument(
        '--dataset_dir',
        type=str,
        required=True,
        help='Path to BimanualDataset directory containing observations.npy and actions.npy'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for checkpoints and logs'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to train on (default: cuda)'
    )
    parser.add_argument(
        '--resume_from',
        type=str,
        default=None,
        help='Optional checkpoint path to resume training from'
    )
    parser.add_argument(
        '--wandb_project',
        type=str,
        default=None,
        help='WandB project name (enables WandB logging)'
    )
    parser.add_argument(
        '--wandb_name',
        type=str,
        default=None,
        help='WandB run name (default: auto-generated)'
    )
    parser.add_argument(
        '--wandb_tags',
        type=str,
        nargs='+',
        default=None,
        help='WandB tags for organizing runs (e.g., --wandb_tags exp1 baseline)'
    )

    args = parser.parse_args()

    train_act(
        config_path=args.config,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        device=args.device,
        resume_from=args.resume_from,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        wandb_tags=args.wandb_tags,
    )


if __name__ == '__main__':
    main()
