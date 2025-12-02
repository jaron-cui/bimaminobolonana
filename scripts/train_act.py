"""
Training script for ACT (Action Chunking Transformer) policy.
"""

import argparse
import sys
from pathlib import Path
import yaml
import torch
import numpy as np
from omegaconf import OmegaConf

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from policy.act import build_act_policy, ACTTrainer, create_temporal_dataloader
from robot.sim import JOINT_OBSERVATION_SIZE
from train.train_utils import Logs

# WandB support
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed.")


# ============================================================
#   ACT Training Function
# ============================================================

def train_act(
    config_path: str,
    dataset_dir: str,
    output_dir: str,
    device: str = 'cuda',
    resume_from: str = None,
    wandb_project: str = None,
    wandb_name: str = None,
    wandb_tags: list = None,
    action_mean_path: str | None = None,
    action_std_path: str | None = None,
):
    print(f"=== ACT Training ===")
    print(f"Config:   {config_path}")
    print(f"Dataset:  {dataset_dir}")
    print(f"Output:   {output_dir}")
    print(f"Device:   {device}")
    print()

    # ----------------------------------------------
    # Load config
    # ----------------------------------------------
    config = OmegaConf.load(config_path)
    print("Configuration:")
    print(OmegaConf.to_yaml(config))
    print()

    # Device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not found → using CPU")
        device = 'cpu'
    device = torch.device(device)

    # ----------------------------------------------
    # Prepare output directory
    # ----------------------------------------------
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Save config
    OmegaConf.save(config, out_path / "config.yaml")

    # ----------------------------------------------
    # WandB
    # ----------------------------------------------
    use_wandb = wandb_project is not None and WANDB_AVAILABLE
    if use_wandb:
        wandb.init(
            project=wandb_project,
            name=wandb_name,
            tags=wandb_tags,
            config=OmegaConf.to_container(config, resolve=True),
            dir=output_dir,
            resume='allow'
        )
        print(f"WandB initialized: {wandb.run.url}\n")

    # ----------------------------------------------
    # Job logger
    # ----------------------------------------------
    from train.train_utils import Job
    from datetime import datetime

    job = Job(path=out_path, date=datetime.now(), tag='act')

    # ----------------------------------------------
    # Build ACT Policy
    # ----------------------------------------------
    print("Building ACT policy...")
    policy = build_act_policy(config).to(device)

    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)

    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}\n")

    if use_wandb:
        wandb.config.update({
            "total_parameters": total_params,
            "trainable_parameters": trainable_params
        })

    # ----------------------------------------------
    # Resume from checkpoint
    # ----------------------------------------------
    if resume_from:
        checkpoint = torch.load(resume_from, map_location=device)
        policy.load_state_dict(checkpoint)
        print(f"Resumed from: {resume_from}\n")

    # ----------------------------------------------
    # Create DataLoader
    # ----------------------------------------------
    print("Creating data loaders...\n")

    train_loader = create_temporal_dataloader(
        dataset_path=dataset_dir,
        temporal_context=config.temporal_context,
        chunk_size=config.chunk_size,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        action_mean_path=action_mean_path,
        action_std_path=action_std_path,
    )

    print(f"Training samples: {len(train_loader.dataset)}\n")

    # ============================================================
    #             ACTION NORMALIZATION (FINAL VERSION)
    # ============================================================

    print("Computing action normalization...")

    # TemporalBimanualDataset holds the true dataset at .base_dataset
    base = train_loader.dataset.base_dataset

    action_array = base._action_array            # tensor [N, action_dim]
    obs_array    = base._observation_array       # tensor [N, obs_dim]

    # compute visual image tensor size
    visual_shape = (1, 2, base.metadata.camera_height,
                    base.metadata.camera_width, 3)
    visual_size = int(np.prod(visual_shape))

    # extract qpos from observation
    qpos = obs_array[:, visual_size:visual_size + JOINT_OBSERVATION_SIZE]  # [N, 16]

    # build approximate "stay-still" action
    approx = torch.cat([
        qpos[:, :7],     # left arm
        qpos[:, 8:15],   # right arm
    ], dim=1)

    approx[:, 6]  = qpos[:, 6]  * 10  # left gripper
    approx[:, 13] = qpos[:, 14] * 10  # right gripper

    # relative actions
    rel = action_array - approx
    rel[:, 6]  = action_array[:, 6]
    rel[:, 13] = action_array[:, 13]

    # compute mean/std
    mean = rel.mean(dim=0)
    std  = rel.std(dim=0)

    policy.action_mean[:] = mean
    policy.action_std[:]  = std

    print("✓ Action normalization computed")
    print("Mean =", mean)
    print("Std  =", std)
    print()

    # ----------------------------------------------
    # Validation loader (optional)
    # ----------------------------------------------
    val_loader = None
    if config.get("val_split", 0) > 0:
        val_loader = create_temporal_dataloader(
            dataset_path=dataset_dir,
            temporal_context=config.temporal_context,
            chunk_size=config.chunk_size,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
            action_mean_path=action_mean_path,
            action_std_path=action_std_path,
        )
        print(f"Validation samples: {len(val_loader.dataset)}\n")

    # ----------------------------------------------
    # Trainer
    # ----------------------------------------------
    print("Creating ACT trainer...\n")

    trainer = ACTTrainer(
        dataloader=train_loader,
        val_dataloader=val_loader,
        checkpoint_frequency=config.checkpoint_frequency,
        job=job,
        kl_weight=config.kl_weight,
        lr=config.lr,
        action_loss_type=config.get("action_loss", "l1"),
        temporal_ensemble=config.get("temporal_ensemble", False),
        on_log_message=print,
        use_wandb=use_wandb,
    )

    # ----------------------------------------------
    # Train
    # ----------------------------------------------
    print("Starting training...")
    print(f"Training for {config.num_epochs} epochs.\n")

    trainer.train(policy, num_epochs=config.num_epochs)

    # ----------------------------------------------
    # Save final model
    # ----------------------------------------------
    final_path = job.save_checkpoint(policy, "act-train", config.num_epochs)
    print(f"\nTraining complete! Final model saved at:\n{final_path}\n")

    if use_wandb:
        wandb.finish()
        print("WandB finished.")


# ============================================================
#   MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Train ACT policy")

    parser.add_argument("--config_path", required=True)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume_from", default=None)

    parser.add_argument("--wandb_project", default=None)
    parser.add_argument("--wandb_name", default=None)
    parser.add_argument("--wandb_tags", nargs="+", default=None)

    parser.add_argument(
        '--action_mean',
        type=str,
        default=None,
        help='Path to action mean .npy for normalization (optional)'
    )
    parser.add_argument(
        '--action_std',
        type=str,
        default=None,
        help='Path to action std .npy for normalization (optional)'
    )

    args = parser.parse_args()
    train_act(
        config_path=args.config_path,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        device=args.device,
        resume_from=args.resume_from,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        wandb_tags=args.wandb_tags,
        action_mean_path=args.action_mean,
        action_std_path=args.action_std,
    )


if __name__ == "__main__":
    main()
