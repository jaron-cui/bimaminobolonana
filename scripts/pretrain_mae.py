#!/usr/bin/env python3
"""
MAE Pretraining Script for Bimanual Visual Encoder.

Uses CLIP ViT backbone with cross-attention between left/right camera views.
Inspired by "Touch in the Wild" cross-modal attention approach.

Usage:
    python scripts/pretrain_mae.py --config configs/pretrain_mae.yaml
    python scripts/pretrain_mae.py --config configs/pretrain_mae.yaml --data_dirs pretrain_encoder_data bc-train-data-test
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from encoder.mae_bimanual import BimanualCrossMAE
from train.mae_dataset import create_mae_dataloader, denormalize_clip

# Try to import wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_optimizer(
    model: BimanualCrossMAE,
    base_lr: float,
    clip_lr: float,
    weight_decay: float,
    betas: tuple = (0.9, 0.95),
) -> AdamW:
    """
    Create optimizer with different learning rates for CLIP backbone.
    """
    # Separate parameters
    clip_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'clip_extractor' in name:
            clip_params.append(param)
        else:
            other_params.append(param)

    param_groups = [
        {'params': clip_params, 'lr': clip_lr, 'name': 'clip'},
        {'params': other_params, 'lr': base_lr, 'name': 'other'},
    ]

    return AdamW(param_groups, betas=betas, weight_decay=weight_decay)


def create_scheduler(
    optimizer: AdamW,
    warmup_epochs: int,
    total_epochs: int,
    steps_per_epoch: int,
) -> SequentialLR:
    """
    Create learning rate scheduler with warmup.
    """
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=warmup_steps,
    )

    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=1e-6,
    )

    return SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )


def visualize_reconstruction(
    model: BimanualCrossMAE,
    batch: Dict[str, torch.Tensor],
    epoch: int,
    save_dir: Path,
    use_wandb: bool = False,
) -> Optional[Dict]:
    """
    Visualize MAE reconstructions.

    For cross_view_mode: shows both cases (left masked, right masked).
    For standard mode: shows both views masked together.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not available for visualization")
        return None

    model.eval()
    with torch.no_grad():
        left_img = batch["left_img"][:4].cuda()
        right_img = batch["right_img"][:4].cuda()

        if model.cross_view_mode:
            # For cross-view mode, show BOTH cases
            result_mask_left = model.get_reconstruction(left_img, right_img, force_mask_view="left")
            result_mask_right = model.get_reconstruction(left_img, right_img, force_mask_view="right")

            # Create larger figure for cross-view mode (showing both cases)
            fig, axes = plt.subplots(4, 8, figsize=(24, 12))

            for i in range(4):
                # Case 1: LEFT masked, RIGHT visible -> reconstruct LEFT
                orig_left = denormalize_clip(result_mask_left["original_left"][i:i+1]).cpu()[0]
                orig_right = denormalize_clip(result_mask_left["original_right"][i:i+1]).cpu()[0]
                masked_left = denormalize_clip(result_mask_left["masked_left"][i:i+1]).cpu()[0]
                recon_left = result_mask_left["recon_left"][i].cpu().clamp(0, 1)

                axes[i, 0].imshow(masked_left.permute(1, 2, 0))
                axes[i, 0].set_title("Left (Masked)" if i == 0 else "")
                axes[i, 0].axis('off')

                axes[i, 1].imshow(orig_right.permute(1, 2, 0))
                axes[i, 1].set_title("Right (Visible)" if i == 0 else "")
                axes[i, 1].axis('off')

                axes[i, 2].imshow(recon_left.permute(1, 2, 0))
                axes[i, 2].set_title("Left Recon" if i == 0 else "")
                axes[i, 2].axis('off')

                axes[i, 3].imshow(orig_left.permute(1, 2, 0))
                axes[i, 3].set_title("Left GT" if i == 0 else "")
                axes[i, 3].axis('off')

                # Case 2: RIGHT masked, LEFT visible -> reconstruct RIGHT
                orig_left2 = denormalize_clip(result_mask_right["original_left"][i:i+1]).cpu()[0]
                masked_right = denormalize_clip(result_mask_right["masked_right"][i:i+1]).cpu()[0]
                recon_right = result_mask_right["recon_right"][i].cpu().clamp(0, 1)
                orig_right2 = denormalize_clip(result_mask_right["original_right"][i:i+1]).cpu()[0]

                axes[i, 4].imshow(orig_left2.permute(1, 2, 0))
                axes[i, 4].set_title("Left (Visible)" if i == 0 else "")
                axes[i, 4].axis('off')

                axes[i, 5].imshow(masked_right.permute(1, 2, 0))
                axes[i, 5].set_title("Right (Masked)" if i == 0 else "")
                axes[i, 5].axis('off')

                axes[i, 6].imshow(recon_right.permute(1, 2, 0))
                axes[i, 6].set_title("Right Recon" if i == 0 else "")
                axes[i, 6].axis('off')

                axes[i, 7].imshow(orig_right2.permute(1, 2, 0))
                axes[i, 7].set_title("Right GT" if i == 0 else "")
                axes[i, 7].axis('off')

            plt.suptitle(f"Epoch {epoch} - Cross-View Completion", fontsize=14)

        else:
            # Standard mode: both views masked
            result = model.get_reconstruction(left_img, right_img)

            # Denormalize for visualization
            orig_left = denormalize_clip(result["original_left"]).cpu()
            orig_right = denormalize_clip(result["original_right"]).cpu()
            masked_left = denormalize_clip(result["masked_left"]).cpu()
            masked_right = denormalize_clip(result["masked_right"]).cpu()
            recon_left = result["recon_left"].cpu().clamp(0, 1)
            recon_right = result["recon_right"].cpu().clamp(0, 1)

            # Create figure
            fig, axes = plt.subplots(4, 6, figsize=(18, 12))

            for i in range(4):
                # Left view: original, masked, reconstructed
                axes[i, 0].imshow(orig_left[i].permute(1, 2, 0))
                axes[i, 0].set_title("Left Original" if i == 0 else "")
                axes[i, 0].axis('off')

                axes[i, 1].imshow(masked_left[i].permute(1, 2, 0))
                axes[i, 1].set_title("Left Masked" if i == 0 else "")
                axes[i, 1].axis('off')

                axes[i, 2].imshow(recon_left[i].permute(1, 2, 0))
                axes[i, 2].set_title("Left Recon" if i == 0 else "")
                axes[i, 2].axis('off')

                # Right view: original, masked, reconstructed
                axes[i, 3].imshow(orig_right[i].permute(1, 2, 0))
                axes[i, 3].set_title("Right Original" if i == 0 else "")
                axes[i, 3].axis('off')

                axes[i, 4].imshow(masked_right[i].permute(1, 2, 0))
                axes[i, 4].set_title("Right Masked" if i == 0 else "")
                axes[i, 4].axis('off')

                axes[i, 5].imshow(recon_right[i].permute(1, 2, 0))
                axes[i, 5].set_title("Right Recon" if i == 0 else "")
                axes[i, 5].axis('off')

            plt.suptitle(f"Epoch {epoch} Reconstructions", fontsize=14)

    plt.tight_layout()

    # Save figure
    save_path = save_dir / f"recon_epoch_{epoch:04d}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')

    # Log to wandb if available
    wandb_result = None
    if use_wandb and WANDB_AVAILABLE:
        wandb_result = {"reconstructions": wandb.Image(fig)}

    plt.close(fig)
    model.train()

    return wandb_result


def train_epoch(
    model: BimanualCrossMAE,
    dataloader: torch.utils.data.DataLoader,
    optimizer: AdamW,
    scheduler: SequentialLR,
    epoch: int,
    config: dict,
    global_step: int,
) -> tuple:
    """
    Train for one epoch.

    Returns:
        avg_loss: Average loss for the epoch
        global_step: Updated global step counter
    """
    model.train()

    total_loss = 0.0
    num_batches = 0

    log_every = config.get("logging", {}).get("log_every", 50)
    grad_clip = config.get("training", {}).get("grad_clip", 1.0)
    use_wandb = config.get("logging", {}).get("use_wandb", False) and WANDB_AVAILABLE

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch in pbar:
        # Move to GPU
        left_img = batch["left_img"].cuda()
        right_img = batch["right_img"].cuda()

        # Forward pass
        # model() returns loss tensor (shape (num_gpus,) with DataParallel)
        optimizer.zero_grad()
        loss = model(left_img, right_img).mean()  # Average across GPUs

        # Backward pass
        loss.backward()

        # Gradient clipping
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        scheduler.step()

        # Accumulate losses
        total_loss += loss.item()
        num_batches += 1
        global_step += 1

        # Update progress bar
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "lr": f"{scheduler.get_last_lr()[0]:.2e}",
        })

        # Log to wandb (every step for detailed monitoring)
        if use_wandb:
            log_dict = {
                "train/loss": loss.item(),
                "train/lr": scheduler.get_last_lr()[0],
                "train/lr_clip": scheduler.get_last_lr()[1] if len(scheduler.get_last_lr()) > 1 else scheduler.get_last_lr()[0],
                "train/epoch": epoch,
                "train/step": global_step,
            }

            # Add gradient norm for monitoring training health (every log_every steps)
            if global_step % log_every == 0:
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
                total_norm = total_norm ** 0.5
                log_dict["train/grad_norm"] = total_norm

            wandb.log(log_dict, step=global_step)

    avg_loss = total_loss / num_batches

    return avg_loss, 0.0, 0.0, global_step  # No per-view loss with DataParallel


def validate(
    model: BimanualCrossMAE,
    dataloader: torch.utils.data.DataLoader,
) -> tuple:
    """
    Validate model.

    Returns:
        avg_loss, avg_loss_left (or 0), avg_loss_right (or 0)
    """
    model.eval()

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            left_img = batch["left_img"].cuda()
            right_img = batch["right_img"].cuda()

            # Forward pass (model() returns loss tensor, works with DataParallel)
            loss = model(left_img, right_img).mean()
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches

    return avg_loss, 0.0, 0.0  # No per-view loss with simplified forward


def main():
    parser = argparse.ArgumentParser(description="MAE Pretraining for Bimanual Encoder")
    parser.add_argument("--config", type=str, default="configs/pretrain_mae.yaml",
                        help="Path to config file")
    parser.add_argument("--data_dirs", nargs="+", type=str, default=None,
                        help="Override data directories from config")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="Override wandb project name")
    parser.add_argument("--wandb_name", type=str, default=None,
                        help="Override wandb run name")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable wandb logging")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override config with command line args
    if args.data_dirs:
        config["data"]["train_dirs"] = args.data_dirs
    if args.wandb_project:
        config["logging"]["wandb_project"] = args.wandb_project
    if args.wandb_name:
        config["logging"]["wandb_name"] = args.wandb_name
    if args.no_wandb:
        config["logging"]["use_wandb"] = False

    # Setup paths
    data_dirs = config["data"]["train_dirs"]
    if isinstance(data_dirs, str):
        data_dirs = [data_dirs]
    data_dirs = [project_root / d for d in data_dirs]

    save_dir = project_root / config["checkpoint"]["save_dir"]
    save_dir.mkdir(parents=True, exist_ok=True)

    # Setup wandb
    use_wandb = config["logging"].get("use_wandb", False) and WANDB_AVAILABLE
    if use_wandb:
        wandb.init(
            project=config["logging"].get("wandb_project", "bimaminobolonana-mae"),
            name=config["logging"].get("wandb_name"),
            config=config,
        )

    print("=" * 60)
    print("MAE Pretraining for Bimanual Encoder")
    print("=" * 60)
    print(f"Data directories: {data_dirs}")
    print(f"Save directory: {save_dir}")
    print(f"Using wandb: {use_wandb}")
    print()

    # Create dataloaders
    print("Loading datasets...")
    data_cfg = config["data"]

    full_dataloader = create_mae_dataloader(
        data_directory=data_dirs if len(data_dirs) > 1 else data_dirs[0],
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        img_size=data_cfg["img_size"],
        augment=data_cfg.get("augment", True),
        shuffle=True,
    )

    # Create validation split
    val_ratio = data_cfg.get("val_ratio", 0.1)
    total_samples = len(full_dataloader.dataset)
    val_samples = int(total_samples * val_ratio)
    train_samples = total_samples - val_samples

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataloader.dataset,
        [train_samples, val_samples],
        generator=torch.Generator().manual_seed(42),
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=data_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
        drop_last=True,
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=data_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
        drop_last=False,
    )

    print(f"Training samples: {train_samples}")
    print(f"Validation samples: {val_samples}")
    print()

    # Create model
    print("Creating model...")
    enc_cfg = config["encoder"]
    cross_cfg = config["cross_attention"]
    dec_cfg = config["decoder"]
    mae_cfg = config["mae"]

    model = BimanualCrossMAE(
        clip_model=enc_cfg["clip_model"],
        pretrained=enc_cfg["pretrained"],
        out_dim=enc_cfg["out_dim"],
        freeze_clip=enc_cfg.get("freeze_clip", False),
        cross_attn_layers=cross_cfg["num_layers"],
        cross_attn_heads=cross_cfg["num_heads"],
        decoder_dim=dec_cfg["embed_dim"],
        decoder_depth=dec_cfg["depth"],
        decoder_heads=dec_cfg["num_heads"],
        mask_ratio=mae_cfg["mask_ratio"],
        norm_pix_loss=mae_cfg.get("norm_pix_loss", True),
        cross_view_mode=mae_cfg.get("cross_view_mode", False),
    ).cuda()

    # Print masking mode info
    if mae_cfg.get("cross_view_mode"):
        print(f"Using Cross-View Completion: mask one view ({mae_cfg['mask_ratio']:.0%}), use other to reconstruct")
        print(f"  Patch size: {model.patch_size}x{model.patch_size}")
        print(f"  Patches per image: {model.num_patches}")
    else:
        print(f"Using standard MAE: mask both views ({mae_cfg['mask_ratio']:.0%})")

    # Multi-GPU support
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f"Using {num_gpus} GPUs with DataParallel")
        model = nn.DataParallel(model)
    else:
        print(f"Using 1 GPU")

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print()

    # Create optimizer and scheduler
    train_cfg = config["training"]

    # Get the underlying model (unwrap DataParallel if needed)
    model_unwrapped = model.module if isinstance(model, nn.DataParallel) else model

    optimizer = create_optimizer(
        model,  # optimizer works with wrapped model
        base_lr=train_cfg["learning_rate"],
        clip_lr=train_cfg.get("clip_lr", train_cfg["learning_rate"] * 0.1),
        weight_decay=train_cfg["weight_decay"],
        betas=tuple(train_cfg.get("betas", [0.9, 0.95])),
    )

    scheduler = create_scheduler(
        optimizer,
        warmup_epochs=train_cfg["warmup_epochs"],
        total_epochs=train_cfg["epochs"],
        steps_per_epoch=len(train_dataloader),
    )

    # Resume from checkpoint
    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")

    if args.resume:
        print(f"Resuming from {args.resume}...")
        checkpoint = torch.load(args.resume)
        # Load to unwrapped model (handles both DataParallel and single GPU)
        model_unwrapped.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint["global_step"]
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        print(f"Resumed from epoch {start_epoch}")

    # Training loop
    print("Starting training...")
    print()

    vis_every = config["logging"].get("vis_every", 5)
    save_every = config["checkpoint"].get("save_every", 10)
    keep_last = config["checkpoint"].get("keep_last", 3)

    # Get a sample batch for visualization
    vis_batch = next(iter(val_dataloader))

    for epoch in range(start_epoch, train_cfg["epochs"]):
        # Train
        train_loss, train_loss_left, train_loss_right, global_step = train_epoch(
            model, train_dataloader, optimizer, scheduler,
            epoch, config, global_step
        )

        # Validate
        val_loss, val_loss_left, val_loss_right = validate(model, val_dataloader)

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        # Log to wandb
        if use_wandb:
            wandb.log({
                "epoch/train_loss": train_loss,
                "epoch/train_loss_left": train_loss_left,
                "epoch/train_loss_right": train_loss_right,
                "epoch/val_loss": val_loss,
                "epoch/val_loss_left": val_loss_left,
                "epoch/val_loss_right": val_loss_right,
                "epoch": epoch,
            }, step=global_step)

        # Visualize reconstructions (use unwrapped model for visualization)
        if epoch % vis_every == 0:
            vis_result = visualize_reconstruction(
                model_unwrapped, vis_batch, epoch, save_dir, use_wandb
            )
            if vis_result and use_wandb:
                wandb.log(vis_result, step=global_step)

        # Save checkpoint (always save unwrapped model state for portability)
        if epoch % save_every == 0 or val_loss < best_val_loss:
            checkpoint = {
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": model_unwrapped.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "best_val_loss": min(best_val_loss, val_loss),
                "config": config,
            }

            # Save periodic checkpoint
            if epoch % save_every == 0:
                ckpt_path = save_dir / f"checkpoint_epoch_{epoch:04d}.pt"
                torch.save(checkpoint, ckpt_path)
                print(f"Saved checkpoint: {ckpt_path}")

                # Remove old checkpoints
                all_ckpts = sorted(save_dir.glob("checkpoint_epoch_*.pt"))
                if len(all_ckpts) > keep_last:
                    for old_ckpt in all_ckpts[:-keep_last]:
                        old_ckpt.unlink()

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = save_dir / "best_model.pt"
                torch.save(checkpoint, best_path)
                print(f"New best model saved: val_loss={val_loss:.4f}")

    # Save final model
    final_path = save_dir / "final_model.pt"
    model_unwrapped.save_pretrained(final_path, config=config)
    print(f"Training complete. Final model saved to {final_path}")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
