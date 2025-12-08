"""
ACT Trainer for Action Chunking Transformer.

Implements training loop with:
- L1/L2 action prediction loss
- KL divergence loss for CVAE regularization
- Temporal ensembling for evaluation
"""

from typing import Callable, Optional
import torch
import torch.nn as nn
from tqdm import tqdm

from policy.act.policy import ACTPolicy
from train.train_utils import Job
from policy.act.dataset import TemporalBimanualDataset

# Import MuJoCo-dependent modules conditionally (only needed for evaluation rollouts)
try:
    from robot.sim import BimanualSim
    MUJOCO_AVAILABLE = True
except ImportError:
    BimanualSim = None
    MUJOCO_AVAILABLE = False

# WandB support
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class ACTTrainer:
    """
    Trainer for ACT policy with CVAE loss.

    Combines action prediction loss (L1 or L2) with KL divergence loss
    for the conditional VAE latent variable.
    """

    def __init__(
        self,
        dataloader: torch.utils.data.DataLoader,
        val_dataloader: Optional[torch.utils.data.DataLoader] = None,
        checkpoint_frequency: int = 10,
        job: Job = None,
        kl_weight: float = 10.0,
        lr: float = 1e-5,
        action_loss_type: str = 'l1',  # 'l1' or 'l2'
        temporal_ensemble: bool = False,
        on_log_message: Callable[[str], None] = print,
        use_wandb: bool = False,
        warmup_epochs: int = 0,
        use_cosine_schedule: bool = False,
    ):
        """
        Initialize ACT trainer.

        Args:
            dataloader: Training data loader with temporal context
            val_dataloader: Validation data loader (optional)
            checkpoint_frequency: Save checkpoint every N epochs
            job: Job instance for saving checkpoints and logs
            kl_weight: Weight for KL divergence loss
            lr: Learning rate (peak learning rate for warmup)
            action_loss_type: Type of action loss ('l1' or 'l2')
            temporal_ensemble: Use temporal ensembling during evaluation
            on_log_message: Logging function
            use_wandb: Enable WandB logging
            warmup_epochs: Number of warmup epochs (0 = no warmup)
            use_cosine_schedule: Use cosine annealing schedule after warmup
        """
        self.dataloader = dataloader
        self.val_dataloader = val_dataloader
        self.checkpoint_frequency = checkpoint_frequency
        self.job = job
        self.kl_weight = kl_weight
        self.lr = lr
        self.temporal_ensemble = temporal_ensemble
        self.log_message = on_log_message
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.warmup_epochs = warmup_epochs
        self.use_cosine_schedule = use_cosine_schedule

        # Loss function for actions
        if action_loss_type == 'l1':
            self.action_criterion = nn.L1Loss()
        elif action_loss_type == 'l2':
            self.action_criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unknown action loss type: {action_loss_type}")

    def train(self, model: ACTPolicy, num_epochs: int):
        """
        Train the ACT policy.

        Args:
            model: ACTPolicy instance to train
            num_epochs: Number of training epochs
        """
        self.log_message(f'Training ACT policy for {num_epochs} epochs.')
        self.log_message(f'KL weight: {self.kl_weight}, LR: {self.lr}')
        if self.warmup_epochs > 0:
            self.log_message(f'Warmup epochs: {self.warmup_epochs}')
        if self.use_cosine_schedule:
            self.log_message(f'Using cosine annealing schedule')

        device = next(model.parameters()).device

        # Use AdamW for better regularization with weight decay
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.lr,
            weight_decay=1e-4
        )

        # Calculate steps per epoch for scheduler
        steps_per_epoch = len(self.dataloader)
        total_steps = num_epochs * steps_per_epoch
        warmup_steps = self.warmup_epochs * steps_per_epoch

        self.log_message(f'Total steps: {total_steps}, Warmup steps: {warmup_steps}')

        # Setup learning rate scheduler
        scheduler = None
        if self.warmup_epochs > 0 or self.use_cosine_schedule:
            scheduler = self._create_scheduler(optimizer, total_steps, warmup_steps)

        global_step = 0  # Track global training step for WandB

        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0
            epoch_action_loss = 0
            epoch_kl_loss = 0
            num_batches = 0

            for obs_sequence, action_chunk in tqdm(self.dataloader, desc=f'Epoch {epoch}'):
                # Move to device
                obs_sequence = obs_sequence.to(device)
                action_chunk = action_chunk.to(device)

                # Forward pass
                optimizer.zero_grad()

                # ACT forward returns first action, but we need the full chunk
                # So we'll manually call the forward and get predictions
                # Encode observations (dataloader provides temporal context: [batch, temporal, 2, H, W, 3])
                if len(obs_sequence.visual.shape) == 6:
                    context_feats = model.encode_temporal_context(obs_sequence)
                else:
                    context_feats = model.encode_observations(obs_sequence)

                batch_size = context_feats.shape[0]

                # CVAE with ground truth actions
                actions_flat = action_chunk.array  # [batch, chunk_size, ACTION_SIZE]
                actions_for_encoder = actions_flat.reshape(batch_size, -1)  # [batch, chunk_size * ACTION_SIZE]

                # Posterior
                latent_input = torch.cat([context_feats, actions_for_encoder], dim=-1)
                latent_params = model.latent_encoder(latent_input)
                mean_post, logvar_post = torch.chunk(latent_params, 2, dim=-1)
                latent = model.sample_latent(mean_post, logvar_post)

                # Prior (for KL)
                latent_params_prior = model.latent_prior(context_feats)
                mean_prior, logvar_prior = torch.chunk(latent_params_prior, 2, dim=-1)

                # Store for KL loss
                model.latent_mean = mean_post
                model.latent_logvar = logvar_post

                # Decode
                latent_feats = model.latent_proj(latent)
                encoder_input = (context_feats + latent_feats).unsqueeze(1)

                query_embed = model.query_embed.weight
                hs = model.transformer(
                    src=encoder_input,
                    query_embed=query_embed,
                    pos_embed=None,
                    mask=None
                )

                action_pred = model.action_head(hs)  # [batch, chunk_size, ACTION_SIZE]

                # Action loss
                action_loss = self.action_criterion(action_pred, actions_flat)

                # KL divergence loss
                kl_loss = model.get_kl_loss()

                # Total loss
                loss = action_loss + self.kl_weight * kl_loss

                # Backward pass
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                # Get current learning rate
                current_lr = optimizer.param_groups[0]['lr']

                # Log per-step metrics to WandB
                if self.use_wandb:
                    wandb.log({
                        'train_step/loss': loss.item(),
                        'train_step/action_loss': action_loss.item(),
                        'train_step/kl_loss': kl_loss.item(),
                        'train_step/learning_rate': current_lr,
                        'epoch': epoch,
                    }, step=global_step)

                global_step += 1

                # Logging for epoch summary
                epoch_loss += loss.item()
                epoch_action_loss += action_loss.item()
                epoch_kl_loss += kl_loss.item()
                num_batches += 1

            # Epoch summary
            avg_loss = epoch_loss / num_batches
            avg_action_loss = epoch_action_loss / num_batches
            avg_kl_loss = epoch_kl_loss / num_batches

            log_msg = (
                f'Epoch {epoch}: '
                f'Loss={avg_loss:.4f}, '
                f'Action={avg_action_loss:.4f}, '
                f'KL={avg_kl_loss:.4f}, '
                f'LR={current_lr:.2e}'
            )

            # Validation
            val_loss = None
            if self.val_dataloader is not None:
                val_loss = self.validate(model, device)
                log_msg += f', Val={val_loss:.4f}'

            self.log_message(log_msg)

            # WandB epoch summary logging
            if self.use_wandb:
                log_dict = {
                    'epoch': epoch,
                    'train_epoch/loss': avg_loss,
                    'train_epoch/action_loss': avg_action_loss,
                    'train_epoch/kl_loss': avg_kl_loss,
                    'train_epoch/learning_rate': current_lr,
                }
                if val_loss is not None:
                    log_dict['val/loss'] = val_loss
                wandb.log(log_dict, step=global_step)

            # Checkpoint
            if epoch % self.checkpoint_frequency == 0 and self.job is not None:
                self.job.save_checkpoint(model, 'act-train', epoch)
                self.log_message(f'  Saved checkpoint at epoch {epoch}')

    def validate(self, model: ACTPolicy, device: torch.device) -> float:
        """
        Run validation on the validation set.

        Args:
            model: ACTPolicy to validate
            device: Device to run on

        Returns:
            Average validation loss
        """
        model.eval()
        val_loss = 0
        num_batches = 0

        with torch.no_grad():
            for obs_sequence, action_chunk in self.val_dataloader:
                obs_sequence = obs_sequence.to(device)
                action_chunk = action_chunk.to(device)

                # Forward (use prior, not posterior)
                context_feats = model.encode_temporal_context(obs_sequence) if len(obs_sequence.visual.shape) == 6 else model.encode_observations(obs_sequence)

                batch_size = context_feats.shape[0]

                # Prior
                latent_params = model.latent_prior(context_feats)
                mean, logvar = torch.chunk(latent_params, 2, dim=-1)
                latent = mean  # Use mean for validation

                # Decode
                latent_feats = model.latent_proj(latent)
                encoder_input = (context_feats + latent_feats).unsqueeze(1)

                query_embed = model.query_embed.weight
                hs = model.transformer(
                    src=encoder_input,
                    query_embed=query_embed,
                    pos_embed=None,
                    mask=None
                )

                action_pred = model.action_head(hs)

                # Action loss only
                loss = self.action_criterion(action_pred, action_chunk.array)
                val_loss += loss.item()
                num_batches += 1

        model.train()
        return val_loss / num_batches if num_batches > 0 else 0.0

    def evaluate_rollout(
        self,
        model: ACTPolicy,
        sim: BimanualSim,
        num_steps: int = 1000,
        temporal_ensemble_window: int = 10,
    ) -> dict:
        """
        Evaluate policy in simulation with temporal ensembling.

        Args:
            model: ACTPolicy to evaluate
            sim: Bimanual simulation environment
            num_steps: Maximum number of simulation steps
            temporal_ensemble_window: Window size for temporal ensembling

        Returns:
            Dictionary with evaluation metrics
        """
        model.eval()
        device = next(model.parameters()).device

        # Action buffer for temporal ensembling
        action_buffer = []
        chunk_buffer = []

        obs = sim.get_obs()
        obs_history = [obs] * model.temporal_context

        trajectory = []

        with torch.no_grad():
            for step in range(num_steps):
                # Convert obs_history to tensor
                from train.dataset import TensorBimanualObs, TensorBimanualState
                import numpy as np

                # Stack visual observations
                visual_stack = np.stack([o.visual for o in obs_history], axis=0)  # [temporal_context, 2, H, W, 3]
                qpos_stack = np.stack([o.qpos.array for o in obs_history], axis=0)
                qvel_stack = np.stack([o.qvel.array for o in obs_history], axis=0)

                tensor_obs = TensorBimanualObs(
                    visual=torch.from_numpy(visual_stack).unsqueeze(0).to(device),  # [1, temporal_context, 2, H, W, 3]
                    qpos=TensorBimanualState(torch.from_numpy(qpos_stack).unsqueeze(0).to(device)),
                    qvel=TensorBimanualState(torch.from_numpy(qvel_stack).unsqueeze(0).to(device))
                )

                # Predict action chunk
                action_chunk = model.predict_action_chunk(tensor_obs)  # [1, chunk_size, ACTION_SIZE]
                chunk_buffer.append(action_chunk[0].cpu())  # [chunk_size, ACTION_SIZE]

                # Temporal ensembling
                if self.temporal_ensemble and len(chunk_buffer) > 1:
                    # Average overlapping predictions
                    ensemble_size = min(len(chunk_buffer), temporal_ensemble_window)
                    actions_to_average = []

                    for i in range(ensemble_size):
                        chunk_age = i
                        chunk = chunk_buffer[-(i+1)]
                        if chunk_age < chunk.shape[0]:
                            actions_to_average.append(chunk[chunk_age])

                    action = torch.stack(actions_to_average).mean(dim=0)
                else:
                    # Use first action from latest chunk
                    action = chunk_buffer[-1][0]

                # Convert to numpy and step
                from robot.sim import BimanualAction
                action_np = BimanualAction(action.numpy())

                # Step simulation
                obs = sim.step(action_np)
                obs_history.append(obs)
                obs_history.pop(0)

                trajectory.append((obs, action_np))

                # Trim chunk buffer
                if len(chunk_buffer) > temporal_ensemble_window:
                    chunk_buffer.pop(0)

        model.train()

        return {
            'trajectory': trajectory,
            'num_steps': num_steps,
        }

    def _create_scheduler(self, optimizer, total_steps, warmup_steps):
        """
        Create learning rate scheduler with warmup + cosine annealing.
        Uses PyTorch's SequentialLR for standard Warmup -> Cosine schedule.
        Updates per STEP (not per epoch).

        Args:
            optimizer: PyTorch optimizer
            total_steps: Total training steps (epochs * steps_per_epoch)
            warmup_steps: Number of warmup steps

        Returns:
            Learning rate scheduler
        """
        from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

        # Warmup scheduler
        # Linearly increase LR from low value to base_lr over warmup_steps
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.001,
            end_factor=1.0,
            total_iters=warmup_steps
        )

        # Main scheduler (Cosine Annealing)
        # Cosine decay from base_lr to min_lr over remaining steps
        main_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=1e-6  # Minimum LR
        )

        # Combine them
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps]
        )

        return scheduler
