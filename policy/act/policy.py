"""
Action Chunking Transformer (ACT) Policy.

Implements the ACT policy from "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware"
(https://github.com/tonyzhaozh/act)

Key features:
- CVAE (Conditional Variational Autoencoder) for action sequence generation
- Transformer encoder-decoder for temporal modeling
- Swappable visual encoders (CLIP, Pri3D, etc.)
- Action chunking for smooth, multi-step predictions
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder import build_encoder
from encoder.base import MultiViewEncoder
from train.dataset import TensorBimanualObs, TensorBimanualAction, JOINT_OBSERVATION_SIZE, ACTION_SIZE
from train.trainer import BimanualActor
from policy.act.detr.transformer import build_transformer


class ACTPolicy(BimanualActor):
    """
    Action Chunking Transformer policy with CVAE for diverse action generation.

    The policy processes visual observations through a visual encoder, combines them
    with proprioceptive information (qpos, qvel), and uses a transformer to predict
    a sequence of future actions (action chunk).
    """

    def __init__(
        self,
        encoder_cfg: Dict,
        chunk_size: int = 50,
        temporal_context: int = 3,
        hidden_dim: int = 512,
        nheads: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 7,
        dim_feedforward: int = 3200,
        latent_dim: int = 32,
        dropout: float = 0.1,
        camera_names: Optional[list] = None,
    ):
        """
        Initialize ACT policy.

        Args:
            encoder_cfg: Configuration dict for building the visual encoder
            chunk_size: Number of future actions to predict
            temporal_context: Number of past observations to use
            hidden_dim: Transformer hidden dimension
            nheads: Number of attention heads
            num_encoder_layers: Number of transformer encoder layers
            num_decoder_layers: Number of transformer decoder layers
            dim_feedforward: Dimension of feedforward network
            latent_dim: Dimension of CVAE latent variable
            dropout: Dropout rate
            camera_names: List of camera names (for reference, not used in forward)
        """
        super().__init__()

        self.chunk_size = chunk_size
        self.temporal_context = temporal_context
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.camera_names = camera_names or ['left', 'right']

        # Visual encoder (swappable via config)
        self.visual_encoder: MultiViewEncoder = build_encoder(encoder_cfg)
        visual_feat_dim = self.visual_encoder.out_dim

        # Proprioception encoder: qpos + qvel -> hidden_dim
        # We have 2 * JOINT_OBSERVATION_SIZE for qpos and qvel
        self.proprio_encoder = nn.Linear(2 * JOINT_OBSERVATION_SIZE, hidden_dim)

        # Temporal context encoder: encodes the temporal_context observations
        # Input: temporal_context * (visual_feat_dim + hidden_dim)
        # Output: hidden_dim
        self.temporal_encoder = nn.Linear(
            temporal_context * (visual_feat_dim + hidden_dim),
            hidden_dim
        )

        # CVAE: Encoder (posterior) and prior networks
        # Encoder: takes current observation + ground truth actions -> latent distribution
        self.latent_encoder = nn.Sequential(
            nn.Linear(hidden_dim + chunk_size * ACTION_SIZE, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim * 2)  # mean and log_var
        )

        # Prior: takes only current observation -> latent distribution
        self.latent_prior = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim * 2)  # mean and log_var
        )

        # Latent projection: project sampled latent to hidden_dim
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)

        # Transformer for action sequence prediction
        self.transformer = build_transformer(
            hidden_dim=hidden_dim,
            dropout=dropout,
            nheads=nheads,
            dim_feedforward=dim_feedforward,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            normalize_before=False,
            return_intermediate_dec=False,
        )

        # Query embeddings for action sequence (learnable)
        self.query_embed = nn.Embedding(chunk_size, hidden_dim)

        # Action head: transformer output -> actions
        self.action_head = nn.Linear(hidden_dim, ACTION_SIZE)

        # Store latent distribution for loss computation
        self.latent_mean = None
        self.latent_logvar = None
        self.is_training = True

    def encode_observations(self, obs: TensorBimanualObs) -> torch.Tensor:
        """
        Encode observations (visual + proprioception) into a single feature vector.

        Args:
            obs: Bimanual observations with visual, qpos, qvel

        Returns:
            Encoded observation features [batch_size, hidden_dim]
        """
        batch_size = obs.visual.shape[0]

        # Visual encoding: [batch, num_cameras, H, W, 3] -> [batch, visual_feat_dim]
        # Need to reshape for encoder: expects (left_imgs, right_imgs)
        # visual shape: [batch, 2, H, W, 3]
        left_imgs = obs.visual[:, 0]  # [batch, H, W, 3]
        right_imgs = obs.visual[:, 1]  # [batch, H, W, 3]

        # Permute to [batch, 3, H, W] for encoder
        left_imgs = left_imgs.permute(0, 3, 1, 2)
        right_imgs = right_imgs.permute(0, 3, 1, 2)

        # Resize to 224x224 if needed for CLIP (handles arbitrary input sizes)
        if left_imgs.shape[2] != 224 or left_imgs.shape[3] != 224:
            import torch.nn.functional as F
            left_imgs = F.interpolate(left_imgs, size=(224, 224), mode='bilinear', align_corners=False)
            right_imgs = F.interpolate(right_imgs, size=(224, 224), mode='bilinear', align_corners=False)

        # Encode images (no gradients needed for frozen encoders)
        with torch.set_grad_enabled(self.training):
            visual_feats = self.visual_encoder.encode((left_imgs, right_imgs))['fused']

        # Proprioception encoding
        proprio = torch.cat([obs.qpos.array, obs.qvel.array], dim=-1)  # [batch, 2*JOINT_OBS_SIZE]
        proprio_feats = self.proprio_encoder(proprio)  # [batch, hidden_dim]

        # Combine visual and proprioception
        combined_feats = torch.cat([visual_feats, proprio_feats], dim=-1)  # [batch, visual_feat_dim + hidden_dim]

        return combined_feats

    def encode_temporal_context(self, obs_sequence: TensorBimanualObs) -> torch.Tensor:
        """
        Encode a sequence of observations with temporal context.

        Args:
            obs_sequence: Observations with shape [batch, temporal_context, ...]

        Returns:
            Temporally encoded features [batch, hidden_dim]
        """
        batch_size = obs_sequence.visual.shape[0]
        temporal_len = obs_sequence.visual.shape[1]

        # Flatten batch and temporal dimensions
        # visual: [batch, temporal_context, 2, H, W, 3]
        flat_visual = obs_sequence.visual.reshape(batch_size * temporal_len, *obs_sequence.visual.shape[2:])
        flat_qpos = obs_sequence.qpos.array.reshape(batch_size * temporal_len, -1)
        flat_qvel = obs_sequence.qvel.array.reshape(batch_size * temporal_len, -1)

        # Create flat observations
        from train.dataset import TensorBimanualObs, TensorBimanualState
        flat_obs = TensorBimanualObs(
            visual=flat_visual,
            qpos=TensorBimanualState(flat_qpos),
            qvel=TensorBimanualState(flat_qvel)
        )

        # Encode all timesteps
        flat_feats = self.encode_observations(flat_obs)  # [batch * temporal_context, feat_dim]

        # Reshape back and concatenate temporal dimension
        feats = flat_feats.reshape(batch_size, temporal_len, -1)  # [batch, temporal_context, feat_dim]
        feats = feats.reshape(batch_size, -1)  # [batch, temporal_context * feat_dim]

        # Project to hidden_dim
        temporal_feats = self.temporal_encoder(feats)  # [batch, hidden_dim]

        return temporal_feats

    def sample_latent(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Sample from latent distribution using reparameterization trick.

        Args:
            mean: Latent mean [batch, latent_dim]
            logvar: Latent log variance [batch, latent_dim]

        Returns:
            Sampled latent [batch, latent_dim]
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + eps * std
        else:
            return mean

    def forward(self, obs: TensorBimanualObs, actions: Optional[TensorBimanualAction] = None) -> TensorBimanualAction:
        """
        Forward pass for training or inference.

        Args:
            obs: Current observations (may include temporal context in the visual dimension)
            actions: Ground truth actions for training [batch, chunk_size, ACTION_SIZE]

        Returns:
            Predicted action chunk [batch, chunk_size, ACTION_SIZE]
        """
        # Check if obs has temporal context (shape: [batch, temporal_context, ...])
        if len(obs.visual.shape) == 6 and obs.visual.shape[1] == self.temporal_context:
            # Has temporal context: [batch, temporal, 2, H, W, 3]
            context_feats = self.encode_temporal_context(obs)
        else:
            # Single timestep, encode directly: [batch, 2, H, W, 3]
            context_feats = self.encode_observations(obs)

        batch_size = context_feats.shape[0]

        # CVAE latent variable
        if self.training and actions is not None:
            # Training: use posterior (encoder)
            # Flatten actions for encoder
            actions_flat = actions.array.reshape(batch_size, -1)  # [batch, chunk_size * ACTION_SIZE]
            latent_input = torch.cat([context_feats, actions_flat], dim=-1)
            latent_params = self.latent_encoder(latent_input)
            mean, logvar = torch.chunk(latent_params, 2, dim=-1)

            # Store for KL loss
            self.latent_mean = mean
            self.latent_logvar = logvar

            # Sample latent
            latent = self.sample_latent(mean, logvar)
        else:
            # Inference: use prior
            latent_params = self.latent_prior(context_feats)
            mean, logvar = torch.chunk(latent_params, 2, dim=-1)
            latent = self.sample_latent(mean, logvar)

            self.latent_mean = mean
            self.latent_logvar = logvar

        # Project latent to hidden_dim
        latent_feats = self.latent_proj(latent)  # [batch, hidden_dim]

        # Combine context and latent
        encoder_input = context_feats + latent_feats  # [batch, hidden_dim]
        encoder_input = encoder_input.unsqueeze(1)  # [batch, 1, hidden_dim]

        # Transformer decoding
        query_embed = self.query_embed.weight  # [chunk_size, hidden_dim]

        # Forward through transformer
        hs = self.transformer(
            src=encoder_input,
            query_embed=query_embed,
            pos_embed=None,
            mask=None
        )  # [batch, chunk_size, hidden_dim]

        # Predict actions
        action_pred = self.action_head(hs)  # [batch, chunk_size, ACTION_SIZE]

        # Wrap in TensorBimanualAction (use first action in chunk)
        return TensorBimanualAction(action_pred[:, 0])

    def predict_action_chunk(self, obs: TensorBimanualObs) -> torch.Tensor:
        """
        Predict a full action chunk for inference.

        Args:
            obs: Current observations

        Returns:
            Predicted action chunk [batch, chunk_size, ACTION_SIZE]
        """
        was_training = self.training
        self.eval()

        with torch.no_grad():
            # Encode observations
            if len(obs.visual.shape) == 6 and obs.visual.shape[1] == self.temporal_context:
                context_feats = self.encode_temporal_context(obs)
            else:
                context_feats = self.encode_observations(obs)

            batch_size = context_feats.shape[0]

            # Use prior for latent
            latent_params = self.latent_prior(context_feats)
            mean, logvar = torch.chunk(latent_params, 2, dim=-1)
            latent = mean  # Use mean for deterministic inference

            # Project and combine
            latent_feats = self.latent_proj(latent)
            encoder_input = (context_feats + latent_feats).unsqueeze(1)

            # Transformer
            query_embed = self.query_embed.weight
            hs = self.transformer(
                src=encoder_input,
                query_embed=query_embed,
                pos_embed=None,
                mask=None
            )

            # Predict actions
            action_chunk = self.action_head(hs)  # [batch, chunk_size, ACTION_SIZE]

        self.train(was_training)
        return action_chunk

    def get_kl_loss(self) -> torch.Tensor:
        """
        Compute KL divergence between posterior and prior.

        Returns:
            KL divergence loss
        """
        if self.latent_mean is None or self.latent_logvar is None:
            return torch.tensor(0.0)

        # KL(q||p) for Gaussian: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + self.latent_logvar - self.latent_mean.pow(2) - self.latent_logvar.exp())
        kl_loss = kl_loss / self.latent_mean.shape[0]  # Average over batch

        return kl_loss


def build_act_policy(config: Dict) -> ACTPolicy:
    """
    Build an ACT policy from a configuration dict.

    Args:
        config: Configuration dictionary with encoder and ACT parameters

    Returns:
        ACTPolicy instance
    """
    return ACTPolicy(
        encoder_cfg=config['encoder'],
        chunk_size=config.get('chunk_size', 50),
        temporal_context=config.get('temporal_context', 3),
        hidden_dim=config.get('hidden_dim', 512),
        nheads=config.get('nheads', 8),
        num_encoder_layers=config.get('num_encoder_layers', 4),
        num_decoder_layers=config.get('num_decoder_layers', 7),
        dim_feedforward=config.get('dim_feedforward', 3200),
        latent_dim=config.get('latent_dim', 32),
        dropout=config.get('dropout', 0.1),
    )
