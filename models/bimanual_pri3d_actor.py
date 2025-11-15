import torch
import torch.nn as nn
import yaml

from encoder import build_encoder
from train.dataset import TensorBimanualObs, TensorBimanualAction


class BimanualPri3DActor(nn.Module):
    def __init__(
        self,
        encoder_cfg_path,
        state_dim=32,
        hidden_dim=512,
        action_dim=14,
        freeze_encoder=False
    ):
        super().__init__()

        # -----------------------
        # Load Pri3D Visual Encoder
        # -----------------------
        with open(encoder_cfg_path, "r") as f:
            cfg = yaml.safe_load(f)

        self.encoder = build_encoder(cfg)

        # Freeze encoder if needed
        if freeze_encoder:
            self.encoder.eval()
            for p in self.encoder.parameters():
                p.requires_grad = False
        else:
            self.encoder.train()
            for p in self.encoder.parameters():
                p.requires_grad = True

        # -----------------------
        # MLP Fusion for Action Prediction
        # -----------------------
        self.mlp = nn.Sequential(
            nn.Linear(512 + state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )

    def forward(self, obs: TensorBimanualObs) -> TensorBimanualAction:
        """
        obs.visual: (B, 2, H, W, C)
        obs.qpos:   (B, state_subdim)
        obs.qvel:   (B, state_subdim)
        """

        # Convert images to PyTorch format (B, C, H, W)
        left_imgs = obs.visual[:, 0].permute(0, 3, 1, 2)
        right_imgs = obs.visual[:, 1].permute(0, 3, 1, 2)

        # Pri3D encoder returns fused features
        feats = self.encoder.encode((left_imgs, right_imgs))
        fused_feat = feats["fused"]      # shape: (B, 512)

        # Joint state features
        qpos = obs.qpos.array if hasattr(obs.qpos, "array") else obs.qpos
        qvel = obs.qvel.array if hasattr(obs.qvel, "array") else obs.qvel

        state_feat = torch.cat([qpos, qvel], dim=-1)

        # Concatenate representation + state
        x = torch.cat([fused_feat, state_feat], dim=-1)

        # Predict action
        out = self.mlp(x)
        return TensorBimanualAction(out)
