import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder import build_encoder
from train.dataset import TensorBimanualObs, TensorBimanualAction, ACTION_SIZE


class VisualOnlyMLPPolicy(nn.Module):
    """
    Visual-only BC policy:
    """

    def __init__(self, encoder_cfg: dict, hidden_dim: int = 512):
        super().__init__()
        self.action_mean = nn.Buffer(torch.zeros(ACTION_SIZE))
        self.action_std = nn.Buffer(torch.ones(ACTION_SIZE))

        self.encoder = build_encoder(encoder_cfg)
        visual_feat_dim = self.encoder.out_dim

        self.mlp = nn.Sequential(
            nn.Linear(visual_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, ACTION_SIZE),
        )

    def forward(self, obs: TensorBimanualObs) -> TensorBimanualAction:
        device = self.mlp[0].weight.device

        # obs.visual shape = [B, 2, H, W, 3]
        visual = obs.visual

        left = visual[:, 0]   # (B, H, W, 3)

        # NHWC -> NCHW
        left = left.permute(0, 3, 1, 2).float().to(device)

        # resize to CLIP input size
        # open-clip preprocess is 224x224 typically
        left = F.interpolate(left, size=(224, 224),
                            mode="bilinear", align_corners=False)

        feats = self.encoder.forward_single(left)   # shape = [B, out_dim]
        # proprio = obs.qpos.array[:, 6:7]

        # action = self.mlp(torch.cat((feats, proprio), dim=-1))
        action = self.mlp(feats)
        action = action * self.action_std + self.action_mean
        
        return TensorBimanualAction(action)