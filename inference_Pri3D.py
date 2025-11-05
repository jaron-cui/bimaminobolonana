import torch
import torch.nn as nn
import numpy as np
import imageio
from pathlib import Path
from robot.sim import BimanualSim, BimanualObs, BimanualAction
from train.dataset import TensorBimanualObs, TensorBimanualAction
from train.trainer import BCTrainer

import yaml
from encoder import build_encoder

class BimanualPri3DActor(nn.Module):
    def __init__(self, encoder_cfg_path="configs/encoder_pri3d_pretrained.yaml",
                 state_dim=32, hidden_dim=512, action_dim=14,
                 freeze_encoder=False):   
        super().__init__()

        # --- Load Pri3D encoder ---
        with open(encoder_cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
        self.encoder = build_encoder(cfg)

        if freeze_encoder:
            self.encoder.eval()
            for p in self.encoder.parameters():
                p.requires_grad = False
        else:
            self.encoder.train()
            for p in self.encoder.parameters():
                p.requires_grad = True

        # --- Fusion MLP ---
        self.mlp = nn.Sequential(
            nn.Linear(512 + state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )

    def forward(self, obs: TensorBimanualObs) -> TensorBimanualAction:
        left_imgs  = obs.visual[:, 0].permute(0, 3, 1, 2)
        right_imgs = obs.visual[:, 1].permute(0, 3, 1, 2)

        feats = self.encoder.encode((left_imgs, right_imgs))
        fused_feat = feats["fused"]

        state_feat = torch.cat([obs.qpos.array, obs.qvel.array], dim=-1)

        x = torch.cat([fused_feat, state_feat], dim=-1)
        out = self.mlp(x)
        return TensorBimanualAction(out)
    
ckpt_path = "out/training-pri3d/pri3d-bc/checkpoints/epoch_100.pt" # path to the trained checkpoint， change if needed
model = BimanualPri3DActor().cuda()
checkpoint = torch.load(ckpt_path, map_location="cuda")
model.load_state_dict(checkpoint["model"])
model.eval()
print("✅ Loaded policy from", ckpt_path)

sim = BimanualSim(camera_dims=(128, 128))   
obs = sim.get_obs()

# method to wrap the model into a policy function
def policy(obs: BimanualObs) -> BimanualAction:
    obs_tensor = TensorBimanualObs(
        visual=torch.from_numpy(obs.visual).unsqueeze(0).to(torch.float32).cuda(),
        qpos=torch.from_numpy(obs.qpos.array).unsqueeze(0).cuda(),
        qvel=torch.from_numpy(obs.qvel.array).unsqueeze(0).cuda()
    )

    with torch.no_grad():
        pred_action = model.forward(obs_tensor)

    return BimanualAction(pred_action.array.squeeze().cpu().numpy())

frames = []
num_steps = 300 # number of steps to run
for step in range(num_steps):
    act = policy(obs)
    obs = sim.step(act)
    left, right = obs.visual
    frame = np.concatenate([left, right], axis=1)    
    frame_uint8 = (frame * 255).astype(np.uint8)
    frames.append(frame_uint8)
    print(f"Step {step+1}/{num_steps}")

save_dir = Path("out/videos")
save_dir.mkdir(parents=True, exist_ok=True)
video_path = save_dir / "pri3d_rollout.mp4"

imageio.mimsave(video_path, frames, fps=30)
print(f"🎬 Video saved to {video_path}")