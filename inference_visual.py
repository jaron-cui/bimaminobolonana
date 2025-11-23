import os
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

from robot.sim import BimanualSim
from robot.visualize import save_frames_to_video

from train.dataset import TensorBimanualObs
from policy.act.policy_visual import VisualOnlyMLPPolicy

import yaml

# =============== CONFIG ==================
CONFIG_PATH = "configs/policy_visual.yaml"   
CHECKPOINT  = "out/visual_only_mlp_bc/epoch_150.pt"  

IMAGE_DIMS = (128, 128)
MAX_STEPS  = 600
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============== LOAD CONFIG & MODEL ==================
with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

encoder_cfg = cfg["encoder"]
hidden_dim  = cfg.get("hidden_dim", 512)

policy = VisualOnlyMLPPolicy(
    encoder_cfg=encoder_cfg,
    hidden_dim=hidden_dim,
).to(DEVICE)

state_dict = torch.load(CHECKPOINT, map_location=DEVICE)
policy.load_state_dict(state_dict)
policy.eval()

print(f"Loaded VisualOnlyMLPPolicy checkpoint from: {CHECKPOINT}")


# =============== CREATE SIM ==================
sim = BimanualSim(
    merge_xml_files=['robot/block.xml'],
    camera_dims=(128, 128),
    obs_camera_names=['wrist_cam_left','wrist_cam_right'],
    on_mujoco_init=lambda m,d:(m,d)   
)

obs = sim.get_obs()


# =============== OBS → TENSOR ==================
def to_tensor_obs_left_only(obs) -> TensorBimanualObs:

    # obs.visual: [2, H, W, 3] (left, right)
    visual = torch.tensor(obs.visual, dtype=torch.float32)[None].to(DEVICE)  # [1, 2, H, W, 3]

    qpos = torch.zeros((1, 1), dtype=torch.float32, device=DEVICE)
    qvel = torch.zeros((1, 1), dtype=torch.float32, device=DEVICE)

    return TensorBimanualObs(visual=visual, qpos=qpos, qvel=qvel)


# =============== ROLLOUT & RECORD FRAMES ==================
left_frames  = []
right_frames = []

for step in tqdm(range(MAX_STEPS)):
    left_frames.append(obs.visual[0])   
    right_frames.append(obs.visual[1]) 

    tensor_obs = to_tensor_obs_left_only(obs)

    with torch.no_grad():
        action = policy(tensor_obs).array[0].cpu().numpy()  # [14]

    obs = sim.step(action)

print("Rollout finished.")


# =============== SAVE VIDEOS ==================
os.makedirs("out/render-visual-mlp", exist_ok=True)

left_path  = "out/render-visual-mlp/left.mp4"
right_path = "out/render-visual-mlp/right.mp4"

save_frames_to_video(left_frames, left_path)
save_frames_to_video(right_frames, right_path)

print("Videos saved:")
print("  Left :", left_path)
print("  Right:", right_path)
