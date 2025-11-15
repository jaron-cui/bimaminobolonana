import torch
import numpy as np
from pathlib import Path
import os
from tqdm import tqdm

from robot.sim import BimanualSim, randomize_block_position
from robot.visualize import save_frames_to_video

from train.dataset import TensorBimanualObs
from train.dataset import TensorBimanualAction
from models.bimanual_pri3d_actor import BimanualPri3DActor   

current_file = Path(__file__).resolve()
project_root = current_file.parent

# =============== CONFIG ==================
CHECKPOINT = "/mnt/data/out/training-pri3d/110625-221758_pri3d-bc-unfrozen/checkpoint/bc-pretrain/95.pt" # Path to trained model checkpoint
IMAGE_DIMS = (128, 128)
MAX_STEPS = 600
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = BimanualPri3DActor(
    encoder_cfg_path=str(Path(project_root) / "configs/encoder_pri3d_pretrained.yaml"),
    freeze_encoder=True
).to(DEVICE)
state_dict = torch.load(CHECKPOINT, map_location=DEVICE)
model.load_state_dict(state_dict)
model.eval()
print(f"Loaded checkpoint: {CHECKPOINT}")

# Create MuJoCo simulator
sim = BimanualSim(
    merge_xml_files=['robot/block.xml'],
    camera_dims=IMAGE_DIMS,
    obs_camera_names=['wrist_cam_left', 'wrist_cam_right'],
    on_mujoco_init=randomize_block_position
)

# Rollout storage
left_frames, right_frames = [], []

obs = sim.get_obs()

# BC expects tensors
def to_tensor_obs(obs):
    visual = torch.tensor(obs.visual, dtype=torch.float32)[None].to(DEVICE)
    if hasattr(obs.qpos, "array"):   
        qpos_np = obs.qpos.array
    else:                            
        qpos_np = obs.qpos

    if hasattr(obs.qvel, "array"):
        qvel_np = obs.qvel.array
    else:
        qvel_np = obs.qvel

    qpos = torch.tensor(qpos_np, dtype=torch.float32)[None].to(DEVICE)
    qvel = torch.tensor(qvel_np, dtype=torch.float32)[None].to(DEVICE)

    return TensorBimanualObs(visual, qpos, qvel)


# Simulation rollout
for step in tqdm(range(MAX_STEPS)):
    # Save images for video
    left_frames.append(obs.visual[0])
    right_frames.append(obs.visual[1])

    # Convert to model input
    torch_obs = to_tensor_obs(obs)

    # Predict next action
    with torch.no_grad():
        action = model(torch_obs).array[0].cpu().numpy()

    # Step simulation
    obs = sim.step(action)

print("Rollout finished.")

# Save outputs
os.makedirs("out/render-pri3d", exist_ok=True)

left_path = "out/render-pri3d/left.mp4"
right_path = "out/render-pri3d/right.mp4"

save_frames_to_video(left_frames, left_path)
save_frames_to_video(right_frames, right_path)

print("Videos saved:")
print(left_path)
print(right_path)
