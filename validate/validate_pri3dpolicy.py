import os
import sys
from pathlib import Path
sys.path.append(str(Path(os.getcwd()).parent.absolute()))

import torch
from tqdm import tqdm

from robot.sim import BimanualSim, randomize_block_position
from robot.sim import BimanualObs, BimanualAction

from validate.evaluation import evaluate_policy
from models.bimanual_pri3d_actor import BimanualPri3DActor
from train.dataset import TensorBimanualObs


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
# ------------------------
# LOAD PRI3D BC POLICY
# ------------------------

CHECKPOINT = "/mnt/data/out/training-pri3d/110625-221758_pri3d-bc-unfrozen/checkpoint/bc-pretrain/95.pt" # Path to trained model checkpoint
ENCODER_CFG = "/configs/encoder_pri3d_pretrained.yaml"

policy_model = BimanualPri3DActor(
    encoder_cfg_path=str(project_root)+ENCODER_CFG,
    freeze_encoder=True
).to(DEVICE)

state_dict = torch.load(CHECKPOINT, map_location=DEVICE)
policy_model.load_state_dict(state_dict)
policy_model.eval()

print("Loaded Pri3D checkpoint.")

# ------------------------
# OBS → TENSOR FORMATTER
# ------------------------

def to_tensor_obs(obs):
    visual = torch.tensor(obs.visual, dtype=torch.float32)[None].to(DEVICE)

    # handle qpos and qvel type (Tensor or BimanualState)
    qpos = obs.qpos.array if hasattr(obs.qpos, "array") else obs.qpos
    qvel = obs.qvel.array if hasattr(obs.qvel, "array") else obs.qvel

    qpos = torch.tensor(qpos, dtype=torch.float32)[None].to(DEVICE)
    qvel = torch.tensor(qvel, dtype=torch.float32)[None].to(DEVICE)

    return TensorBimanualObs(visual, qpos, qvel)

# ------------------------
# POLICY FUNCTION (Pri3D)
# ------------------------

def pri3d_policy(obs: BimanualObs) -> BimanualAction:
    torch_obs = to_tensor_obs(obs)

    with torch.no_grad():
        action_tensor = policy_model(torch_obs).array[0].cpu().numpy()

    return BimanualAction(action_tensor)

# ------------------------
# SIM CREATOR
# ------------------------

def create_sim() -> BimanualSim:
    sim = BimanualSim(
        merge_xml_files=['block.xml'],
        on_mujoco_init=randomize_block_position,
        obs_camera_names=['wrist_cam_left', 'wrist_cam_right'],
        camera_dims=(128,128)
    )
    return sim

# ------------------------
# RUN EVALUATION
# ------------------------

success_rate = evaluate_policy(
    policy=pri3d_policy,
    create_sim=create_sim,
    num_rollouts=100,
    verbose=True
)

print(f"Pri3D policy success rate: {success_rate * 100:.2f}%")
