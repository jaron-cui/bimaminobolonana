"""
Visualize ACT (Action Chunking Transformer) Policy Rollouts

This script loads a trained ACT checkpoint and runs a rollout with visualization,
saving videos of the left and right camera views.

Usage:
    python inference_ACT.py
"""

import sys
import torch
import numpy as np
from pathlib import Path
import os
from tqdm import tqdm
from collections import deque
from omegaconf import OmegaConf

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent))

from policy.act import build_act_policy
from robot.sim import BimanualSim, BimanualObs, BimanualAction, randomize_block_position
from robot.visualize import save_frames_to_video
from train.dataset import TensorBimanualObs

current_file = Path(__file__).resolve()
project_root = current_file.parent

# =============== CONFIG ==================
CHECKPOINT = "runs/act_test_1/checkpoint/act-train/20.pt"  # Path to trained ACT checkpoint
CONFIG_PATH = "configs/policy_act.yaml"  # Path to ACT config
MAX_STEPS = 600
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============== LOAD MODEL ==================
print(f"Loading config from: {CONFIG_PATH}")
config = OmegaConf.load(CONFIG_PATH)
print(f"Config: {config.name}")
print(f"  - Chunk size: {config.chunk_size}")
print(f"  - Temporal context: {config.temporal_context}")
print(f"  - Encoder: {config.encoder.name}")
print(f"  - Image size: {config.image_size}")
print()

print("Building ACT policy...")
model = build_act_policy(config).to(DEVICE)
model.eval()

print(f"Loading checkpoint from: {CHECKPOINT}")
state_dict = torch.load(CHECKPOINT, map_location=DEVICE)
model.load_state_dict(state_dict)
print("✓ Checkpoint loaded successfully!")
print()

# =============== ACT POLICY WRAPPER ==================
class ACTPolicyWrapper:
    """
    Wrapper for ACT policy that handles temporal context and action chunking,
    AND denormalizes the predicted actions.
    """
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        self.temporal_context = config.temporal_context
        self.chunk_size = config.chunk_size
        self.ensemble_window = config.get('ensemble_window', 10) if config.get('temporal_ensemble', False) else 1

        # Temporal context buffer
        self.obs_history = deque(maxlen=self.temporal_context)

        # Action chunk buffer
        self.action_buffer = deque(maxlen=self.ensemble_window)

        # ============================================
        # 🔥 Load action normalization from model
        # ============================================
        if hasattr(model, "action_mean") and hasattr(model, "action_std"):
            self.action_mean = model.action_mean.detach().to(device)     # [A]
            self.action_std = model.action_std.detach().to(device)       # [A]
            print("✓ Loaded action normalization from checkpoint.")
            print("  action_mean:", self.action_mean)
            print("  action_std:", self.action_std)
        else:
            self.action_mean = None
            self.action_std = None
            print("⚠ No action normalization found in model → assuming raw actions.")

    # Reset buffers
    def reset(self):
        self.obs_history.clear()
        self.action_buffer.clear()

    # ====================================================
    # 🔥 Main API: predict denormalized BimanualAction
    # ====================================================
    def __call__(self, obs: BimanualObs) -> BimanualAction:

        # Update obs history
        self.obs_history.append(obs)
        while len(self.obs_history) < self.temporal_context:
            self.obs_history.appendleft(obs)

        # Convert to TensorBimanualObs
        tensor_obs = self._to_tensor_obs(list(self.obs_history))

        # --- Model forward ---
        with torch.no_grad():
            action_chunk = self.model.predict_action_chunk(tensor_obs)  # shape: [1, chunk, A]

        # Save chunk for temporal ensembling
        self.action_buffer.append(action_chunk.cpu())

        # ----------------------------
        # Temporal ensembling (same as before)
        # ----------------------------
        if len(self.action_buffer) > 0:
            current_actions = []
            for i, chunk in enumerate(self.action_buffer):
                step_in_chunk = len(self.action_buffer) - 1 - i
                if step_in_chunk < chunk.shape[1]:
                    current_actions.append(chunk[0, step_in_chunk])

            if current_actions:
                action_tensor = torch.stack(current_actions).mean(dim=0)  # [A]
            else:
                action_tensor = action_chunk[0, 0].cpu()
        else:
            action_tensor = action_chunk[0, 0].cpu()

        # ====================================================
        # 🔥 DENORMALIZE ACTION (from normalized relative to relative)
        # ====================================================
        normalized_action = action_tensor.clone()  # 保存用于调试
        if self.action_mean is not None and self.action_std is not None:
            action_tensor = (
                action_tensor.to(self.device) * self.action_std + self.action_mean
            ).cpu()

        # ====================================================
        # 🔥 CONVERT RELATIVE ACTION TO ABSOLUTE ACTION
        # ====================================================
        # Get current qpos from the last observation in history
        current_obs = list(self.obs_history)[-1]
        qpos = torch.from_numpy(current_obs.qpos.array).float()  # [16]
        
        # Build approximate "stay-still" action (same as training)
        approx = torch.cat([
            qpos[:7],      # left arm
            qpos[8:15],    # right arm
        ], dim=0)  # [14]
        
        approx[6] = qpos[6] * 10    # left gripper
        approx[13] = qpos[14] * 10  # right gripper
        
        # Convert relative to absolute for arm joints
        # relative = absolute - approx
        # => absolute = relative + approx
        absolute_action = action_tensor.clone()
        
        # Arm joints (indices 0-5, 7-12): convert from relative to absolute
        absolute_action[:6] = action_tensor[:6] + approx[:6]
        absolute_action[7:13] = action_tensor[7:13] + approx[7:13]
        
        # Gripper joints (indices 6, 13): already absolute, keep as is
        # (These were not converted to relative during training)
        absolute_action[6] = action_tensor[6]
        absolute_action[13] = action_tensor[13]
        
        # ====================================================
        # 🔥 GRIPPER THRESHOLDING: Binary open/close
        # ====================================================
        # Apply threshold to convert continuous gripper values to discrete open/close
        # Left gripper (index 6)
        absolute_action[6] = 0.0 if absolute_action[6] < 0.25 else 0.36
        # Right gripper (index 13)
        absolute_action[13] = 0.0 if absolute_action[13] < 0.25 else 0.36

        # Convert to numpy for simulator
        action_array = absolute_action.numpy()

        return BimanualAction(action_array)

    # ====================================================
    # Helper: Convert list of BimanualObs → TensorBimanualObs
    # ====================================================
    def _to_tensor_obs(self, obs_list):
        visual = np.stack([o.visual for o in obs_list], axis=0)  # [T, C, H, W, 3]
        visual = torch.from_numpy(visual).float().unsqueeze(0).to(self.device)

        qpos_arrays = [torch.from_numpy(o.qpos.array).float() for o in obs_list]
        qvel_arrays = [torch.from_numpy(o.qvel.array).float() for o in obs_list]
        qpos = torch.stack(qpos_arrays).unsqueeze(0).to(self.device)
        qvel = torch.stack(qvel_arrays).unsqueeze(0).to(self.device)

        class TemporalState:
            def __init__(self, arr): self.array = arr

        return TensorBimanualObs(
            visual=visual,
            qpos=TemporalState(qpos),
            qvel=TemporalState(qvel)
        )



# Create policy wrapper
act_policy = ACTPolicyWrapper(model, config, DEVICE)

# =============== CREATE SIMULATOR ==================
print("Creating MuJoCo simulator...")
sim = BimanualSim(
    merge_xml_files=[Path('robot/block.xml')],
    camera_dims=(config.image_size, config.image_size),
    obs_camera_names=config.camera_names,
    # on_mujoco_init=randomize_block_position
    on_mujoco_init=lambda m,d:(m,d)
)
print("✓ Simulator created!")
print()

# =============== ROLLOUT ==================
print(f"Starting rollout (max {MAX_STEPS} steps)...")
print("Note: First policy call may take 10-30 seconds due to CLIP encoding initialization.")
print()

# Rollout storage
left_frames, right_frames = [], []

# Use context manager for proper cleanup
with sim:
    # Reset policy for fresh rollout
    act_policy.reset()

    # Get initial observation
    obs = sim.get_obs()

    # Simulation rollout
    for step in tqdm(range(MAX_STEPS), desc="Rollout progress"):
        # Save images for video
        # Images from simulator are already in [0, 1] float range
        left_frames.append(obs.visual[0].copy())
        right_frames.append(obs.visual[1].copy())

        # Predict next action using ACT policy
        action = act_policy(obs)

        # Step simulation
        obs = sim.step(action)

print("✓ Rollout finished!")
print()

# =============== SAVE VIDEOS ==================
output_dir = Path("out/render-act")
output_dir.mkdir(parents=True, exist_ok=True)

left_path = output_dir / "left.mp4"
right_path = output_dir / "right.mp4"

print("Saving videos...")
save_frames_to_video(left_frames, left_path)
save_frames_to_video(right_frames, right_path)

print("✓ Videos saved:")
print(f"  Left camera:  {left_path}")
print(f"  Right camera:  {right_path}")
print()