import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

from policy.privileged_policy import PrivilegedPolicy
from robot.sim import BimanualAction, BimanualObs, BimanualSim, JOINT_OBSERVATION_SIZE, ACTION_SIZE, randomize_block_position


def record_bc_data(
  save_dir: Path,
  total_sample_count: int,
  max_steps_per_rollout: int,
  camera_dims: Tuple[int, int],
  skip_frames: int = 0,
  resume: bool = True
):
  # 2 x image_size + qpos_size + qvel_size
  observation_size = 2 * (camera_dims[0] * camera_dims[1] * 3) + JOINT_OBSERVATION_SIZE + JOINT_OBSERVATION_SIZE
  observation_array_shape = (total_sample_count, observation_size)
  action_array_shape = (total_sample_count, ACTION_SIZE)
  gb = total_sample_count * (observation_size + ACTION_SIZE) * 4 / 1e9
  response = input(f'This function will allocate {gb:.2f}GB in `{save_dir}`. Are you sure you want to proceed? (y/[n])')
  if response != 'y':
    print('Canceled.')
    return

  metadata_path = save_dir / 'metadata.txt'
  observation_file_path = save_dir / 'observations.npy'
  action_file_path = save_dir / 'actions.npy'
  rollout_length_file_path = save_dir / 'rollout_length.npy'
  metadata_values = {
    'total_sample_count': total_sample_count,
    'max_steps_per_rollout': max_steps_per_rollout,
    'camera_height': camera_dims[0],
    'camera_width': camera_dims[1],
    'skip_frames': skip_frames
  }
  def update_metadata():
    with open(metadata_path, 'w') as file:
      file.write(' '.join([f'{field}={value}' for field, value in metadata_values.items()]) + f' sample_count={sample_count} rollout_count={rollout_count}')
    
  if not resume or not observation_file_path.exists() or not action_file_path.exists() or not rollout_length_file_path.exists():
    print(f'Allocating {gb:.2f}GB in `{save_dir}`...')
    sample_count = 0
    rollout_count = 0
    os.makedirs(save_dir, exist_ok=True)
    update_metadata()
    np.save(observation_file_path, np.zeros(observation_array_shape, dtype=np.float32))
    np.save(action_file_path, np.zeros(action_array_shape, dtype=np.float32))
    np.save(rollout_length_file_path, np.zeros(0, dtype=np.uint16))
  elif metadata_path.exists():
    with open(metadata_path, 'r') as file:
      parts = file.read().split(' ')
    # error if incompatible config
    for i, (field, value) in enumerate(metadata_values.items()):
      config_value = int(parts[i][len(field + '='):])
      if value != config_value:
        raise ValueError(f'Cannot resume data generation in {save_dir}; metadata values don\'t match: {field}=={value}!={config_value}')
    sample_count = int(parts[-2][len('sample_count='):])
    rollout_count = int(parts[-1][len('rollout_count='):])

  observation_array = np.memmap(observation_file_path, dtype=np.float32, mode='r+', shape=observation_array_shape)
  action_array = np.memmap(action_file_path, dtype=np.float32, mode='r+', shape=action_array_shape)
  rollout_lengths = list(np.load(rollout_length_file_path))[:rollout_count]

  observation_buffer: List[BimanualObs] = []
  action_buffer: List[BimanualAction] = []
  while True:
    with BimanualSim(merge_xml_files=['block.xml'], camera_dims=camera_dims, on_mujoco_init=randomize_block_position) as sim:
      policy = PrivilegedPolicy(sim.model, sim.data)
      success = False
      obs = sim.get_obs()
      rollout_length = 0
      for sim_step in tqdm(range(max_steps_per_rollout), desc=f'Attempting rollout {rollout_count}.'):
        action = policy(obs)
        if sim_step % (skip_frames + 1) == 0:
          observation_buffer.append(obs)
          action_buffer.append(action.copy())
          rollout_length += 1
        obs = sim.step(action)
        if policy.succeeded(obs):
          success = True
          break
    # save rollout samples
    if success:
      for observation, action in zip(observation_buffer, action_buffer):
        observation_array[sample_count] = np.concat((observation.visual.flatten(), observation.qpos.array, observation.qvel.array))
        action_array[sample_count] = action.array
        sample_count += 1
        if sample_count == total_sample_count:
          break
      rollout_lengths.append(rollout_length)
      rollout_count += 1
      np.save(rollout_length_file_path, np.array(rollout_lengths, dtype=np.uint16))
      update_metadata()
    print(f' - Rollout {f"succeeded. Saved" if success else "failed. Discarded"} {rollout_length} samples.')
    observation_buffer, action_buffer = [], []
    if sample_count == total_sample_count:
      break
  print(f'Finished generating {total_sample_count} samples in {save_dir}.')
