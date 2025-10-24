from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from policy.privileged_policy import PrivilegedPolicy
from robot.sim import BimanualAction, BimanualObs, BimanualSim, JOINT_OBSERVATION_SIZE, ACTION_SIZE, randomize_block_position


class BimanualDataset(torch.utils.data.Dataset):
  def __init__(self, data_directory: Path | str):
    super().__init__()
    self._observation_array = None


@dataclass
class BimanualDatasetMetadata:
  save_dir: Path
  total_sample_count: int
  max_steps_per_rollout: int
  camera_height: int
  camera_width: int
  skip_frames: int
  sample_count: int = 0

  rollout_lengths: List[int] = field(default_factory=list)

  @property
  def rollout_count(self) -> int:
    return len(self.rollout_lengths)

  @property
  def metadata_file_path(self) -> Path:
    return self.save_dir / 'metadata.txt'
  
  @property
  def observation_file_path(self) -> Path:
    return self.save_dir / 'observations.npy'
  
  @property
  def action_file_path(self) -> Path:
    return self.save_dir / 'actions.npy'
  
  @property
  def rollout_length_file_path(self) -> Path:
    return self.save_dir / 'rollout_length.npy'
  
  @property
  def observation_size(self) -> int:
    return 2 * (self.camera_height * self.camera_width * 3) + JOINT_OBSERVATION_SIZE + JOINT_OBSERVATION_SIZE
  
  @property
  def action_size(self) -> int:
    return ACTION_SIZE
  
  @property
  def size_in_gigabytes(self) -> float:
    return self.total_sample_count * (self.observation_size + ACTION_SIZE) * 4 / 1e9
  
  def memmap_data(
    self,
  ) -> Tuple[np.ndarray, np.ndarray] | None:
    existing_metadata = BimanualDatasetMetadata.from_file(self.save_dir)
    if existing_metadata is not None:
      for field, current, existing in [
        ('total_sample_count', self.total_sample_count, existing_metadata.total_sample_count),
        ('max_steps_per_rollout', self.max_steps_per_rollout, existing_metadata.max_steps_per_rollout),
        ('camera_height', self.camera_height, existing_metadata.camera_height),
        ('camera_width', self.camera_width, existing_metadata.camera_width),
        ('skip_frames', self.skip_frames, existing_metadata.skip_frames)
      ]:
        if current != existing:
          raise ValueError(f'Bimanual dataset metadata values don\'t match: {field}=={existing}!={current}')
        
    if any([not path.exists() for path in [
      self.metadata_file_path,
      self.observation_file_path,
      self.action_file_path,
      self.rollout_length_file_path
    ]]):
      response = input(
        f'This will allocate {self.size_in_gigabytes:.2f}GB in `{self.save_dir}`. '
        'Are you sure you want to proceed? (y/[n])'
      )
      if response != 'y':
        print('Canceled.')
        return None
      print(f'Allocating {self.size_in_gigabytes:.2f}GB in `{self.save_dir}`...')
      self.sample_count = 0
      self.rollout_lengths = []
      os.makedirs(self.save_dir, exist_ok=True)
      np.save(self.observation_file_path, np.zeros((self.total_sample_count, self.observation_size), dtype=np.float32))
      np.save(self.action_file_path, np.zeros((self.total_sample_count, self.action_size), dtype=np.float32))
      np.save(self.rollout_length_file_path, np.array(self.rollout_lengths, dtype=np.uint16))
      self.update_data_pointers()

    observation_array = np.memmap(
      self.observation_file_path, dtype=np.float32, mode='r+', shape=(self.total_sample_count, self.observation_size))
    action_array = np.memmap(
      self.action_file_path, dtype=np.float32, mode='r+', shape=(self.total_sample_count, self.action_size))
    return observation_array, action_array
  
  def update_data_pointers(self, new_rollout_length: int | None = None):
    if new_rollout_length:
      self.sample_count += new_rollout_length
      self.rollout_lengths.append(new_rollout_length)
    metadata_values = {
      'total_sample_count': self.total_sample_count,
      'max_steps_per_rollout': self.max_steps_per_rollout,
      'camera_height': self.camera_height,
      'camera_width': self.camera_width,
      'skip_frames': self.skip_frames,
      'sample_count': self.sample_count,
      'rollout_count': self.rollout_count
    }
    with open(self.metadata_file_path, 'w') as file:
      file.write(' '.join([f'{field}={value}' for field, value in metadata_values.items()]))

  @staticmethod
  def from_file(save_dir: Path | str) -> 'BimanualDatasetMetadata | None':
    metadata_file_path = Path(save_dir) / 'metadata.txt'
    rollout_length_file_path = Path(save_dir) / 'rollout_length.npy'
    if not metadata_file_path.exists() or not rollout_length_file_path:
      return None
    
    with open(metadata_file_path, 'r') as file:
      parts = file.read().split(' ')

    return BimanualDatasetMetadata(
      save_dir=Path(save_dir),
      total_sample_count=int(parts[0][len('total_sample_count='):]),
      max_steps_per_rollout=int(parts[1][len('max_steps_per_rollout='):]),
      camera_height=int(parts[2][len('camera_height='):]),
      camera_width=int(parts[3][len('camera_width='):]),
      skip_frames=int(parts[4][len('skip_frames='):]),
      sample_count=int(parts[5][len('sample_count='):]),
      rollout_lengths=np.load(rollout_length_file_path)[:int(parts[6][len('rollout_count='):])]
    )


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
