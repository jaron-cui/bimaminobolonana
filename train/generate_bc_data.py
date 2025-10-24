from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from policy.privileged_policy import PrivilegedPolicy
from robot.sim import BimanualAction, BimanualObs, BimanualSim, JOINT_OBSERVATION_SIZE, ACTION_SIZE, BimanualState, randomize_block_position


class BimanualDataset(torch.utils.data.Dataset):
  def __init__(self, data_directory: Path | str):
    super().__init__()
    metadata = BimanualDatasetMetadata.from_file(data_directory, read_only=True)
    if metadata is None:
      raise FileNotFoundError(f'Dataset not found in {data_directory}.')
    self._metadata = metadata

    memmap = self._metadata.memmap_data(overwrite=False)
    assert memmap is not None
    self._observation_array, self._action_array = memmap

  def __len__(self):
    return self._metadata.sample_count

  def __getitem__(self, index) -> Tuple[np.ndarray, np.ndarray]:
    return self._observation_array[index], self._action_array[index]


class HumanReadableBimanualDataset(BimanualDataset):
  def __getitem__(self, index) -> Tuple[BimanualObs, BimanualAction]:
    visual_shape = (2, self._metadata.camera_height, self._metadata.camera_width, 3)
    visual_size = np.array(visual_shape).prod()
    observation = self._observation_array[index]
    action = self._action_array[index]
    return (
      BimanualObs(
        visual=observation[:visual_size].reshape(visual_shape),
        qpos=BimanualState(observation[visual_size:visual_size + JOINT_OBSERVATION_SIZE]),
        qvel=BimanualState(observation[visual_size + JOINT_OBSERVATION_SIZE:])
      ),
      BimanualAction(action)
    )


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

  read_only: bool = True

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
    # 2 x image_size + qpos_size + qvel_size
    return 2 * (self.camera_height * self.camera_width * 3) + JOINT_OBSERVATION_SIZE + JOINT_OBSERVATION_SIZE
  
  @property
  def action_size(self) -> int:
    return ACTION_SIZE
  
  @property
  def size_in_gigabytes(self) -> float:
    return self.total_sample_count * (self.observation_size + ACTION_SIZE) * 4 / 1e9
  
  def memmap_data(self, overwrite: bool) -> Tuple[np.ndarray, np.ndarray] | None:
    if overwrite and self.read_only:
      raise ValueError('Cannot overwrite bimanual dataset in read-only mode.')
    
    # verify that any existing metadata file doesn't conflict with the way the dataset is being interpreted now
    existing_metadata = BimanualDatasetMetadata.from_file(self.save_dir, read_only=True)
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
    
    # check whether the dataset is missing files
    missing_data = False
    for path in [
      self.metadata_file_path,
      self.observation_file_path,
      self.action_file_path,
      self.rollout_length_file_path
    ]:
      if not path.exists():
        missing_data = True
        if self.read_only:
          raise FileNotFoundError(f'Bimanual dataset is being accessed in read-only mode, but {path} is missing.')
    
    # allocate array files if required
    if overwrite or missing_data:
      if self.read_only:
        raise FileNotFoundError(f'Missing data files in {self.save_dir}.')
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
      self.update_data_pointers()

    # memmap
    file_mode = 'r' if self.read_only else 'r+'
    observation_array = np.memmap(
      self.observation_file_path, dtype=np.float32, mode=file_mode, shape=(self.total_sample_count, self.observation_size))
    action_array = np.memmap(
      self.action_file_path, dtype=np.float32, mode=file_mode, shape=(self.total_sample_count, self.action_size))
    return observation_array, action_array
  
  def update_data_pointers(self, new_rollout_length: int | None = None):
    if self.read_only:
      raise RuntimeError('Cannot update bimanual dataset pointers in read-only mode.')
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
    np.save(self.rollout_length_file_path, np.array(self.rollout_lengths, dtype=np.uint16))

  @staticmethod
  def from_file(save_dir: Path | str, read_only: bool) -> 'BimanualDatasetMetadata | None':
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
      rollout_lengths=np.load(rollout_length_file_path)[:int(parts[6][len('rollout_count='):])],
      read_only=read_only
    )


def record_bc_data(
  save_dir: Path,
  total_sample_count: int,
  max_steps_per_rollout: int,
  camera_dims: Tuple[int, int],
  skip_frames: int = 0,
  overwrite: bool = False
):
  metadata = BimanualDatasetMetadata.from_file(save_dir, read_only=False) if not overwrite else None
  if metadata is None:
    metadata = BimanualDatasetMetadata(
      save_dir=save_dir,
      total_sample_count=total_sample_count,
      max_steps_per_rollout=max_steps_per_rollout,
      camera_height=camera_dims[0],
      camera_width=camera_dims[1],
      skip_frames=skip_frames
    )
  memmap = metadata.memmap_data(overwrite=overwrite)
  if memmap is None:  # canceled
    return
  observation_array, action_array = memmap

  observation_buffer: List[BimanualObs] = []
  action_buffer: List[BimanualAction] = []
  while True:
    with BimanualSim(merge_xml_files=['block.xml'], camera_dims=camera_dims, on_mujoco_init=randomize_block_position) as sim:
      policy = PrivilegedPolicy(sim.model, sim.data)
      success = False
      obs = sim.get_obs()
      rollout_length = 0
      for sim_step in tqdm(range(max_steps_per_rollout), desc=f'Attempting rollout {metadata.rollout_count}.'):
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
      sample_index = metadata.sample_count
      for observation, action in zip(observation_buffer, action_buffer):
        observation_array[sample_index] = np.concat((observation.visual.flatten(), observation.qpos.array, observation.qvel.array))
        action_array[sample_index] = action.array
        sample_index += 1
        if sample_index == total_sample_count:
          break
      metadata.update_data_pointers(new_rollout_length=rollout_length)
    print(f' - Rollout {f"succeeded. Saved" if success else "failed. Discarded"} {rollout_length} samples.')
    observation_buffer, action_buffer = [], []
    if sample_index == total_sample_count:
      break
  print(f'Finished generating {total_sample_count} samples in {save_dir}.')
