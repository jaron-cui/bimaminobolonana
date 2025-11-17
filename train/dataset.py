from dataclasses import dataclass, field
from datetime import datetime
import os
from pathlib import Path
from typing import Callable, List, Tuple

import mujoco
import numpy as np
import torch
from tqdm import tqdm

from validate.evaluation import TaskEvaluator
from robot.sim import BimanualAction, BimanualObs, BimanualSim, JOINT_OBSERVATION_SIZE, ACTION_SIZE


@dataclass
class TensorBimanualObs:
  """
  Tensor version of robot.sim.BimanualObs.

  :param visual: RGB camera images, of shape (batch, num_cameras, height, width, 3), dtype np.float32, and in range [0.0, 1.0].
  :param qpos: The positions of all joints, of shape (batch, num_joints) and dtype np.float32.
  :param qvel: The velocities of all joints, of shape (batch, num_joints) and dtype np.float32.
  """
  visual: torch.Tensor
  qpos: 'TensorBimanualState'
  qvel: 'TensorBimanualState'

  @property
  def device(self) -> torch.device:
    if not self.visual.device == self.qpos.device == self.qvel.device:
      raise RuntimeError(
        'Inconsistent TensorBimanualObs tensor devices: '
        f'{self.visual.device}, {self.qpos.device}, {self.qvel.device}.'
      )
    return self.visual.device

  def cpu(self) -> 'TensorBimanualObs':
    return TensorBimanualObs(
      visual=self.visual.cpu(),
      qpos=self.qpos.cpu(),
      qvel=self.qvel.cpu()
    )

  def cuda(self) -> 'TensorBimanualObs':
    return TensorBimanualObs(
      visual=self.visual.cuda(),
      qpos=self.qpos.cuda(),
      qvel=self.qvel.cuda()
    )

  def to(self, *args, **kwargs) -> 'TensorBimanualObs':
    return TensorBimanualObs(
      visual=self.visual.to(*args, **kwargs),
      qpos=self.qpos.to(*args, **kwargs),
      qvel=self.qvel.to(*args, **kwargs)
    )


class TensorBimanualState:
  """
  Tensor version of robot.sim.BimanualState.
  This is the low-level observation-space array used for `qpos` and `qvel`, larger than the action space used for `ctrl`.
  Namely, the gripper is represented by each finger's position instead of a single value.
  """
  def __init__(self, array: torch.Tensor | None = None):
    self.array = torch.zeros((1, JOINT_OBSERVATION_SIZE)) if array is None else array
    assert self.array.shape[1] == JOINT_OBSERVATION_SIZE and len(self.array.shape) == 2

  def to_approximate_action(self) -> 'TensorBimanualAction':
    return TensorBimanualAction(torch.cat((self.array[:, :7], self.array[:, 8:15]), dim=-1))

  @property
  def device(self) -> torch.device:
    return self.array.device

  def cpu(self) -> 'TensorBimanualState':
    return TensorBimanualState(self.array.cpu())

  def cuda(self) -> 'TensorBimanualState':
    return TensorBimanualState(self.array.cuda())

  def to(self, *args, **kwargs) -> 'TensorBimanualState':
    return TensorBimanualState(self.array.to(*args, **kwargs))


class TensorBimanualAction:
  """
  A convenient wrapper class for a bimanual robot action array.
  This is the array used for the action space of the robot.
  """
  def __init__(self, array: torch.Tensor | None = None):
    self.array = torch.zeros((1, ACTION_SIZE)) if array is None else array
    assert self.array.shape[1] == ACTION_SIZE and len(self.array.shape) == 2

  def copy(self) -> 'TensorBimanualAction':
    return TensorBimanualAction(self.array.clone())

  @property
  def device(self) -> torch.device:
    return self.array.device

  def cpu(self) -> 'TensorBimanualAction':
    return TensorBimanualAction(self.array.cpu())

  def cuda(self) -> 'TensorBimanualAction':
    return TensorBimanualAction(self.array.cuda())

  def to(self, *args, **kwargs) -> 'TensorBimanualAction':
    return TensorBimanualAction(self.array.to(*args, **kwargs))


class BimanualDataset(torch.utils.data.Dataset):
  def __init__(self, data_directory: Path | str):
    super().__init__()
    metadata = BimanualDatasetMetadata.from_file(data_directory, read_only=True)
    if metadata is None:
      raise FileNotFoundError(f'Dataset not found in {data_directory}.')
    if metadata.sample_count != metadata.total_sample_count:
      print(
        f'Warning: The bimanual dataset loaded from `{data_directory}` is incomplete, '
        f'with only {metadata.sample_count}/{metadata.total_sample_count} sample slots filled. '
        'This dataset is perfectly usable, but its filespace is not completely utilized.'
      )
    self.metadata = metadata

    memmap = self.metadata.memmap_data(overwrite=False)
    assert memmap is not None
    observation_array, action_array = memmap
    self._observation_array = torch.from_numpy(observation_array)
    self._action_array = torch.from_numpy(action_array)

  def __len__(self):
    return self.metadata.sample_count

  def __getitem__(self, index) -> Tuple[TensorBimanualObs, TensorBimanualAction]:
    if index >= self.metadata.sample_count:
      raise IndexError()
    visual_shape = (1, 2, self.metadata.camera_height, self.metadata.camera_width, 3)
    visual_size = np.array(visual_shape).prod()
    observation = self._observation_array[index:index + 1]
    action = self._action_array[index:index + 1]
    return (
      TensorBimanualObs(
        visual=observation[:, :visual_size].reshape(visual_shape),
        qpos=TensorBimanualState(observation[:, visual_size:visual_size + JOINT_OBSERVATION_SIZE]),
        qvel=TensorBimanualState(observation[:, visual_size + JOINT_OBSERVATION_SIZE:])
      ),
      TensorBimanualAction(action)
    )

  @staticmethod
  def collate_fn(
    batch: List[Tuple[TensorBimanualObs, TensorBimanualAction]]
  ) -> Tuple[TensorBimanualObs, TensorBimanualAction]:
    return (
      TensorBimanualObs(
        visual=torch.cat([t[0].visual for t in batch], dim=0),
        qpos=TensorBimanualState(torch.cat([t[0].qpos.array for t in batch], dim=0)),
        qvel=TensorBimanualState(torch.cat([t[0].qvel.array for t in batch], dim=0))
      ),
      TensorBimanualAction(
        array=torch.cat([t[1].array for t in batch], dim=0)
      )
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

  def memmap_data(self, overwrite: bool, force_allocate_storage_space: bool = False) -> Tuple[np.ndarray, np.ndarray] | None:
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
      response = 'y' if force_allocate_storage_space else input(
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
      rollout_lengths=list(np.load(rollout_length_file_path)[:int(parts[6][len('rollout_count='):])]),
      read_only=read_only
    )


def generate_bimanual_dataset(
  save_dir: Path,
  create_sim: Callable[[], BimanualSim],
  create_privileged_policy: Callable[[mujoco.MjModel, mujoco.MjData], Callable[[BimanualObs], BimanualAction]],
  create_task_evaluator: Callable[[], TaskEvaluator],
  total_sample_count: int,
  max_steps_per_rollout: int,
  camera_dims: Tuple[int, int],
  skip_frames: int = 0,
  resume: bool = True,
  force_allocate_storage_space: bool = False
):
  """
  Generates a dataset of (BimanualObs, BimanualAction) samples by repeatedly rolling out BimanualSim
  with the handcrafted policy.PrivilegedPolicy on the "Pass block" task.

  :param save_dir: The directory in which to save the data.
  :param create_sim: A function called per rollout that creates the BimanualSim.
  :param create_privileged_policy: A function called per rollout that creates the policy object.
  :param create_task_evaluator: A function called per rollout that defines the criteria for task success or failure.
  :param total_sample_count: The number of samples for which space will be allocated.
  :param max_steps_per_rollout: The maximum number of steps a given rollout will be allowed before designated a failure.
  :param camera_dims: The (height, width) pixel resolution with which to save camera images.
  :param skip_frames: The number of samples to skip between each sample recording.
  :param resume: Whether we should resume from a previous call to this function instead of overwriting all existing samples.
  :param force_allocate_storage_space: Whether we should skip the storage space allocation safety prompt. Use with caution!
  """
  print(f'Bimanual dataset save directory is set to `{save_dir}`.')
  metadata = None
  if resume:
    metadata = BimanualDatasetMetadata.from_file(save_dir, read_only=False)
    if metadata is not None:
      print(f'Resuming from sample {metadata.sample_count}/{metadata.total_sample_count}.')
  if metadata is None:
    metadata = BimanualDatasetMetadata(
      save_dir=save_dir,
      total_sample_count=total_sample_count,
      max_steps_per_rollout=max_steps_per_rollout,
      camera_height=camera_dims[0],
      camera_width=camera_dims[1],
      skip_frames=skip_frames,
      read_only=False
    )
  memmap = metadata.memmap_data(overwrite=not resume, force_allocate_storage_space=force_allocate_storage_space)
  if memmap is None:  # canceled
    return
  if metadata.sample_count == metadata.total_sample_count:
    print(f'Dataset is already complete with {metadata.sample_count} samples.')
    return
  observation_array, action_array = memmap

  observation_buffer: List[BimanualObs] = []
  action_buffer: List[BimanualAction] = []
  while True:
    task_evaluator = create_task_evaluator()
    with create_sim() as sim:
      policy = create_privileged_policy(sim.model, sim.data)
      success = False
      obs = sim.get_obs()
      for sim_step in tqdm(range(max_steps_per_rollout), desc=f'Attempting rollout {metadata.rollout_count}.'):
        action = policy(obs)

        if sim_step % (skip_frames + 1) == 0:
          observation_buffer.append(obs)
          action_buffer.append(action.copy())

        task_status = task_evaluator.determine_status(sim.model, sim.data, obs, action)
        if task_status == 'succeeded':
          success = True
          break
        elif task_status == 'failed':
          break

        obs = sim.step(action)
    # save rollout samples
    if success:
      sample_index = metadata.sample_count
      for observation, action in zip(observation_buffer, action_buffer):
        observation_array[sample_index] = np.concat((observation.visual.flatten(), observation.qpos.array, observation.qvel.array))
        action_array[sample_index] = action.array
        sample_index += 1
        if sample_index == metadata.total_sample_count:
          break
      rollout_length = sample_index - metadata.sample_count
      metadata.update_data_pointers(new_rollout_length=rollout_length)
    else:
      rollout_length = len(observation_buffer)
    print(
      f' - Rollout {f"succeeded. Saved" if success else "failed. Discarded"} '
      f'{rollout_length} samples at {datetime.now()}. '
      f'({metadata.sample_count}/{metadata.total_sample_count})'
    )
    observation_buffer, action_buffer = [], []
    if metadata.sample_count == total_sample_count:
      break
  print(f'Finished generating {total_sample_count} samples in {save_dir}.')
