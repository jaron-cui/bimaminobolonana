from dataclasses import dataclass
import os
from pathlib import Path
from typing import Callable, Sequence, Tuple
from xml.etree import ElementTree as ET

import mujoco
import numpy as np
from robot_descriptions import aloha_mj_description


OBSERVATION_SIZE = 16
ACTION_SIZE = 14


@dataclass
class BimanualObs:
  """
  An observation from a bimanual robot.

  :param visual: RGB camera images, of shape (num_cameras, height, width, 3).
  :param qpos: The positions of all joints, of shape (num_joints,).
  :param qvel: The velocities of all joints, of shape (num_joints,).
  """
  visual: np.ndarray
  qpos: 'BimanualState'
  qvel: 'BimanualState'


class BimanualState:
  """
  A convenient wrapper class for a bimanual robot state array.
  This is the low-level observation-space array used for `qpos` and `qvel`, larger than the action space used for `ctrl`.
  Namely, the gripper is represented by each finger's position instead of a single value.
  """
  def __init__(self, array: np.ndarray | None = None):
    self.array = np.zeros(OBSERVATION_SIZE) if array is None else array
    assert self.array.shape == (OBSERVATION_SIZE,), f'Expected action of shape ({OBSERVATION_SIZE},), but got {self.array.shape}.'

  @property
  def left_waist(self) -> np.float64:
    return self.array[0]

  @property
  def left_shoulder(self) -> np.float64:
    return self.array[1]

  @property
  def left_elbow(self) -> np.float64:
    return self.array[2]

  @property
  def left_forearm_roll(self) -> np.float64:
    return self.array[3]

  @property
  def left_wrist_angle(self) -> np.float64:
    return self.array[4]

  @property
  def left_wrist_rotate(self) -> np.float64:
    return self.array[5]

  @property
  def left_left_finger(self) -> np.float64:
    return self.array[6]
  
  @property
  def left_right_finger(self) -> np.float64:
    return self.array[7]

  @property
  def right_waist(self) -> np.float64:
    return self.array[8]

  @property
  def right_shoulder(self) -> np.float64:
    return self.array[9]

  @property
  def right_elbow(self) -> np.float64:
    return self.array[10]

  @property
  def right_forearm_roll(self) -> np.float64:
    return self.array[11]

  @property
  def right_wrist_angle(self) -> np.float64:
    return self.array[12]

  @property
  def right_wrist_rotate(self) -> np.float64:
    return self.array[13]

  @property
  def right_left_finger(self) -> np.float64:
    return self.array[14]

  @property
  def right_right_finger(self) -> np.float64:
    return self.array[15]
  
  @left_waist.setter
  def left_waist(self, v):
    self.array[0] = v

  @left_shoulder.setter
  def left_shoulder(self, v):
    self.array[1] = v

  @left_elbow.setter
  def left_elbow(self, v):
    self.array[2] = v

  @left_forearm_roll.setter
  def left_forearm_roll(self, v):
    self.array[3] = v

  @left_wrist_angle.setter
  def left_wrist_angle(self, v):
    self.array[4] = v

  @left_wrist_rotate.setter
  def left_wrist_rotate(self, v):
    self.array[5] = v

  @left_left_finger.setter
  def left_left_finger(self, v):
    self.array[6] = v

  @left_right_finger.setter
  def left_right_finger(self, v):
    self.array[7] = v

  @right_waist.setter
  def right_waist(self, v):
    self.array[8] = v

  @right_shoulder.setter
  def right_shoulder(self, v):
    self.array[9] = v

  @right_elbow.setter
  def right_elbow(self, v):
    self.array[10] = v

  @right_forearm_roll.setter
  def right_forearm_roll(self, v):
    self.array[11] = v

  @right_wrist_angle.setter
  def right_wrist_angle(self, v):
    self.array[12] = v

  @right_wrist_rotate.setter
  def right_wrist_rotate(self, v):
    self.array[13] = v

  @right_left_finger.setter
  def right_left_finger(self, v):
    self.array[14] = v

  @right_right_finger.setter
  def right_right_finger(self, v):
    self.array[15] = v

  def to_approximate_action(self) -> 'BimanualAction':
    return BimanualAction(np.concat((self.array[:7], self.array[8:15])))


class BimanualAction:
  """
  A convenient wrapper class for a bimanual robot action array.
  This is the array used for the action space of the robot.
  """
  def __init__(self, array: np.ndarray | None = None):
    self.array = np.zeros(ACTION_SIZE) if array is None else array
    assert self.array.shape == (ACTION_SIZE,), f'Expected action of shape ({ACTION_SIZE},), but got {self.array.shape}.'

  @property
  def left_waist(self) -> np.float64:
    return self.array[0]

  @property
  def left_shoulder(self) -> np.float64:
    return self.array[1]

  @property
  def left_elbow(self) -> np.float64:
    return self.array[2]

  @property
  def left_forearm_roll(self) -> np.float64:
    return self.array[3]

  @property
  def left_wrist_angle(self) -> np.float64:
    return self.array[4]

  @property
  def left_wrist_rotate(self) -> np.float64:
    return self.array[5]

  @property
  def left_gripper(self) -> np.float64:
    return self.array[6]

  @property
  def right_waist(self) -> np.float64:
    return self.array[7]

  @property
  def right_shoulder(self) -> np.float64:
    return self.array[8]

  @property
  def right_elbow(self) -> np.float64:
    return self.array[9]

  @property
  def right_forearm_roll(self) -> np.float64:
    return self.array[10]

  @property
  def right_wrist_angle(self) -> np.float64:
    return self.array[11]

  @property
  def right_wrist_rotate(self) -> np.float64:
    return self.array[12]

  @property
  def right_gripper(self) -> np.float64:
    return self.array[13]
  
  @left_waist.setter
  def left_waist(self, v):
    self.array[0] = v

  @left_shoulder.setter
  def left_shoulder(self, v):
    self.array[1] = v

  @left_elbow.setter
  def left_elbow(self, v):
    self.array[2] = v

  @left_forearm_roll.setter
  def left_forearm_roll(self, v):
    self.array[3] = v

  @left_wrist_angle.setter
  def left_wrist_angle(self, v):
    self.array[4] = v

  @left_wrist_rotate.setter
  def left_wrist_rotate(self, v):
    self.array[5] = v

  @left_gripper.setter
  def left_gripper(self, v):
    self.array[6] = v

  @right_waist.setter
  def right_waist(self, v):
    self.array[7] = v

  @right_shoulder.setter
  def right_shoulder(self, v):
    self.array[8] = v

  @right_elbow.setter
  def right_elbow(self, v):
    self.array[9] = v

  @right_forearm_roll.setter
  def right_forearm_roll(self, v):
    self.array[10] = v

  @right_wrist_angle.setter
  def right_wrist_angle(self, v):
    self.array[11] = v

  @right_wrist_rotate.setter
  def right_wrist_rotate(self, v):
    self.array[12] = v

  @right_gripper.setter
  def right_gripper(self, v):
    self.array[13] = v


class BimanualSim:
  def __init__(
    self,
    initial_pos: BimanualState | None = None,
    substeps: int = 5,
    camera_dims: Tuple[int, int] = (480, 540),
    obs_camera_names: Sequence[str] = ('wrist_cam_left', 'wrist_cam_right'),
    merge_xml_files: Sequence[Path] = tuple(),
    on_mujoco_init: Callable[[mujoco.MjModel, mujoco.MjData], Tuple[mujoco.MjModel, mujoco.MjData]] = lambda m, d: (m, d)
  ):
    """
    A bimanual simulation instance which can be progressed with BimanualSim.step().

    :param initial_pos: The initial pose and equilibriating action for the robot.
    :param substeps: The number of simulation substeps to take between each step.
    :param camera_dims: The pixel (height, width) of the visual observations taken from each camera.
    :param obs_camera_names: The names of the cameras in the MuJoCo XML schema that will be used to render the visual observations.
    :param merge_xml_files: XML files to merge into the scene, such as those containing task-specific objects.
    :param on_mujoco_init: General-purpose function called after MuJoCo initialization to modify the model and data.
    """
    if initial_pos is None:
      initial_pos = BimanualState()
      initial_pos.left_shoulder = -1
      initial_pos.left_elbow = 0.5
      initial_pos.left_wrist_angle = 1
      initial_pos.left_left_finger = 0.01
      initial_pos.left_right_finger = 0.01
      initial_pos.right_shoulder = -1
      initial_pos.right_elbow = 0.5
      initial_pos.right_wrist_angle = 1
      initial_pos.right_left_finger = 0.01
      initial_pos.right_right_finger = 0.01
    initial_ctrl = initial_pos.to_approximate_action()

    self.model: mujoco.MjModel = merge_xml_into_mujoco_scene(Path(aloha_mj_description.MJCF_PATH), merge_xml_files)
    # self.model: mujoco.MjModel = mujoco.MjModel.from_xml_path(aloha_mj_description.MJCF_PATH)
    self.data: mujoco.MjData = mujoco.MjData(self.model)

    self.data.qpos[:OBSERVATION_SIZE] = initial_pos.array
    self.data.ctrl[:ACTION_SIZE] = initial_ctrl.array

    self.model, self.data = on_mujoco_init(self.model, self.data)

    self.substeps = substeps
    self.camera_dims = camera_dims

    # configure renderers
    self.renderers = [
      (
        camera_name,
        mujoco.Renderer(self.model, *camera_dims)
      ) for camera_name in obs_camera_names
    ]

    # ensure off-screen rendering buffer is at least the size of all camera frames
    self.model.vis.global_.offheight = max(self.model.vis.global_.offheight, camera_dims[0])
    self.model.vis.global_.offwidth  = max(self.model.vis.global_.offwidth,  camera_dims[1])

  def get_obs(self) -> BimanualObs:
    # extract camera images (num_cameras, height, width, 3)
    visual_obs = np.zeros((len(self.renderers), self.camera_dims[0], self.camera_dims[1], 3))
    for i, (camera_name, renderer) in enumerate(self.renderers):
      renderer.update_scene(self.data, camera=camera_name)
      visual_obs[i] = renderer.render()
    
    return BimanualObs(
      visual=visual_obs,
      qpos=BimanualState(self.data.qpos.copy()),
      qvel=BimanualState(self.data.qvel.copy())
    )

  def step(self, action: np.ndarray | BimanualAction) -> BimanualObs:
    """
    Given an action, step the simulation, and then return an observation of the stepped simulation.

    :param action: New target positions for all joints, of shape (num_joints,).
    :return: The observation taken after a number of simulation substeps are applied.
    """
    if isinstance(action, BimanualAction):
      action = action.array
    assert action.shape == (self.model.nu,), f'Expected action of shape ({self.model.nu},), but got {action.shape}.'

    # apply action and simulate for a number of timesteps
    self.data.ctrl[:] = action
    for _ in range(self.substeps):
      mujoco.mj_step(self.model, self.data)

    return self.get_obs()

  def launch_viewer(self):
    import mujoco.viewer
    mujoco.viewer.launch(self.model, self.data)
    

def merge_xml_into_mujoco_scene(scene_path: Path, merge_paths: Sequence[Path]):
  # cache current directory
  original_dir = os.getcwd()
  
  # resolve absolute paths
  scene_abs = os.path.abspath(scene_path)
  abs_merge_paths = [os.path.abspath(merge_path) for merge_path in merge_paths]
  
  # switch to scene XML dir so local imports work correctly during loading
  scene_dir = os.path.dirname(scene_abs)
  os.chdir(scene_dir)

  # parse scene XML
  scene_tree = ET.parse(scene_abs)
  scene_root = scene_tree.getroot()
  scene_worldbody = scene_root.find('worldbody')
  assert scene_worldbody is not None, f'Scene {scene_path} needs to contain a worldbody.'
  
  try:
    for merge_path, abs_path in zip(merge_paths, abs_merge_paths):
      # parse merge XML
      extra_tree = ET.parse(abs_path)
      extra_worldbody = extra_tree.getroot().find('worldbody')
      assert extra_worldbody is not None, f'Dependency {merge_path} needs to contain a worldbody.'
      
      # merge objects
      object_bodies = extra_worldbody.findall('body')
      for body in object_bodies:
        new_body = ET.fromstring(ET.tostring(body))
        scene_worldbody.append(new_body)
    
    # compile
    xml_string = ET.tostring(scene_root, encoding='unicode')
    model = mujoco.MjModel.from_xml_string(xml_string)
      
  finally:
    os.chdir(original_dir)
  
  return model
