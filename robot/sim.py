from dataclasses import dataclass
from typing import Sequence, Tuple

import mujoco
import numpy as np
from robot_descriptions import aloha_mj_description


@dataclass
class BimanualObs:
  """
  An observation from a bimanual robot.

  :param visual: RGB camera images, of shape (num_cameras, height, width, 3).
  :param qpos: The positions of all joints, of shape (num_joints,).
  :param qvel: The velocities of all joints, of shape (num_joints,).
  """
  visual: np.ndarray
  qpos: np.ndarray
  qvel: np.ndarray


class BimanualSim:
  def __init__(
    self,
    substeps: int = 5,
    camera_dims: Tuple[int, int] = (480, 540),
    obs_camera_names: Sequence[str] = ('wrist_cam_left', 'wrist_cam_right')
  ):
    """
    A bimanual simulation instance which can be progressed with BimanualSim.step().

    :param substeps: The number of simulation substeps to take between each step.
    :param camera_height: The pixel height of the visual observations taken from each camera.
    :param camera_width: The pixel width of the visual observations taken from each camera.
    :param obs_camera_names: The names of the cameras in the MuJoCo XML schema that will be used to render the visual observations.
    """
    self.model: mujoco.MjModel = mujoco.MjModel.from_xml_path(aloha_mj_description.MJCF_PATH)
    self.data: mujoco.MjData = mujoco.MjData(self.model)

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
      qpos=self.data.qpos.copy(),
      qvel=self.data.qvel.copy()
    )

  def step(self, action: np.ndarray) -> BimanualObs:
    """
    Given an action, step the simulation, and then return an observation of the stepped simulation.

    :param action: New target positions for all joints, of shape (num_joints,).
    :return: The observation taken after a number of simulation substeps are applied.
    """
    assert action.shape == (self.model.nu,), f'Expected action of shape ({self.model.nu},), but got {action.shape}.'

    # apply action and simulate for a number of timesteps
    self.data.ctrl[:] = action
    for _ in range(self.substeps):
      mujoco.mj_step(self.model, self.data)

    return self.get_obs()

    
