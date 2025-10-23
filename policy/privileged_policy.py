from typing import Any, Callable, Dict, Literal, Tuple, get_args

import mujoco
import numpy as np
from robot_descriptions import aloha_mj_description
from scipy.spatial.transform import Rotation as scipyrotation

from robot import kinematics
from robot.sim import BimanualAction, BimanualObs


class PrivilegedPolicy:
  """
  A handcrafted bimanual block-passing policy that uses kinematics with privileged simulation state information.
  """
  Stage = Literal[
    'left-greet-block',
    'left-approach-block',
    'left-grasp-block',
    'left-raise-block',
    'right-greet-block',
    'right-approach-block',
    'right-grasp-block',
    'left-release-block',
    'right-retract-block',
    'done'
  ]
  def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
    self.model = model
    self.data = data
    self.policy_stage: PrivilegedPolicy.Stage = 'left-greet-block'
    self.subpolicies: Dict[PrivilegedPolicy.Stage, Callable[[BimanualObs, Dict], BimanualAction]] = {
      'left-greet-block': self._left_greet_block,
      'left-approach-block': self._left_approach_block,
      'left-grasp-block': self._left_grasp_block,
      'left-raise-block': self._left_raise_block,
      'right-greet-block': self._right_greet_block,
      'right-approach-block': self._right_approach_block,
      'right-grasp-block': self._right_grasp_block,
      'left-release-block': self._left_release_block,
      'right-retract-block': self._right_retract_block,
      'done': self._done,
    }
    self.subpolicy_state: Dict[PrivilegedPolicy.Stage, Any] = {stage: {} for stage in get_args(PrivilegedPolicy.Stage)}
    self.left_kinematic_chain = kinematics.parse_kinematic_chain(aloha_mj_description.MJCF_PATH, 'left/base_link', 'left/gripper_base')
    self.right_kinematic_chain = kinematics.parse_kinematic_chain(aloha_mj_description.MJCF_PATH, 'right/base_link', 'right/gripper_base')
    self.left_base_position = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'left/base_link')]
    self.right_base_position = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'right/base_link')]
    self.previous_action = BimanualAction()

  def __call__(self, obs: BimanualObs) -> BimanualAction:
    action = self.subpolicies[self.policy_stage](obs, self.subpolicy_state[self.policy_stage])
    self.previous_action = action
    return action

  def _left_greet_block(self, obs: BimanualObs, state: Dict) -> BimanualAction:
    target_pos, block_axes = get_block_orientation(self.model, self.data, with_respect_to=self.left_base_position)
    action = self._inverse_kinematics_pass(
      'left',
      self.previous_action,
      obs,
      state,
      target_pos,
      block_axes[1],
      end_effector_displacement=np.array([0.16, 0.0, 0.0]),
      on_target_reached='left-approach-block'
    )
    target_pos, _ = get_block_orientation(self.model, self.data, with_respect_to=self.right_base_position)
    action = self._inverse_kinematics_pass(
      'right',
      action,
      obs,
      {},
      target_pos,
      np.array([0.0, 1.0, 0.0]),
      end_effector_displacement=np.array([0.4, 0.0, 0.0])
    )
    action.left_gripper = 0.37
    return action

  def _left_approach_block(self, obs: BimanualObs, state: Dict) -> BimanualAction:
    target_pos, block_axes = get_block_orientation(self.model, self.data, with_respect_to=self.left_base_position)
    return self._inverse_kinematics_pass(
      'left',
      self.previous_action,
      obs,
      state,
      target_pos,
      block_axes[1],
      end_effector_displacement=np.array([0.1, 0.0, 0.0]),
      on_target_reached='left-grasp-block'
    )
  
  def _left_grasp_block(self, obs: BimanualObs, state: Dict) -> BimanualAction:
    self._maintain_stage(state, steps=10, then='left-raise-block')
    action = obs.qpos.to_approximate_action()
    action.left_gripper = 0
    return action
  
  def _left_raise_block(self, obs: BimanualObs, state: Dict) -> BimanualAction:
    target_pos, gripper_axis = np.array([-0.05, 0.0, 0.4]), np.array([0.0, 1.0, 0.0])
    action = self._inverse_kinematics_pass(
      'left',
      self.previous_action,
      obs,
      state,
      target_pos,
      gripper_axis,
      end_effector_displacement=np.array([0.1, 0.0, 0.0]),
      on_target_reached='right-greet-block'
    )
    action = self._inverse_kinematics_pass(
      'right',
      action,
      obs,
      {},
      target_pos,
      np.array([0.0, 1.0, 0.0]),
      end_effector_displacement=np.array([0.3, 0.0, 0.0])
    )
    action.left_gripper = 0
    return action
    
  def _right_greet_block(self, obs: BimanualObs, state: Dict) -> BimanualAction:
    base_action = self.previous_action

    target_pos, block_axes = get_block_orientation(self.model, self.data, with_respect_to=self.right_base_position)
    action = self._inverse_kinematics_pass(
      'right',
      base_action,
      obs,
      state,
      target_pos,
      block_axes[2],
      end_effector_displacement=np.array([0.2, 0.0, 0.0]),
      on_target_reached='right-approach-block'
    )
    action.left_gripper = 0
    action.right_gripper = 0.37

    return action
  
  def _right_approach_block(self, obs: BimanualObs, state: Dict) -> BimanualAction:
    base_action = self.previous_action

    target_pos, block_axes = get_block_orientation(self.model, self.data, with_respect_to=self.right_base_position)
    action = self._inverse_kinematics_pass(
      'right',
      base_action,
      obs,
      state,
      target_pos,
      block_axes[2],
      end_effector_displacement=np.array([0.1, 0.0, 0.0]),
      on_target_reached='right-grasp-block'
    )
    action.left_gripper = 0
    action.right_gripper = 0.37

    return action
  
  def _right_grasp_block(self, _: BimanualObs, state: Dict) -> BimanualAction:
    self._maintain_stage(state, steps=10, then='left-release-block')
    action = self.previous_action
    action.left_gripper = 0
    action.right_gripper = 0
    return action
  
  def _left_release_block(self, _: BimanualObs, state: Dict) -> BimanualAction:
    self._maintain_stage(state, steps=10, then='right-retract-block')
    action = self.previous_action
    action.left_gripper = 0.37
    action.right_gripper = 0
    return action
  
  def _right_retract_block(self, obs: BimanualObs, state: Dict) -> BimanualAction:
    target_pos, gripper_axis = np.array([0.2, 0.0, 0.4]), np.array([0.0, 1.0, 0.0])
    action = self._inverse_kinematics_pass(
      'right',
      self.previous_action,
      obs,
      state,
      target_pos,
      gripper_axis,
      end_effector_displacement=np.array([0.1, 0.0, 0.0]),
      on_target_reached='done'
    )
    action.right_gripper = 0
    return action
  
  def _done(self, _: BimanualObs, __: Dict) -> BimanualAction:
    return self.previous_action

  def _maintain_stage(self, state: Dict, steps: int, then: 'PrivilegedPolicy.Stage'):
    if 'steps' not in state:
      state['steps'] = 0
    state['steps'] += 1
    if state['steps'] > steps:
      self.policy_stage = then

  def _inverse_kinematics_pass(
    self,
    arm: Literal['left', 'right'],
    base_action: BimanualAction,
    obs: BimanualObs,
    state: Dict,
    target_pos: np.ndarray,
    grasp_axis: np.ndarray,
    *,
    target_tolerance_distance: float = 0.04,
    end_effector_displacement: np.ndarray | None = None,
    on_target_reached: 'PrivilegedPolicy.Stage | None' = None
  ) -> BimanualAction:
    if 'settling-steps' not in state:
      state['settling-steps'] = 0

    if arm == 'left':
      arm_joints_obs, arm_joints_action, kinematic_chain = slice(0, 6), slice(0, 6), self.left_kinematic_chain
    else:
      arm_joints_obs, arm_joints_action, kinematic_chain = slice(8, 14), slice(7, 13), self.right_kinematic_chain
    if end_effector_displacement is None:
      end_effector_displacement = np.array([0.0, 0.0, 0.0])

    # perform forward kinematics to get gripper position and retrieve block grasping info
    joint_angles = obs.qpos.array[arm_joints_obs]  # exclude the fingers
    gripper_pos, _ = kinematics.augmented_forward(kinematic_chain, joint_angles, end_effector_displacement)

    # progress the policy stage if the block is within grasping distance (4cm) continuously for 10 timesteps
    if on_target_reached is not None and np.linalg.norm(gripper_pos - target_pos) <= target_tolerance_distance:
      state['settling-steps'] += 1
      if state['settling-steps'] > 10:
        self.policy_stage = on_target_reached
    else:
      state['settling-steps'] = 0

    # calculate the target position as at most 5cm in the direction of the block
    gripper_to_target = target_pos - gripper_pos
    target_distance = min(0.05, np.linalg.norm(gripper_to_target).item())
    target_dir = gripper_to_target / np.linalg.norm(gripper_to_target)
    constrained_target_pos = gripper_pos + target_dir * target_distance
    constrained_target_pos[2] = max(constrained_target_pos[2], 0.07)

    # calculate target joint angles using inverse kinematics
    base_action.array[arm_joints_action] = kinematics.inverse(
      kinematic_chain, joint_angles, end_effector_displacement, constrained_target_pos, grasp_axis
    )
    return base_action


def wxyz_to_xyzw(quat: np.ndarray) -> np.ndarray:
  return np.concat((quat[1:], quat[0:1]))


def get_block_orientation(model: mujoco.MjModel, data: mujoco.MjData, *, with_respect_to: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  """
  Get the position and orientation information of the block object in the bimanual simulation.

  :return: The block position and a (3, 3) array where the rows are the facing, side-to-side, and vertical axis.
  """
  block_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'block')
  if block_id == -1:
    raise RuntimeError('There is no body named `block` in the MuJoCo XML. Did you forget to call `BimanualSim(merge_xml_files=[\'block.xml\'])`?')

  axes = scipyrotation.from_quat(wxyz_to_xyzw(data.xquat[block_id])).apply(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))

  vertical_axis_index = np.abs(axes @ np.array([0, 0, 1]).T).argmax()
  vertical_axis = axes[vertical_axis_index]
  axes = np.delete(axes, vertical_axis_index, axis=0)
  
  base_to_block = data.xpos[block_id] - with_respect_to
  facing_axis_index = np.abs(axes @ base_to_block.T).argmax()
  facing_axis = axes[facing_axis_index]
  axes = np.delete(axes, facing_axis_index, axis=0)

  side_axis = axes[0]
  ordered_block_axes = np.stack((facing_axis, side_axis, vertical_axis))
  
  return data.xpos[block_id], ordered_block_axes
