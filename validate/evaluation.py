import abc
from typing import Callable, Literal

import mujoco
import numpy as np
from tqdm import tqdm

from robot import kinematics
from robot.sim import LEFT_KINEMATIC_CHAIN, RIGHT_KINEMATIC_CHAIN, BimanualAction, BimanualObs, BimanualSim


class TaskEvaluator(abc.ABC):
  @abc.abstractmethod
  def determine_status(
    self,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    obs: BimanualObs,
    action: BimanualAction
  ) -> Literal['ambiguous', 'succeeded', 'failed']:
    pass


class PassBlockTaskEvaluator(TaskEvaluator):
  def __init__(self) -> None:
    super().__init__()
    self.left_gripper_tracker = GripperTracker('left')
    self.right_gripper_tracker = GripperTracker('right')

  def determine_status(
    self,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    obs: BimanualObs,
    action: BimanualAction
  ) -> Literal['ambiguous', 'succeeded', 'failed']:
    self.left_gripper_tracker.update(action, obs)
    self.right_gripper_tracker.update(action, obs)

    block_pos = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'block')]
    right_gripper_distance = np.linalg.norm(block_pos - self.right_gripper_tracker.pos()).item()
    left_gripper_distance = np.linalg.norm(block_pos - self.left_gripper_tracker.pos()).item()
    if self.right_gripper_tracker.is_gripping() and right_gripper_distance < 0.05 and left_gripper_distance > 0.2:
      return 'succeeded'
    return 'ambiguous'


class PickupBlockTaskEvaluator(TaskEvaluator):
  def __init__(self) -> None:
    super().__init__()
    self.left_gripper_tracker = GripperTracker('left')

  def determine_status(
    self,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    obs: BimanualObs,
    action: BimanualAction
  ) -> Literal['ambiguous', 'succeeded', 'failed']:
    self.left_gripper_tracker.update(action, obs)

    block_pos = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'block')]
    left_gripper_pos = self.left_gripper_tracker.pos()
    left_gripper_distance = np.linalg.norm(block_pos - left_gripper_pos).item()
    if self.left_gripper_tracker.is_gripping() and left_gripper_distance < 0.05 and left_gripper_pos[2] > 0.2:
      return 'succeeded'
    return 'ambiguous'


def evaluate_policy(
  policy: Callable[[BimanualObs], BimanualAction],
  create_sim: Callable[[], BimanualSim],
  create_task_evaluator: Callable[[], TaskEvaluator] = PassBlockTaskEvaluator,
  num_rollouts: int = 100,
  max_steps_per_rollout: int = 600,
  verbose: bool = False
) -> float:
  rollouts = tqdm(range(num_rollouts), desc=f'Rollouts of {policy}') if verbose else range(num_rollouts)
  success_count = 0
  for _ in rollouts:
    if run_evaluation_rollout(policy, create_sim(), create_task_evaluator(), max_steps_per_rollout, verbose=False):
      success_count += 1
  return success_count / num_rollouts


def run_evaluation_rollout(
  policy: Callable[[BimanualObs], BimanualAction],
  sim: BimanualSim,
  task_evaluator: TaskEvaluator,
  max_steps_per_rollout: int,
  verbose: bool = True
) -> bool:
  with sim as sim:
    obs = sim.get_obs()
    rollout = tqdm(range(max_steps_per_rollout), desc=f'Policy rollout: {policy}') if verbose else range(max_steps_per_rollout)
    for _ in rollout:
      action = policy(obs)

      task_status = task_evaluator.determine_status(sim.model, sim.data, obs, action)
      if task_status == 'succeeded':
        return True
      elif task_status == 'failed':
        return False
      
      obs = sim.step(action)
  return False


class GripperTracker:
  def __init__(self, arm: Literal['left', 'right']) -> None:
    if arm == 'left':
      self.kinematic_chain = LEFT_KINEMATIC_CHAIN
      self.get_joint_pos = lambda obs: obs.qpos.left_arm[:-2]
      self.get_gripper_error = lambda action, obs: obs.qpos.to_approximate_action().left_gripper - action.left_gripper
    else:
      self.kinematic_chain = RIGHT_KINEMATIC_CHAIN
      self.get_joint_pos = lambda obs: obs.qpos.right_arm[:-2]
      self.get_gripper_error = lambda action, obs: obs.qpos.to_approximate_action().right_gripper - action.right_gripper
    self.last_obs: BimanualObs | None = None
    self.last_stable_error = 0.0
    self.stability_duration = 0
  
  def update(self, action: BimanualAction, obs: BimanualObs):
    stability_threshold = 0.01
    self.last_obs = obs
    gripper_error = self.get_gripper_error(action, obs)
    if self.last_stable_error - stability_threshold < gripper_error < self.last_stable_error + stability_threshold:
      self.stability_duration += 1
    else:
      self.last_stable_error = gripper_error
      self.stability_duration = 0

  def pos(self) -> np.ndarray:
    return kinematics.augmented_forward(self.kinematic_chain, self.get_joint_pos(self.last_obs), np.array([0.1, 0.0, 0.0]))[0]
  
  def is_stable(self) -> bool:
    return self.stability_duration > 10
  
  def is_gripping(self) -> bool:
    return self.last_stable_error > 0.2
