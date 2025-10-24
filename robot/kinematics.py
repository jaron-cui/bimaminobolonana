from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple
import mujoco
import numpy as np
import scipy
from scipy.spatial.transform import Rotation as scipyrotation


@dataclass
class KinematicLink:
  joint_name: str
  joint_limits: Tuple[float, float]
  rotation_axis: Tuple[float, float, float]
  origin_pos: Tuple[float, float, float]
  quat: Tuple[float, float, float, float]


def wxyz_to_xyzw(quat: np.ndarray) -> np.ndarray:
  return np.concat((quat[1:], quat[0:1]))


def parse_kinematic_chain(
  mujoco_xml_path: Path | str, 
  root_body_name: str, 
  end_effector_body_name: str
) -> List[KinematicLink]:
  """
  Parse the kinematic chain from a root body to a descendant.
  Each link corresponds to a MuJoCo joint element. The origin position and quaternion of a link are relative to the
  origin position and quaternion of the previous link, as is the case for nested bodies in the MuJoCo XML schema.

  :param mujoco_xml_path: The path to the source MuJoCo XML file.
  :param root_body_name: The name of the root body of the kinematic chain as specified in the MuJoCo file.
  :param end_effector_body_name: The name of the end effector body as specified in the MuJoCo file.
  """
  model = mujoco.MjModel.from_xml_path(str(mujoco_xml_path))
  root_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_body_name)
  ee_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, end_effector_body_name)
  
  if root_body_id == -1:
      raise ValueError(f"Root body '{root_body_name}' not found.")
  if ee_body_id == -1:
      raise ValueError(f"End effector body '{end_effector_body_name}' not found.")
  
  # build path from end effector to root
  path = []
  current_body_id = ee_body_id
  while current_body_id != root_body_id:
      if current_body_id == 0:
          raise ValueError(f"No path found from {root_body_name} to {end_effector_body_name}.")
      path.append(current_body_id)
      current_body_id = model.body_parentid[current_body_id]
  path.append(root_body_id)
  path.reverse()
  
  # create kinematic chain representation
  kinematic_chain = []
  pos_relative_to_last_joint = np.zeros(3)
  quat_relative_to_last_joint = scipyrotation.from_quat([0.0, 0.0, 0.0, 1.0])
  for i in range(len(path)):
    body_id = path[i]

    # accumulated positional and rotational offsets between fixed connections
    pos_relative_to_last_joint += model.body_pos[body_id]
    quat_relative_to_last_joint *= scipyrotation.from_quat(wxyz_to_xyzw(model.body_quat[body_id]))
    
    # retrieve the joint element in this body
    joint_id = None
    joint_name = ""
    for jnt_id in range(model.njnt):
      if model.jnt_bodyid[jnt_id] == body_id:
        joint_id = jnt_id
        joint_name = model.joint(jnt_id).name
        break
    
    if joint_id is None:
      continue
    
    # reset the succeeding joint pos and quat accumulators to be relative to this joint
    relative_joint_pos = pos_relative_to_last_joint + model.jnt_pos[joint_id]
    relative_joint_quat = quat_relative_to_last_joint
    pos_relative_to_last_joint = -model.jnt_pos[joint_id]
    quat_relative_to_last_joint = scipyrotation.from_quat([0.0, 0.0, 0.0, 1.0])

    kinematic_chain = kinematic_chain + [
      KinematicLink(
        joint_name=joint_name,
        joint_limits=tuple(model.jnt_range[joint_id]) if model.jnt_limited[joint_id] else (-np.inf, np.inf),
        rotation_axis=tuple(model.jnt_axis[joint_id]),
        origin_pos=relative_joint_pos,
        quat=tuple(relative_joint_quat.as_quat())
      )
    ]
  
  if not kinematic_chain:
     raise ValueError('No joints found in path from {root_body_name} to {end_effector_body_name}.')
  
  return kinematic_chain


def forward(kinematic_chain: List[KinematicLink], joint_q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  """
  Performs a forward kinematics pass.

  :param kinematic_chain: A list of kinematic links, ordered from root to end effector.
  :param joint_q: Angles for each of the kinematic links, in radians.

  :return: The global position (n, 3) and quaternion (n, 4) for each joint. The quaternions are XYZW-ordered.
  """
  assert joint_q.shape == (len(kinematic_chain),), (
    f'Number of joint angles ({joint_q.shape[0]}) must match length of kinematic chain ({len(kinematic_chain)}).'
  )
  global_pos = np.zeros((len(kinematic_chain), 3))
  global_quat = np.zeros((len(kinematic_chain), 4))
  global_quat[:, -1] = 1
  for i, (link, q) in enumerate(zip(reversed(kinematic_chain), reversed(joint_q))):
    rotation = scipyrotation.from_quat(link.quat) * scipyrotation.from_rotvec(q * np.array(link.rotation_axis))
    global_quat[-i - 1:] = (scipyrotation.from_quat(global_quat[-i - 1:]) * rotation).as_quat()
    global_pos[-i - 1:] = rotation.apply(global_pos[-i - 1:]) + np.array(link.origin_pos)
  return global_pos, global_quat


def inverse(
  kinematic_chain: List[KinematicLink],
  starting_joint_q: np.ndarray,
  end_effector_displacement: np.ndarray,
  target_pos: np.ndarray,
  grasp_axis: np.ndarray
) -> np.ndarray:
  """
  Performs an inverse kinematics pass.

  :param kinematic chain: A list of `n` kinematic links, ordered from root to end effector.
  :param starting_joint_q: The initial joint angles used by the solver; shape (n,).
  :param end_effector_displacement: Custom positional displacement for the final link in the chain; shape (3,).
  :param target_pos: The target position; shape (3,).
  :param grasp_axis: The axis with which the end effector should be horizontally aligned; shape (3,).

  :return: An array of joint positions of shape (n,).
  """
  lower_bound = np.array([l.joint_limits[0] for l in kinematic_chain])
  upper_bound = np.array([l.joint_limits[1] for l in kinematic_chain])
  return scipy.optimize.least_squares(
    build_grasp_ik_objective(kinematic_chain, end_effector_displacement, target_pos, grasp_axis),
    starting_joint_q.clip(min=lower_bound, max=upper_bound),
    bounds=(lower_bound, upper_bound)
  ).x


def build_grasp_ik_objective(
  kinematic_chain: List[KinematicLink],
  end_effector_displacement: np.ndarray,
  target_pos: np.ndarray,
  grasp_axis: np.ndarray
):
  def objective(joint_q: np.ndarray):
    gripper_pos, gripper_quat = augmented_forward(kinematic_chain, joint_q, end_effector_displacement)
    gripper_grasp_plane = scipyrotation.from_quat(gripper_quat).apply(np.array([[1, 0, 0], [0, 0, 1]]))
    gripper_direction = gripper_grasp_plane[0:1]
    target_direction = target_pos - gripper_pos
    target_direction /= np.linalg.norm(target_direction)

    pos_error = gripper_pos - target_pos
    grasp_angle_error = np.abs(gripper_grasp_plane.dot(grasp_axis))
    pointing_angle_error = np.abs(gripper_direction.dot(target_direction))
    # upside_down_error = np.array([max(gripper_grasp_plane[1].dot(np.array([0, 0, -1])), 0)])
    return np.concat([10 * pos_error, grasp_angle_error, pointing_angle_error])
  return objective


def augmented_forward(
  kinematic_chain: List[KinematicLink],
  joint_q: np.ndarray,
  end_effector_displacement: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
  augmented_kinematic_chain = kinematic_chain + [
    KinematicLink(
      'end_effector',
      (0, 0),
      (0, 1, 0),
      tuple(end_effector_displacement),
      (0, 0, 0, 1)
    )
  ]
  joint_pos, joint_quat = forward(augmented_kinematic_chain, np.concat((joint_q, np.array([0.0]))))
  return joint_pos[-1], joint_quat[-1]
