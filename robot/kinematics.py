from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple
import mujoco
import numpy as np
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


def extract_kinematic_info(
  mujoco_xml_path: Path, 
  root_body_name: str, 
  end_effector_body_name: str
) -> List[KinematicLink]:
  """
  Extract kinematic chain from MuJoCo model from root to end effector.
  
  Args:
      mujoco_xml_path: Path to MuJoCo XML file
      root_body_name: Name of the root body
      end_effector_body_name: Name of the end effector body
      
  Returns:
      KinematicLink representing the kinematic chain (from root to end effector)
  """
  model = mujoco.MjModel.from_xml_path(str(mujoco_xml_path))
  
  # Get body IDs
  root_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_body_name)
  ee_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, end_effector_body_name)
  
  if root_body_id == -1:
      raise ValueError(f"Root body '{root_body_name}' not found")
  if ee_body_id == -1:
      raise ValueError(f"End effector body '{end_effector_body_name}' not found")
  
  # Build path from end effector to root
  path = []
  current_body_id = ee_body_id
  while current_body_id != root_body_id:
      if current_body_id == 0:  # Reached world body without finding root
          raise ValueError(f"No path found from {root_body_name} to {end_effector_body_name}")
      path.append(current_body_id)
      current_body_id = model.body_parentid[current_body_id]
  path.append(root_body_id)
  
  # Reverse to go from root to end effector
  path.reverse()
  
  # Build kinematic chain (from root to end effector)
  kinematic_chain = []
  pos_relative_to_last_joint = np.zeros(3)
  quat_relative_to_last_joint = scipyrotation.from_quat([0.0, 0.0, 0.0, 1.0])
  for i in range(len(path)):  # Start from first child of root
    body_id = path[i]
    pos_relative_to_last_joint += model.body_pos[body_id]
    quat_relative_to_last_joint *= scipyrotation.from_quat(wxyz_to_xyzw(model.body_quat[body_id]))
    
    # Find joint associated with this body
    joint_id = None
    joint_name = ""
    
    # Search for a joint whose body is this body_id
    for jnt_id in range(model.njnt):
      if model.jnt_bodyid[jnt_id] == body_id:
        joint_id = jnt_id
        joint_name = model.joint(jnt_id).name
        break
    
    if joint_id is None:
      # Body has no joint (e.g., fixed connection)
      continue
    
    # Get joint limits
    jnt_limited = model.jnt_limited[joint_id]
    if jnt_limited:
      joint_limits = tuple(model.jnt_range[joint_id])
    else:
      joint_limits = (-np.inf, np.inf)
    
    # Get rotation axis
    rotation_axis = tuple(model.jnt_axis[joint_id])
    
    # Calculate origin_pos: accumulate body positions from last joint to this joint
    # and add this joint's offset, minus the last joint's offset
    relative_joint_pos = pos_relative_to_last_joint + model.jnt_pos[joint_id]
    pos_relative_to_last_joint = -model.jnt_pos[joint_id]
    # accumulated_pos = np.zeros(3)
    
    # Accumulate body positions from the body after the last joint to current body
    # for j in range(last_joint_body_idx + 1, i + 1):
    #   accumulated_pos += model.body_pos[path[j]]
    
    # Add current joint's offset within its body
    # accumulated_pos += model.jnt_pos[joint_id]
    
    # Subtract the previous joint's offset (if there was a previous joint)
    # if kinematic_chain:
    #   # Find the previous joint
    #   prev_body_id = path[last_joint_body_idx]
    #   for jnt_id in range(model.njnt):
    #     if model.jnt_bodyid[jnt_id] == prev_body_id:
    #       accumulated_pos -= model.jnt_pos[jnt_id]
    #       break
    
    # origin_pos = tuple(accumulated_pos)
    
    kinematic_chain = kinematic_chain + [
      KinematicLink(
        joint_name=joint_name,
        joint_limits=joint_limits,
        rotation_axis=rotation_axis,
        origin_pos=relative_joint_pos,
        quat=tuple(quat_relative_to_last_joint.as_quat())
      )
    ]
    
    last_joint_body_idx = i
  
  if kinematic_chain is None:
    raise ValueError(f"No joints found in kinematic chain from {root_body_name} to {end_effector_body_name}")
  
  return kinematic_chain

def forward_kinematics(kinematic_chain: List[KinematicLink], joint_q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  assert joint_q.shape == (len(kinematic_chain),), (
    f'Number of joint angles ({joint_q.shape[0]}) must match length of kinematic chain ({len(kinematic_chain)}).'
  )
  global_pos = np.zeros((len(kinematic_chain), 3))
  global_quat = np.zeros((len(kinematic_chain), 4))
  global_quat[:, -1] = 1
  for i, (link, q) in enumerate(zip(reversed(kinematic_chain), reversed(joint_q))):
    rotation = scipyrotation.from_rotvec(q * np.array(link.rotation_axis))
    global_pos[-i - 1:] = rotation.apply(global_pos[-i - 1:]) + np.array(link.origin_pos)
    global_quat[-i - 1] = link.quat
    global_quat[-i - 1:] = (scipyrotation.from_quat(global_quat[-i - 1:]) * rotation).as_quat()
  return global_pos, global_quat

from robot_descriptions import aloha_mj_description
k = extract_kinematic_info(aloha_mj_description.MJCF_PATH, 'left/base_link', 'left/gripper_base')

print(forward_kinematics(k, np.zeros(len(k))))
