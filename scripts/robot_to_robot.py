"""
Robot to Robot Motion Retargeting Script

This script converts motion data from one robot format to another robot format.
It maps joint angles directly between robots with similar kinematic structures.

Usage:
    python scripts/robot_to_robot.py \
        --src_file data/TWIST2_dataset/AMASS_g1_GMR8/motion.pkl \
        --src_robot unitree_g1 \
        --tgt_robot taks_t1 \
        --save_path output.pkl
"""

import argparse
import pickle
import numpy as np
import torch
from rich import print
from tqdm import tqdm
import os
import sys

# Add GMR to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from general_motion_retargeting import ROBOT_XML_DICT, KinematicsModel

# Global z-axis offset for root position adjustment (in meters)
Z_OFFSET = -0.02

# Joint mapping from G1 (29 DOF) to Taks_T1 (32 DOF, excluding neck)
# G1 joint order (29 DOF):
#   0-5: left leg (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)
#   6-11: right leg (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)
#   12-14: waist (yaw, roll, pitch)
#   15-22: left arm (shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw)
#   23-28: right arm (shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw)

# Taks_T1 joint order (32 DOF, based on XML):
#   0-5: left leg (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)
#   6-11: right leg (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)
#   12-14: waist (yaw, roll, pitch)
#   15-21: left arm (shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_yaw, wrist_pitch)
#   22-28: right arm (shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_yaw, wrist_pitch)
#   29-31: neck (yaw, roll, pitch)

# Note: G1 wrist order is (roll, pitch, yaw), Taks_T1 wrist order is (roll, yaw, pitch)

G1_TO_TAKS_T1_MAPPING = {
    # Left leg (same order)
    0: 0,   # left_hip_pitch
    1: 1,   # left_hip_roll
    2: 2,   # left_hip_yaw
    3: 3,   # left_knee
    4: 4,   # left_ankle_pitch
    5: 5,   # left_ankle_roll
    # Right leg (same order)
    6: 6,   # right_hip_pitch
    7: 7,   # right_hip_roll
    8: 8,   # right_hip_yaw
    9: 9,   # right_knee
    10: 10, # right_ankle_pitch
    11: 11, # right_ankle_roll
    # Waist (same order)
    12: 12, # waist_yaw
    13: 13, # waist_roll
    14: 14, # waist_pitch
    # Left arm
    15: 15, # left_shoulder_pitch
    16: 16, # left_shoulder_roll
    17: 17, # left_shoulder_yaw
    18: 18, # left_elbow
    19: 19, # left_wrist_roll
    20: 21, # left_wrist_pitch -> taks_t1 index 21
    21: 20, # left_wrist_yaw -> taks_t1 index 20
    # Right arm
    22: 22, # right_shoulder_pitch
    23: 23, # right_shoulder_roll
    24: 24, # right_shoulder_yaw
    25: 25, # right_elbow
    26: 26, # right_wrist_roll
    27: 28, # right_wrist_pitch -> taks_t1 index 28
    28: 27, # right_wrist_yaw -> taks_t1 index 27
}

# Reverse mapping for Taks_T1 to G1
TAKS_T1_TO_G1_MAPPING = {v: k for k, v in G1_TO_TAKS_T1_MAPPING.items()}


def convert_g1_to_taks_t1(dof_pos_g1):
    """
    Convert G1 DOF positions to Taks_T1 DOF positions.
    
    Args:
        dof_pos_g1: numpy array of shape (N, 29) - G1 joint angles
        
    Returns:
        dof_pos_t1: numpy array of shape (N, 32) - Taks_T1 joint angles
    """
    num_frames = dof_pos_g1.shape[0]
    dof_pos_t1 = np.zeros((num_frames, 32), dtype=dof_pos_g1.dtype)
    
    for g1_idx, t1_idx in G1_TO_TAKS_T1_MAPPING.items():
        dof_pos_t1[:, t1_idx] = dof_pos_g1[:, g1_idx]
    
    # Neck joints (29-31) remain at zero
    return dof_pos_t1


def convert_taks_t1_to_g1(dof_pos_t1):
    """
    Convert Taks_T1 DOF positions to G1 DOF positions.
    
    Args:
        dof_pos_t1: numpy array of shape (N, 32) - Taks_T1 joint angles
        
    Returns:
        dof_pos_g1: numpy array of shape (N, 29) - G1 joint angles
    """
    num_frames = dof_pos_t1.shape[0]
    dof_pos_g1 = np.zeros((num_frames, 29), dtype=dof_pos_t1.dtype)
    
    for t1_idx, g1_idx in TAKS_T1_TO_G1_MAPPING.items():
        dof_pos_g1[:, g1_idx] = dof_pos_t1[:, t1_idx]
    
    return dof_pos_g1


def load_robot_motion(file_path):
    """Load robot motion data from pkl file."""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def save_robot_motion(data, file_path):
    """Save robot motion data to pkl file."""
    os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else '.', exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def get_taks_t1_link_body_list():
    """Get the link body list for Taks_T1 robot."""
    return [
        'pelvis',
        'left_hip_pitch_link', 'left_hip_roll_link', 'left_hip_yaw_link',
        'left_knee_link', 'left_ankle_pitch_link', 'left_ankle_roll_link',
        'right_hip_pitch_link', 'right_hip_roll_link', 'right_hip_yaw_link',
        'right_knee_link', 'right_ankle_pitch_link', 'right_ankle_roll_link',
        'waist_yaw_link', 'waist_roll_link', 'torso_link',
        'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link',
        'left_elbow_link', 'left_wrist_roll_link', 'left_wrist_yaw_link', 'left_wrist_pitch_link',
        'right_shoulder_pitch_link', 'right_shoulder_roll_link', 'right_shoulder_yaw_link',
        'right_elbow_link', 'right_wrist_roll_link', 'right_wrist_yaw_link', 'right_wrist_pitch_link',
        'neck_yaw_link', 'neck_roll_link', 'head_link'
    ]


def compute_local_body_pos(dof_pos, robot_type, device="cuda:0"):
    """
    Compute local body positions using forward kinematics.
    
    Args:
        dof_pos: numpy array of shape (N, num_dof)
        robot_type: robot type string
        device: torch device
        
    Returns:
        local_body_pos: numpy array of shape (N, num_bodies, 3)
        body_names: list of body names
    """
    xml_path = str(ROBOT_XML_DICT[robot_type])
    kinematics_model = KinematicsModel(xml_path, device=device)
    
    num_frames = dof_pos.shape[0]
    
    # Use identity root pose to get local body positions
    identity_root_pos = torch.zeros((num_frames, 3), device=device)
    identity_root_rot = torch.zeros((num_frames, 4), device=device)
    identity_root_rot[:, -1] = 1.0  # w=1 for identity quaternion (xyzw format)
    
    dof_pos_tensor = torch.from_numpy(dof_pos).to(device=device, dtype=torch.float)
    
    local_body_pos, _ = kinematics_model.forward_kinematics(
        identity_root_pos,
        identity_root_rot,
        dof_pos_tensor
    )
    
    return local_body_pos.detach().cpu().numpy(), kinematics_model.body_names


def convert_motion(src_data, src_robot, tgt_robot, device="cuda:0"):
    """
    Convert motion data from source robot to target robot.
    
    Args:
        src_data: dict with keys 'fps', 'root_pos', 'root_rot', 'dof_pos', etc.
        src_robot: source robot name
        tgt_robot: target robot name
        device: torch device for FK computation
        
    Returns:
        tgt_data: converted motion data dict
    """
    # Validate input
    required_keys = ['fps', 'root_pos', 'root_rot', 'dof_pos']
    for key in required_keys:
        if key not in src_data:
            raise ValueError(f"Missing required key: {key}")
    
    # Convert DOF positions
    if src_robot == 'unitree_g1' and tgt_robot == 'taks_t1':
        tgt_dof_pos = convert_g1_to_taks_t1(src_data['dof_pos'])
    elif src_robot == 'taks_t1' and tgt_robot == 'unitree_g1':
        tgt_dof_pos = convert_taks_t1_to_g1(src_data['dof_pos'])
    else:
        raise ValueError(f"Unsupported conversion: {src_robot} -> {tgt_robot}")
    
    # Compute local body positions using FK for both source and target
    src_local_body_pos, src_body_names = compute_local_body_pos(
        src_data['dof_pos'], src_robot, device)
    tgt_local_body_pos, tgt_body_names = compute_local_body_pos(
        tgt_dof_pos, tgt_robot, device)
    
    # Find ankle indices to calculate height adjustment
    src_ankle_idx = None
    tgt_ankle_idx = None
    for i, name in enumerate(src_body_names):
        if 'left_ankle_roll_link' in name:
            src_ankle_idx = i
            break
    for i, name in enumerate(tgt_body_names):
        if 'left_ankle_roll_link' in name:
            tgt_ankle_idx = i
            break
    
    # Adjust root_pos.z based on the difference in ankle height
    tgt_root_pos = src_data['root_pos'].copy()
    if src_ankle_idx is not None and tgt_ankle_idx is not None:
        # Use first frame to calculate height difference
        src_ankle_z = src_local_body_pos[0, src_ankle_idx, 2]
        tgt_ankle_z = tgt_local_body_pos[0, tgt_ankle_idx, 2]
        height_diff = src_ankle_z - tgt_ankle_z
        tgt_root_pos[:, 2] += height_diff
        print(f"  Height adjustment: {height_diff:.4f}m "
              f"(src ankle z: {src_ankle_z:.4f}, tgt ankle z: {tgt_ankle_z:.4f})")
    
    # Apply global z-axis offset
    if Z_OFFSET != 0.0:
        tgt_root_pos[:, 2] += Z_OFFSET
        print(f"  Applied global Z offset: {Z_OFFSET:.4f}m")
    
    # Create target data
    tgt_data = {
        'fps': src_data['fps'],
        'root_pos': tgt_root_pos,
        'root_rot': src_data['root_rot'].copy(),
        'dof_pos': tgt_dof_pos,
        'local_body_pos': tgt_local_body_pos,
        'link_body_list': tgt_body_names,
    }
    
    return tgt_data


def main():
    parser = argparse.ArgumentParser(description='Convert robot motion between different robot formats')
    parser.add_argument('--src_file', type=str, required=True,
                        help='Path to source robot motion pkl file')
    parser.add_argument('--src_robot', type=str, required=True,
                        choices=['unitree_g1', 'taks_t1'],
                        help='Source robot type')
    parser.add_argument('--tgt_robot', type=str, required=True,
                        choices=['unitree_g1', 'taks_t1'],
                        help='Target robot type')
    parser.add_argument('--save_path', type=str, required=True,
                        help='Path to save converted motion pkl file')
    
    args = parser.parse_args()
    
    if args.src_robot == args.tgt_robot:
        print("[yellow]Warning: Source and target robots are the same. No conversion needed.[/yellow]")
        return
    
    print(f"[bold]Converting motion from {args.src_robot} to {args.tgt_robot}[/bold]")
    print(f"  Source: {args.src_file}")
    print(f"  Target: {args.save_path}")
    
    # Load source motion
    print("Loading source motion...")
    src_data = load_robot_motion(args.src_file)
    
    print(f"  Frames: {src_data['dof_pos'].shape[0]}")
    print(f"  FPS: {src_data['fps']}")
    print(f"  DOF: {src_data['dof_pos'].shape[1]}")
    
    # Convert motion
    print("Converting motion...")
    tgt_data = convert_motion(src_data, args.src_robot, args.tgt_robot)
    
    print(f"  Target DOF: {tgt_data['dof_pos'].shape[1]}")
    
    # Save target motion
    print("Saving converted motion...")
    save_robot_motion(tgt_data, args.save_path)
    
    print(f"[bold green]Done! Saved to {args.save_path}[/bold green]")


if __name__ == '__main__':
    main()
