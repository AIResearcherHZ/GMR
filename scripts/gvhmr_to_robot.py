import argparse
import pathlib
import os
import time

import numpy as np
from scipy.spatial.transform import Rotation as sRot

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.utils.smpl import load_gvhmr_pred_file, get_gvhmr_data_offline_fast

from rich import print


# H1 rotation axis for each DOF (19 joints)
H1_ROTATION_AXIS = np.array([
    [0, 0, 1],  # 0: l_hip_yaw
    [1, 0, 0],  # 1: l_hip_roll
    [0, 1, 0],  # 2: l_hip_pitch
    [0, 1, 0],  # 3: l_knee
    [0, 1, 0],  # 4: l_ankle
    [0, 0, 1],  # 5: r_hip_yaw
    [1, 0, 0],  # 6: r_hip_roll
    [0, 1, 0],  # 7: r_hip_pitch
    [0, 1, 0],  # 8: r_knee
    [0, 1, 0],  # 9: r_ankle
    [0, 0, 1],  # 10: torso
    [0, 1, 0],  # 11: l_shoulder_pitch
    [1, 0, 0],  # 12: l_shoulder_roll
    [0, 0, 1],  # 13: l_shoulder_yaw
    [0, 1, 0],  # 14: l_elbow
    [0, 1, 0],  # 15: r_shoulder_pitch
    [1, 0, 0],  # 16: r_shoulder_roll
    [0, 0, 1],  # 17: r_shoulder_yaw
    [0, 1, 0],  # 18: r_elbow
])


def convert_to_hover_format(qpos_list, motion_fps, base_name, segment_length=0, segment_overlap=0):
    """
    Convert GMR qpos data to HOVER/Neural-WBC format.
    
    Key differences from raw data:
    1. root_trans_offset: relative to first frame (XY starts at 0)
    2. root_rot: relative to first frame orientation (starts as identity)
    3. pose_aa: computed from dof using H1 rotation axis
    4. All quaternions in xyzw format
    """
    num_frames = len(qpos_list)
    
    # Extract data from qpos (wxyz format from GMR)
    root_pos_raw = np.array([qpos[:3] for qpos in qpos_list])
    root_rot_wxyz = np.array([qpos[3:7] for qpos in qpos_list])
    dof_pos = np.array([qpos[7:] for qpos in qpos_list])
    
    # Convert root_rot from wxyz to xyzw for scipy
    root_rot_xyzw = root_rot_wxyz[:, [1, 2, 3, 0]]
    
    # Get first frame rotation for relative computation
    first_rot = sRot.from_quat(root_rot_xyzw[0])
    first_rot_inv = first_rot.inv()
    
    # Compute relative rotations (relative to first frame)
    all_rots = sRot.from_quat(root_rot_xyzw)
    relative_rots = first_rot_inv * all_rots
    root_rot_relative = relative_rots.as_quat()  # xyzw format
    
    # Compute relative positions (relative to first frame XY, keep Z as height)
    root_trans_offset = root_pos_raw.copy()
    root_trans_offset[:, 0] -= root_pos_raw[0, 0]  # X relative to start
    root_trans_offset[:, 1] -= root_pos_raw[0, 1]  # Y relative to start
    # Z stays as absolute height
    
    # Rotate positions by inverse of first frame heading
    heading_rot_matrix = first_rot_inv.as_matrix()
    for i in range(num_frames):
        xy = root_trans_offset[i, :2]
        xy_rotated = heading_rot_matrix[:2, :2] @ xy
        root_trans_offset[i, :2] = xy_rotated
    
    # Compute pose_aa from dof (22 joints x 3)
    pose_aa = np.zeros((num_frames, 22, 3), dtype=np.float32)
    for i in range(19):
        pose_aa[:, i + 1, :] = H1_ROTATION_AXIS[i] * dof_pos[:, i:i+1]
    
    def create_motion_entry(start, end, seg_name):
        return {
            "root_trans_offset": root_trans_offset[start:end].astype(np.float64),
            "pose_aa": pose_aa[start:end].astype(np.float32),
            "dof": dof_pos[start:end].astype(np.float32),
            "root_rot": root_rot_relative[start:end].astype(np.float64),
            "smpl_joints": np.zeros((end - start, 24, 3), dtype=np.float32),
            "fps": int(motion_fps),
        }
    
    if segment_length > 0:
        motion_data = {}
        step = segment_length - segment_overlap
        seg_idx = 0
        start = 0
        while start < num_frames:
            end = min(start + segment_length, num_frames)
            if end - start < 30:
                break
            motion_name = f"{seg_idx}-{base_name}_seg{seg_idx:03d}_poses"
            motion_data[motion_name] = create_motion_entry(start, end, motion_name)
            seg_idx += 1
            start += step
        print(f"Split into {len(motion_data)} segments")
    else:
        motion_name = f"0-{base_name}_poses"
        motion_data = {motion_name: create_motion_entry(0, num_frames, motion_name)}
    
    return motion_data

if __name__ == "__main__":
    
    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gvhmr_pred_file",
        help="SMPLX motion file to load.",
        type=str,
        # required=True,
        default="/home/yanjieze/projects/g1_wbc/GMR/GVHMR/outputs/demo/tennis/hmr4d_results.pt",
    )
    
    parser.add_argument(
        "--robot",
        choices=["unitree_g1", "unitree_g1_with_hands", "unitree_h1", "unitree_h1_2",
                 "booster_t1", "booster_t1_29dof","stanford_toddy", "fourier_n1", 
                "engineai_pm01", "kuavo_s45", "hightorque_hi", "galaxea_r1pro", "berkeley_humanoid_lite", "booster_k1",
                "pnd_adam_lite", "openloong", "tienkung", "taks_t1"],
        default="unitree_g1",
    )
    
    parser.add_argument(
        "--save_path",
        default=None,
        help="Path to save the robot motion.",
    )
    
    parser.add_argument(
        "--loop",
        default=False,
        action="store_true",
        help="Loop the motion.",
    )

    parser.add_argument(
        "--record_video",
        default=False,
        action="store_true",
        help="Record the video.",
    )

    parser.add_argument(
        "--rate_limit",
        default=False,
        action="store_true",
        help="Limit the rate of the retargeted robot motion to keep the same as the human motion.",
    )

    parser.add_argument(
        "--output_format",
        choices=["gmr", "hover"],
        default="gmr",
        help="Output format: 'gmr' for GMR format, 'hover' for HOVER/Neural-WBC format.",
    )

    parser.add_argument(
        "--segment_length",
        type=int,
        default=0,
        help="For hover format: split motion into segments of this length (frames). 0 means no split.",
    )

    parser.add_argument(
        "--segment_overlap",
        type=int,
        default=0,
        help="For hover format: overlap between segments (frames).",
    )

    args = parser.parse_args()


    SMPLX_FOLDER = HERE / ".." / "assets" / "body_models"
    
    
    # Load SMPLX trajectory
    smplx_data, body_model, smplx_output, actual_human_height = load_gvhmr_pred_file(
        args.gvhmr_pred_file, SMPLX_FOLDER
    )
    
    # align fps
    tgt_fps = 30
    smplx_data_frames, aligned_fps = get_gvhmr_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=tgt_fps)
    
    
   
    # Initialize the retargeting system
    retarget = GMR(
        actual_human_height=actual_human_height,
        src_human="smplx",
        tgt_robot=args.robot,
    )
    
    robot_motion_viewer = RobotMotionViewer(robot_type=args.robot,
                                            motion_fps=aligned_fps,
                                            transparent_robot=0,
                                            record_video=args.record_video,
                                            video_path=f"videos/{args.robot}_{args.gvhmr_pred_file.split('/')[-1].split('.')[0]}.mp4",)
    

    curr_frame = 0
    # FPS measurement variables
    fps_counter = 0
    fps_start_time = time.time()
    fps_display_interval = 2.0  # Display FPS every 2 seconds
    
    if args.save_path is not None:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:  # Only create directory if it's not empty
            os.makedirs(save_dir, exist_ok=True)
        qpos_list = []
    
    # Start the viewer
    i = 0

    while True:
        if args.loop:
            i = (i + 1) % len(smplx_data_frames)
        else:
            i += 1
            if i >= len(smplx_data_frames):
                break
        
        # FPS measurement
        fps_counter += 1
        current_time = time.time()
        if current_time - fps_start_time >= fps_display_interval:
            actual_fps = fps_counter / (current_time - fps_start_time)
            print(f"Actual rendering FPS: {actual_fps:.2f}")
            fps_counter = 0
            fps_start_time = current_time
        
        # Update task targets.
        smplx_data = smplx_data_frames[i]

        # retarget
        qpos = retarget.retarget(smplx_data)

        # visualize
        robot_motion_viewer.step(
            root_pos=qpos[:3],
            root_rot=qpos[3:7],
            dof_pos=qpos[7:],
            human_motion_data=retarget.scaled_human_data,
            # human_motion_data=smplx_data,
            human_pos_offset=np.array([0.0, 0.0, 0.0]),
            show_human_body_name=False,
            rate_limit=args.rate_limit,
        )
        if args.save_path is not None:
            qpos_list.append(qpos)
            
    if args.save_path is not None:
        import pickle
        import joblib
        
        if args.output_format == "gmr":
            root_pos = np.array([qpos[:3] for qpos in qpos_list])
            root_rot = np.array([qpos[3:7][[1, 2, 3, 0]] for qpos in qpos_list])
            dof_pos = np.array([qpos[7:] for qpos in qpos_list])
            motion_data = {
                "fps": aligned_fps,
                "root_pos": root_pos,
                "root_rot": root_rot,
                "dof_pos": dof_pos,
                "local_body_pos": None,
                "link_body_list": None,
            }
            with open(args.save_path, "wb") as f:
                pickle.dump(motion_data, f)
            n_motions = 1
        else:  # hover format
            base_name = os.path.splitext(os.path.basename(args.gvhmr_pred_file))[0]
            motion_data = convert_to_hover_format(
                qpos_list, aligned_fps, base_name,
                segment_length=args.segment_length,
                segment_overlap=args.segment_overlap
            )
            joblib.dump(motion_data, args.save_path)
            n_motions = len(motion_data)
        
        print(f"Saved to {args.save_path} (format: {args.output_format}, motions: {n_motions})")

            
      
    
    robot_motion_viewer.close()
