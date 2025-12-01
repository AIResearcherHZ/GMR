import argparse
import pathlib
import time
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from rich import print
from tqdm import tqdm
import os
import numpy as np
import pickle
from scipy.spatial.transform import Rotation as sRot


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
    """Convert GMR qpos data to HOVER/Neural-WBC format."""
    num_frames = len(qpos_list)
    root_pos_raw = np.array([qpos[:3] for qpos in qpos_list])
    root_rot_wxyz = np.array([qpos[3:7] for qpos in qpos_list])
    dof_pos = np.array([qpos[7:] for qpos in qpos_list])

    root_rot_xyzw = root_rot_wxyz[:, [1, 2, 3, 0]]
    first_rot = sRot.from_quat(root_rot_xyzw[0])
    first_rot_inv = first_rot.inv()

    all_rots = sRot.from_quat(root_rot_xyzw)
    relative_rots = first_rot_inv * all_rots
    root_rot_relative = relative_rots.as_quat()

    root_trans_offset = root_pos_raw.copy()
    root_trans_offset[:, 0] -= root_pos_raw[0, 0]
    root_trans_offset[:, 1] -= root_pos_raw[0, 1]

    heading_rot_matrix = first_rot_inv.as_matrix()
    for i in range(num_frames):
        xy = root_trans_offset[i, :2]
        xy_rotated = heading_rot_matrix[:2, :2] @ xy
        root_trans_offset[i, :2] = xy_rotated

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

def load_optitrack_fbx_motion_file(motion_file):
    with open(motion_file, "rb") as f:
        motion_data = pickle.load(f)
    return motion_data

def offset_to_ground(retargeter: GMR, motion_data):
    offset = np.inf
    for human_data in motion_data:
        human_data = retargeter.to_numpy(human_data)
        human_data = retargeter.scale_human_data(human_data, retargeter.human_root_name, retargeter.human_scale_table)
        human_data = retargeter.offset_human_data(human_data, retargeter.pos_offsets1, retargeter.rot_offsets1)
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            if pos[2] < offset:
                offset = pos[2]

    return offset

if __name__ == "__main__":
    
    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--motion_file",
        help="FBX motion file to load (OptiTrack motion).",
        required=True,
        type=str,
    )
    
    parser.add_argument(
        "--robot",
        choices=["unitree_g1", "booster_t1", "stanford_toddy", "fourier_n1", "engineai_pm01", "taks_t1"],
        default="unitree_g1",
    )
        
    parser.add_argument(
        "--record_video",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--video_path",
        type=str,
        default="videos/optitrack_example.mp4",
    )

    parser.add_argument(
        "--rate_limit",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--save_path",
        default=None,
        help="Path to save the robot motion.",
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
        help="For hover format: split motion into segments (frames). 0 = no split.",
    )

    parser.add_argument(
        "--segment_overlap",
        type=int,
        default=0,
        help="For hover format: overlap between segments (frames).",
    )

    args = parser.parse_args()
    

    if args.save_path is not None:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:  # Only create directory if it's not empty
            os.makedirs(save_dir, exist_ok=True)
        qpos_list = []

    
    # Load OptiTrack FMB motion trajectory
    print(f"Loading OptiTrack FBX motion file: {args.motion_file}")
    data_frames = load_optitrack_fbx_motion_file(args.motion_file)
    print(f"Loaded {len(data_frames)} frames")
    
    
    # Initialize the retargeting system with fbx configuration
    retargeter = GMR(
        src_human="fbx_offline",  # Use the new fbx configuration
        tgt_robot=args.robot,
        actual_human_height=1.8,
    )

    height_offset = offset_to_ground(retargeter, data_frames)
    retargeter.set_ground_offset(height_offset)

    motion_fps = 120
    
    robot_motion_viewer = RobotMotionViewer(robot_type=args.robot,
                                            motion_fps=motion_fps,
                                            transparent_robot=1,
                                            record_video=args.record_video,
                                            video_path=args.video_path,
                                            camera_follow=False,
                                            # video_width=2080,
                                            # video_height=1170
                                            )
    
    # FPS measurement variables
    fps_counter = 0
    fps_start_time = time.time()
    fps_display_interval = 2.0  # Display FPS every 2 seconds
    
    print(f"mocap_frame_rate: {motion_fps}")
    
    # Create tqdm progress bar for the total number of frames
    pbar = tqdm(total=len(data_frames), desc="Retargeting OptiTrack motion")
    
    # Start the viewer
    i = 0

    try:
        while i < len(data_frames):
            
            # FPS measurement
            fps_counter += 1
            current_time = time.time()
            if current_time - fps_start_time >= fps_display_interval:
                actual_fps = fps_counter / (current_time - fps_start_time)
                print(f"Actual rendering FPS: {actual_fps:.2f}")
                fps_counter = 0
                fps_start_time = current_time
                
            # Update progress bar
            pbar.update(1)

            # Update task targets.
            smplx_data = data_frames[i]

            # retarget
            qpos = retargeter.retarget(smplx_data)

            # visualize
            robot_motion_viewer.step(
                root_pos=qpos[:3],
                root_rot=qpos[3:7],
                dof_pos=qpos[7:],
                human_motion_data=retargeter.scaled_human_data,
                rate_limit=args.rate_limit,
                # human_pos_offset=np.array([0.0, 0.0, 0.0])
            )

            i += 1

            if args.save_path is not None:
                qpos_list.append(qpos)
    except KeyboardInterrupt:
        print("\nInterrupted by user, cleaning up...")
    finally:
        # Close progress bar
        pbar.close()
        robot_motion_viewer.close()

    if args.save_path is not None:
        import joblib

        if args.output_format == "gmr":
            root_pos = np.array([qpos[:3] for qpos in qpos_list])
            root_rot = np.array([qpos[3:7][[1, 2, 3, 0]] for qpos in qpos_list])
            dof_pos = np.array([qpos[7:] for qpos in qpos_list])
            motion_data = {
                "fps": motion_fps,
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
            base_name = os.path.splitext(os.path.basename(args.motion_file))[0]
            motion_data = convert_to_hover_format(
                qpos_list, motion_fps, base_name,
                segment_length=args.segment_length,
                segment_overlap=args.segment_overlap
            )
            joblib.dump(motion_data, args.save_path)
            n_motions = len(motion_data)

        print(f"Saved to {args.save_path} (format: {args.output_format}, motions: {n_motions})")
 