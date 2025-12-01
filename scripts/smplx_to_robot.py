import argparse
import pathlib
import os
import time

import numpy as np

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.utils.smpl import load_smplx_file, get_smplx_data_offline_fast

from rich import print

if __name__ == "__main__":
    
    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smplx_file",
        help="SMPLX motion file to load.",
        type=str,
        # required=True,
        default="/home/yanjieze/projects/g1_wbc/GMR/motion_data/ACCAD/Male1General_c3d/General_A1_-_Stand_stageii.npz",
        # default="/home/yanjieze/projects/g1_wbc/GMR/motion_data/ACCAD/Male2MartialArtsKicks_c3d/G8_-__roundhouse_left_stageii.npz"
        # default="/home/yanjieze/projects/g1_wbc/TWIST-dev/motion_data/AMASS/KIT_572_dance_chacha11_stageii.npz"
        # default="/home/yanjieze/projects/g1_wbc/GMR/motion_data/ACCAD/Male2MartialArtsPunches_c3d/E1_-__Jab_left_stageii.npz",
        # default="/home/yanjieze/projects/g1_wbc/GMR/motion_data/ACCAD/Male1Running_c3d/Run_C24_-_quick_side_step_left_stageii.npz",
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
        help="For hover format: split motion into segments (frames). 0 = no split.",
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
    smplx_data, body_model, smplx_output, actual_human_height = load_smplx_file(
        args.smplx_file, SMPLX_FOLDER
    )
    
    # align fps
    tgt_fps = 30
    smplx_data_frames, aligned_fps = get_smplx_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=tgt_fps)
    
   
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
                                            video_path=f"videos/{args.robot}_{args.smplx_file.split('/')[-1].split('.')[0]}.mp4",)
    

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

    try:
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
    except KeyboardInterrupt:
        print("\nInterrupted by user, cleaning up...")
    finally:
        robot_motion_viewer.close()
            
    if args.save_path is not None:
        import pickle
        import joblib
        root_pos = np.array([qpos[:3] for qpos in qpos_list])
        # save from wxyz to xyzw
        root_rot = np.array([qpos[3:7][[1,2,3,0]] for qpos in qpos_list])
        dof_pos = np.array([qpos[7:] for qpos in qpos_list])
        
        if args.output_format == "gmr":
            local_body_pos = None
            body_names = None
            motion_data = {
                "fps": aligned_fps,
                "root_pos": root_pos,
                "root_rot": root_rot,
                "dof_pos": dof_pos,
                "local_body_pos": local_body_pos,
                "link_body_list": body_names,
            }
            with open(args.save_path, "wb") as f:
                pickle.dump(motion_data, f)
        else:  # hover format
            base_name = os.path.splitext(os.path.basename(args.smplx_file))[0]
            num_frames = len(qpos_list)
            
            if args.segment_length > 0:
                motion_data = {}
                seg_len = args.segment_length
                step = seg_len - args.segment_overlap
                seg_idx = 0
                start = 0
                while start < num_frames:
                    end = min(start + seg_len, num_frames)
                    if end - start < 30:
                        break
                    motion_name = f"{seg_idx}-{base_name}_seg{seg_idx:03d}_poses"
                    motion_data[motion_name] = {
                        "root_trans_offset": root_pos[start:end].astype(np.float64),
                        "pose_aa": np.zeros((end-start, 22, 3), dtype=np.float32),
                        "dof": dof_pos[start:end].astype(np.float32),
                        "root_rot": root_rot[start:end].astype(np.float64),
                        "smpl_joints": np.zeros((end-start, 24, 3), dtype=np.float32),
                        "fps": int(aligned_fps),
                    }
                    seg_idx += 1
                    start += step
                print(f"Split into {len(motion_data)} segments")
            else:
                motion_name = f"0-{base_name}_poses"
                motion_data = {
                    motion_name: {
                        "root_trans_offset": root_pos.astype(np.float64),
                        "pose_aa": np.zeros((num_frames, 22, 3), dtype=np.float32),
                        "dof": dof_pos.astype(np.float32),
                        "root_rot": root_rot.astype(np.float64),
                        "smpl_joints": np.zeros((num_frames, 24, 3), dtype=np.float32),
                        "fps": int(aligned_fps),
                    }
                }
            joblib.dump(motion_data, args.save_path)
        n_motions = len(motion_data) if args.output_format == 'hover' else 1
        print(f"Saved to {args.save_path} (format: {args.output_format}, motions: {n_motions})")
