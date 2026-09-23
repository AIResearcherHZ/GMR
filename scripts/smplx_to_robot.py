import argparse
import pathlib
import os
import time
import platform
import multiprocessing

import mujoco as mj
import numpy as np
import torch
from scipy.spatial.transform import Rotation as sRot

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.utils.smpl import load_smplx_file, get_smplx_data_offline_fast
from tqdm import tqdm

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

def model_has_free_root(model) -> bool:
    return model.njnt > 0 and int(model.jnt_type[0]) == int(mj.mjtJoint.mjJNT_FREE)

def split_qpos(qpos, model):
    qpos = np.asarray(qpos)
    if model_has_free_root(model):
        return qpos[:3].copy(), qpos[3:7].copy(), qpos[7:].copy()
    root_pos = model.body_pos[1].copy().astype(qpos.dtype)
    root_rot = model.body_quat[1].copy().astype(qpos.dtype)
    return root_pos, root_rot, qpos.copy()

def print_hardware_info():
    table = Table(title="硬件信息")
    table.add_column("项目", style="cyan")
    table.add_column("信息", style="green")
    
    table.add_row("系统", platform.system())
    table.add_row("处理器", platform.processor() or platform.machine())
    table.add_row("CPU核心数", str(multiprocessing.cpu_count()))
    
    if torch.cuda.is_available():
        table.add_row("CUDA可用", "是")
        table.add_row("GPU数量", str(torch.cuda.device_count()))
        for i in range(torch.cuda.device_count()):
            table.add_row(f"GPU {i}", torch.cuda.get_device_name(i))
    else:
        table.add_row("CUDA可用", "否")
    
    console.print(table)


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
    """
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

if __name__ == "__main__":
    console.print(Panel.fit("[bold cyan]SMPL-X转机器人工具[/bold cyan]", border_style="cyan"))
    print_hardware_info()
    
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
                "pnd_adam_lite", "openloong", "tienkung", "fourier_gr3", "taks_t1", "semi_taks_t1", "semi_taks_lv1", "semi_taks_lv1_chassis"],
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
    
    console.print(f"[cyan]总帧数: {len(smplx_data_frames)}[/cyan]")
    console.print(f"[cyan]预计时长: {len(smplx_data_frames)/aligned_fps:.2f}秒[/cyan]")
    
    if args.save_path is not None:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:  # Only create directory if it's not empty
            os.makedirs(save_dir, exist_ok=True)
        qpos_list = []
    
    # Start the viewer
    i = 0
    process_start_time = time.time()
    pbar = tqdm(total=len(smplx_data_frames), desc="重定向处理中", unit="帧", ncols=100)

    try:
        while True:
            if args.loop:
                i = (i + 1) % len(smplx_data_frames)
            else:
                i += 1
                if i >= len(smplx_data_frames):
                    break
            
            pbar.update(1)
            
            # FPS measurement
            fps_counter += 1
            current_time = time.time()
            if current_time - fps_start_time >= fps_display_interval:
                actual_fps = fps_counter / (current_time - fps_start_time)
                elapsed = current_time - process_start_time
                progress_pct = (i / len(smplx_data_frames)) * 100
                console.print(f"[yellow]渲染FPS: {actual_fps:.2f} | 进度: {progress_pct:.1f}% | 已用时: {elapsed:.1f}秒[/yellow]")
                fps_counter = 0
                fps_start_time = current_time
            
            # Update task targets.
            smplx_data = smplx_data_frames[i]

            # retarget
            qpos = retarget.retarget(smplx_data)

            # visualize
            rp, rr, dp = split_qpos(qpos, retarget.model)
            robot_motion_viewer.step(
                root_pos=rp,
                root_rot=rr,
                dof_pos=dp,
                human_motion_data=retarget.scaled_human_data,
                # human_motion_data=smplx_data,
                human_pos_offset=np.array([0.0, 0.0, 0.0]),
                show_human_body_name=False,
                rate_limit=args.rate_limit,
                follow_camera=False,
            )
            if args.save_path is not None:
                qpos_list.append(qpos)
    except KeyboardInterrupt:
        console.print("[yellow]\n用户中断，清理中...[/yellow]")
    finally:
        pbar.close()
        robot_motion_viewer.close()
        total_time = time.time() - process_start_time
        if i > 0:
            console.print(f"[cyan]处理了 {i}/{len(smplx_data_frames)} 帧, 总耗时: {total_time:.2f}秒[/cyan]")
            
    if args.save_path is not None:
        import pickle
        import joblib

        if args.output_format == "gmr":
            # Fixed-base robots (e.g. semi_taks_t1) carry no free root in qpos -- their
            # qpos is just the actuated DoFs. Detect that and take the (constant) root
            # pose from the base body's MJCF transform instead of mis-slicing qpos.
            _m = retarget.model
            if _m.njnt > 0 and int(_m.jnt_type[0]) == int(mj.mjtJoint.mjJNT_FREE):  # floating base: qpos = [root_pos(3), root_rot_wxyz(4), dof]
                root_pos = np.array([qpos[:3] for qpos in qpos_list])
                root_rot = np.array([qpos[3:7][[1, 2, 3, 0]] for qpos in qpos_list])
                dof_pos = np.array([qpos[7:] for qpos in qpos_list])
            else:  # fixed base: constant root from base body XML pose; qpos == dof
                _n = len(qpos_list)
                root_pos = np.tile(_m.body_pos[1].astype(np.float64), (_n, 1))
                root_rot = np.tile(_m.body_quat[1][[1, 2, 3, 0]].astype(np.float64), (_n, 1))
                dof_pos = np.array([qpos for qpos in qpos_list])

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
            base_name = os.path.splitext(os.path.basename(args.smplx_file))[0]
            motion_data = convert_to_hover_format(
                qpos_list, aligned_fps, base_name,
                segment_length=args.segment_length,
                segment_overlap=args.segment_overlap
            )
            joblib.dump(motion_data, args.save_path)
            n_motions = len(motion_data)

        console.print(f"[bold green]已保存到 {args.save_path} (格式: {args.output_format}, 动作数: {n_motions})[/bold green]")

