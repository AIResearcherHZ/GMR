import argparse
import pathlib
import time
import platform
import multiprocessing
import mujoco as mj
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.utils.lafan1 import load_bvh_file
from general_motion_retargeting.kinematics_model import KinematicsModel
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from tqdm import tqdm
import os
import numpy as np
import torch
from scipy.spatial.transform import Rotation as sRot

console = Console()


def model_has_free_root(model) -> bool:
    """robot 的 base 是浮动 (freejoint) 还是焊死? 决定 qpos 前 7 位是否是 root_pos+root_rot."""
    return model.njnt > 0 and int(model.jnt_type[0]) == int(mj.mjtJoint.mjJNT_FREE)


def split_qpos(qpos, model):
    """把 qpos 拆成 (root_pos[3], root_rot_wxyz[4], dof_pos[N]).
    fixed-base 时 root_pos 取 base body 的 XML pos, root_rot = identity."""
    qpos = np.asarray(qpos)
    if model_has_free_root(model):
        return qpos[:3].copy(), qpos[3:7].copy(), qpos[7:].copy()
    # fixed base: base body is id 1 (child of worldbody)
    root_pos = model.body_pos[1].copy().astype(qpos.dtype)
    root_rot = model.body_quat[1].copy().astype(qpos.dtype)  # MuJoCo stores body_quat as wxyz
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


def convert_to_hover_format(qpos_list, motion_fps, base_name, segment_length=0, segment_overlap=0, model=None):
    """
    Convert GMR qpos data to HOVER/Neural-WBC format.

    Key differences from raw data:
    1. root_trans_offset: relative to first frame (XY starts at 0)
    2. root_rot: relative to first frame orientation (starts as identity)
    3. pose_aa: computed from dof using H1 rotation axis
    4. All quaternions in xyzw format
    """
    num_frames = len(qpos_list)

    # Extract data from qpos (wxyz format from GMR). For fixed-base models, the
    # mujoco qpos has no leading 7 freejoint values; split_qpos synthesizes
    # constant root_pos/root_rot from the base body's XML attrs in that case.
    if model is None:
        # Fallback: assume floating base layout
        root_pos_raw = np.array([qpos[:3] for qpos in qpos_list])
        root_rot_wxyz = np.array([qpos[3:7] for qpos in qpos_list])
        dof_pos = np.array([qpos[7:] for qpos in qpos_list])
    else:
        rp_rr_dp = [split_qpos(q, model) for q in qpos_list]
        root_pos_raw = np.array([t[0] for t in rp_rr_dp])
        root_rot_wxyz = np.array([t[1] for t in rp_rr_dp])
        dof_pos = np.array([t[2] for t in rp_rr_dp])
    
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
    # This ensures the motion starts facing forward
    heading_rot_matrix = first_rot_inv.as_matrix()
    for i in range(num_frames):
        xy = root_trans_offset[i, :2]
        xy_rotated = heading_rot_matrix[:2, :2] @ xy
        root_trans_offset[i, :2] = xy_rotated
    
    # Compute pose_aa from dof (22 joints x 3)
    # Joint 0 is root (identity), joints 1-19 from dof, joints 20-21 are hands (zeros)
    pose_aa = np.zeros((num_frames, 22, 3), dtype=np.float32)
    for i in range(19):
        pose_aa[:, i + 1, :] = H1_ROTATION_AXIS[i] * dof_pos[:, i:i+1]
    
    # Create motion data dict
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
            if end - start < 30:  # Skip segments shorter than 1 second
                break
            motion_name = f"{seg_idx}-{base_name}_seg{seg_idx:03d}_poses"
            motion_data[motion_name] = create_motion_entry(start, end, motion_name)
            seg_idx += 1
            start += step
        console.print(f"[cyan]分割为 {len(motion_data)} 个片段[/cyan]")
    else:
        motion_name = f"0-{base_name}_poses"
        motion_data = {motion_name: create_motion_entry(0, num_frames, motion_name)}
    
    return motion_data

if __name__ == "__main__":
    console.print(Panel.fit("[bold cyan]BVH转机器人工具[/bold cyan]", border_style="cyan"))
    print_hardware_info()
    
    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bvh_file",
        help="BVH motion file to load.",
        required=True,
        type=str,
    )
    
    parser.add_argument(
        "--format",
        choices=["lafan1", "nokov", "mixamo"],
        default="lafan1",
    )
    
    parser.add_argument(
        "--loop",
        default=False,
        action="store_true",
        help="Loop the motion.",
    )
    
    parser.add_argument(
        "--robot",
        choices=["unitree_g1", "unitree_g1_with_hands", "booster_t1", "stanford_toddy", "fourier_n1", "engineai_pm01", "pal_talos", "taks_t1", "semi_taks_t1", "semi_taks_lv1", "semi_taks_lv1_chassis"],
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
        default="videos/example.mp4",
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
        "--motion_fps",
        default=30,
        type=int,
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

    
    # Load SMPLX trajectory
    lafan1_data_frames, actual_human_height = load_bvh_file(args.bvh_file, format=args.format)
    
    
    # Initialize the retargeting system
    retargeter = GMR(
        src_human=f"bvh_{args.format}",
        tgt_robot=args.robot,
        actual_human_height=actual_human_height,
    )

    motion_fps = args.motion_fps
    
    robot_motion_viewer = RobotMotionViewer(robot_type=args.robot,
                                            motion_fps=motion_fps,
                                            transparent_robot=0,
                                            record_video=args.record_video,
                                            video_path=args.video_path,
                                            # video_width=2080,
                                            # video_height=1170
                                            )
    
    # FPS measurement variables
    fps_counter = 0
    fps_start_time = time.time()
    fps_display_interval = 2.0  # Display FPS every 2 seconds
    
    console.print(f"[cyan]动捕帧率: {motion_fps}[/cyan]")
    console.print(f"[cyan]总帧数: {len(lafan1_data_frames)}[/cyan]")
    console.print(f"[cyan]预计时长: {len(lafan1_data_frames)/motion_fps:.2f}秒[/cyan]")
    
    # Create tqdm progress bar for the total number of frames
    pbar = tqdm(total=len(lafan1_data_frames), desc="重定向处理中", unit="帧", ncols=100)
    
    # Start the viewer
    i = 0
    process_start_time = time.time()

    try:
        while True:
            # FPS measurement
            fps_counter += 1
            current_time = time.time()
            if current_time - fps_start_time >= fps_display_interval:
                actual_fps = fps_counter / (current_time - fps_start_time)
                elapsed = current_time - process_start_time
                progress_pct = (i / len(lafan1_data_frames)) * 100
                console.print(f"[yellow]渲染FPS: {actual_fps:.2f} | 进度: {progress_pct:.1f}% | 已用时: {elapsed:.1f}秒[/yellow]")
                fps_counter = 0
                fps_start_time = current_time
                
            # Update progress bar
            pbar.update(1)

            # Update task targets.
            smplx_data = lafan1_data_frames[i]

            # retarget
            qpos = retargeter.retarget(smplx_data)

            # visualize
            rp, rr, dp = split_qpos(qpos, retargeter.model)
            robot_motion_viewer.step(
                root_pos=rp,
                root_rot=rr,
                dof_pos=dp,
                human_motion_data=retargeter.scaled_human_data,
                rate_limit=args.rate_limit,
                follow_camera=True,
            )

            if args.loop:
                i = (i + 1) % len(lafan1_data_frames)
            else:
                i += 1
                if i >= len(lafan1_data_frames):
                    break
            
            if args.save_path is not None:
                qpos_list.append(qpos)
    except KeyboardInterrupt:
        console.print("[yellow]\n用户中断，清理中...[/yellow]")
    finally:
        # Close progress bar
        pbar.close()
        robot_motion_viewer.close()
        
        # 显示处理统计
        total_time = time.time() - process_start_time
        if i > 0:
            avg_time_per_frame = total_time / i
            console.print(f"[cyan]处理了 {i}/{len(lafan1_data_frames)} 帧[/cyan]")
            console.print(f"[cyan]总耗时: {total_time:.2f}秒, 平均: {avg_time_per_frame*1000:.2f}毫秒/帧[/cyan]")
    
    if args.save_path is not None:
        import pickle
        import joblib
        
        console.print(f"[cyan]开始保存数据...[/cyan]")
        save_start_time = time.time()
        
        if args.output_format == "gmr":
            rp_rr_dp = [split_qpos(q, retargeter.model) for q in qpos_list]
            root_pos = np.array([t[0] for t in rp_rr_dp])
            root_rot = np.array([t[1][[1, 2, 3, 0]] for t in rp_rr_dp])  # wxyz to xyzw
            dof_pos = np.array([t[2] for t in rp_rr_dp])

            # Normalize root_pos to start from origin (keep only Z offset for height)
            first_frame_pos = root_pos[0].copy()
            root_pos[:, 0] -= first_frame_pos[0]  # X offset to 0
            root_pos[:, 1] -= first_frame_pos[1]  # Y offset to 0
            # Keep Z as is for height
            
            # Compute local_body_pos using forward kinematics
            device = "cpu"
            kinematics_model = KinematicsModel(retargeter.xml_file, device=device)
            num_frames = root_pos.shape[0]
            
            identity_root_pos = torch.zeros((num_frames, 3), device=device)
            identity_root_rot = torch.zeros((num_frames, 4), device=device)
            identity_root_rot[:, -1] = 1.0  # w=1 for identity quaternion (xyzw format)
            
            local_body_pos, _ = kinematics_model.forward_kinematics(
                identity_root_pos,
                identity_root_rot,
                torch.from_numpy(dof_pos).to(device=device, dtype=torch.float)
            )
            body_names = kinematics_model.body_names
            
            motion_data = {
                "fps": motion_fps,
                "root_pos": root_pos,
                "root_rot": root_rot,
                "dof_pos": dof_pos,
                "local_body_pos": local_body_pos.detach().cpu().numpy(),
                "link_body_list": body_names,
            }
            with open(args.save_path, "wb") as f:
                pickle.dump(motion_data, f)
            n_motions = 1
        else:  # hover format
            base_name = os.path.splitext(os.path.basename(args.bvh_file))[0]
            motion_data = convert_to_hover_format(
                qpos_list, motion_fps, base_name,
                segment_length=args.segment_length,
                segment_overlap=args.segment_overlap,
                model=retargeter.model,
            )
            joblib.dump(motion_data, args.save_path)
            n_motions = len(motion_data)
        
        save_time = time.time() - save_start_time
        console.print(f"[bold green]已保存到 {args.save_path}[/bold green]")
        console.print(f"[bold green]格式: {args.output_format}, 动作数: {n_motions}, 保存耗时: {save_time:.2f}秒[/bold green]")

       
