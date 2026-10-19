import argparse
import pathlib
import time
import platform
import multiprocessing
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.motion_data import split_qpos
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

H1_ROTATION_AXIS = np.array([
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0],
    [0, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0],
    [0, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
])

def convert_to_hover_format(qpos_list, motion_fps, base_name, segment_length=0, segment_overlap=0, *, model):
    """
    Convert GMR qpos data to HOVER/Neural-WBC format.

    Key differences from raw data:
    1. root_trans_offset: relative to first frame (XY starts at 0)
    2. root_rot: relative to first frame orientation (starts as identity)
    3. pose_aa: computed from dof using H1 rotation axis
    4. All quaternions in xyzw format
    """
    num_frames = len(qpos_list)

    root_pos_raw, root_rot_wxyz, dof_pos = split_qpos(qpos_list, model)

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

    if args.output_format == "hover" and args.robot != "unitree_h1":
        parser.error("HOVER 输出使用 H1 的关节定义，仅支持 unitree_h1；其他机器人请使用 gmr。")

    if args.save_path is not None:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        qpos_list = []

    lafan1_data_frames, actual_human_height = load_bvh_file(args.bvh_file, format=args.format)

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

                                            )

    fps_counter = 0
    fps_start_time = time.time()
    fps_display_interval = 2.0

    console.print(f"[cyan]动捕帧率: {motion_fps}[/cyan]")
    console.print(f"[cyan]总帧数: {len(lafan1_data_frames)}[/cyan]")
    console.print(f"[cyan]预计时长: {len(lafan1_data_frames)/motion_fps:.2f}秒[/cyan]")

    pbar = tqdm(total=len(lafan1_data_frames), desc="重定向处理中", unit="帧", ncols=100)

    i = 0
    processed_frames = 0
    process_start_time = time.time()

    try:
        while i < len(lafan1_data_frames):
            fps_counter += 1
            current_time = time.time()
            if current_time - fps_start_time >= fps_display_interval:
                actual_fps = fps_counter / (current_time - fps_start_time)
                elapsed = current_time - process_start_time
                progress_pct = (i / len(lafan1_data_frames)) * 100
                console.print(f"[yellow]渲染FPS: {actual_fps:.2f} | 进度: {progress_pct:.1f}% | 已用时: {elapsed:.1f}秒[/yellow]")
                fps_counter = 0
                fps_start_time = current_time

            smplx_data = lafan1_data_frames[i]

            qpos = retargeter.retarget(smplx_data)

            if args.save_path is not None and len(qpos_list) < len(lafan1_data_frames):
                qpos_list.append(qpos)
            processed_frames += 1
            pbar.update(1)

            rp, rr, dp = split_qpos(qpos, retargeter.model)
            robot_motion_viewer.step(
                root_pos=rp,
                root_rot=rr,
                dof_pos=dp,
                human_motion_data=retargeter.scaled_human_data,
                rate_limit=args.rate_limit,
                follow_camera=True,
            )

            i = (i + 1) % len(lafan1_data_frames) if args.loop else i + 1

    except KeyboardInterrupt:
        console.print("[yellow]\n用户中断，清理中...[/yellow]")
    finally:
        pbar.close()
        robot_motion_viewer.close()

        total_time = time.time() - process_start_time
        if processed_frames > 0:
            avg_time_per_frame = total_time / processed_frames
            console.print(f"[cyan]处理了 {processed_frames}/{len(lafan1_data_frames)} 帧[/cyan]")
            console.print(f"[cyan]总耗时: {total_time:.2f}秒, 平均: {avg_time_per_frame*1000:.2f}毫秒/帧[/cyan]")

    if args.save_path is not None and qpos_list:
        import pickle
        import joblib

        console.print(f"[cyan]开始保存数据...[/cyan]")
        save_start_time = time.time()

        if args.output_format == "gmr":
            root_pos, root_rot, dof_pos = split_qpos(qpos_list, retargeter.model)
            root_rot = root_rot[:, [1, 2, 3, 0]]

            first_frame_pos = root_pos[0].copy()
            root_pos[:, 0] -= first_frame_pos[0]
            root_pos[:, 1] -= first_frame_pos[1]

            device = "cpu"
            kinematics_model = KinematicsModel(retargeter.xml_file, device=device)
            num_frames = root_pos.shape[0]

            identity_root_pos = torch.zeros((num_frames, 3), device=device)
            identity_root_rot = torch.zeros((num_frames, 4), device=device)
            identity_root_rot[:, -1] = 1.0

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
        else:
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
