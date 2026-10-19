import argparse
import pathlib
import time
import platform
import multiprocessing
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.motion_data import split_qpos
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.kinematics_model import KinematicsModel
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from tqdm import tqdm
import os
import numpy as np
import pickle
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
    """Convert GMR qpos data to HOVER/Neural-WBC format."""
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
    console.print(Panel.fit("[bold cyan]FBX离线转机器人工具[/bold cyan]", border_style="cyan"))
    print_hardware_info()

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

    if args.output_format == "hover" and args.robot != "unitree_h1":
        parser.error("HOVER 输出使用 H1 的关节定义，仅支持 unitree_h1；其他机器人请使用 gmr。")

    if args.save_path is not None:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        qpos_list = []

    console.print(f"[cyan]加载OptiTrack FBX动作文件: {args.motion_file}[/cyan]")
    data_frames = load_optitrack_fbx_motion_file(args.motion_file)
    console.print(f"[green]已加载 {len(data_frames)} 帧[/green]")

    retargeter = GMR(
        src_human="fbx_offline",
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

                                            )

    fps_counter = 0
    fps_start_time = time.time()
    fps_display_interval = 2.0

    console.print(f"[cyan]动捕帧率: {motion_fps}[/cyan]")
    console.print(f"[cyan]总帧数: {len(data_frames)}[/cyan]")
    console.print(f"[cyan]预计时长: {len(data_frames)/motion_fps:.2f}秒[/cyan]")

    pbar = tqdm(total=len(data_frames), desc="重定向处理中", unit="帧", ncols=100)

    i = 0
    process_start_time = time.time()

    try:
        while i < len(data_frames):
            fps_counter += 1
            current_time = time.time()
            if current_time - fps_start_time >= fps_display_interval:
                actual_fps = fps_counter / (current_time - fps_start_time)
                elapsed = current_time - process_start_time
                progress_pct = (i / len(data_frames)) * 100
                console.print(f"[yellow]渲染FPS: {actual_fps:.2f} | 进度: {progress_pct:.1f}% | 已用时: {elapsed:.1f}秒[/yellow]")
                fps_counter = 0
                fps_start_time = current_time

            pbar.update(1)

            smplx_data = data_frames[i]

            qpos = retargeter.retarget(smplx_data)

            rp, rr, dp = split_qpos(qpos, retargeter.model)
            robot_motion_viewer.step(
                root_pos=rp,
                root_rot=rr,
                dof_pos=dp,
                human_motion_data=retargeter.scaled_human_data,
                rate_limit=args.rate_limit,

            )

            i += 1

            if args.save_path is not None:
                qpos_list.append(qpos)
    except KeyboardInterrupt:
        console.print("[yellow]\n用户中断，清理中...[/yellow]")
    finally:
        pbar.close()
        robot_motion_viewer.close()

        total_time = time.time() - process_start_time
        if i > 0:
            console.print(f"[cyan]处理了 {i}/{len(data_frames)} 帧, 总耗时: {total_time:.2f}秒[/cyan]")

    if args.save_path is not None and qpos_list:
        import joblib

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
            base_name = os.path.splitext(os.path.basename(args.motion_file))[0]
            motion_data = convert_to_hover_format(
                qpos_list, motion_fps, base_name,
                segment_length=args.segment_length,
                segment_overlap=args.segment_overlap,
                model=retargeter.model,
            )
            joblib.dump(motion_data, args.save_path)
            n_motions = len(motion_data)

        console.print(f"[bold green]已保存到 {args.save_path} (格式: {args.output_format}, 动作数: {n_motions})[/bold green]")
