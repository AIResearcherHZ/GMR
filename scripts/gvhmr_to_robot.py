import argparse
import pathlib
import os
import time
import platform
import multiprocessing

import numpy as np
from scipy.spatial.transform import Rotation as sRot
from scipy.interpolate import CubicSpline
import mujoco as mj

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import ROBOT_XML_DICT
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.utils.smpl import load_gvhmr_pred_file, get_gvhmr_data_offline_fast
from general_motion_retargeting.kinematics_model import KinematicsModel
import torch

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

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


def fit_parabola_and_sample(y0, y1, N, g=9.81):
    T = np.sqrt(2.0 * (y0 - y1) / g)
    t_all = np.linspace(0.0, T, N + 2)
    y_all = y0 - 0.5 * g * t_all**2
    t_mid = t_all[1:-1]
    y_mid = y_all[1:-1]
    return t_all, y_all, t_mid, y_mid

def find_local_maxima_indices(z, resolution_factor=100):
    if len(z) < 3:
        raise ValueError("Input array must have at least 3 elements")
    
    t = len(z)
    x_original = np.arange(t)
    cs = CubicSpline(x_original, z)
    
    def cs_derivative(x):
        return cs(x, 1)
    
    def cs_second_derivative(x):
        return cs(x, 2)
    
    potential_maxima = []
    for i in range(t - 1):
        x_fine = np.linspace(i, i + 1, resolution_factor)
        derivative_values = cs_derivative(x_fine)
        
        for j in range(len(x_fine) - 1):
            if derivative_values[j] * derivative_values[j + 1] <= 0:
                if derivative_values[j] == 0:
                    candidate = x_fine[j]
                elif derivative_values[j + 1] == 0:
                    candidate = x_fine[j + 1]
                else:
                    left = x_fine[j]
                    right = x_fine[j + 1]
                    for _ in range(10):
                        mid = (left + right) / 2
                        if cs_derivative(left) * cs_derivative(mid) <= 0:
                            right = mid
                        else:
                            left = mid
                    candidate = (left + right) / 2
                
                if cs_second_derivative(candidate) < 0:
                    potential_maxima.append(candidate)
    
    if potential_maxima:
        potential_maxima = np.array(potential_maxima)
        unique_maxima = []
        threshold = 0.5
        
        sorted_indices = np.argsort(potential_maxima)
        for idx in sorted_indices:
            x = potential_maxima[idx]
            if not unique_maxima or abs(x - unique_maxima[-1]) > threshold:
                unique_maxima.append(x)
        
        maxima_indices = []
        for x_max in unique_maxima:
            nearest_idx = np.argmin(np.abs(x_original - x_max))
            if nearest_idx == 0:
                if z[nearest_idx] > z[nearest_idx + 1]:
                    maxima_indices.append(nearest_idx)
            elif nearest_idx == t - 1:
                if z[nearest_idx] > z[nearest_idx - 1]:
                    maxima_indices.append(nearest_idx)
            else:
                if (z[nearest_idx] > z[nearest_idx - 1] and 
                    z[nearest_idx] > z[nearest_idx + 1]):
                    maxima_indices.append(nearest_idx)
        
        maxima_indices = sorted(set(maxima_indices))
        return maxima_indices
    else:
        return []

def find_local_minima_indices(z, resolution_factor=100):
    if len(z) < 3:
        raise ValueError("Input array must have at least 3 elements")
    
    t = len(z)
    x_original = np.arange(t)
    cs = CubicSpline(x_original, z)
    
    def cs_derivative(x):
        return cs(x, 1)
    
    def cs_second_derivative(x):
        return cs(x, 2)
    
    potential_minima = []
    for i in range(t - 1):
        x_fine = np.linspace(i, i + 1, resolution_factor)
        derivative_values = cs_derivative(x_fine)
        
        for j in range(len(x_fine) - 1):
            if derivative_values[j] * derivative_values[j + 1] <= 0:
                if derivative_values[j] == 0:
                    candidate = x_fine[j]
                elif derivative_values[j + 1] == 0:
                    candidate = x_fine[j + 1]
                else:
                    left = x_fine[j]
                    right = x_fine[j + 1]
                    for _ in range(10):
                        mid = (left + right) / 2
                        if cs_derivative(left) * cs_derivative(mid) <= 0:
                            right = mid
                        else:
                            left = mid
                    candidate = (left + right) / 2
                
                if cs_second_derivative(candidate) > 0:
                    potential_minima.append(candidate)
    
    if potential_minima:
        potential_minima = np.array(potential_minima)
        unique_minima = []
        threshold = 0.5
        
        sorted_indices = np.argsort(potential_minima)
        for idx in sorted_indices:
            x = potential_minima[idx]
            if not unique_minima or abs(x - unique_minima[-1]) > threshold:
                unique_minima.append(x)
        
        minima_indices = []
        for x_min in unique_minima:
            nearest_idx = np.argmin(np.abs(x_original - x_min))
            if nearest_idx == 0:
                if z[nearest_idx] < z[nearest_idx + 1]:
                    minima_indices.append(nearest_idx)
            elif nearest_idx == t - 1:
                if z[nearest_idx] < z[nearest_idx - 1]:
                    minima_indices.append(nearest_idx)
            else:
                if (z[nearest_idx] < z[nearest_idx - 1] and 
                    z[nearest_idx] < z[nearest_idx + 1]):
                    minima_indices.append(nearest_idx)
        
        minima_indices = sorted(set(minima_indices))
        return minima_indices
    else:
        return []

def get_min_body_z_from_qpos(qpos, mj_model, robot_data):
    robot_data.qpos[:] = qpos
    mj.mj_forward(mj_model, robot_data)

    z_vals = robot_data.xpos[:, 2]
    robot_z_vals = z_vals[1:]

    min_idx = int(np.argmin(robot_z_vals))
    min_z = float(robot_z_vals[min_idx])

    return min_z

def adjust_root_z(qpos_list, mj_model, robot_data):
    qpos = np.array(qpos_list)
    T, N = qpos.shape

    root_z = qpos[:, 2]
    root_vel = np.zeros_like(root_z)
    root_vel[:-1] = root_z[1:] - root_z[:-1]

    rebuild_root_z = np.zeros_like(root_z)
    minidxes = find_local_minima_indices(qpos[:,2])
    maxidxes = find_local_maxima_indices(qpos[:,2])
    rebuild_root_z[0] = root_z[0] - get_min_body_z_from_qpos(qpos[0], mj_model, robot_data)
    
    skip = []
    if len(minidxes) >= 3:
        skip = [minidxes[i] for i in [-3]]

    for t in range(1,T):
        if t in minidxes and t not in skip:
            new_z = root_z[t] - get_min_body_z_from_qpos(qpos[t], mj_model, robot_data)
            new_qpos_t = qpos[t].copy()
            new_qpos_t[2] = new_z
            if get_min_body_z_from_qpos(new_qpos_t, mj_model, robot_data) >= 0:
                rebuild_root_z[t] = new_z
                continue
        rebuild_root_z[t] = rebuild_root_z[t-1] + root_vel[t-1]

    for i in maxidxes:
        for j in minidxes:
            if j in skip:
                break
            if j > i and rebuild_root_z[i] > rebuild_root_z[j]:
                rebuild_root_z[i+1:j] = fit_parabola_and_sample(rebuild_root_z[i],rebuild_root_z[j], j - i - 1)[-1]
                break
    
    qpos[:,2] = rebuild_root_z

    for t in range(T):
        minz = get_min_body_z_from_qpos(qpos[t], mj_model, robot_data)
        if minz < 0:
            rebuild_root_z[t] = root_z[t] - get_min_body_z_from_qpos(qpos[t], mj_model, robot_data)

    qpos[:,2] = rebuild_root_z
    return qpos


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
    console.print(Panel.fit("[bold cyan]GVHMR转机器人工具[/bold cyan]", border_style="cyan"))
    print_hardware_info()
    
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
    
    console.print(f"[cyan]总帧数: {len(smplx_data_frames)}[/cyan]")
    console.print(f"[cyan]预计时长: {len(smplx_data_frames)/aligned_fps:.2f}秒[/cyan]")
    
    if args.save_path is not None:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:  # Only create directory if it's not empty
            os.makedirs(save_dir, exist_ok=True)
        qpos_list = []
        
        robot_xml_path = ROBOT_XML_DICT[args.robot]
        mj_model = mj.MjModel.from_xml_path(str(robot_xml_path))
        robot_data = mj.MjData(mj_model)
    
    # Start the viewer
    i = 0
    process_start_time = time.time()
    from tqdm import tqdm
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
        console.print("[yellow]\n用户中断，清理中...[/yellow]")
    finally:
        pbar.close()
        total_time = time.time() - process_start_time
        if i > 0:
            console.print(f"[cyan]处理了 {i}/{len(smplx_data_frames)} 帧, 总耗时: {total_time:.2f}秒[/cyan]")
            
    if args.save_path is not None:
        import pickle
        import joblib
        
        qpos_array = np.array(qpos_list)
        console.print(f"[cyan]应用极值点修正前: root_z范围 [{qpos_array[:, 2].min():.3f}, {qpos_array[:, 2].max():.3f}][/cyan]")
        qpos_array = adjust_root_z(qpos_array, mj_model, robot_data)
        console.print(f"[green]应用极值点修正后: root_z范围 [{qpos_array[:, 2].min():.3f}, {qpos_array[:, 2].max():.3f}][/green]")

        qpos_list = [qpos_array[i] for i in range(len(qpos_array))]
        
        if args.output_format == "gmr":
            root_pos = np.array([qpos[:3] for qpos in qpos_list])
            root_rot = np.array([qpos[3:7][[1, 2, 3, 0]] for qpos in qpos_list])
            dof_pos = np.array([qpos[7:] for qpos in qpos_list])
            
            # Normalize root_pos to start from origin (keep only Z offset for height)
            first_frame_pos = root_pos[0].copy()
            root_pos[:, 0] -= first_frame_pos[0]  # X offset to 0
            root_pos[:, 1] -= first_frame_pos[1]  # Y offset to 0
            # Keep Z as is for height
            
            # Compute local_body_pos using forward kinematics
            device = "cpu"
            kinematics_model = KinematicsModel(retarget.xml_file, device=device)
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
                "fps": aligned_fps,
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
            base_name = os.path.splitext(os.path.basename(args.gvhmr_pred_file))[0]
            motion_data = convert_to_hover_format(
                qpos_list, aligned_fps, base_name,
                segment_length=args.segment_length,
                segment_overlap=args.segment_overlap
            )
            joblib.dump(motion_data, args.save_path)
            n_motions = len(motion_data)
        
        console.print(f"[bold green]已保存到 {args.save_path} (格式: {args.output_format}, 动作数: {n_motions})[/bold green]")

            
      
    
    robot_motion_viewer.close()
