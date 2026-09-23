"""
GVHMR to Robot Motion Retargeting Dataset Script

This script batch processes GVHMR prediction files and converts them to robot motion format.
It walks through a source folder containing .pt files and saves retargeted motions to a target folder.

Usage:
    python scripts/gvhmr_to_robot_dataset.py \
        --src_folder /path/to/gvhmr_outputs \
        --tgt_folder /path/to/output \
        --robot taks_t1
"""

import argparse
import pathlib
import os
import platform
import multiprocessing
import numpy as np
from tqdm import tqdm
import torch
import pickle
from scipy.interpolate import CubicSpline
import mujoco as mj

from concurrent.futures import ProcessPoolExecutor

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import ROBOT_XML_DICT
from general_motion_retargeting.kinematics_model import KinematicsModel
from general_motion_retargeting.utils.smpl import load_gvhmr_pred_file, get_gvhmr_data_offline_fast
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


def process_file(args_tuple):
    """处理单个pt文件的worker函数"""
    pt_file_path, tgt_file_path, robot, SMPLX_FOLDER, target_fps, override = args_tuple

    if os.path.exists(tgt_file_path) and not override:
        return (pt_file_path, True, "skipped", 0)

    import time
    start_time = time.time()

    try:
        smplx_data, body_model, smplx_output, actual_human_height = load_gvhmr_pred_file(
            pt_file_path, str(SMPLX_FOLDER)
        )
        smplx_data_frames, aligned_fps = get_gvhmr_data_offline_fast(
            smplx_data, body_model, smplx_output, tgt_fps=target_fps
        )
    except Exception as e:
        return (pt_file_path, False, str(e), 0)
    
    retarget = GMR(
        actual_human_height=actual_human_height,
        src_human="smplx",
        tgt_robot=robot,
        verbose=False,
    )
    
    qpos_list = []
    for frame_data in smplx_data_frames:
        qpos = retarget.retarget(frame_data)
        qpos_list.append(qpos.copy())
    qpos_list = np.array(qpos_list)
    
    # 加载mujoco模型做root_z修正
    robot_xml_path = ROBOT_XML_DICT[robot]
    mj_model = mj.MjModel.from_xml_path(str(robot_xml_path))
    robot_data = mj.MjData(mj_model)
    qpos_list = adjust_root_z(qpos_list, mj_model, robot_data)

    root_pos = qpos_list[:, :3]
    root_rot = qpos_list[:, 3:7]
    root_rot[:, [0, 1, 2, 3]] = root_rot[:, [1, 2, 3, 0]]
    dof_pos = qpos_list[:, 7:]
    num_frames = root_pos.shape[0]
    
    first_frame_pos = root_pos[0].copy()
    root_pos[:, 0] -= first_frame_pos[0]
    root_pos[:, 1] -= first_frame_pos[1]
    
    device = "cpu"
    kinematics_model = KinematicsModel(retarget.xml_file, device=device)
    identity_root_pos = torch.zeros((num_frames, 3), device=device)
    identity_root_rot = torch.zeros((num_frames, 4), device=device)
    identity_root_rot[:, -1] = 1.0
    local_body_pos, _ = kinematics_model.forward_kinematics(
        identity_root_pos, identity_root_rot,
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
    
    os.makedirs(os.path.dirname(tgt_file_path), exist_ok=True)
    with open(tgt_file_path, "wb") as f:
        pickle.dump(motion_data, f)
    
    elapsed = time.time() - start_time
    return (pt_file_path, True, "ok", elapsed)


if __name__ == "__main__":
    console.print(Panel.fit("[bold cyan]GVHMR转机器人数据集工具[/bold cyan]", border_style="cyan"))
    print_hardware_info()
    
    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_folder",
        help="Folder containing GVHMR prediction files (.pt) to load.",
        required=True,
        type=str,
    )
    
    parser.add_argument(
        "--tgt_folder",
        help="Folder to save the retargeted motion files.",
        default="../../motion_data/GVHMR_g1_gmr"
    )
    
    parser.add_argument(
        "--robot",
        choices=["unitree_g1", "unitree_g1_with_hands", "unitree_h1", "unitree_h1_2",
                 "booster_t1", "booster_t1_29dof", "stanford_toddy", "fourier_n1", 
                 "engineai_pm01", "kuavo_s45", "hightorque_hi", "galaxea_r1pro", 
                 "berkeley_humanoid_lite", "booster_k1", "pnd_adam_lite", "openloong", 
                 "tienkung", "taks_t1"],
        default="unitree_g1",
    )
    
    parser.add_argument(
        "--override",
        default=False,
        action="store_true",
        help="Override existing files.",
    )
    
    parser.add_argument(
        "--target_fps",
        default=30,
        type=int,
        help="Target FPS for the output motion.",
    )
    
    args = parser.parse_args()
    
    src_folder = args.src_folder
    tgt_folder = args.tgt_folder
    
    # 每个worker加载大型模型，保守设置worker数防止内存耗尽
    total_cpus = multiprocessing.cpu_count()
    n_workers = max(1, min(total_cpus // 4, 8))
    print_hardware_info()
    console.print(f"[cyan]使用Worker数: {n_workers}[/cyan]")
    
    SMPLX_FOLDER = HERE / ".." / "assets" / "body_models"
    
    # Collect all .pt files
    pt_files = []
    for dirpath, _, filenames in os.walk(src_folder):
        for filename in sorted(filenames):
            if filename.endswith(".pt"):
                pt_files.append(os.path.join(dirpath, filename))
    
    console.print(f"[cyan]找到 {len(pt_files)} 个.pt文件待处理[/cyan]")
    
    # 构建任务列表
    task_list = []
    for pt_file_path in pt_files:
        rel_path = os.path.relpath(pt_file_path, src_folder)
        tgt_file_path = os.path.join(tgt_folder, rel_path).replace(".pt", ".pkl")
        task_list.append((
            pt_file_path, tgt_file_path, args.robot, SMPLX_FOLDER,
            args.target_fps, args.override
        ))
    
    import time
    start_time = time.time()
    success_count, fail_count, skip_count = 0, 0, 0
    total_process_time = 0.0
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for result in tqdm(executor.map(process_file, task_list),
                           total=len(task_list), desc="重定向中", unit="文件", ncols=100):
            path, ok, msg, elapsed = result
            if not ok:
                fail_count += 1
                console.print(f"[red]失败 {os.path.basename(path)}: {msg}[/red]")
            elif msg == "skipped":
                skip_count += 1
            else:
                success_count += 1
                total_process_time += elapsed
                if success_count % 10 == 0:
                    avg_time = total_process_time / success_count
                    console.print(f"[cyan]已处理 {success_count} 个文件，平均耗时: {avg_time:.2f}秒/文件[/cyan]")
    
    total_time = time.time() - start_time
    console.print(f"[bold green]完成! 成功: {success_count}, 跳过: {skip_count}, 失败: {fail_count}[/bold green]")
    console.print(f"[bold green]总耗时: {total_time:.2f}秒[/bold green]")
    if success_count > 0:
        console.print(f"[bold green]平均处理时间: {total_process_time/success_count:.2f}秒/文件[/bold green]")
    console.print(f"[bold green]已保存到 {tgt_folder}[/bold green]")
