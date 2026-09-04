#!/usr/bin/env python3
"""
将kungfu_athlete_for_g1_29dof的NPZ格式转换为TWIST2 G1 PKL格式

NPZ结构:
  - fps: int
  - qpos: (N, 36) = root_pos(3) + root_quat(4, wxyz) + dof(29)

PKL结构:
  - fps: float
  - root_pos: (N, 3)
  - root_rot: (N, 4) - xyzw格式
  - dof_pos: (N, 29)
  - local_body_pos: (N, bodies, 3)
  - link_body_list: body名称列表
"""

import argparse
import os
import pickle
import numpy as np
import torch
import platform
import multiprocessing
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import sys
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))
from general_motion_retargeting.kinematics_model import KinematicsModel

console = Console()

def print_hardware_info(n_workers=None):
    table = Table(title="硬件信息")
    table.add_column("项目", style="cyan")
    table.add_column("信息", style="green")
    
    table.add_row("系统", platform.system())
    table.add_row("处理器", platform.processor() or platform.machine())
    table.add_row("CPU核心数", str(multiprocessing.cpu_count()))
    if n_workers:
        table.add_row("使用Worker数", str(n_workers))
    
    if torch.cuda.is_available():
        table.add_row("CUDA可用", "是")
        table.add_row("GPU数量", str(torch.cuda.device_count()))
        for i in range(torch.cuda.device_count()):
            table.add_row(f"GPU {i}", torch.cuda.get_device_name(i))
    else:
        table.add_row("CUDA可用", "否")
    
    console.print(table)


G1_XML_PATH = Path(__file__).parent.parent / "assets/unitree_g1/g1_mocap_29dof.xml"


def convert_quat_wxyz_to_xyzw(quat_wxyz: np.ndarray) -> np.ndarray:
    """wxyz -> xyzw"""
    return quat_wxyz[..., [1, 2, 3, 0]]


def convert_quat_xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    """xyzw -> wxyz"""
    return quat_xyzw[..., [3, 0, 1, 2]]


def load_npz(npz_path: str) -> dict:
    """加载NPZ文件"""
    data = np.load(npz_path, allow_pickle=True)
    
    # 检查文件中的键
    keys = list(data.keys())
    
    # 处理fps
    if 'fps' in keys:
        fps_data = data['fps']
        # 处理三种情况：标量、单元素数组、多元素数组
        if hasattr(fps_data, 'shape') and fps_data.shape == ():
            # 标量
            fps = int(fps_data)
        elif hasattr(fps_data, '__len__') and len(fps_data) > 0:
            # 数组
            fps = int(fps_data[0])
        else:
            # 其他情况
            fps = int(fps_data)
    else:
        raise ValueError(f"NPZ文件缺少'fps'键。可用键: {keys}")
    
    # 处理两种格式
    if 'qpos' in keys:
        # 格式1: 包含qpos键 (root_pos + root_quat + dof_pos)
        qpos = data['qpos']
        root_pos = qpos[:, :3]
        root_quat_wxyz = qpos[:, 3:7]
        dof_pos = qpos[:, 7:]
    elif 'joint_pos' in keys:
        # 格式2: 包含joint_pos键 (需要从body_pos_w和body_quat_w提取root)
        if 'body_pos_w' not in keys or 'body_quat_w' not in keys:
            raise ValueError(f"NPZ文件格式不支持。可用键: {keys}")
        
        joint_pos = data['joint_pos']
        body_pos_w = data['body_pos_w']
        body_quat_w = data['body_quat_w']
        
        # 提取root信息 (假设第一个body是root/pelvis)
        root_pos = body_pos_w[:, 0, :]
        # body_quat_w是wxyz格式，需要转换
        root_quat_wxyz = body_quat_w[:, 0, :]
        
        # joint_pos的维度判断
        # 36维: 前7列是root信息(3 pos + 4 quat)，后29列是关节角度
        # 29维: 只有关节角度
        if joint_pos.shape[1] == 36:
            dof_pos = joint_pos[:, 7:]
        elif joint_pos.shape[1] == 29:
            dof_pos = joint_pos
        else:
            raise ValueError(f"joint_pos维度不支持: {joint_pos.shape}")
    else:
        raise ValueError(f"NPZ文件缺少'qpos'或'joint_pos'键。可用键: {keys}")
    
    # 检查四元数格式并转换
    # 检查是否为单位四元数的wxyz格式 (w分量接近±1)
    w_component_mean = np.abs(root_quat_wxyz[:, 0]).mean()
    last_component_mean = np.abs(root_quat_wxyz[:, -1]).mean()
    
    # 如果第一个分量的绝对值明显大于最后一个，说明是wxyz格式
    if w_component_mean > 0.7 and w_component_mean > last_component_mean:
        # wxyz格式，需要转换为xyzw
        root_rot = convert_quat_wxyz_to_xyzw(root_quat_wxyz)
    else:
        # 已经是xyzw格式
        root_rot = root_quat_wxyz
    
    # 归一化四元数
    root_rot = root_rot / np.linalg.norm(root_rot, axis=-1, keepdims=True)
    
    return {
        'fps': fps,
        'root_pos': root_pos,
        'root_rot': root_rot,
        'dof_pos': dof_pos,
    }


def compute_local_body_pos(root_pos: np.ndarray, root_rot: np.ndarray, 
                           dof_pos: np.ndarray, xml_path: str,
                           device: str = "cpu") -> tuple:
    """使用FK计算local_body_pos (相对于根的位置)"""
    kinematics_model = KinematicsModel(str(xml_path), device=device)
    
    num_frames = root_pos.shape[0]
    
    # 使用单位根位置和旋转计算局部body位置
    identity_root_pos = torch.zeros((num_frames, 3), device=device, dtype=torch.float)
    identity_root_rot = torch.zeros((num_frames, 4), device=device, dtype=torch.float)
    identity_root_rot[:, -1] = 1.0  # xyzw格式，w=1
    
    dof_tensor = torch.from_numpy(dof_pos).to(device=device, dtype=torch.float)
    
    local_body_pos, _ = kinematics_model.forward_kinematics(
        identity_root_pos,
        identity_root_rot,
        dof_tensor
    )
    
    return local_body_pos.detach().cpu().numpy(), kinematics_model.body_names


def convert_npz_to_pkl(npz_path: str, output_path: str,
                       target_fps: float = None,
                       device: str = "cpu") -> dict:
    """转换单个NPZ文件为PKL格式"""
    # 加载NPZ
    data = load_npz(npz_path)

    fps = data['fps']
    root_pos = data['root_pos']
    root_rot = data['root_rot']
    dof_pos = data['dof_pos']

    # 重采样到目标帧率
    if target_fps is not None and target_fps != fps:
        num_frames = root_pos.shape[0]
        duration = num_frames / fps
        new_num_frames = int(duration * target_fps)
        
        old_times = np.linspace(0, duration, num_frames)
        new_times = np.linspace(0, duration, new_num_frames)
        
        # 线性插值
        root_pos = np.array([np.interp(new_times, old_times, root_pos[:, i]) 
                            for i in range(3)]).T
        root_rot = np.array([np.interp(new_times, old_times, root_rot[:, i]) 
                            for i in range(4)]).T
        # 归一化四元数
        root_rot = root_rot / np.linalg.norm(root_rot, axis=-1, keepdims=True)
        dof_pos = np.array([np.interp(new_times, old_times, dof_pos[:, i]) 
                           for i in range(dof_pos.shape[1])]).T
        fps = target_fps
    
    # 计算local_body_pos
    local_body_pos, body_names = compute_local_body_pos(
        root_pos, root_rot, dof_pos, G1_XML_PATH, device
    )
    
    # 构建输出数据
    motion_data = {
        'fps': float(fps),
        'root_pos': root_pos.astype(np.float64),
        'root_rot': root_rot.astype(np.float64),
        'dof_pos': dof_pos.astype(np.float64),
        'local_body_pos': local_body_pos.astype(np.float32),
        'link_body_list': body_names,
    }
    
    # 保存
    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(motion_data, f)
    
    return motion_data


def _convert_worker(args):
    """并行处理worker"""
    npz_file, out_file, target_fps, device, override = args

    # 跳过已存在的文件
    if os.path.exists(out_file) and not override:
        return (str(npz_file), True, "skipped", 0)

    import time
    start_time = time.time()

    try:
        convert_npz_to_pkl(str(npz_file), str(out_file), target_fps, device)
        elapsed = time.time() - start_time
        return (str(npz_file), True, "ok", elapsed)
    except Exception as e:
        return (str(npz_file), False, str(e), 0)


def batch_convert(input_dir: str, output_dir: str,
                  target_fps: float = None,
                  n_workers: int = None,
                  device: str = "cpu",
                  override: bool = False):
    """批量转换目录下所有NPZ文件（并行）"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    npz_files = list(input_path.rglob("*.npz"))
    
    # 优化worker数：每个worker需要加载KinematicsModel
    if n_workers is None:
        cpu_count = multiprocessing.cpu_count()
        if cpu_count <= 4:
            n_workers = 2
        elif cpu_count <= 8:
            n_workers = 4
        elif cpu_count <= 16:
            n_workers = 8
        else:
            n_workers = min(24, cpu_count // 2)
    
    console.print(f"[cyan]找到 {len(npz_files)} 个NPZ文件[/cyan]")
    console.print(f"[yellow]提示: 每个worker需要加载运动学模型，启动可能需要一些时间[/yellow]")
    
    # 准备任务
    tasks = []
    for npz_file in npz_files:
        rel_path = npz_file.relative_to(input_path)
        out_file = output_path / rel_path.with_suffix('.pkl')
        os.makedirs(out_file.parent, exist_ok=True)
        tasks.append((npz_file, out_file, target_fps, device, override))
    
    # 检查有多少文件需要处理
    files_to_process = sum(1 for _, out_file, *_ in tasks 
                          if override or not os.path.exists(out_file))
    if files_to_process == 0:
        console.print(f"[green]所有文件已存在，无需处理。使用 --override 强制重新处理[/green]")
        return
    
    console.print(f"[cyan]需要处理 {files_to_process} 个文件，跳过 {len(tasks) - files_to_process} 个已存在文件[/cyan]")
    
    # 并行处理
    import time
    start_time = time.time()
    success, failed, skipped = 0, 0, 0
    total_process_time = 0.0
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for result in tqdm(executor.map(_convert_worker, tasks), 
                          total=len(tasks), desc="转换中", unit="文件", ncols=100):
            path, ok, msg, elapsed = result
            if not ok:
                failed += 1
                console.print(f"[red]转换失败: {os.path.basename(path)}, 错误: {msg}[/red]")
            elif msg == "skipped":
                skipped += 1
            else:
                success += 1
                total_process_time += elapsed
                # 每10个文件显示一次平均速度
                if success % 10 == 0:
                    avg_time = total_process_time / success
                    console.print(f"[cyan]已处理 {success} 个文件，平均耗时: {avg_time:.2f}秒/文件[/cyan]")
    
    total_time = time.time() - start_time
    
    console.print(f"[bold green]完成! 成功: {success}, 跳过: {skipped}, 失败: {failed}[/bold green]")
    console.print(f"[bold green]总耗时: {total_time:.2f}秒[/bold green]")
    if success > 0:
        avg_time = total_process_time / success
        console.print(f"[bold green]平均处理时间: {avg_time:.2f}秒/文件[/bold green]")
    console.print(f"[bold green]输出目录: {output_dir}[/bold green]")


def main():
    console.print(Panel.fit("[bold cyan]NPZ转G1 PKL格式工具[/bold cyan]", border_style="cyan"))
    
    parser = argparse.ArgumentParser(description="将NPZ转换为G1 PKL格式")
    parser.add_argument("--input", "-i", required=True,
                        help="输入NPZ文件或目录")
    parser.add_argument("--output", "-o", required=True,
                        help="输出PKL文件或目录")
    parser.add_argument("--target_fps", type=float, default=None,
                        help="目标帧率(默认保持原始帧率)")
    parser.add_argument("--device", default="cpu",
                        help="计算设备(cpu/cuda)")
    parser.add_argument("--batch", action="store_true",
                        help="批量转换模式")
    parser.add_argument("--n_workers", type=int, default=None,
                        help="并行工作进程数(默认自动检测)")
    parser.add_argument("--override", action="store_true", default=False,
                        help="覆盖已存在的文件")
    
    args = parser.parse_args()
    
    # 优化worker数
    if args.n_workers:
        n_workers = args.n_workers
    else:
        cpu_count = multiprocessing.cpu_count()
        if cpu_count <= 4:
            n_workers = 2
        elif cpu_count <= 8:
            n_workers = 4
        elif cpu_count <= 16:
            n_workers = 8
        else:
            n_workers = min(24, cpu_count // 2)
    
    print_hardware_info(n_workers)
    
    if args.batch or os.path.isdir(args.input):
        batch_convert(args.input, args.output, args.target_fps,
                     args.n_workers, args.device, args.override)
    else:
        convert_npz_to_pkl(args.input, args.output, args.target_fps,
                          args.device)
        console.print(f"[bold green]完成! 输出: {args.output}[/bold green]")


if __name__ == "__main__":
    main()
