import argparse
import pathlib
import os
import numpy as np
import platform
import multiprocessing
from tqdm import tqdm
import torch
import pickle
from concurrent.futures import ProcessPoolExecutor

from general_motion_retargeting.utils.lafan1 import load_bvh_file
from general_motion_retargeting.kinematics_model import KinematicsModel
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

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


def process_file(args_tuple):
    """处理单个bvh文件的worker函数"""
    bvh_file_path, tgt_file_path, robot, override = args_tuple
    
    # 跳过已存在的文件
    if os.path.exists(tgt_file_path) and not override:
        return (bvh_file_path, True, "skipped", 0)
    
    import time
    start_time = time.time()
    
    try:
        lafan1_data_frames, actual_human_height = load_bvh_file(bvh_file_path, format="lafan1")
        src_fps = 30
    except Exception as e:
        return (bvh_file_path, False, str(e), 0)
    
    retarget = GMR(
        src_human="bvh_lafan1",
        tgt_robot=robot,
        actual_human_height=actual_human_height,
        verbose=False,
    )
    
    qpos_list = []
    for frame_data in lafan1_data_frames:
        qpos = retarget.retarget(frame_data)
        qpos_list.append(qpos.copy())
    qpos_list = np.array(qpos_list)

    device = "cpu"
    kinematics_model = KinematicsModel(retarget.xml_file, device=device)
    
    root_pos = qpos_list[:, :3]
    root_rot = qpos_list[:, 3:7]
    root_rot[:, [0, 1, 2, 3]] = root_rot[:, [1, 2, 3, 0]]
    dof_pos = qpos_list[:, 7:]
    num_frames = root_pos.shape[0]
    
    first_frame_pos = root_pos[0].copy()
    root_pos[:, 0] -= first_frame_pos[0]
    root_pos[:, 1] -= first_frame_pos[1]
    
    identity_root_pos = torch.zeros((num_frames, 3), device=device)
    identity_root_rot = torch.zeros((num_frames, 4), device=device)
    identity_root_rot[:, -1] = 1.0
    local_body_pos, _ = kinematics_model.forward_kinematics(
        identity_root_pos, identity_root_rot,
        torch.from_numpy(dof_pos).to(device=device, dtype=torch.float)
    )
    body_names = kinematics_model.body_names
    
    motion_data = {
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": dof_pos,
        "local_body_pos": local_body_pos.detach().cpu().numpy(),
        "fps": src_fps,
        "link_body_list": body_names,
    }
    
    os.makedirs(os.path.dirname(tgt_file_path), exist_ok=True)
    with open(tgt_file_path, "wb") as f:
        pickle.dump(motion_data, f)
    
    elapsed = time.time() - start_time
    return (bvh_file_path, True, "ok", elapsed)


if __name__ == "__main__":
    console.print(Panel.fit("[bold cyan]BVH转机器人数据集工具[/bold cyan]", border_style="cyan"))
    
    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_folder",
        help="Folder containing BVH motion files to load.",
        required=True,
        type=str,
    )
    
    parser.add_argument(
        "--tgt_folder",
        help="Folder to save the retargeted motion files.",
        default="../../motion_data/LAFAN1_g1_gmr"
    )
    
    parser.add_argument(
        "--robot",
        default="unitree_g1",
    )
    
    parser.add_argument(
        "--override",
        default=False,
        action="store_true",
    )
    
    parser.add_argument(
        "--target_fps",
        default=30,
        type=int,
    )
    
    parser.add_argument(
        "--n_workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: auto-detect CPU cores-1).",
    )

    args = parser.parse_args()
    
    # 优化worker数：每个worker需要加载模型，使用适中的并行度
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
    console.print(f"[yellow]提示: 每个worker需要加载GMR模型，启动可能需要一些时间[/yellow]")
    
    src_folder = args.src_folder
    tgt_folder = args.tgt_folder

    # 收集所有bvh文件
    task_list = []
    for dirpath, _, filenames in os.walk(src_folder):
        for filename in sorted(filenames):
            if not filename.endswith(".bvh"):
                continue
            bvh_file_path = os.path.join(dirpath, filename)
            tgt_file_path = bvh_file_path.replace(src_folder, tgt_folder).replace(".bvh", ".pkl")
            task_list.append((bvh_file_path, tgt_file_path, args.robot, args.override))
    
    console.print(f"[cyan]找到 {len(task_list)} 个BVH文件[/cyan]")
    
    # 检查有多少文件需要处理
    files_to_process = sum(1 for _, tgt, _, override, *_ in task_list 
                          if override or not os.path.exists(tgt))
    if files_to_process == 0:
        console.print(f"[green]所有文件已存在，无需处理。使用 --override 强制重新处理[/green]")
        import sys
        sys.exit(0)
    
    console.print(f"[cyan]需要处理 {files_to_process} 个文件，跳过 {len(task_list) - files_to_process} 个已存在文件[/cyan]")
    
    import time
    start_time = time.time()
    success_count, fail_count, skip_count = 0, 0, 0
    total_process_time = 0.0
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for result in tqdm(executor.map(process_file, task_list),
                           total=len(task_list), desc="重定向中", 
                           unit="文件", ncols=100):
            path, ok, msg, elapsed = result
            if not ok:
                fail_count += 1
                console.print(f"[red]失败 {os.path.basename(path)}: {msg}[/red]")
            elif msg == "skipped":
                skip_count += 1
            else:
                success_count += 1
                total_process_time += elapsed
                # 每10个文件显示一次平均速度
                if success_count % 10 == 0:
                    avg_time = total_process_time / success_count
                    console.print(f"[cyan]已处理 {success_count} 个文件，平均耗时: {avg_time:.2f}秒/文件[/cyan]")
    
    total_time = time.time() - start_time
    
    console.print(f"[bold green]完成! 成功: {success_count}, 跳过: {skip_count}, 失败: {fail_count}[/bold green]")
    console.print(f"[bold green]总耗时: {total_time:.2f}秒[/bold green]")
    if success_count > 0:
        avg_time = total_process_time / success_count
        console.print(f"[bold green]平均处理时间: {avg_time:.2f}秒/文件[/bold green]")
    console.print(f"[bold green]已保存到 {tgt_folder}[/bold green]")
