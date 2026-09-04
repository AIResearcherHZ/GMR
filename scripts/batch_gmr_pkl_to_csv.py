import argparse
import pickle
import os
import sys
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import platform

import numpy as np
import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    """处理单个文件的worker函数"""
    file_path, out_path, override = args_tuple

    if os.path.exists(out_path) and not override:
        return (file_path, True, "skipped", 0)

    import time
    start_time = time.time()

    try:
        with open(file_path, "rb") as f:
            motion_data = pickle.load(f)

        dof_pos = motion_data["dof_pos"]
        frame_rate = motion_data["fps"]
        motion = np.zeros((dof_pos.shape[0], dof_pos.shape[1] + 7), dtype=np.float32)
        motion[:, :3] = motion_data["root_pos"]
        motion[:, 3:7] = motion_data["root_rot"]
        motion[:, 7:] = dof_pos
        
        if frame_rate > 30:
            downsample_factor = frame_rate / 30.0
            indices = np.arange(0, motion.shape[0], downsample_factor).astype(int)
            motion = motion[indices]
        
        np.savetxt(out_path, motion, delimiter=",")
        elapsed = time.time() - start_time
        return (file_path, True, "ok", elapsed)
    except Exception as e:
        return (file_path, False, str(e), 0)


if __name__ == "__main__":
    console.print(Panel.fit("[bold cyan]GMR PKL转CSV批处理工具[/bold cyan]", border_style="cyan"))
    
    parser = argparse.ArgumentParser(description="Convert GMR pickle files to CSV (for beyondmimic)")
    parser.add_argument(
        "--folder", type=str, help="Path to the folder containing pickle files from GMR",
    )
    parser.add_argument("--n_workers", type=int, default=None,
                        help="Number of parallel workers (default: auto-detect CPU cores-1)")
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
        else:
            n_workers = min(8, cpu_count // 2)
    print_hardware_info(n_workers)

    out_folder = os.path.join(args.folder, "csv")
    os.makedirs(out_folder, exist_ok=True)
    
    tasks = []
    for file in os.listdir(args.folder):
        if file.endswith(".pkl"):
            file_path = os.path.join(args.folder, file)
            out_path = os.path.join(out_folder, file.replace(".pkl", ".csv"))
            tasks.append((file_path, out_path, args.override))
    
    console.print(f"[cyan]找到 {len(tasks)} 个pkl文件待转换[/cyan]")
    
    # 检查有多少文件需要处理
    files_to_process = sum(1 for _, out_path, *_ in tasks 
                          if args.override or not os.path.exists(out_path))
    console.print(f"[cyan]需要处理 {files_to_process} 个文件，跳过 {len(tasks) - files_to_process} 个已存在文件[/cyan]")
    
    from tqdm import tqdm
    import time
    start_time = time.time()
    success, failed, skipped = 0, 0, 0
    total_process_time = 0.0
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for result in tqdm(executor.map(process_file, tasks), 
                          total=len(tasks), desc="转换中", unit="文件", ncols=100):
            path, ok, msg, elapsed = result
            if not ok:
                failed += 1
                console.print(f"[red]失败: {os.path.basename(path)}, 错误: {msg}[/red]")
            elif msg == "skipped":
                skipped += 1
            else:
                success += 1
                total_process_time += elapsed
    
    total_time = time.time() - start_time
    console.print(f"[bold green]完成! 成功: {success}, 跳过: {skipped}, 失败: {failed}[/bold green]")
    console.print(f"[bold green]总耗时: {total_time:.2f}秒[/bold green]")
    if success > 0:
        console.print(f"[bold green]平均处理时间: {total_process_time/success:.2f}秒/文件[/bold green]")
    console.print(f"[bold green]输出目录: {out_folder}[/bold green]")
