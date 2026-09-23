"""
Batch Robot to Robot Motion Retargeting Script

Usage:
    python scripts/robot_to_robot_dataset.py \
        --src_folder data/TWIST2_dataset/AMASS_g1_GMR8 \
        --tgt_folder data/TWIST2_dataset/AMASS_taks_t1 \
        --src_robot unitree_g1 \
        --tgt_robot taks_t1
"""

import argparse
import os
import platform
import multiprocessing
import torch
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

from robot_to_robot import load_robot_motion, save_robot_motion, convert_motion

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
    """处理单个文件的转换"""
    pkl_file, src_robot, tgt_robot, tgt_folder, device, override = args_tuple

    tgt_file = Path(tgt_folder) / Path(pkl_file).name
    if tgt_file.exists() and not override:
        return (str(pkl_file), True, "skipped", 0)

    import time
    start_time = time.time()

    try:
        src_data = load_robot_motion(str(pkl_file))

        tgt_data = convert_motion(src_data, src_robot, tgt_robot, device=device)
        
        save_robot_motion(tgt_data, str(tgt_file))
        elapsed = time.time() - start_time
        return (str(pkl_file), True, "ok", elapsed)
    except Exception as e:
        return (str(pkl_file), False, str(e), 0)


def main():
    console.print(Panel.fit("[bold cyan]批量机器人到机器人运动转换工具[/bold cyan]", border_style="cyan"))
    
    parser = argparse.ArgumentParser(
        description='Batch convert robot motion between different robot formats'
    )
    parser.add_argument('--src_folder', type=str, required=True,
                        help='Path to source folder containing pkl files')
    parser.add_argument('--tgt_folder', type=str, required=True,
                        help='Path to target folder to save converted pkl files')
    parser.add_argument('--src_robot', type=str, required=True,
                        choices=['unitree_g1', 'taks_t1'],
                        help='Source robot type')
    parser.add_argument('--tgt_robot', type=str, required=True,
                        choices=['unitree_g1', 'taks_t1'],
                        help='Target robot type')
    parser.add_argument('--n_workers', type=int, default=None,
                        help='Number of parallel workers (default: auto-detect CPU cores-1)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use for computation (default: cpu, can be cuda:0)')
    parser.add_argument('--override', action='store_true', default=False,
                        help='覆盖已存在的文件')
    
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
    
    if args.src_robot == args.tgt_robot:
        console.print("[yellow]警告: 源机器人和目标机器人相同。[/yellow]")
        return
    
    src_folder = Path(args.src_folder)
    tgt_folder = Path(args.tgt_folder)
    
    pkl_files = list(src_folder.glob('*.pkl'))
    
    if not pkl_files:
        console.print(f"[red]在 {src_folder} 中未找到pkl文件[/red]")
        return
    
    console.print(f"[bold cyan]转换 {len(pkl_files)} 个文件[/bold cyan]")
    console.print(f"  从: {args.src_robot} -> {args.tgt_robot}")
    console.print(f"  源目录: {src_folder}")
    console.print(f"  目标目录: {tgt_folder}")
    
    # Create target folder
    tgt_folder.mkdir(parents=True, exist_ok=True)
    
    # Convert each file
    success_count = 0
    fail_count = 0
    
    # 准备参数元组列表
    args_list = [
        (pkl_file, args.src_robot, args.tgt_robot, tgt_folder, args.device, args.override)
        for pkl_file in pkl_files
    ]
    
    # 检查有多少文件需要处理
    files_to_process = sum(1 for pf in pkl_files 
                          if args.override or not (tgt_folder / pf.name).exists())
    console.print(f"[cyan]需要处理 {files_to_process} 个文件，跳过 {len(pkl_files) - files_to_process} 个已存在文件[/cyan]")
    
    import time
    start_time = time.time()
    skip_count = 0
    total_process_time = 0.0
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for result in tqdm(executor.map(process_file, args_list), 
                          total=len(pkl_files), desc="转换中", unit="文件", ncols=100):
            path, ok, msg, elapsed = result
            if not ok:
                console.print(f"[red]转换失败 {os.path.basename(path)}: {msg}[/red]")
                fail_count += 1
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


if __name__ == '__main__':
    main()