import os
import argparse
import numpy as np
import platform
import multiprocessing
import torch
from tqdm import tqdm
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

def convert_smpl_to_smplx(input_path, output_path, gender='neutral'):
    # Load SMPL data
    smpl_data = np.load(input_path, allow_pickle=True)
    data_dict = dict(smpl_data)  # Convert to dict for modification

    # Handle betas padding for SMPL-X (pad from 10 to 16 if necessary)
    if 'betas' in data_dict:
        betas = data_dict['betas']
        if betas.shape == (10,):
            data_dict['betas'] = np.concatenate([betas, np.zeros(6, dtype=betas.dtype)])
            console.print(f"[cyan]将betas从10填充到16: {input_path}[/cyan]")
        elif betas.shape not in [(16,), (1, 16)]:
            raise ValueError(f"Unexpected betas shape: {betas.shape}. Expected (10,), (16,), or (1,16) for padding to SMPL-X.")

    # Handle mocap_frame_rate variations
    if 'mocap_framerate' in data_dict:
        data_dict['mocap_frame_rate'] = data_dict.pop('mocap_framerate')
        console.print(f"[cyan]将'mocap_framerate'重命名为'mocap_frame_rate': {input_path}[/cyan]")

    if 'poses' not in data_dict:
        raise ValueError("Input file does not contain 'poses' key. Is this an SMPL file?")

    poses = data_dict['poses']
    if poses.shape[1] > 72:
        poses = poses[:, :72]

    # Map to SMPL-X format
    data_dict['root_orient'] = poses[:, :3]
    data_dict['pose_body'] = poses[:, 3:66]  # 21 joints x 3 = 63, ignoring SMPL hand poses

    # Ensure gender is set
    if 'gender' not in data_dict:
        data_dict['gender'] = np.array(gender)

    # Remove original poses key
    del data_dict['poses']

    # Save as SMPL-X npz
    np.savez(output_path, **data_dict)
    console.print(f"[green]已转换 {input_path} 到 {output_path}[/green]")

def process_directory(src_folder, tgt_folder, gender='neutral'):
    os.makedirs(tgt_folder, exist_ok=True)
    for filename in tqdm(os.listdir(src_folder)):
        if filename.endswith('.npz'):
            input_path = os.path.join(src_folder, filename)
            output_path = os.path.join(tgt_folder, filename)
            convert_smpl_to_smplx(input_path, output_path, gender)

if __name__ == "__main__":
    console.print(Panel.fit("[bold cyan]SMPL转SMPL-X格式工具[/bold cyan]", border_style="cyan"))
    print_hardware_info()
    
    parser = argparse.ArgumentParser(description="Convert SMPL motion data to SMPL-X format.")
    parser.add_argument("--src_folder", type=str, help="Source directory of SMPL .npz files")
    parser.add_argument("--tgt_folder", type=str, help="Target directory for SMPL-X .npz files")
    parser.add_argument("--input_file", type=str, help="Single input SMPL .npz file")
    parser.add_argument("--output_file", type=str, help="Single output SMPL-X .npz file")
    parser.add_argument("--gender", type=str, default="neutral", choices=["male", "female", "neutral"],
                        help="Gender for SMPL-X model if not present in file.")
    args = parser.parse_args()

    if args.src_folder and args.tgt_folder:
        process_directory(args.src_folder, args.tgt_folder, args.gender)
    elif args.input_file and args.output_file:
        convert_smpl_to_smplx(args.input_file, args.output_file, args.gender)
    else:
        parser.print_help()
