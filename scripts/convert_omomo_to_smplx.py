import os
import joblib
import numpy as np
import pickle
import platform
import multiprocessing
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


console.print(Panel.fit("[bold cyan]OMOMO转SMPLX工具[/bold cyan]", border_style="cyan"))
print_hardware_info()

# these paths are from the original OMOMO dataset
motion_path1 = "/home/yanjieze/projects/g1_wbc/motion_data/omomo_data/train_diffusion_manip_seq_joints24.p"
motion_path2 = "/home/yanjieze/projects/g1_wbc/motion_data/omomo_data/test_diffusion_manip_seq_joints24.p"
all_motion_data1 = joblib.load(motion_path1)
all_motion_data2 = joblib.load(motion_path2)

# save as individual files
target_dir = "/home/yanjieze/projects/g1_wbc/motion_data/OMOMO_smplx"
os.makedirs(target_dir, exist_ok=True)
for motion_data in [all_motion_data1, all_motion_data2]:
    for data_name in motion_data.keys():
        
        smpl_data = motion_data[data_name]
        seq_name = smpl_data['seq_name']
        # save as npz
        num_frames = smpl_data["pose_body"].shape[0]
        mocap_frame_rate = 30
        poses = np.concatenate([smpl_data["pose_body"], 
                                np.zeros((num_frames, 102))],
                                axis=1)
        smpl_data["poses"] = poses
        smpl_data["mocap_frame_rate"] = np.array(mocap_frame_rate)
        # use pickle to save
        with open(f"{target_dir}/{seq_name}.pkl", "wb") as f:
            pickle.dump(smpl_data, f)
        console.print(f"[green]已保存 {seq_name}[/green]")