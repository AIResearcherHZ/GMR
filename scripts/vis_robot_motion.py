from general_motion_retargeting import RobotMotionViewer, load_robot_motion
import argparse
import os
import platform
import multiprocessing
import numpy as np
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

if __name__ == "__main__":
    console.print(Panel.fit("[bold cyan]机器人运动可视化工具[/bold cyan]", border_style="cyan"))
    print_hardware_info()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default="unitree_g1")
                        
    parser.add_argument("--robot_motion_path", type=str, required=True)

    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--video_path", type=str, 
                        default="videos/example.mp4")
                        
    args = parser.parse_args()
    
    robot_type = args.robot
    robot_motion_path = args.robot_motion_path
    
    if not os.path.exists(robot_motion_path):
        raise FileNotFoundError(f"Motion file {robot_motion_path} not found")
    
    motion_data, motion_fps, motion_root_pos, motion_root_rot, motion_dof_pos, motion_local_body_pos, motion_link_body_list = load_robot_motion(robot_motion_path)
    
    env = RobotMotionViewer(robot_type=robot_type,
                            motion_fps=motion_fps,
                            camera_follow=False,
                            record_video=args.record_video, video_path=args.video_path)
    
    frame_idx = 0
    while True:
        human_motion_data = None
        if motion_local_body_pos is not None and motion_link_body_list is not None:
            human_motion_data = {}
            for i, body_name in enumerate(motion_link_body_list):
                pos = motion_local_body_pos[frame_idx][i][:3]
                rot = np.array([1.0, 0.0, 0.0, 0.0])  # identity quaternion (wxyz)
                human_motion_data[body_name] = (pos, rot)
        
        env.step(motion_root_pos[frame_idx], 
                motion_root_rot[frame_idx], 
                motion_dof_pos[frame_idx],
                human_motion_data=human_motion_data,
                human_pos_offset=np.array([0.0, 0.0, 0.0]),
                rate_limit=True)
        frame_idx += 1
        if frame_idx >= len(motion_root_pos):
            frame_idx = 0
    env.close()