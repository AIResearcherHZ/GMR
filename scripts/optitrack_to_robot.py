from general_motion_retargeting.optitrack_vendor.NatNetClient import setup_optitrack
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
import threading
import argparse
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

def main(args):
    console.print(Panel.fit("[bold cyan]OptiTrack转机器人工具[/bold cyan]", border_style="cyan"))
    print_hardware_info()
    
    console.print("[yellow]请确保在两台机器上禁用防火墙:[/yellow]")
    console.print("[yellow]OptiTrack计算机: 禁用Windows防火墙[/yellow]")
    console.print("[yellow]本机: sudo ufw disable[/yellow]")

    client = setup_optitrack(
        server_address=args.server_ip,
        client_address=args.client_ip,
        use_multicast=args.use_multicast,
    )

    # start a thread to client.run()
    thread = threading.Thread(target=client.run)
    thread.start()

    if not client:
        console.print("[red]OptiTrack客户端设置失败[/red]")
        exit(1)

    console.print(f"[green]OptiTrack客户端已连接: {client.connected()}[/green]")
    console.print("[cyan]开始运动重定向...[/cyan]")

    retarget = GMR(
            src_human="fbx",
            tgt_robot=args.robot,
            actual_human_height=1.6,
        )
    viewer = RobotMotionViewer(robot_type="unitree_g1")

    try:
        while True:
            frame = client.get_frame()
            frame_number = client.get_frame_number()
            qpos = retarget.retarget(frame)
            viewer.step(
                root_pos=qpos[:3],
                root_rot=qpos[3:7],
                dof_pos=qpos[7:],
                rate_limit=False,
            )
    except KeyboardInterrupt:
        console.print("[yellow]\n用户中断，清理中...[/yellow]")
    finally:
        viewer.close()
        client.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_ip", type=str, default="192.168.200.160")
    parser.add_argument("--client_ip", type=str, default="192.168.200.117")
    parser.add_argument("--use_multicast", type=bool, default=False)
    parser.add_argument("--robot", type=str, default="unitree_g1")
    args = parser.parse_args()
    main(args)
    