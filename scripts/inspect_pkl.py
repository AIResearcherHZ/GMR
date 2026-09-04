#!/usr/bin/env python3
"""
PKL 文件格式检查和验证工具
支持 GMR 格式和 HOVER 格式的 pkl 文件
"""
import argparse
import pickle
import joblib
import signal
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from rich import print
from rich.table import Table
from rich.console import Console

# 全局变量用于信号处理
_animation_running = False

def _signal_handler(signum, frame):
    """处理 Ctrl+C 信号，优雅退出"""
    global _animation_running
    if _animation_running:
        plt.close('all')
        _animation_running = False
    print("\n[yellow]用户中断，退出...[/yellow]")
    sys.exit(0)

console = Console()


def detect_format(data):
    """检测 pkl 文件格式"""
    if isinstance(data, dict):
        keys = set(data.keys())
        # GMR 格式特征
        gmr_keys = {"fps", "root_pos", "root_rot", "dof_pos"}
        if gmr_keys.issubset(keys):
            return "gmr"
        # HOVER 格式特征：包含动作名称作为 key，每个动作有特定字段
        for k, v in data.items():
            if isinstance(v, dict):
                hover_keys = {"root_trans_offset", "dof", "root_rot", "fps"}
                if hover_keys.issubset(set(v.keys())):
                    return "hover"
    return "unknown"


def load_pkl(file_path):
    """尝试用不同方式加载 pkl 文件"""
    # 先尝试 joblib
    try:
        data = joblib.load(file_path)
        return data, "joblib"
    except Exception:
        pass
    
    # 再尝试 pickle
    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        return data, "pickle"
    except Exception:
        pass
    
    raise ValueError(f"无法加载文件: {file_path}")


def print_array_info(name, arr, indent=2):
    """打印数组信息"""
    prefix = " " * indent
    if arr is None:
        print(f"{prefix}[yellow]{name}[/yellow]: None")
    elif isinstance(arr, np.ndarray):
        print(f"{prefix}[cyan]{name}[/cyan]: shape={arr.shape}, dtype={arr.dtype}")
        if arr.size > 0:
            print(f"{prefix}  min={arr.min():.4f}, max={arr.max():.4f}, mean={arr.mean():.4f}")
    elif isinstance(arr, (int, float)):
        print(f"{prefix}[cyan]{name}[/cyan]: {arr}")
    elif isinstance(arr, list):
        print(f"{prefix}[cyan]{name}[/cyan]: list, len={len(arr)}")
    else:
        print(f"{prefix}[cyan]{name}[/cyan]: {type(arr).__name__}")


def inspect_gmr_format(data):
    """检查 GMR 格式"""
    print("\n[bold green]═══ GMR 格式 ═══[/bold green]")
    
    print_array_info("fps", data.get("fps"))
    print_array_info("root_pos", data.get("root_pos"))
    print_array_info("root_rot", data.get("root_rot"))
    print_array_info("dof_pos", data.get("dof_pos"))
    print_array_info("local_body_pos", data.get("local_body_pos"))
    print_array_info("link_body_list", data.get("link_body_list"))
    
    # 验证
    print("\n[bold]验证结果:[/bold]")
    errors = []
    
    root_pos = data.get("root_pos")
    root_rot = data.get("root_rot")
    dof_pos = data.get("dof_pos")
    
    if root_pos is not None and root_rot is not None:
        if root_pos.shape[0] != root_rot.shape[0]:
            errors.append(f"root_pos 和 root_rot 帧数不匹配: {root_pos.shape[0]} vs {root_rot.shape[0]}")
    
    if root_pos is not None and dof_pos is not None:
        if root_pos.shape[0] != dof_pos.shape[0]:
            errors.append(f"root_pos 和 dof_pos 帧数不匹配: {root_pos.shape[0]} vs {dof_pos.shape[0]}")
    
    if root_rot is not None and root_rot.shape[-1] != 4:
        errors.append(f"root_rot 应该是四元数 (最后一维=4), 当前: {root_rot.shape[-1]}")
    
    if root_pos is not None and root_pos.shape[-1] != 3:
        errors.append(f"root_pos 应该是 3D 位置 (最后一维=3), 当前: {root_pos.shape[-1]}")
    
    if errors:
        for e in errors:
            print(f"  [red]✗[/red] {e}")
    else:
        print("  [green]✓[/green] 格式验证通过")
        if root_pos is not None:
            print(f"  [green]✓[/green] 总帧数: {root_pos.shape[0]}")
            fps = data.get("fps", 30)
            duration = root_pos.shape[0] / fps
            print(f"  [green]✓[/green] 时长: {duration:.2f} 秒 (fps={fps})")


def inspect_hover_format(data):
    """检查 HOVER 格式"""
    print("\n[bold green]═══ HOVER 格式 ═══[/bold green]")
    print(f"[bold]动作数量:[/bold] {len(data)}")
    
    table = Table(title="动作列表")
    table.add_column("动作名称", style="cyan")
    table.add_column("帧数", justify="right")
    table.add_column("FPS", justify="right")
    table.add_column("时长(秒)", justify="right")
    table.add_column("DOF维度", justify="right")
    
    total_frames = 0
    for motion_name, motion_data in data.items():
        if not isinstance(motion_data, dict):
            continue
        
        frames = motion_data.get("root_trans_offset", np.array([])).shape[0]
        fps = motion_data.get("fps", 30)
        duration = frames / fps if fps > 0 else 0
        dof_dim = motion_data.get("dof", np.array([])).shape[-1] if motion_data.get("dof") is not None else "N/A"
        
        table.add_row(
            motion_name[:50] + "..." if len(motion_name) > 50 else motion_name,
            str(frames),
            str(fps),
            f"{duration:.2f}",
            str(dof_dim)
        )
        total_frames += frames
    
    console.print(table)
    
    # 详细查看第一个动作
    first_motion_name = list(data.keys())[0]
    first_motion = data[first_motion_name]
    
    print(f"\n[bold]第一个动作详情: [cyan]{first_motion_name}[/cyan][/bold]")
    for key, value in first_motion.items():
        print_array_info(key, value)
    
    # 验证
    print("\n[bold]验证结果:[/bold]")
    errors = []
    
    for motion_name, motion_data in data.items():
        if not isinstance(motion_data, dict):
            continue
        
        root_trans = motion_data.get("root_trans_offset")
        root_rot = motion_data.get("root_rot")
        dof = motion_data.get("dof")
        
        if root_trans is not None and root_rot is not None:
            if root_trans.shape[0] != root_rot.shape[0]:
                errors.append(f"[{motion_name}] root_trans_offset 和 root_rot 帧数不匹配")
        
        if root_trans is not None and dof is not None:
            if root_trans.shape[0] != dof.shape[0]:
                errors.append(f"[{motion_name}] root_trans_offset 和 dof 帧数不匹配")
    
    if errors:
        for e in errors:
            print(f"  [red]✗[/red] {e}")
    else:
        print("  [green]✓[/green] 格式验证通过")
        print(f"  [green]✓[/green] 总动作数: {len(data)}")
        print(f"  [green]✓[/green] 总帧数: {total_frames}")


def animate_motion(root_pos, root_rot, dof, fps, title="Motion Animation"):
    """动画播放动作"""
    global _animation_running
    _animation_running = True
    
    # 注册信号处理器
    original_handler = signal.signal(signal.SIGINT, _signal_handler)
    
    n_frames = len(root_pos)
    
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f"{title}\nFrames: {n_frames}, FPS: {fps}\n(Press Q or close window to exit)", fontsize=12)
    
    # 3D 轨迹视图
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D Trajectory')
    
    # 设置 3D 轴范围
    margin = 0.2
    ax1.set_xlim(root_pos[:, 0].min() - margin, root_pos[:, 0].max() + margin)
    ax1.set_ylim(root_pos[:, 1].min() - margin, root_pos[:, 1].max() + margin)
    ax1.set_zlim(root_pos[:, 2].min() - margin, root_pos[:, 2].max() + margin)
    
    # XY 平面视图 - 使用 set_aspect 而不是 axis('equal') 避免 limits 警告
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('XY Plane')
    ax2.grid(True, alpha=0.3)
    # 计算合适的范围以保持等比例
    x_range = root_pos[:, 0].max() - root_pos[:, 0].min()
    y_range = root_pos[:, 1].max() - root_pos[:, 1].min()
    max_range = max(x_range, y_range) / 2 + margin
    x_center = (root_pos[:, 0].max() + root_pos[:, 0].min()) / 2
    y_center = (root_pos[:, 1].max() + root_pos[:, 1].min()) / 2
    ax2.set_xlim(x_center - max_range, x_center + max_range)
    ax2.set_ylim(y_center - max_range, y_center + max_range)
    ax2.set_aspect('equal', adjustable='box')
    
    # 关节角度视图
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_xlabel('Joint Index')
    ax3.set_ylabel('Angle (rad)')
    ax3.set_title('Joint Angles')
    ax3.grid(True, alpha=0.3)
    if dof is not None:
        ax3.set_xlim(-0.5, dof.shape[1] - 0.5)
        ax3.set_ylim(dof.min() - 0.1, dof.max() + 0.1)
    
    # 信息文本
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    
    # 初始化绘图对象
    traj_line_3d, = ax1.plot([], [], [], 'b-', linewidth=0.5, alpha=0.5)
    current_point_3d, = ax1.plot([], [], [], 'ro', markersize=10)
    
    traj_line_2d, = ax2.plot([], [], 'b-', linewidth=0.5, alpha=0.5)
    current_point_2d, = ax2.plot([], [], 'ro', markersize=10)
    
    joint_bars = ax3.bar(range(dof.shape[1]) if dof is not None else [], 
                         [0] * (dof.shape[1] if dof is not None else 0),
                         color='steelblue')
    
    info_text = ax4.text(0.1, 0.5, '', fontsize=14, family='monospace',
                         verticalalignment='center', transform=ax4.transAxes)
    
    def init():
        traj_line_3d.set_data([], [])
        traj_line_3d.set_3d_properties([])
        current_point_3d.set_data([], [])
        current_point_3d.set_3d_properties([])
        traj_line_2d.set_data([], [])
        current_point_2d.set_data([], [])
        return [traj_line_3d, current_point_3d, traj_line_2d, current_point_2d, info_text] + list(joint_bars)
    
    def update(frame):
        # 更新 3D 轨迹
        traj_line_3d.set_data(root_pos[:frame+1, 0], root_pos[:frame+1, 1])
        traj_line_3d.set_3d_properties(root_pos[:frame+1, 2])
        current_point_3d.set_data([root_pos[frame, 0]], [root_pos[frame, 1]])
        current_point_3d.set_3d_properties([root_pos[frame, 2]])
        
        # 更新 2D 轨迹
        traj_line_2d.set_data(root_pos[:frame+1, 0], root_pos[:frame+1, 1])
        current_point_2d.set_data([root_pos[frame, 0]], [root_pos[frame, 1]])
        
        # 更新关节角度柱状图
        if dof is not None:
            for bar, val in zip(joint_bars, dof[frame]):
                bar.set_height(val)
                bar.set_color('steelblue' if val >= 0 else 'coral')
        
        # 更新信息文本
        time_sec = frame / fps
        info = f"Frame: {frame:4d} / {n_frames}\n"
        info += f"Time:  {time_sec:.2f} s\n\n"
        info += f"Position:\n"
        info += f"  X: {root_pos[frame, 0]:+.4f}\n"
        info += f"  Y: {root_pos[frame, 1]:+.4f}\n"
        info += f"  Z: {root_pos[frame, 2]:+.4f}\n\n"
        if root_rot is not None:
            info += f"Rotation (quat):\n"
            info += f"  {root_rot[frame, 0]:+.3f}, {root_rot[frame, 1]:+.3f}\n"
            info += f"  {root_rot[frame, 2]:+.3f}, {root_rot[frame, 3]:+.3f}"
        info_text.set_text(info)
        
        return [traj_line_3d, current_point_3d, traj_line_2d, current_point_2d, info_text] + list(joint_bars)
    
    interval = 1000 / fps  # 毫秒
    anim = FuncAnimation(fig, update, frames=n_frames, init_func=init,
                         interval=interval, blit=False, repeat=True)
    
    # 添加键盘事件处理
    def on_key(event):
        if event.key in ['q', 'Q', 'escape']:
            plt.close('all')
    
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    plt.tight_layout()
    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        _animation_running = False
        signal.signal(signal.SIGINT, original_handler)
        plt.close('all')
    return anim


def plot_gmr_data(data, title="GMR Motion Data", animate=False):
    """可视化 GMR 格式数据"""
    root_pos = data.get("root_pos")
    root_rot = data.get("root_rot")
    dof_pos = data.get("dof_pos")
    fps = data.get("fps", 30)
    
    if animate:
        return animate_motion(root_pos, root_rot, dof_pos, fps, title)
    
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(title, fontsize=14)
    
    # 1. 3D 根轨迹
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    if root_pos is not None:
        ax1.plot(root_pos[:, 0], root_pos[:, 1], root_pos[:, 2], 'b-', linewidth=0.5)
        ax1.scatter(root_pos[0, 0], root_pos[0, 1], root_pos[0, 2], c='g', s=100, label='Start')
        ax1.scatter(root_pos[-1, 0], root_pos[-1, 1], root_pos[-1, 2], c='r', s=100, label='End')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.legend()
    ax1.set_title('Root Trajectory (3D)')
    
    # 2. 根位置随时间变化
    ax2 = fig.add_subplot(2, 3, 2)
    if root_pos is not None:
        t = np.arange(len(root_pos)) / fps
        ax2.plot(t, root_pos[:, 0], 'r-', label='X', linewidth=0.8)
        ax2.plot(t, root_pos[:, 1], 'g-', label='Y', linewidth=0.8)
        ax2.plot(t, root_pos[:, 2], 'b-', label='Z', linewidth=0.8)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Position (m)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    ax2.set_title('Root Position vs Time')
    
    # 3. 根旋转四元数
    ax3 = fig.add_subplot(2, 3, 3)
    if root_rot is not None:
        t = np.arange(len(root_rot)) / fps
        ax3.plot(t, root_rot[:, 0], 'r-', label='X', linewidth=0.8)
        ax3.plot(t, root_rot[:, 1], 'g-', label='Y', linewidth=0.8)
        ax3.plot(t, root_rot[:, 2], 'b-', label='Z', linewidth=0.8)
        ax3.plot(t, root_rot[:, 3], 'k-', label='W', linewidth=0.8)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Quaternion')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    ax3.set_title('Root Rotation (Quaternion)')
    
    # 4. XY 平面轨迹
    ax4 = fig.add_subplot(2, 3, 4)
    if root_pos is not None:
        ax4.plot(root_pos[:, 0], root_pos[:, 1], 'b-', linewidth=0.8)
        ax4.scatter(root_pos[0, 0], root_pos[0, 1], c='g', s=100, label='Start', zorder=5)
        ax4.scatter(root_pos[-1, 0], root_pos[-1, 1], c='r', s=100, label='End', zorder=5)
        ax4.set_xlabel('X')
        ax4.set_ylabel('Y')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.axis('equal')
    ax4.set_title('Root Trajectory (XY Plane)')
    
    # 5. DOF 关节角度热力图
    ax5 = fig.add_subplot(2, 3, 5)
    if dof_pos is not None:
        im = ax5.imshow(dof_pos.T, aspect='auto', cmap='RdBu_r', 
                        extent=[0, len(dof_pos)/fps, dof_pos.shape[1]-0.5, -0.5])
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Joint Index')
        plt.colorbar(im, ax=ax5, label='Angle (rad)')
    ax5.set_title('DOF Angles Heatmap')
    
    # 6. 选定关节角度曲线
    ax6 = fig.add_subplot(2, 3, 6)
    if dof_pos is not None:
        t = np.arange(len(dof_pos)) / fps
        n_joints = min(6, dof_pos.shape[1])
        for i in range(n_joints):
            ax6.plot(t, dof_pos[:, i], linewidth=0.8, label=f'Joint {i}')
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('Angle (rad)')
        ax6.legend(loc='upper right', fontsize=8)
        ax6.grid(True, alpha=0.3)
    ax6.set_title('Selected Joint Angles')
    
    plt.tight_layout()
    plt.show()


def plot_hover_data(data, motion_idx=0, title="HOVER Motion Data", animate=False):
    """可视化 HOVER 格式数据"""
    motion_names = list(data.keys())
    if motion_idx >= len(motion_names):
        motion_idx = 0
    
    motion_name = motion_names[motion_idx]
    motion_data = data[motion_name]
    
    root_pos = motion_data.get("root_trans_offset")
    root_rot = motion_data.get("root_rot")
    dof = motion_data.get("dof")
    fps = motion_data.get("fps", 30)
    
    full_title = f"{title}\nMotion: {motion_name} ({motion_idx+1}/{len(motion_names)})"
    
    if animate:
        return animate_motion(root_pos, root_rot, dof, fps, full_title)
    
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(full_title, fontsize=12)
    
    # 1. 3D 根轨迹
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    if root_pos is not None:
        ax1.plot(root_pos[:, 0], root_pos[:, 1], root_pos[:, 2], 'b-', linewidth=0.5)
        ax1.scatter(root_pos[0, 0], root_pos[0, 1], root_pos[0, 2], c='g', s=100, label='Start')
        ax1.scatter(root_pos[-1, 0], root_pos[-1, 1], root_pos[-1, 2], c='r', s=100, label='End')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.legend()
    ax1.set_title('Root Trajectory (3D)')
    
    # 2. 根位置随时间变化
    ax2 = fig.add_subplot(2, 3, 2)
    if root_pos is not None:
        t = np.arange(len(root_pos)) / fps
        ax2.plot(t, root_pos[:, 0], 'r-', label='X', linewidth=0.8)
        ax2.plot(t, root_pos[:, 1], 'g-', label='Y', linewidth=0.8)
        ax2.plot(t, root_pos[:, 2], 'b-', label='Z', linewidth=0.8)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Position (m)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    ax2.set_title('Root Position vs Time')
    
    # 3. 根旋转四元数
    ax3 = fig.add_subplot(2, 3, 3)
    if root_rot is not None:
        t = np.arange(len(root_rot)) / fps
        ax3.plot(t, root_rot[:, 0], 'r-', label='X', linewidth=0.8)
        ax3.plot(t, root_rot[:, 1], 'g-', label='Y', linewidth=0.8)
        ax3.plot(t, root_rot[:, 2], 'b-', label='Z', linewidth=0.8)
        ax3.plot(t, root_rot[:, 3], 'k-', label='W', linewidth=0.8)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Quaternion')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    ax3.set_title('Root Rotation (Quaternion)')
    
    # 4. XY 平面轨迹
    ax4 = fig.add_subplot(2, 3, 4)
    if root_pos is not None:
        ax4.plot(root_pos[:, 0], root_pos[:, 1], 'b-', linewidth=0.8)
        ax4.scatter(root_pos[0, 0], root_pos[0, 1], c='g', s=100, label='Start', zorder=5)
        ax4.scatter(root_pos[-1, 0], root_pos[-1, 1], c='r', s=100, label='End', zorder=5)
        ax4.set_xlabel('X')
        ax4.set_ylabel('Y')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.axis('equal')
    ax4.set_title('Root Trajectory (XY Plane)')
    
    # 5. DOF 关节角度热力图
    ax5 = fig.add_subplot(2, 3, 5)
    if dof is not None:
        im = ax5.imshow(dof.T, aspect='auto', cmap='RdBu_r',
                        extent=[0, len(dof)/fps, dof.shape[1]-0.5, -0.5])
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Joint Index')
        plt.colorbar(im, ax=ax5, label='Angle (rad)')
    ax5.set_title('DOF Angles Heatmap')
    
    # 6. 选定关节角度曲线
    ax6 = fig.add_subplot(2, 3, 6)
    if dof is not None:
        t = np.arange(len(dof)) / fps
        n_joints = min(6, dof.shape[1])
        for i in range(n_joints):
            ax6.plot(t, dof[:, i], linewidth=0.8, label=f'Joint {i}')
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('Angle (rad)')
        ax6.legend(loc='upper right', fontsize=8)
        ax6.grid(True, alpha=0.3)
    ax6.set_title('Selected Joint Angles')
    
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="PKL 文件格式检查和验证工具")
    parser.add_argument("pkl_file", help="要检查的 pkl 文件路径")
    parser.add_argument("--show-samples", "-s", type=int, default=0,
                        help="显示前 N 帧的数据样本")
    parser.add_argument("--plot", "-p", action="store_true",
                        help="静态图表可视化")
    parser.add_argument("--animate", "-a", action="store_true",
                        help="动画播放动作")
    parser.add_argument("--motion-idx", "-m", type=int, default=0,
                        help="HOVER 格式时选择第几个动作进行可视化 (从0开始)")
    args = parser.parse_args()
    
    print(f"\n[bold]文件:[/bold] {args.pkl_file}")
    
    # 加载文件
    try:
        data, loader = load_pkl(args.pkl_file)
        print(f"[bold]加载方式:[/bold] {loader}")
    except Exception as e:
        print(f"[red]加载失败:[/red] {e}")
        return
    
    # 检测格式
    fmt = detect_format(data)
    print(f"[bold]检测格式:[/bold] {fmt}")
    
    if fmt == "gmr":
        inspect_gmr_format(data)
    elif fmt == "hover":
        inspect_hover_format(data)
    else:
        print("\n[yellow]未知格式，显示原始结构:[/yellow]")
        if isinstance(data, dict):
            print(f"Keys: {list(data.keys())}")
            for k, v in data.items():
                print_array_info(k, v)
        else:
            print(f"Type: {type(data)}")
    
    # 显示数据样本
    if args.show_samples > 0:
        print(f"\n[bold]前 {args.show_samples} 帧数据样本:[/bold]")
        if fmt == "gmr":
            root_pos = data.get("root_pos")
            root_rot = data.get("root_rot")
            dof_pos = data.get("dof_pos")
            n = min(args.show_samples, len(root_pos) if root_pos is not None else 0)
            for i in range(n):
                print(f"\n[cyan]帧 {i}:[/cyan]")
                if root_pos is not None:
                    print(f"  root_pos: {root_pos[i]}")
                if root_rot is not None:
                    print(f"  root_rot: {root_rot[i]}")
                if dof_pos is not None:
                    print(f"  dof_pos: {dof_pos[i][:5]}... (前5个)")
        elif fmt == "hover":
            first_motion = list(data.values())[0]
            root_trans = first_motion.get("root_trans_offset")
            root_rot = first_motion.get("root_rot")
            dof = first_motion.get("dof")
            n = min(args.show_samples, len(root_trans) if root_trans is not None else 0)
            for i in range(n):
                print(f"\n[cyan]帧 {i}:[/cyan]")
                if root_trans is not None:
                    print(f"  root_trans_offset: {root_trans[i]}")
                if root_rot is not None:
                    print(f"  root_rot: {root_rot[i]}")
                if dof is not None:
                    print(f"  dof: {dof[i][:5]}... (前5个)")

    # 可视化
    if args.plot or args.animate:
        import os
        filename = os.path.basename(args.pkl_file)
        if fmt == "gmr":
            plot_gmr_data(data, title=f"GMR: {filename}", animate=args.animate)
        elif fmt == "hover":
            plot_hover_data(data, motion_idx=args.motion_idx, 
                           title=f"HOVER: {filename}", animate=args.animate)
        else:
            print("[yellow]未知格式，无法可视化[/yellow]")


if __name__ == "__main__":
    main()
