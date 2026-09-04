import math
import platform
import multiprocessing
from isaacgym import gymapi
from isaacgym import gymutil
import numpy as np
from isaacgym.torch_utils import *
from termcolor import cprint
import argparse
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

console.print(Panel.fit("[bold cyan]机器人 URDF 可视化工具[/bold cyan]", border_style="cyan"))
print_hardware_info()


# initialize gym
gym = gymapi.acquire_gym()

# create a simulator
sim_params = gymapi.SimParams()
sim_params.substeps = 2
sim_params.dt = 1.0 / 60.0

sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 4
sim_params.physx.num_velocity_iterations = 1

sim_params.physx.num_threads = 4
sim_params.physx.use_gpu = True

sim_params.use_gpu_pipeline = False
device_id = 1
# sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
sim = gym.create_sim(device_id, device_id, gymapi.SIM_PHYSX, sim_params)
if sim is None:
    print("*** Failed to create sim")
    quit()

# create viewer using the default camera properties
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise ValueError('*** Failed to create viewer')

# add ground plane
plane_params = gymapi.PlaneParams()
gym.add_ground(sim, gymapi.PlaneParams())

# set up the env grid
num_envs = 4
spacing = 1.5
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, 0.0, spacing)


asset_root = f"../assets/"
# asset_file = "agibot_a2/urdf/model.urdf"
# asset_file = "unitree_h1/h1.urdf"
# asset_file = "adam_lite/adam_lite.urdf"
# asset_file = "openloong/AzureLoong.urdf"
# asset_file = "booster_t1/T1_serial.xml"
# asset_file = "unitree_h1_2/h1_2.urdf"
# asset_file = "unitree_h1_2/h1_2_handless.urdf"
asset_file = "pnd_adam_lite/adam_lite.urdf"

# Load asset with default control type of position for all joints
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
asset_options.collapse_fixed_joints = False
asset_options.use_mesh_materials = False
# asset_options.vhacd_enabled = False
asset_options.vhacd_enabled = True 
asset_options.vhacd_params = gymapi.VhacdParams() 
asset_options.vhacd_params.resolution = 200000 

asset_options.flip_visual_attachments = True
# asset_options.flip_visual_attachments = False

console.print(f"[cyan]加载资产 '{asset_file}' 从 '{asset_root}'[/cyan]")
robot_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

robot_link_dict = gym.get_asset_rigid_body_dict(robot_asset)
robot_dof_dict = gym.get_asset_dof_dict(robot_asset)
ordered_body_dict = {k: v for k, v in sorted(robot_link_dict.items(), key=lambda item: item[1])}
cprint(f'[HumanoidRobot] Full robot body dict. NumBody: {len(robot_link_dict.keys())}', 'blue')
for k, v in ordered_body_dict.items():
    cprint(f'\t {k} {v}', 'blue')

ordered_dof_dict = {k: v for k, v in sorted(robot_dof_dict.items(), key=lambda item: item[1])}
cprint(f'[HumanoidRobot] Full robot dof dict. DoF: {len(ordered_dof_dict.keys())}', 'green')
for k, v in ordered_dof_dict.items():
    cprint(f'\t {k} {v}', 'green')

# print body names and dof names in the format: [""]
console.print(f"[cyan]身体名称: {list(ordered_body_dict.keys())}[/cyan]")
console.print(f"[cyan]DoF名称: {list(ordered_dof_dict.keys())}[/cyan]")

# initial root pose
initial_pose = gymapi.Transform()
initial_pose.p = gymapi.Vec3(0.0, 1.5, 0.0) # since isaacgym uses y-up coordinate system
initial_pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)
# initial_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

env = gym.create_env(sim, env_lower, env_upper, 0)
robot = gym.create_actor(env, robot_asset, initial_pose, 'robot', 0, 1)


# Configure DOF properties
props = gym.get_actor_dof_properties(env, robot)
props['driveMode'][:] = gymapi.DOF_MODE_POS  # Set all joints to position control
props['stiffness'][:] = 400.0  # Set stiffness for better control
props['damping'][:] = 40.0    # Set damping to reduce oscillations
gym.set_actor_dof_properties(env, robot, props)


# Look at the first env
cam_pos = gymapi.Vec3(8, 4, 1.5)
cam_target = gymapi.Vec3(0, 2, 1.5)
gym.viewer_camera_look_at(viewer, env, cam_pos, cam_target)

num_dofs = gym.get_actor_dof_count(env, robot)
initial_dof_positions = np.zeros(num_dofs, dtype=np.float32)  # Adjust as per robot
gym.set_actor_dof_position_targets(env, robot, initial_dof_positions)


# Simulate
while not gym.query_viewer_has_closed(viewer):

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)



console.print('[bold green]完成[/bold green]')

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)