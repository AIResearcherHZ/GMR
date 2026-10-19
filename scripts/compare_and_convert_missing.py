#!/usr/bin/env python3
"""
数据集对比工具 - 找出目标数据集中缺失的文件并生成转换命令

用于对比两个数据集目录(如g1和taks_t1),找出taks_t1中缺失的文件,
并生成单独的转换命令或直接执行转换。
"""

import os
import argparse
import shlex
from pathlib import Path
from typing import List, Tuple, Set


def get_files_recursive(folder: Path, extensions: Tuple[str, ...] = ('.pkl', '.npz')) -> Set[str]:
    """
    递归获取文件夹中所有指定扩展名的文件的相对路径
    
    Args:
        folder: 文件夹路径
        extensions: 文件扩展名元组
    
    Returns:
        相对路径集合(相对于folder)
    """
    files = set()
    if not folder.exists():
        return files
    
    for root, _, filenames in os.walk(folder):
        for filename in filenames:
            if filename.endswith(extensions):
                full_path = Path(root) / filename
                rel_path = full_path.relative_to(folder)
                files.add(str(rel_path))
    
    return files


def compare_datasets(src_folder: Path, tgt_folder: Path) -> List[str]:
    """
    对比源数据集和目标数据集,找出目标数据集中缺失的文件
    
    Args:
        src_folder: 源数据集文件夹(如g1)
        tgt_folder: 目标数据集文件夹(如taks_t1)
    
    Returns:
        缺失文件的相对路径列表
    """
    src_files = get_files_recursive(src_folder)
    tgt_files = get_files_recursive(tgt_folder)
    
    # 找出源数据集有但目标数据集没有的文件
    missing_files = src_files - tgt_files
    
    return sorted(list(missing_files))


def generate_conversion_commands(
    src_folder: Path,
    tgt_folder: Path,
    missing_files: List[str],
    src_robot: str,
    tgt_robot: str,
    script_type: str = 'robot_to_robot'
) -> List[str]:
    """
    生成转换命令
    
    Args:
        src_folder: 源数据集文件夹
        tgt_folder: 目标数据集文件夹
        missing_files: 缺失文件列表
        src_robot: 源机器人类型
        tgt_robot: 目标机器人类型
        script_type: 脚本类型 ('robot_to_robot', 'bvh_to_robot', 'npz_to_taks_t1', 'npz_to_g1')
    
    Returns:
        转换命令列表
    """
    commands = []
    
    for missing_file in missing_files:
        src_file = src_folder / missing_file
        tgt_file = tgt_folder / missing_file
        
        # 确保目标文件夹存在
        tgt_file.parent.mkdir(parents=True, exist_ok=True)
        
        if script_type == 'robot_to_robot':
            cmd = (
                f"python scripts/robot_to_robot.py "
                f"--src_file {shlex.quote(str(src_file))} "
                f"--src_robot {src_robot} "
                f"--tgt_robot {tgt_robot} "
                f"--save_path {shlex.quote(str(tgt_file))}"
            )
        elif script_type == 'bvh_to_robot':
            cmd = (
                f"python scripts/bvh_to_robot.py "
                f"--bvh_file {shlex.quote(str(src_file))} "
                f"--robot {tgt_robot} "
                f"--save_path {shlex.quote(str(tgt_file))}"
            )
        elif script_type == 'npz_to_taks_t1':
            cmd = (
                f"python scripts/npz_to_taks_t1_pkl.py "
                f"-i {shlex.quote(str(src_file))} "
                f"-o {shlex.quote(str(tgt_file))}"
            )
        elif script_type == 'npz_to_g1':
            cmd = (
                f"python scripts/npz_to_g1_pkl.py "
                f"-i {shlex.quote(str(src_file))} "
                f"-o {shlex.quote(str(tgt_file))}"
            )
        else:
            raise ValueError(f"Unknown script_type: {script_type}")
        
        commands.append(cmd)
    
    return commands


def main():
    parser = argparse.ArgumentParser(
        description='对比数据集并找出缺失文件,生成或执行转换命令'
    )
    parser.add_argument(
        '--src_folder',
        type=str,
        required=True,
        help='源数据集文件夹路径(如 data/TWIST2_dataset/AMASS_g1)'
    )
    parser.add_argument(
        '--tgt_folder',
        type=str,
        required=True,
        help='目标数据集文件夹路径(如 data/TWIST2_dataset/AMASS_taks_t1)'
    )
    parser.add_argument(
        '--src_robot',
        type=str,
        default='unitree_g1',
        help='源机器人类型(默认: unitree_g1)'
    )
    parser.add_argument(
        '--tgt_robot',
        type=str,
        default='taks_t1',
        help='目标机器人类型(默认: taks_t1)'
    )
    parser.add_argument(
        '--script_type',
        type=str,
        default='robot_to_robot',
        choices=['robot_to_robot', 'bvh_to_robot', 'npz_to_taks_t1', 'npz_to_g1'],
        help='转换脚本类型(默认: robot_to_robot)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='输出命令到文件(可选,不指定则打印到屏幕)'
    )
    parser.add_argument(
        '--execute',
        action='store_true',
        help='直接执行转换命令而不是仅生成命令'
    )
    
    args = parser.parse_args()
    
    src_folder = Path(args.src_folder)
    tgt_folder = Path(args.tgt_folder)
    
    # 检查源文件夹是否存在
    if not src_folder.exists():
        print(f"错误: 源文件夹不存在: {src_folder}")
        return
    
    # 对比数据集
    print(f"正在对比数据集...")
    print(f"  源文件夹: {src_folder}")
    print(f"  目标文件夹: {tgt_folder}")
    
    missing_files = compare_datasets(src_folder, tgt_folder)
    
    if not missing_files:
        print("\n✓ 目标数据集完整,没有缺失文件!")
        return
    
    print(f"\n找到 {len(missing_files)} 个缺失文件:")
    for i, file in enumerate(missing_files[:10], 1):
        print(f"  {i}. {file}")
    if len(missing_files) > 10:
        print(f"  ... 还有 {len(missing_files) - 10} 个文件")
    
    # 生成转换命令
    print(f"\n正在生成转换命令...")
    commands = generate_conversion_commands(
        src_folder,
        tgt_folder,
        missing_files,
        args.src_robot,
        args.tgt_robot,
        args.script_type
    )
    
    # 输出或执行命令
    if args.execute:
        print(f"\n开始执行转换命令...")
        import subprocess
        for i, cmd in enumerate(commands, 1):
            print(f"\n[{i}/{len(commands)}] 执行: {cmd}")
            result = subprocess.run(cmd, shell=True)
            if result.returncode != 0:
                print(f"  ✗ 命令执行失败!")
            else:
                print(f"  ✓ 完成")
    elif args.output:
        output_file = Path(args.output)
        with open(output_file, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write("# 自动生成的转换命令\n\n")
            for cmd in commands:
                f.write(cmd + "\n")
        print(f"\n✓ 转换命令已保存到: {output_file}")
        print(f"  共 {len(commands)} 条命令")
    else:
        print(f"\n生成的转换命令:")
        print("=" * 80)
        for cmd in commands:
            print(cmd)
        print("=" * 80)
        print(f"\n共 {len(commands)} 条命令")


if __name__ == '__main__':
    main()
