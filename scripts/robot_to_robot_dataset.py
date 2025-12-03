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
from pathlib import Path
from rich import print
from tqdm import tqdm

from robot_to_robot import load_robot_motion, save_robot_motion, convert_motion


def main():
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
    
    args = parser.parse_args()
    
    if args.src_robot == args.tgt_robot:
        print("[yellow]Warning: Source and target robots are the same.[/yellow]")
        return
    
    # Find all pkl files
    src_folder = Path(args.src_folder)
    tgt_folder = Path(args.tgt_folder)
    
    pkl_files = list(src_folder.glob('*.pkl'))
    
    if not pkl_files:
        print(f"[red]No pkl files found in {src_folder}[/red]")
        return
    
    print(f"[bold]Converting {len(pkl_files)} files[/bold]")
    print(f"  From: {args.src_robot} -> {args.tgt_robot}")
    print(f"  Source: {src_folder}")
    print(f"  Target: {tgt_folder}")
    
    # Create target folder
    tgt_folder.mkdir(parents=True, exist_ok=True)
    
    # Convert each file
    success_count = 0
    fail_count = 0
    
    for pkl_file in tqdm(pkl_files, desc="Converting"):
        try:
            src_data = load_robot_motion(str(pkl_file))
            tgt_data = convert_motion(src_data, args.src_robot, args.tgt_robot)
            
            tgt_file = tgt_folder / pkl_file.name
            save_robot_motion(tgt_data, str(tgt_file))
            success_count += 1
        except Exception as e:
            print(f"[red]Failed to convert {pkl_file.name}: {e}[/red]")
            fail_count += 1
    
    print(f"[bold green]Done! Success: {success_count}, Failed: {fail_count}[/bold green]")


if __name__ == '__main__':
    main()
