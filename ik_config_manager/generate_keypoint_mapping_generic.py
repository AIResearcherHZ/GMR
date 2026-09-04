"""
通用 generate_keypoint_mapping 脚本。
基于 SMPLX T-pose 数据 + 关节名映射，为任意格式生成优化后的 ik_config。
支持格式: bvh_nokov, fbx, fbx_offline, xrobot, bvh_xsens 等。
"""
import argparse
import pathlib
import os
import time
import json

import numpy as np
from scipy.spatial.transform import Rotation as R

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.utils.smpl import (
    load_smplx_file,
    get_smplx_data_offline_fast,
)

from rich import print

from utils.optimize_human_scale import optimize_human_scale_table
from utils.fk_solver import MuJoCoFK
from utils.data_processor import load_robot_init, scale_human_data, align_robot_data, offset_human_data, write_all_data_to_ik
from utils.compute_offsets import compute_position_offsets, compute_quaternion_offsets


# SMPLX关节名 -> 语义标签
SMPLX_SEMANTIC = {
    "pelvis": "root",
    "left_hip": "l_hip",
    "left_knee": "l_knee",
    "left_foot": "l_foot",
    "right_hip": "r_hip",
    "right_knee": "r_knee",
    "right_foot": "r_foot",
    "spine3": "torso",
    "left_shoulder": "l_shoulder",
    "left_elbow": "l_elbow",
    "left_wrist": "l_wrist",
    "right_shoulder": "r_shoulder",
    "right_elbow": "r_elbow",
    "right_wrist": "r_wrist",
}

# 从ik_config中提取human关节名到语义标签的映射
SEMANTIC_ORDER = [
    "root", "l_hip", "l_knee", "l_foot",
    "r_hip", "r_knee", "r_foot", "torso",
    "l_shoulder", "l_elbow", "l_wrist",
    "r_shoulder", "r_elbow", "r_wrist",
]


def build_name_mapping(ik_config):
    """从ik_config中提取human关节名，建立与SMPLX关节名的映射。"""
    # 提取目标格式的human关节名（按robot link顺序）
    target_human_names = []
    for entry in ik_config["ik_match_table1"].values():
        target_human_names.append(entry[0])

    # SMPLX的human关节名（同样顺序）
    smplx_cfg_path = pathlib.Path(__file__).parent.parent / "general_motion_retargeting" / "ik_configs" / "smplx_to_taks_t1.json"
    # 不依赖特定文件，直接用语义顺序
    smplx_names_ordered = list(SMPLX_SEMANTIC.keys())

    # 建立映射: smplx_name -> target_name (按位置对应)
    mapping = {}
    for smplx_name, target_name in zip(smplx_names_ordered, target_human_names):
        mapping[smplx_name] = target_name

    return mapping


def remap_frame(smplx_frame, name_mapping, target_root_name):
    """将SMPLX格式的帧数据重命名为目标格式。"""
    remapped = {}
    for smplx_name, target_name in name_mapping.items():
        if smplx_name in smplx_frame:
            remapped[target_name] = smplx_frame[smplx_name]
    # 复制所有未映射的关节
    for key, value in smplx_frame.items():
        mapped_targets = set(name_mapping.keys())
        if key not in mapped_targets:
            # 保留原始数据（可能有额外关节）
            if key not in [name_mapping.get(k) for k in mapped_targets]:
                remapped[key] = value
    return remapped


if __name__ == "__main__":
    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser(description="通用 generate_keypoint_mapping，基于SMPLX T-pose数据")
    parser.add_argument(
        "--smplx_file",
        help="SMPLX T-pose文件路径",
        type=str,
        default="ik_config_manager/SMPLX_TPOSE_UNIFIED_AMASS.npz",
    )
    parser.add_argument(
        "--src_human",
        help="目标人体数据格式",
        type=str,
        required=True,
        choices=["bvh_nokov", "fbx", "fbx_offline", "xrobot", "bvh_xsens"],
    )
    parser.add_argument(
        "--robot",
        choices=["unitree_g1", "unitree_g1_with_hands", "unitree_h1", "unitree_h1_2",
                 "booster_t1", "booster_t1_29dof", "stanford_toddy", "fourier_n1",
                 "engineai_pm01", "kuavo_s45", "hightorque_hi", "galaxea_r1pro",
                 "berkeley_humanoid_lite", "booster_k1", "pnd_adam_lite",
                 "openloong", "tienkung", "joyin", "joyin_add",
                 "roboparty_atom01", "roboparty_atom01_long_base_link", "roboparty_atom02", "taks_t1"],
        default="taks_t1",
    )
    parser.add_argument("--loop", default=True, action="store_true")
    parser.add_argument("--record_video", default=False, action="store_true")
    parser.add_argument("--rate_limit", default=False, action="store_true")
    parser.add_argument("--save_path", default=None)
    parser.add_argument("--robot_qpos_init", type=str,
                        default="ik_config_manager/pose_inits/taks_t1_tpose.json")
    parser.add_argument("--ik_config_in", type=str, required=True,
                        help="输入 IK 配置路径")
    parser.add_argument("--ik_config_out", type=str, required=True,
                        help="输出 IK 配置路径")

    args = parser.parse_args()

    SMPLX_FOLDER = HERE / ".." / "assets" / "body_models"

    # 加载SMPLX T-pose数据
    print(f"=== 加载SMPLX T-pose数据: {args.smplx_file} ===")
    smplx_data, body_model, smplx_output, actual_human_height = load_smplx_file(
        args.smplx_file, SMPLX_FOLDER
    )
    tgt_fps = 30
    smplx_data_frames, aligned_fps = get_smplx_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=tgt_fps
    )

    # 加载目标格式的IK配置
    with open(args.ik_config_in, "r", encoding="utf-8") as f:
        ik_cfg_tmp = json.load(f)

    # 建立SMPLX -> 目标格式的关节名映射
    name_mapping = build_name_mapping(ik_cfg_tmp)
    target_root_name = ik_cfg_tmp["human_root_name"]
    print(f"=== 关节名映射 ({args.src_human}) ===")
    for s, t in name_mapping.items():
        print(f"  {s} -> {t}")

    # 将SMPLX T-pose帧转换为目标格式
    smplx_first_frame = smplx_data_frames[0]
    first_frame_data = remap_frame(smplx_first_frame, name_mapping, target_root_name)

    # 步骤1: 参数生成阶段
    print(f"=== 开始参数生成阶段 (src={args.src_human}) ===")

    # 用smplx初始化GMR（只为了获取xml_file）
    retarget = GMR(
        actual_human_height=actual_human_height,
        src_human="smplx",
        tgt_robot=args.robot,
    )

    # 加载固定机器人初始姿态
    fixed_root_pos, fixed_root_rot, fixed_dof_pos, joint_match = load_robot_init(args.robot_qpos_init)

    # 初始化FK求解器
    fk_solver = MuJoCoFK(retarget.xml_file)
    joint_order = fk_solver.joint_order

    # 映射关节顺序
    nd_expected = len(joint_order)
    vec = np.zeros(nd_expected, dtype=np.float32)
    assigned = 0
    for i, joint_name in enumerate(joint_order):
        if i < nd_expected and joint_name in fixed_dof_pos.keys():
            vec[i] = fixed_dof_pos[joint_name]
            assigned += 1
    print(f"[INFO] 映射关节: {assigned}/{len(fixed_dof_pos)}")
    fixed_dof_pos = vec

    # 计算FK得到机器人T-pose
    qpos_fk = np.concatenate([
        fixed_root_pos.astype(np.float64),
        fixed_root_rot.astype(np.float64),
        fixed_dof_pos.astype(np.float64)
    ], axis=0)
    centers, Rs_fk = fk_solver.get_specific_body_positions(qpos_fk, fk_solver.body_names)

    # 优化缩放系数
    ratio = actual_human_height / ik_cfg_tmp["human_height_assumption"]
    print("=== 优化缩放系数 ===")
    optimized_scales = optimize_human_scale_table(
        human_data=first_frame_data,
        robot_centers=centers,
        body_names=fk_solver.body_names,
        ik_config=ik_cfg_tmp,
        human_root_name=target_root_name,
        initial_scales=ik_cfg_tmp.get("human_scale_table", None),
        bounds=(0.1, 10.0),
        max_iter=10000,
        device='cpu',
        plot_loss=False,
        plot_save_path=None
    )

    # 创建用于GMR的缩放系数
    gmr_scales = {}
    for key, value in optimized_scales.items():
        gmr_scales[key] = float(value / ratio)

    # 确保optimized_scales包含human_root_name
    if target_root_name not in optimized_scales:
        for k, v in list(optimized_scales.items()):
            if k.lower() == target_root_name.lower():
                optimized_scales[target_root_name] = v
                gmr_scales[target_root_name] = gmr_scales.get(k, float(v / ratio))
                break

    # 缩放人体数据
    scaled_human_data = scale_human_data(
        first_frame_data,
        target_root_name,
        optimized_scales
    )

    # 计算四元数偏移
    print("=== 计算四元数偏移 ===")
    quat_offsets, missing_links = compute_quaternion_offsets(
        scaled_human_data,
        centers, Rs_fk, fk_solver.body_names,
        ik_cfg_tmp
    )

    # 对齐机器人数据到人体根节点
    human_root_pos = np.array(scaled_human_data[target_root_name][0], dtype=np.float64)
    robot_root_name = ik_cfg_tmp.get("robot_root_name", "pelvis")

    robot_link_indices = {}
    for idx, name in enumerate(fk_solver.body_names):
        if not isinstance(name, str):
            name = name.decode("utf-8")
        robot_link_indices[name] = idx

    if robot_root_name in robot_link_indices:
        robot_root_idx = robot_link_indices[robot_root_name]
        robot_root_pos = centers[robot_root_idx]
    else:
        robot_root_pos = centers[0] if len(centers) > 0 else np.zeros(3)

    aligned_robot_centers = align_robot_data(
        centers, robot_root_pos, human_root_pos
    )

    # 计算位置偏移
    print("=== 计算位置偏移 ===")
    pos_offsets, missing_links = compute_position_offsets(
        scaled_human_data,
        aligned_robot_centers, Rs_fk, fk_solver.body_names,
        ik_cfg_tmp, quat_offsets
    )

    # 保存所有优化数据
    print("=== 保存优化配置 ===")
    output_file = write_all_data_to_ik(
        ik_config_path=args.ik_config_in,
        output_path=args.ik_config_out,
        human_scale_table=gmr_scales,
        pos_offsets=pos_offsets,
        quat_offsets=quat_offsets
    )

    if output_file:
        print(f"[SUCCESS] {args.src_human} 参数生成完成！")
        print(f"  - 优化了 {len(optimized_scales)} 个缩放系数")
        print(f"  - 计算了 {len(pos_offsets)} 个位置偏移")
        print(f"  - 计算了 {len(quat_offsets)} 个四元数偏移")
        print(f"  - 所有数据已保存到: {output_file}")
    else:
        print(f"[ERROR] {args.src_human} 参数生成失败！")
        exit(1)

    # 步骤2: 可视化验证
    with open(output_file, "r", encoding="utf-8") as f:
        ik_cfg = json.load(f)

    ik_match_table1 = ik_cfg.get("ik_match_table1", {})
    rot_offsets = {}
    for frame_name, entry in ik_match_table1.items():
        body_name, pos_weight, rot_weight, pos_offset, rot_offset = entry
        if pos_weight != 0 or rot_weight != 0:
            pos_offsets[body_name] = np.array(pos_offset)
            rot_offset_xyzw = [rot_offset[1], rot_offset[2], rot_offset[3], rot_offset[0]]
            rot_offsets[body_name] = R.from_quat(rot_offset_xyzw)

    new_human_data = offset_human_data(scaled_human_data, pos_offsets, rot_offsets)

    print("\n=== 开始可视化阶段 ===")

    # 用smplx重定向器做可视化（T-pose数据来自SMPLX）
    retarget_new = GMR(
        actual_human_height=actual_human_height,
        src_human="smplx",
        tgt_robot=args.robot,
    )

    robot_motion_viewer = RobotMotionViewer(
        robot_type=args.robot,
        motion_fps=aligned_fps,
        transparent_robot=0,
        record_video=args.record_video,
        video_path=f"videos/{args.robot}_{args.src_human}_generic.mp4",
    )

    fps_counter = 0
    fps_start_time = time.time()
    fps_display_interval = 2.0

    if args.save_path is not None:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        qpos_list = []

    desired_dt = 1.0 / float(aligned_fps)
    next_frame_time = time.perf_counter()
    i = 0

    while True:
        if args.rate_limit:
            now = time.perf_counter()
            if now < next_frame_time:
                time.sleep(max(0.0, next_frame_time - now))
            next_frame_time += desired_dt

        if args.loop:
            i = (i + 1) % len(smplx_data_frames)
        else:
            i += 1
            if i >= len(smplx_data_frames):
                break

        fps_counter += 1
        current_time = time.time()
        if current_time - fps_start_time >= fps_display_interval:
            actual_fps = fps_counter / (current_time - fps_start_time)
            print(f"Actual rendering FPS: {actual_fps:.2f}")
            fps_counter = 0
            fps_start_time = current_time

        smplx_frame = smplx_data_frames[i]
        qpos = retarget_new.retarget(smplx_frame)

        robot_motion_viewer.step(
            root_pos=scaled_human_data[target_root_name][0],
            root_rot=fixed_root_rot,
            dof_pos=fixed_dof_pos,
            human_motion_data=new_human_data,
            human_pos_offset=np.array([0.0, 0.0, 0.0]),
            show_human_body_name=True,
            rate_limit=args.rate_limit
        )

        if args.save_path is not None:
            qpos_list.append(qpos.copy())

    if args.save_path is not None and qpos_list:
        np.save(args.save_path, np.array(qpos_list))
        print(f"Motion saved to {args.save_path}")

    robot_motion_viewer.close()
    print("=== 可视化完成 ===")
