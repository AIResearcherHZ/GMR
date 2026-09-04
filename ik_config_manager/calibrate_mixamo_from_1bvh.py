"""
基于 1.bvh frame 0 的 Mixamo / RoboCAP IK 配置自动校准。

由于 1.bvh frame 0 右臂未在 T-pose、左右不对称（这是 1.bvh 数据本身的问题），
本脚本将"左侧（已是 T-pose）镜像到右侧"得到一个真正对称的合成 T-pose，
再用该合成姿态作为参考调用项目标准的校准逻辑（与 generate_keypoint_mapping_bvh.py 一致）。

使用：
    python ik_config_manager/calibrate_mixamo_from_1bvh.py            # 校准全部三个机器人
    python ik_config_manager/calibrate_mixamo_from_1bvh.py --robot taks_t1
"""
import argparse
import copy
import json
import pathlib
import shutil
import sys

import mujoco
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.utils.lafan1 import load_bvh_file
from utils.compute_offsets import compute_position_offsets, compute_quaternion_offsets
from utils.data_processor import (
    load_robot_init,
    scale_human_data,
    write_all_data_to_ik,
)
from utils.fk_solver import MuJoCoFK
from utils.optimize_human_scale import optimize_human_scale_table

ROOT = HERE.parent
LAFAN1_TEMPLATE = {
    "unitree_g1": ROOT / "general_motion_retargeting/ik_configs/bvh_lafan1_to_g1.json",
    "taks_t1": ROOT / "general_motion_retargeting/ik_configs/bvh_lafan1_to_taks_t1.json",
    # semi_taks_t1 / semi_taks_lv1 没有 LAFAN1 模板，直接复用现有自定义模板
    "semi_taks_t1": ROOT / "general_motion_retargeting/ik_configs/bvh_mixamo_to_semi_taks_t1.json",
    "semi_taks_lv1": ROOT / "general_motion_retargeting/ik_configs/bvh_mixamo_to_semi_taks_lv1.json",
}
TARGET_CFG = {
    "unitree_g1": ROOT / "general_motion_retargeting/ik_configs/bvh_mixamo_to_g1.json",
    "taks_t1": ROOT / "general_motion_retargeting/ik_configs/bvh_mixamo_to_taks_t1.json",
    "semi_taks_t1": ROOT / "general_motion_retargeting/ik_configs/bvh_mixamo_to_semi_taks_t1.json",
    "semi_taks_lv1": ROOT / "general_motion_retargeting/ik_configs/bvh_mixamo_to_semi_taks_lv1.json",
}
TPOSE_JSON = {
    "unitree_g1": ROOT / "ik_config_manager/pose_inits/unitree_g1_tpose.json",
    "taks_t1": ROOT / "ik_config_manager/pose_inits/taks_t1_tpose.json",
    "semi_taks_t1": ROOT / "ik_config_manager/pose_inits/semi_taks_t1_tpose.json",
    "semi_taks_lv1": ROOT / "ik_config_manager/pose_inits/semi_taks_lv1_tpose.json",
}

# 左右对称的关节对（去前缀后的标准命名）
LR_PAIRS = [
    ("LeftShoulder", "RightShoulder"),
    ("LeftArm", "RightArm"),
    ("LeftForeArm", "RightForeArm"),
    ("LeftHand", "RightHand"),
    ("LeftUpLeg", "RightUpLeg"),
    ("LeftLeg", "RightLeg"),
    ("LeftFoot", "RightFoot"),
    ("LeftToeBase", "RightToeBase"),
    ("LeftFootMod", "RightFootMod"),
]


def mirror_left_to_right(frame):
    """沿世界 X=0 平面镜像左侧关节到右侧。

    位置：x -> -x；四元数 (wxyz)：y、z 分量取反。
    """
    out = copy.deepcopy(frame)
    for left, right in LR_PAIRS:
        if left in out:
            pos, q = out[left]
            mp = np.array([-pos[0], pos[1], pos[2]], dtype=np.float64)
            mq = np.array([q[0], q[1], -q[2], -q[3]], dtype=np.float64)
            out[right] = [mp, mq]
    return out


WRIST_END_LINK_REMAP = {
    "taks_t1": {
        "left_wrist_yaw_link": ("left_wrist_pitch_link", [0.0, 0.0225, 0.0333]),
        "right_wrist_yaw_link": ("right_wrist_pitch_link", [0.0, -0.0225, 0.0333]),
    },
    "semi_taks_t1": {
        "left_wrist_yaw_link": ("left_wrist_pitch_link", [0.0, 0.0225, 0.0333]),
        "right_wrist_yaw_link": ("right_wrist_pitch_link", [0.0, -0.0225, 0.0333]),
    },
}


def _remap_wrist_end_link(target_path, robot):
    mapping = WRIST_END_LINK_REMAP.get(robot)
    if not mapping:
        return
    with open(target_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    for tbl_key in ("ik_match_table1", "ik_match_table2"):
        if tbl_key not in cfg:
            continue
        tbl = cfg[tbl_key]
        for old_link, (new_link, delta) in mapping.items():
            if old_link not in tbl or new_link in tbl:
                continue
            entry = tbl.pop(old_link)
            entry[3] = [entry[3][0] + delta[0], entry[3][1] + delta[1], entry[3][2] + delta[2]]
            tbl[new_link] = entry
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def calibrate(robot, bvh_file):
    template = LAFAN1_TEMPLATE[robot]
    target = TARGET_CFG[robot]
    tpose_json = TPOSE_JSON[robot]

    if template != target:
        shutil.copy(template, target)

    frames, h = load_bvh_file(str(bvh_file), format="mixamo")
    f0 = mirror_left_to_right(frames[0])

    retarget = GMR(
        actual_human_height=h, src_human="bvh_mixamo", tgt_robot=robot, verbose=False
    )
    with open(target, "r", encoding="utf-8") as f:
        ik_cfg_tmp = json.load(f)

    fixed_root_pos, fixed_root_rot, fixed_dof_pos, _ = load_robot_init(str(tpose_json))
    _robot_xml = str(pathlib.Path(retarget.xml_file).with_name(pathlib.Path(retarget.xml_file).name.replace("scene_", "")))
    fk = MuJoCoFK(_robot_xml)
    fk.body_names = [mujoco.mj_id2name(fk.model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(1, fk.model.nbody)]
    qpos_fk = fk.build_qpos(fixed_root_pos, fixed_root_rot, fixed_dof_pos)
    centers, Rs = fk.get_specific_body_positions(qpos_fk, fk.body_names)
    ratio = h / ik_cfg_tmp["human_height_assumption"]

    optimized_scales = optimize_human_scale_table(
        human_data=f0,
        robot_centers=centers,
        body_names=fk.body_names,
        ik_config=ik_cfg_tmp,
        human_root_name=retarget.human_root_name,
        initial_scales=ik_cfg_tmp.get("human_scale_table", None),
        bounds=(0.1, 10.0),
        max_iter=10000,
        device="cpu",
        plot_loss=False,
        plot_save_path=None,
    )
    gmr_scales = {k: float(v / ratio) for k, v in optimized_scales.items()}
    scaled = scale_human_data(f0, retarget.human_root_name, optimized_scales)

    quat_offsets, _ = compute_quaternion_offsets(
        scaled, centers, Rs, fk.body_names, ik_cfg_tmp
    )

    rl_idx = {
        (n if isinstance(n, str) else n.decode()): i
        for i, n in enumerate(fk.body_names)
    }
    rrn = ik_cfg_tmp.get("robot_root_name", "pelvis")
    robot_root_pos = centers[rl_idx[rrn]] if rrn in rl_idx else centers[0]
    human_root_pos = np.array(scaled[retarget.human_root_name][0], dtype=np.float64)
    aligned = [c - robot_root_pos + human_root_pos for c in centers]

    pos_offsets, _ = compute_position_offsets(
        scaled, aligned, Rs, fk.body_names, ik_cfg_tmp, quat_offsets
    )

    out = write_all_data_to_ik(
        str(target), str(target), gmr_scales, pos_offsets, quat_offsets
    )
    _remap_wrist_end_link(str(target), robot)
    print(f"[OK] {robot} -> {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bvh_file",
        type=str,
        default=str(ROOT / "2.bvh"),
        help="作为 T-pose 来源的 Mixamo BVH（默认：2.bvh，对称 T-pose）",
    )
    parser.add_argument(
        "--robot",
        type=str,
        choices=["unitree_g1", "taks_t1", "semi_taks_t1", "semi_taks_lv1", "all"],
        default="all",
    )
    args = parser.parse_args()

    robots = (
        ["unitree_g1", "taks_t1", "semi_taks_t1", "semi_taks_lv1"]
        if args.robot == "all"
        else [args.robot]
    )
    for r in robots:
        print(f"\n========== Calibrating {r} (synthetic symmetric T-pose from {args.bvh_file}) ==========")
        calibrate(r, args.bvh_file)
