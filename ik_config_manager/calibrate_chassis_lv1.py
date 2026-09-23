import argparse
import json
import pathlib
import sys

import numpy as np
import mujoco

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.params import IK_CONFIG_DICT, ROBOT_XML_DICT
from general_motion_retargeting.utils.lafan1 import load_bvh_file
from utils.fk_solver import MuJoCoFK
from utils.data_processor import load_robot_init, scale_human_data, write_all_data_to_ik
from utils.compute_offsets import compute_quaternion_offsets, compute_position_offsets
from utils.optimize_human_scale import optimize_human_scale_table

LR_PAIRS = [
    ("LeftShoulder", "RightShoulder"),
    ("LeftArm", "RightArm"),
    ("LeftForeArm", "RightForeArm"),
    ("LeftHand", "RightHand"),
    ("LeftUpLeg", "RightUpLeg"),
    ("LeftLeg", "RightLeg"),
    ("LeftFoot", "RightFoot"),
    ("LeftToeBase", "RightToeBase"),
]


def mirror_left_to_right(frame):
    out = {}
    for k, v in frame.items():
        out[k] = [np.array(v[0], dtype=np.float64), np.array(v[1], dtype=np.float64)]
    for left, right in LR_PAIRS:
        if left in out:
            pos, q = out[left]
            out[right] = [
                np.array([-pos[0], pos[1], pos[2]], dtype=np.float64),
                np.array([q[0], q[1], -q[2], -q[3]], dtype=np.float64),
            ]
    return out


def calibrate(robot, tpose_json, bvh_file, fmt="mixamo"):
    src_human = f"bvh_{fmt}"
    xml_file = str(ROBOT_XML_DICT[robot].with_name(ROBOT_XML_DICT[robot].name.replace("scene_", "")))
    target = str(IK_CONFIG_DICT[src_human][robot])

    frames, h = load_bvh_file(str(bvh_file), format=fmt)
    f0 = mirror_left_to_right(frames[0]) if fmt == "mixamo" else frames[0]

    GMR(src_human=src_human, tgt_robot=robot, verbose=False)

    with open(target, "r", encoding="utf-8") as f:
        ik_cfg = json.load(f)

    src_frame_rot = ik_cfg.get("src_frame_rot", None)
    if src_frame_rot is not None:
        from scipy.spatial.transform import Rotation as Rot
        rot = Rot.from_quat([src_frame_rot[1], src_frame_rot[2], src_frame_rot[3], src_frame_rot[0]])
        root_pos = np.array(f0["Hips"][0], dtype=np.float64)
        rotated = {}
        for name, (pos, quat) in f0.items():
            new_pos = rot.apply(np.array(pos, dtype=np.float64) - root_pos) + root_pos
            q_xyzw = [quat[1], quat[2], quat[3], quat[0]]
            new_q = (rot * Rot.from_quat(q_xyzw)).as_quat()
            rotated[name] = [new_pos, np.array([new_q[3], new_q[0], new_q[1], new_q[2]])]
        f0 = rotated

    rp, rr, jd, _ = load_robot_init(str(tpose_json))
    fk = MuJoCoFK(xml_file)
    qpos = fk.build_qpos(rp, rr, jd)

    all_body_names = [
        mujoco.mj_id2name(fk.model, mujoco.mjtObj.mjOBJ_BODY, i)
        for i in range(1, fk.model.nbody)
    ]

    centers, Rs = fk.get_specific_body_positions(qpos, all_body_names)

    ratio = h / ik_cfg["human_height_assumption"]
    optimized = ik_cfg["human_scale_table"]
    gmr_scales = {k: float(v / ratio) for k, v in optimized.items()}
    scaled = scale_human_data(f0, "Hips", optimized)

    quat_offsets, miss_q = compute_quaternion_offsets(
        scaled, centers, Rs, all_body_names, ik_cfg
    )
    print("[quat] missing:", miss_q)

    rl_idx = {n: i for i, n in enumerate(all_body_names)}
    rrn = ik_cfg.get("robot_root_name", "Hips")
    robot_root_pos = centers[rl_idx[rrn]] if rrn in rl_idx else centers[0]
    human_root_pos = np.array(scaled["Hips"][0], dtype=np.float64)
    aligned = [c - robot_root_pos + human_root_pos for c in centers]

    pos_offsets, miss_p = compute_position_offsets(
        scaled, aligned, Rs, all_body_names, ik_cfg, quat_offsets
    )
    print("[pos] missing:", miss_p)

    out = write_all_data_to_ik(target, target, gmr_scales, pos_offsets, quat_offsets)
    print(f"[OK] {robot} -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Semi_Taks_LV1_chassis bvh 标定(用 mujoco 全 body 名绕过 attach)")
    ap.add_argument("--robot", default="semi_taks_lv1_chassis")
    ap.add_argument(
        "--tpose",
        default=str(HERE / "pose_inits" / "semi_taks_lv1_chassis_tpose.json"),
    )
    ap.add_argument("--bvh_file", default=str(HERE.parent / "2.bvh"))
    ap.add_argument("--format", default="mixamo", choices=["mixamo", "lafan1"])
    args = ap.parse_args()
    calibrate(args.robot, args.tpose, args.bvh_file, args.format)