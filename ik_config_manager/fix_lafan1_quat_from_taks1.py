import argparse
import json
import pathlib
import sys

import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as Rot

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from utils.fk_solver import MuJoCoFK
from utils.data_processor import load_robot_init
from general_motion_retargeting.params import IK_CONFIG_DICT, ROBOT_XML_DICT


def qwxyz_to_R(qw):
    return Rot.from_quat([qw[1], qw[2], qw[3], qw[0]])


def R_to_qwxyz(r):
    q = r.as_quat()
    return [float(q[3]), float(q[0]), float(q[1]), float(q[2])]


def body_world_rots(robot_xml, tpose_json):
    fk = MuJoCoFK(robot_xml)
    bn = [mujoco.mj_id2name(fk.model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(1, fk.model.nbody)]
    rp, rr, jd, _ = load_robot_init(tpose_json)
    q = fk.build_qpos(rp, rr, jd)
    _, R = fk.get_specific_body_positions(q, bn)
    return {n: Rot.from_matrix(R[i]) for i, n in enumerate(bn)}


def derive_lafan1_rot():
    t1cfg = json.load(open(HERE.parent / "general_motion_retargeting/ik_configs/bvh_lafan1_to_taks_t1.json"))
    t1_rot = body_world_rots(str(ROBOT_XML_DICT["taks_t1"].with_name(ROBOT_XML_DICT["taks_t1"].name.replace("scene_", ""))),
                             str(HERE / "pose_inits" / "taks_t1_tpose.json"))
    lafan = {}
    for rl, e in t1cfg["ik_match_table1"].items():
        if rl in t1_rot:
            lafan[e[0]] = t1_rot[rl] * qwxyz_to_R(e[4]).inv()
    lafan["LeftFoot"] = lafan["LeftFootMod"]
    lafan["Spine1"] = lafan["Spine2"]
    lafan["Head"] = lafan["Spine2"]
    return lafan


def fix(robot):
    lafan = derive_lafan1_rot()
    rzyaw = Rot.from_euler("z", -90, degrees=True)
    robot_xml = str(ROBOT_XML_DICT[robot].with_name(ROBOT_XML_DICT[robot].name.replace("scene_", "")))
    tpose = str(HERE / "pose_inits" / (robot + "_tpose.json"))
    semi_rot = body_world_rots(robot_xml, tpose)

    cfg_path = str(IK_CONFIG_DICT["bvh_lafan1"][robot])
    cfg = json.load(open(cfg_path))
    for t in ["ik_match_table1", "ik_match_table2"]:
        for rl, e in cfg[t].items():
            bone = e[0]
            if robot == "semi_taks_lv1_chassis" and rl == "base_link":
                e[4] = [1.0, 0.0, 0.0, 0.0]
                continue
            if bone in lafan and rl in semi_rot:
                e[4] = R_to_qwxyz(lafan[bone].inv() * rzyaw * semi_rot[rl])
    json.dump(cfg, open(cfg_path, "w"), indent=2, ensure_ascii=False)
    print(f"[OK] {robot} LAFAN1 quat 偏移已用 taks_t1 delta + Rz(-90) 修正 -> {cfg_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="用 taks_t1 LAFAN1 干净偏移反推 LAFAN1 T-pose,修正 semi_taks_lv1(_chassis) 的 quat 偏移(保留 pos/scale)")
    ap.add_argument("--robot", default="all", choices=["semi_taks_lv1", "semi_taks_lv1_chassis", "all"])
    args = ap.parse_args()
    robots = ["semi_taks_lv1", "semi_taks_lv1_chassis"] if args.robot == "all" else [args.robot]
    for r in robots:
        fix(r)