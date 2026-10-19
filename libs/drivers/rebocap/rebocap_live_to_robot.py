from __future__ import annotations

import argparse
import os
import pickle
import signal
import sys
import threading
import time
from pathlib import Path

import mujoco as mj
import numpy as np
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.utils.lafan_vendor.utils import quat_fk, quat_mul
from rebocap_to_bvh import (
    _LR_NO_OFFSET,
    _LR_WITH_OFFSET,
    _extract_smpl_quats,
    _globals_to_locals,
    make_bone_smpl_lookup,
    qmul,
)
from rebocap_udp_receiver import LatestRebocapUdpReceiver
from scipy.spatial.transform import Rotation as sRot

_HERE = Path(__file__).resolve()

_R_AXIS = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
_R_AXIS_T = _R_AXIS.T
_qxyzw = sRot.from_matrix(_R_AXIS).as_quat()
_AXIS_QUAT_WXYZ = np.array(
    [_qxyzw[3], _qxyzw[0], _qxyzw[1], _qxyzw[2]], dtype=np.float64
)
_BVH_SCALE = 100.0
_HUMAN_HEIGHT_MIXAMO = 1.75


def parse_skeleton(bvh_path: Path):
    text = bvh_path.read_text()
    motion_idx = text.find("\nMOTION")
    header = text[:motion_idx] if motion_idx >= 0 else text

    names: list[str] = []
    parents: list[int] = []
    offsets: list[list[float]] = []
    stack: list[int] = []
    pending_name: str | None = None
    in_end_site = False

    for raw in header.splitlines():
        s = raw.strip()
        if s.startswith(("ROOT ", "JOINT ")):
            pending_name = s.split(None, 1)[1]
        elif s.startswith("End Site"):
            in_end_site = True
        elif s == "{":
            if in_end_site:
                continue
            idx = len(names)
            names.append(pending_name)
            parents.append(stack[-1] if stack else -1)
            offsets.append([0.0, 0.0, 0.0])
            stack.append(idx)
            pending_name = None
        elif s == "}":
            if in_end_site:
                in_end_site = False
            elif stack:
                stack.pop()
        elif s.startswith("OFFSET") and not in_end_site and stack:
            p = s.split()
            offsets[stack[-1]] = [float(p[1]), float(p[2]), float(p[3])]

    return names, parents, np.asarray(offsets, dtype=np.float64)


def _build_local_quats(payload, bone_smpl, lr_table, rotation_mode):
    if rotation_mode == "auto":
        rotation_mode = getattr(payload, "rotation_mode", "local")
    if rotation_mode not in ("local", "global"):
        raise ValueError(f"rotation_mode must be local/global, got {rotation_mode!r}")
    quats = _extract_smpl_quats(payload)
    if rotation_mode == "global":
        quats = _globals_to_locals(quats)

    n = len(bone_smpl)
    out = np.zeros((n, 4), dtype=np.float64)
    out[:, 0] = 1.0
    for i, smpl_name in enumerate(bone_smpl):
        if smpl_name is None:
            continue
        q = quats.get(smpl_name)
        if q is None:
            continue
        L, R = lr_table[smpl_name]
        out[i] = qmul(qmul(L, q), R)
    return out


def payload_to_frame_dict(
    payload, names_stripped, parents, base_offsets, bone_smpl, lr_table, rotation_mode
):
    local_quats = _build_local_quats(payload, bone_smpl, lr_table, rotation_mode)
    local_pos = base_offsets.copy()
    rt = getattr(payload, "root_translation", None)
    if rt:
        local_pos[0, 0] = float(rt[0]) * _BVH_SCALE
        local_pos[0, 1] = float(rt[2]) * _BVH_SCALE
        local_pos[0, 2] = -float(rt[1]) * _BVH_SCALE
    else:
        local_pos[0, :] = 0.0

    gq, gp = quat_fk(local_quats, local_pos, parents)
    world_pos = gp @ _R_AXIS_T / _BVH_SCALE
    world_quat = quat_mul(_AXIS_QUAT_WXYZ, gq)

    frame = {n: (world_pos[i], world_quat[i]) for i, n in enumerate(names_stripped)}
    if "LeftFoot" in frame and "LeftToeBase" in frame:
        frame["LeftFootMod"] = (frame["LeftFoot"][0], frame["LeftToeBase"][1])
    if "RightFoot" in frame and "RightToeBase" in frame:
        frame["RightFootMod"] = (frame["RightFoot"][0], frame["RightToeBase"][1])
    return frame


def _model_has_free_root(model) -> bool:
    return model.njnt > 0 and int(model.jnt_type[0]) == int(mj.mjtJoint.mjJNT_FREE)


def _split_qpos(qpos: np.ndarray, model):
    qpos = np.asarray(qpos)
    if _model_has_free_root(model):
        return qpos[:3].copy(), qpos[3:7].copy(), qpos[7:].copy()
    rp = model.body_pos[1].copy().astype(qpos.dtype)
    rr = model.body_quat[1].copy().astype(qpos.dtype)
    return rp, rr, qpos.copy()


class QposStream:
    __slots__ = ("dim", "fp", "n", "tmp_path")

    def __init__(self, save_path: str, dim: int):
        self.dim = dim
        self.tmp_path = save_path + ".qpos.tmp"
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        self.fp = open(self.tmp_path, "wb", buffering=1024 * 1024)
        self.n = 0

    def write(self, qpos: np.ndarray) -> None:

        self.fp.write(np.ascontiguousarray(qpos, dtype=np.float64).tobytes())
        self.n += 1

    def close(self) -> None:
        if not self.fp.closed:
            self.fp.close()

    def load(self) -> np.ndarray:
        self.close()
        if self.n == 0:
            return np.empty((0, self.dim), dtype=np.float64)
        return np.fromfile(self.tmp_path, dtype=np.float64).reshape(self.n, self.dim)

    def cleanup(self) -> None:
        try:
            os.remove(self.tmp_path)
        except OSError:
            pass


def _install_stop() -> threading.Event:
    ev = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: ev.set())
    if sys.platform != "win32":
        signal.signal(signal.SIGTERM, lambda *_: ev.set())
    return ev


def _finalize_save(
    stream: QposStream, model, xml_file: str, args: argparse.Namespace
) -> None:
    import torch
    from general_motion_retargeting.kinematics_model import KinematicsModel

    qpos_all = stream.load()
    nf = qpos_all.shape[0]
    if nf == 0:
        print("[save] 0 frames, 跳过")
        stream.cleanup()
        return

    rp_rr_dp = [_split_qpos(q, model) for q in qpos_all]
    root_pos = np.array([t[0] for t in rp_rr_dp])
    root_rot = np.array([t[1][[1, 2, 3, 0]] for t in rp_rr_dp])
    dof_pos = np.array([t[2] for t in rp_rr_dp])

    root_pos[:, 0] -= root_pos[0, 0]
    root_pos[:, 1] -= root_pos[0, 1]

    km = KinematicsModel(xml_file, device="cpu")
    ip = torch.zeros((nf, 3))
    ir = torch.zeros((nf, 4))
    ir[:, -1] = 1.0
    local_body_pos, _ = km.forward_kinematics(ip, ir, torch.from_numpy(dof_pos).float())

    data = {
        "fps": int(args.motion_fps),
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": dof_pos,
        "local_body_pos": local_body_pos.detach().cpu().numpy(),
        "link_body_list": km.body_names,
    }
    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    with open(args.save_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    stream.cleanup()


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Rebocap UDP -> robot 实时重定向")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=9010)
    ap.add_argument(
        "--template",
        default=str(_HERE.parent.parent.parent.parent / "2.bvh"),
        help="Mixamo BVH 模板 (提供 52 骨骼层级 + offsets)",
    )
    ap.add_argument(
        "--robot",
        choices=[
            "unitree_g1",
            "unitree_g1_with_hands",
            "booster_t1",
            "stanford_toddy",
            "fourier_n1",
            "engineai_pm01",
            "pal_talos",
            "taks_t1",
            "semi_taks_t1",
            "semi_taks_lv1",
            "semi_taks_lv1_chassis",
        ],
        default="semi_taks_t1",
    )
    ap.add_argument("--motion_fps", type=int, default=30)
    ap.add_argument("--max-stale", type=float, default=0.5)
    ap.add_argument(
        "--rotation-mode", choices=("auto", "local", "global"), default="auto"
    )
    ap.add_argument("--no-rest-offset", action="store_true")
    ap.add_argument("--rate_limit", action="store_true", default=False)
    ap.add_argument("--record_video", action="store_true", default=False)
    ap.add_argument("--video_path", default="videos/rebocap_live.mp4")
    ap.add_argument("--save_path", default=None)
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    template_path = Path(args.template)
    names_full, parents, offsets = parse_skeleton(template_path)
    names_stripped = [n.split(":", 1)[-1] for n in names_full]
    bone_smpl = make_bone_smpl_lookup(names_full)
    lr_table = _LR_NO_OFFSET if args.no_rest_offset else _LR_WITH_OFFSET

    print(
        f"[init] template={template_path}  bones={len(names_full)}  robot={args.robot}"
    )

    retargeter = GMR(
        src_human="bvh_mixamo",
        tgt_robot=args.robot,
        actual_human_height=_HUMAN_HEIGHT_MIXAMO,
    )
    viewer = RobotMotionViewer(
        robot_type=args.robot,
        motion_fps=args.motion_fps,
        transparent_robot=0,
        record_video=args.record_video,
        video_path=args.video_path,
    )

    receiver = LatestRebocapUdpReceiver(host=args.host, port=args.port)
    receiver.start()
    print(
        f"[udp ] listening on {args.host}:{args.port}  motion_fps={args.motion_fps}  "
        f"rate_limit={args.rate_limit}  rot_mode={args.rotation_mode}"
    )

    stream = (
        QposStream(args.save_path, int(retargeter.model.nq)) if args.save_path else None
    )
    stop = _install_stop()
    last_seq = -1
    last_log = time.perf_counter()
    rendered_in_sec = 0
    started_at = last_log

    try:
        while not stop.is_set():
            payload = receiver.get_latest_payload(timeout=0.2, max_age=args.max_stale)
            if payload is None:
                continue
            seq = receiver.seq
            if seq == last_seq:
                continue
            last_seq = seq

            frame = payload_to_frame_dict(
                payload,
                names_stripped,
                parents,
                offsets,
                bone_smpl,
                lr_table,
                args.rotation_mode,
            )
            qpos = retargeter.retarget(frame)
            rp, rr, dp = _split_qpos(qpos, retargeter.model)
            viewer.step(
                root_pos=rp,
                root_rot=rr,
                dof_pos=dp,
                human_motion_data=retargeter.scaled_human_data,
                rate_limit=args.rate_limit,
                follow_camera=True,
            )
            if stream is not None:
                stream.write(qpos)
            rendered_in_sec += 1

            now = time.perf_counter()
            if now - last_log >= 1.0:
                print(
                    f"[live] frames={rendered_in_sec}/s  total={int(now - started_at)}s  "
                    f"udp_seq={receiver.seq}  dropped={receiver.dropped}",
                    flush=True,
                )
                rendered_in_sec = 0
                last_log = now
    except KeyboardInterrupt:
        pass
    finally:
        receiver.stop()
        viewer.close()
        if stream is not None:
            stream.close()
            if stream.n > 0:
                print(f"[save] {stream.n} frames -> {args.save_path}")
                _finalize_save(stream, retargeter.model, retargeter.xml_file, args)
                print(f"[save] done. size={os.path.getsize(args.save_path)}B")
            else:
                stream.cleanup()


if __name__ == "__main__":
    main()
