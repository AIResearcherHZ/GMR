from __future__ import annotations

import argparse
import math
import os
import signal
import sys
import threading
import time
from collections.abc import Iterable
from pathlib import Path

from rebocap_udp_receiver import LatestRebocapUdpReceiver

Quat = tuple[float, float, float, float]
Vec3 = tuple[float, float, float]
Bone = tuple[str, Vec3]
IDENTITY: Quat = (1.0, 0.0, 0.0, 0.0)


def qmul(a: Quat, b: Quat) -> Quat:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def qconj(q: Quat) -> Quat:
    w, x, y, z = q
    return (w, -x, -y, -z)


def quat_to_zxy_deg(q: Quat) -> Vec3:

    w, x, y, z = q
    sx = 2.0 * (y * z + w * x)
    if sx > 1.0:
        sx = 1.0
    elif sx < -1.0:
        sx = -1.0
    rx = math.asin(sx)
    rz = math.atan2(2.0 * (w * z - x * y), 1.0 - 2.0 * (x * x + z * z))
    ry = math.atan2(2.0 * (w * y - x * z), 1.0 - 2.0 * (x * x + y * y))
    return math.degrees(rz), math.degrees(rx), math.degrees(ry)


SMPL_JOINT_ORDER: tuple[str, ...] = (
    "Pelvis",
    "L_Hip",
    "R_Hip",
    "Spine1",
    "L_Knee",
    "R_Knee",
    "Spine2",
    "L_Ankle",
    "R_Ankle",
    "Spine3",
    "L_Foot",
    "R_Foot",
    "Neck",
    "L_Collar",
    "R_Collar",
    "Head",
    "L_Shoulder",
    "R_Shoulder",
    "L_Elbow",
    "R_Elbow",
    "L_Wrist",
    "R_Wrist",
    "L_Hand",
    "R_Hand",
)
SMPL_PARENT_INDEX: tuple[int, ...] = (
    -1,
    0,
    0,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    9,
    9,
    12,
    13,
    14,
    16,
    17,
    18,
    19,
    20,
    21,
)
_SMPL_NAME_TO_IDX = {n: i for i, n in enumerate(SMPL_JOINT_ORDER)}

REBOCAP_TO_MIXAMO: dict[str, str] = {
    "Pelvis": "mixamorig:Hips",
    "Spine1": "mixamorig:Spine",
    "Spine2": "mixamorig:Spine1",
    "Spine3": "mixamorig:Spine2",
    "Neck": "mixamorig:Neck",
    "Head": "mixamorig:Head",
    "L_Collar": "mixamorig:LeftShoulder",
    "R_Collar": "mixamorig:RightShoulder",
    "L_Shoulder": "mixamorig:LeftArm",
    "R_Shoulder": "mixamorig:RightArm",
    "L_Elbow": "mixamorig:LeftForeArm",
    "R_Elbow": "mixamorig:RightForeArm",
    "L_Wrist": "mixamorig:LeftHand",
    "R_Wrist": "mixamorig:RightHand",
    "L_Hip": "mixamorig:LeftUpLeg",
    "R_Hip": "mixamorig:RightUpLeg",
    "L_Knee": "mixamorig:LeftLeg",
    "R_Knee": "mixamorig:RightLeg",
    "L_Ankle": "mixamorig:LeftFoot",
    "R_Ankle": "mixamorig:RightFoot",
    "L_Foot": "mixamorig:LeftToeBase",
    "R_Foot": "mixamorig:RightToeBase",
}

_S = math.sqrt(0.5)
_Q_FIX: Quat = (_S, -_S, 0.0, 0.0)
_Q_FIX_INV: Quat = (_S, _S, 0.0, 0.0)

Q_ALIGN_SMPL_TO_MIXAMO: dict[str, Quat] = {
    "Pelvis": IDENTITY,
    "Spine1": IDENTITY,
    "Spine2": IDENTITY,
    "Spine3": IDENTITY,
    "Neck": IDENTITY,
    "Head": IDENTITY,
    "L_Collar": (+0.5, +0.5, +0.5, -0.5),
    "R_Collar": (+0.5, +0.5, -0.5, +0.5),
    "L_Shoulder": (+0.5, +0.5, +0.5, -0.5),
    "R_Shoulder": (+0.5, +0.5, -0.5, +0.5),
    "L_Elbow": (+0.5, +0.5, +0.5, -0.5),
    "R_Elbow": (+0.5, +0.5, -0.5, +0.5),
    "L_Wrist": (+0.5, +0.5, +0.5, -0.5),
    "R_Wrist": (+0.5, +0.5, -0.5, +0.5),
    "L_Hip": (0.0, 0.0, 0.0, 1.0),
    "R_Hip": (0.0, 0.0, 0.0, 1.0),
    "L_Knee": (0.0, 0.0, 0.0, 1.0),
    "R_Knee": (0.0, 0.0, 0.0, 1.0),
    "L_Ankle": (+0.0329465681, -0.0156536438, +0.4596812682, +0.8873345585),
    "R_Ankle": (-0.0329512081, +0.0156518976, +0.4596813656, +0.8873343666),
    "L_Foot": (+0.0362878733, -0.0037028146, +0.7305516176, +0.6818825510),
    "R_Foot": (-0.0362914591, +0.0036995102, +0.7305516486, +0.6818823450),
}


def _build_lr_table(apply_rest_offset: bool) -> dict[str, tuple[Quat, Quat]]:
    table: dict[str, tuple[Quat, Quat]] = {}
    for i, name in enumerate(SMPL_JOINT_ORDER):
        if name not in REBOCAP_TO_MIXAMO:
            continue
        if not apply_rest_offset:
            table[name] = (_Q_FIX, _Q_FIX_INV)
            continue
        qa_self = Q_ALIGN_SMPL_TO_MIXAMO.get(name, IDENTITY)
        p = SMPL_PARENT_INDEX[i]
        pn = SMPL_JOINT_ORDER[p] if p >= 0 else None
        qa_parent = Q_ALIGN_SMPL_TO_MIXAMO.get(pn, IDENTITY) if pn else IDENTITY
        table[name] = (qmul(qconj(qa_parent), _Q_FIX), qmul(_Q_FIX_INV, qa_self))
    return table


_LR_WITH_OFFSET = _build_lr_table(apply_rest_offset=True)
_LR_NO_OFFSET = _build_lr_table(apply_rest_offset=False)


def parse_hierarchy(bvh_path: Path) -> tuple[str, list[Bone]]:
    text = bvh_path.read_text()
    motion_idx = text.find("\nMOTION")
    if motion_idx < 0:
        raise ValueError(f"{bvh_path} missing MOTION section")
    header = text[: motion_idx + 1]

    bones: list[Bone] = []
    pending_name: str | None = None
    pending_offset: Vec3 | None = None
    for raw in header.splitlines():
        s = raw.strip()
        if s.startswith(("ROOT ", "JOINT ")):
            pending_name = s.split(None, 1)[1]
            pending_offset = None
        elif s.startswith("End Site"):
            pending_name = None
        elif pending_name is None:
            continue
        elif s.startswith("OFFSET") and pending_offset is None:
            p = s.split()
            pending_offset = (float(p[1]), float(p[2]), float(p[3]))
        elif s.startswith("CHANNELS") and pending_offset is not None:
            bones.append((pending_name, pending_offset))
            pending_name = None
    return header, bones


class BvhWriter:
    _FRAMES_WIDTH = 10
    _BUF = 1024 * 1024

    def __init__(self, out_path: Path, header: str, frame_time: float):
        self.path = out_path
        self.fp = open(
            out_path, "w", encoding="ascii", newline="\n", buffering=self._BUF
        )
        self.fp.write(header)
        self.fp.write("MOTION\n")
        self._frames_pos = self.fp.tell()
        self.fp.write(f"Frames:\t{'0':>{self._FRAMES_WIDTH}}\n")
        self.fp.write(f"Frame Time:\t{frame_time:.7f}\n")
        self.n_frames = 0
        self._fmt: str | None = None

    def write_frame(self, values: list[float]) -> None:
        if self._fmt is None:
            self._fmt = "%.7g " * len(values) + "\n"
        self.fp.write(self._fmt % tuple(values))
        self.n_frames += 1

    def close(self) -> None:
        if self.fp.closed:
            return
        self.fp.flush()
        self.fp.seek(self._frames_pos)
        self.fp.write(f"Frames:\t{self.n_frames:>{self._FRAMES_WIDTH}}\n")
        self.fp.flush()
        self.fp.close()


_ZERO3: Vec3 = (0.0, 0.0, 0.0)


def make_bone_smpl_lookup(bone_names: Iterable[str]) -> tuple[str | None, ...]:
    mixamo_to_smpl = {m: s for s, m in REBOCAP_TO_MIXAMO.items()}
    return tuple(mixamo_to_smpl.get(n) for n in bone_names)


def _extract_smpl_quats(payload) -> dict[str, Quat]:
    out: dict[str, Quat] = {}
    for j in payload.joints:
        if j.name in _SMPL_NAME_TO_IDX:
            q = j.quaternion
            out[j.name] = (float(q.w), float(q.x), float(q.y), float(q.z))
    return out


def _globals_to_locals(quats: dict[str, Quat]) -> dict[str, Quat]:
    g = [quats.get(n, IDENTITY) for n in SMPL_JOINT_ORDER]
    local: list[Quat] = [IDENTITY] * len(SMPL_JOINT_ORDER)
    for i, parent in enumerate(SMPL_PARENT_INDEX):
        local[i] = g[i] if parent < 0 else qmul(qconj(g[parent]), g[i])
    return dict(zip(SMPL_JOINT_ORDER, local))


def build_frame_channels(
    payload,
    bones: list[Bone],
    bone_smpl: tuple[str | None, ...],
    pose_scale: float,
    rotation_mode: str = "auto",
    apply_rest_offset: bool = True,
) -> list[float]:
    if rotation_mode == "auto":
        rotation_mode = getattr(payload, "rotation_mode", "local")
    if rotation_mode not in ("local", "global"):
        raise ValueError(
            f"rotation_mode must be 'local' or 'global', got {rotation_mode!r}"
        )

    quats = _extract_smpl_quats(payload)
    if rotation_mode == "global":
        quats = _globals_to_locals(quats)

    lr = _LR_WITH_OFFSET if apply_rest_offset else _LR_NO_OFFSET
    rot_by_smpl: dict[str, Vec3] = {}
    for name, q in quats.items():
        pair = lr.get(name)
        if pair is None:
            continue
        L, R = pair
        rot_by_smpl[name] = quat_to_zxy_deg(qmul(qmul(L, q), R))

    rt = getattr(payload, "root_translation", None) or _ZERO3
    root_pos: Vec3 = (
        float(rt[0]) * pose_scale,
        float(rt[2]) * pose_scale,
        -float(rt[1]) * pose_scale,
    )

    out: list[float] = []
    extend = out.extend
    extend(root_pos)
    extend(rot_by_smpl.get(bone_smpl[0], _ZERO3))
    for (_, offset), smpl_name in zip(bones[1:], bone_smpl[1:]):
        extend(offset)
        extend(rot_by_smpl.get(smpl_name, _ZERO3))
    return out


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Rebocap UDP -> BVH")
    ap.add_argument("-o", "--out", default="rebocap_live.bvh")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=9010)
    ap.add_argument(
        "--template", default=str(Path(__file__).resolve().parent.parent.parent.parent / "2.bvh")
    )
    ap.add_argument("--fps", type=float, default=60.0)
    ap.add_argument("--scale", type=float, default=100.0)
    ap.add_argument("--max-stale", type=float, default=0.5)
    ap.add_argument(
        "--rotation-mode", choices=("auto", "local", "global"), default="auto"
    )
    ap.add_argument(
        "--no-rest-offset",
        action="store_true",
        help="禁用 SMPL→Mixamo rest-pose 偏移补偿",
    )
    return ap.parse_args()


def _install_stop_handler() -> threading.Event:
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    if sys.platform != "win32":
        signal.signal(signal.SIGTERM, lambda *_: stop.set())
    return stop


def _record_loop(
    receiver: LatestRebocapUdpReceiver,
    writer: BvhWriter,
    bones: list[Bone],
    bone_smpl: tuple[str | None, ...],
    args: argparse.Namespace,
    apply_rest_offset: bool,
) -> None:
    stop = _install_stop_handler()
    frame_time = 1.0 / args.fps
    next_tick = time.perf_counter()
    last_log = next_tick
    last_seq = -1
    written_in_sec = 0

    while not stop.is_set():
        payload = receiver.get_latest_payload(timeout=0.2, max_age=args.max_stale)
        if payload is None:
            next_tick = time.perf_counter() + frame_time
            continue

        seq = receiver.seq
        if seq != last_seq:
            last_seq = seq
            writer.write_frame(
                build_frame_channels(
                    payload,
                    bones,
                    bone_smpl,
                    args.scale,
                    rotation_mode=args.rotation_mode,
                    apply_rest_offset=apply_rest_offset,
                )
            )
            written_in_sec += 1

        now = time.perf_counter()
        if now - last_log >= 1.0:
            print(
                f"[BVH] frames={writer.n_frames}  +{written_in_sec}/s  "
                f"udp_seq={receiver.seq}  dropped={receiver.dropped}",
                flush=True,
            )
            written_in_sec = 0
            last_log = now

        next_tick += frame_time
        sleep_for = next_tick - time.perf_counter()
        if sleep_for > 0:
            time.sleep(sleep_for)
        else:
            next_tick = time.perf_counter()


def main() -> None:
    args = _parse_args()
    template_path, out_path = Path(args.template), Path(args.out)
    header, bones = parse_hierarchy(template_path)
    bone_smpl = make_bone_smpl_lookup(name for name, _ in bones)
    apply_rest_offset = not args.no_rest_offset
    print(
        f"[BVH] template={template_path}  bones={len(bones)}  out={out_path}  "
        f"rest_offset={'on' if apply_rest_offset else 'off'}  rot_mode={args.rotation_mode}"
    )

    writer = BvhWriter(out_path, header, 1.0 / args.fps)
    receiver = LatestRebocapUdpReceiver(host=args.host, port=args.port)
    receiver.start()
    print(
        f"[UDP] listening on {args.host}:{args.port}  scale={args.scale}  fps={args.fps:g}"
    )

    try:
        _record_loop(receiver, writer, bones, bone_smpl, args, apply_rest_offset)
    finally:
        receiver.stop()
        writer.close()
        print(
            f"\n[BVH] done. frames={writer.n_frames}  "
            f"size={os.path.getsize(out_path)}B  -> {out_path}",
            flush=True,
        )


if __name__ == "__main__":
    main()
