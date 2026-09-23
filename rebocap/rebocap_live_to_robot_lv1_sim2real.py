from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
BACKEND = PROJECT_ROOT / "backend"
for path in (HERE, BACKEND, BACKEND / "libs" / "SDK"):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.utils.lafan_vendor.utils import quat_fk, quat_mul
from rebocap_live_to_robot import (
    _HERE,
    _LR_NO_OFFSET,
    _LR_WITH_OFFSET,
    make_bone_smpl_lookup,
    parse_skeleton,
    payload_to_frame_dict,
)
from rebocap_udp_receiver import LatestRebocapUdpReceiver
from lv1_real_backend import (
    DEFAULT_CAN_MAP,
    JOINT_KPKD,
    MODEL_XML,
    SCENE_XML,
    LV1_ASSET_ROOT,
    TAU_SCALE,
    CompModel,
    RateLimiter,
    RealBackend,
    _parse_can_map,
    _force_exit_after,
    ramp,
)

sys.path.insert(0, str(LV1_ASSET_ROOT / "tools"))
from wrist_yaw_closed_chain import WristYawClosedChain

HUMAN_HEIGHT = 1.75


def parse_args():
    parser = argparse.ArgumentParser(description="Rebocap UDP -> Semi-Taks-LV1 真机实时重定向")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9010)
    parser.add_argument("--template", default=str(_HERE.parent.parent / "2.bvh"))
    parser.add_argument("--motion_fps", type=float, default=30.0)
    parser.add_argument("--control_fps", type=float, default=200.0)
    parser.add_argument("--max-stale", type=float, default=0.5)
    parser.add_argument("--rotation-mode", choices=("auto", "local", "global"), default="auto")
    parser.add_argument("--no-rest-offset", action="store_true")
    parser.add_argument("--rate_limit", action="store_true")
    parser.add_argument("-T", "--time", type=float, default=5.0)
    parser.add_argument("--stop-time", type=float, default=None)
    parser.add_argument("--hold-time", type=float, default=1.0)
    parser.add_argument("--mode", choices=("q", "qd", "qdd", "none"), default="qd")
    parser.add_argument("--no-comp", action="store_true")
    parser.add_argument("--ff-scale", type=float, default=TAU_SCALE)
    parser.add_argument("--kp-scale", type=float, default=1.0)
    parser.add_argument("--kd-scale", type=float, default=1.0)
    parser.add_argument("--cans", default=None)
    parser.add_argument("--no-view", action="store_true")
    parser.add_argument("--save_path", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.stop_time is None:
        args.stop_time = args.time
    names, parents, offsets = parse_skeleton(Path(args.template))
    names_stripped = [name.split(":", 1)[-1] for name in names]
    bone_smpl = make_bone_smpl_lookup(names)
    lr_table = _LR_NO_OFFSET if args.no_rest_offset else _LR_WITH_OFFSET
    retargeter = GMR(src_human="bvh_mixamo", tgt_robot="semi_taks_lv1", actual_human_height=HUMAN_HEIGHT)
    xml = SCENE_XML if SCENE_XML.exists() else MODEL_XML
    comp = CompModel(xml, tau_scale=args.ff_scale)
    wrist_ik = {
        side: WristYawClosedChain(side, model=comp.m)
        for side in ("left", "right")
    }
    wrist_indices = {
        side: (
            comp.m.joint(f"{side}_wrist_yaw_joint").qposadr[0],
            comp.act_names.index(f"{side}_wrist_yaw_motor_joint"),
        )
        for side in wrist_ik
    }
    can_map = _parse_can_map(args.cans, DEFAULT_CAN_MAP)
    backend = RealBackend(comp, xml, can_map, view=not args.no_view, kpkd=JOINT_KPKD)
    receiver = LatestRebocapUdpReceiver(host=args.host, port=args.port)
    kp = np.array([JOINT_KPKD[name][0] for name in comp.act_names]) * args.kp_scale
    kd = np.array([JOINT_KPKD[name][1] for name in comp.act_names]) * args.kd_scale
    zero = np.zeros(comp.nu)
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    last_q = None
    target_lock = threading.Lock()
    retarget_error = None
    prev_pos = None
    last_time = None
    last_log = 0.0
    current_a = 0.0
    current_pos = None
    q_start = None
    ramp_start = None
    hold_start = None
    def update_target():
        nonlocal last_q, retarget_error
        last_seq = -1
        try:
            while not stop.is_set():
                payload = receiver.get_latest_payload(timeout=0.05, max_age=args.max_stale)
                seq = receiver.seq
                if payload is None or seq == last_seq:
                    stop.wait(0.01)
                    continue
                last_seq = seq
                frame = payload_to_frame_dict(payload, names_stripped, parents, offsets, bone_smpl, lr_table, args.rotation_mode)
                qpos = retargeter.retarget(frame)
                if not np.all(np.isfinite(qpos)):
                    continue
                target = qpos[comp.act_qadr].copy()
                with target_lock:
                    previous = None if last_q is None else last_q.copy()
                for side, solver in wrist_ik.items():
                    output_index, motor_index = wrist_indices[side]
                    output = float(np.clip(qpos[output_index], *solver.output_limits))
                    try:
                        target[motor_index] = solver.ik(output)
                    except ValueError:
                        target[motor_index] = previous[motor_index] if previous is not None else q_start[motor_index]
                target = np.clip(target, comp.m.actuator_ctrlrange[:, 0], comp.m.actuator_ctrlrange[:, 1])
                with target_lock:
                    last_q = target
        except Exception as exc:
            retarget_error = exc
            stop.set()

    worker = None
    print(f"[init] robot=semi_taks_lv1 udp={args.host}:{args.port} control={args.control_fps:g}Hz")
    try:
        backend.enable()
        deadline = time.perf_counter() + 3.0
        while backend.online_count() < len(backend.specs) and time.perf_counter() < deadline:
            backend.read()
            time.sleep(0.02)
        q_start = backend.current_motor_pos()
        backend.seed_embedded()
        receiver.start()
        worker = threading.Thread(target=update_target, name="lv1-retarget", daemon=True)
        worker.start()
        ramp_start = time.perf_counter()
        rl = RateLimiter(args.control_fps)
        while not stop.is_set() and backend.is_running():
            now = time.perf_counter()
            with target_lock:
                target_pos = q_start.copy() if last_q is None else last_q.copy()
            if now - ramp_start < args.time:
                a = ramp((now - ramp_start) / args.time) if args.time > 0 else 1.0
                phase = "缓启动"
                pos = q_start + a * (target_pos - q_start)
            elif hold_start is None:
                hold_start = now
                a = 1.0
                phase = "保持"
                pos = target_pos.copy()
            elif now - hold_start < args.hold_time:
                a = 1.0
                phase = "保持"
                pos = target_pos.copy()
            else:
                a = 1.0
                phase = "重定向"
                pos = target_pos.copy()
            qpos, qvel, qacc, tau_meas = backend.read()
            vel = zero if prev_pos is None else (pos - prev_pos) / max(now - last_time, 1e-4)
            prev_pos = pos.copy()
            last_time = now
            tau_ff = a * comp.tau(qpos, qvel, qacc, args.mode) if not args.no_comp and args.mode != "none" else zero
            tau_cmd = backend.command(a * kp, a * kd, pos, vel, tau_ff)
            current_a = a
            current_pos = pos.copy()
            if now - last_log >= 1.0:
                last_log = now
                print(f"[{phase}] udp={receiver.seq} online={backend.online_count()}/{len(backend.specs)} |q|={np.abs(pos).max():.3f} |tau|={np.abs(tau_cmd).max():.2f}", flush=True)
            backend.sync()
            rl.sleep()
    finally:
        stop.set()
        receiver.stop()
        if worker is not None:
            worker.join(timeout=1.0)
        try:
            if current_pos is not None and current_a > 0.0:
                print("[缓停止] kp/kd/重力补偿缓慢归零", flush=True)
                stop_rl = RateLimiter(args.control_fps)
                stop_t0 = time.perf_counter()
                while True:
                    now = time.perf_counter()
                    u = (now - stop_t0) / args.stop_time if args.stop_time > 0.0 else 1.0
                    a = current_a * (1.0 - ramp(min(u, 1.0)))
                    qpos, qvel, qacc, _ = backend.read()
                    tau_ff = a * comp.tau(qpos, qvel, qacc, args.mode) if not args.no_comp and args.mode != "none" else zero
                    backend.command(a * kp, a * kd, current_pos, zero, tau_ff)
                    backend.sync()
                    if u >= 1.0:
                        break
                    stop_rl.sleep()
        except Exception as exc:
            print(f"[缓停止] 执行失败: {exc}", flush=True)
        wd = _force_exit_after(5.0)
        try:
            backend.command(zero, zero, current_pos if current_pos is not None else (q_start if q_start is not None else zero), zero, zero)
            backend.disable()
        finally:
            wd.cancel()
    if retarget_error is not None:
        raise RuntimeError("LV1 重定向线程失败") from retarget_error


if __name__ == "__main__":
    main()
