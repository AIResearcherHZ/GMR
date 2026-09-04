import argparse
import json
import os
import socket
import sys
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import wuji_sdk
from wuji_sdk import SdkManager

PROJECT_ROOT = Path(__file__).resolve().parent

try:
    os.getcwd()
except OSError:
    os.chdir(PROJECT_ROOT)

LIB_DIR = PROJECT_ROOT / "lib"
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))

from handctl import Retargeter, WujiGloveDevice
from libs.drivers.rate_limiter import perf_counter, sleep

_ENABLE_TIMEOUT_SEC = 5.0
_INVERTER_READY_STATE = 4


def _scan_hand2(manager, options):
    devs = manager.scan()
    cands = [d for d in devs if d.sn.startswith("WH2")]
    if not cands:
        raise SystemExit("No Wuji Hand 2 discovered (no SN starts with 'WH2').")
    return manager.connect(sn=cands[0].sn, device_name="wuji_hand_2", options=options)


def _enable_hand(hand, kp, kd, effort, broken_idx):
    hand.control_mode().set("mit")
    hand.effort_limit().set(effort)
    kp_grid = [[kp] * 4 for _ in range(5)]
    kd_grid = [[kd] * 4 for _ in range(5)]
    if broken_idx is not None:
        bfi, bji = divmod(broken_idx, 4)
        kp_grid[bfi][bji] = 0.0
        kd_grid[bfi][bji] = 0.0
        print(
            f"WH120: joint idx {broken_idx} broken -> kp/kd forced to 0 (not enabled)"
        )
    hand.mit_params().set(kp=kp_grid, kd=kd_grid)
    try:
        hand.clear_fault().set(1)
    except Exception as e:
        print(f"WH120: clear_fault warn: {e}")
    hand.enable()

    deadline = perf_counter() + _ENABLE_TIMEOUT_SEC
    last_diags = []
    while perf_counter() < deadline:
        sleep(0.2)
        last_diags = hand.diagnostics().get()
        live = [
            d
            for i, d in enumerate(last_diags)
            if d is not None and d.vbus > 0.5 and i != broken_idx
        ]
        if live and all(d.inverter_state == _INVERTER_READY_STATE for d in live):
            print(f"WH120 enabled (kp={kp}, kd={kd}, effort={effort}A)")
            return
    print("WH120: enable timeout. Per-joint state:")
    for i, d in enumerate(last_diags):
        if d is not None:
            fi, ji = divmod(i, 4)
            print(
                f"  finger{fi + 1}/j{ji} (idx {i}): "
                f"inverter_state={d.inverter_state} vbus={d.vbus:.2f}"
            )
    hand.disable()
    raise SystemExit(f"WH120: enable timeout after {_ENABLE_TIMEOUT_SEC}s")


def _read_actual_pos(hand, timeout=2.0):
    sub = hand.joint_state().subscribe()
    deadline = perf_counter() + timeout
    try:
        while perf_counter() < deadline:
            state = sub.recv()
            if state and state.num_joints >= 20:
                return [state.joints[i].actual_pos for i in range(20)]
        raise SystemExit("WH120: timeout reading joint_state for ramp")
    finally:
        sub.close()


def _ramp_to(publisher, hand, target, zeros, ramp_sec=1.0, ramp_rate=200.0):
    actual = _read_actual_pos(hand)
    n = max(1, int(ramp_sec * ramp_rate))
    print(f"WH120 ramp: {ramp_sec:.1f}s @ {ramp_rate:.0f}Hz ({n} steps)")
    print(
        f"WH120 ramp: actual[0..3]={[round(x, 3) for x in actual[:4]]} "
        f"-> target[0..3]={[round(x, 3) for x in target[:4]]}"
    )
    t0 = perf_counter()
    for i in range(1, n + 1):
        a = i / n
        pos = [(1 - a) * actual[j] + a * target[j] for j in range(20)]
        deadline = t0 + i / ramp_rate
        wait = deadline - perf_counter()
        if wait > 0:
            sleep(wait)
        publisher.send(pos, zeros, zeros)
    print("WH120 ramp: done")


def main():
    parser = argparse.ArgumentParser(
        description="Drive the real WH120 from the Wuji Glove via retargeting (glove -> real)"
    )
    parser.add_argument(
        "--ip", default="", help="optional hand override; auto-scan when empty"
    )
    parser.add_argument(
        "--mjcf", default="", help="MuJoCo 模型路径 (默认 mjcf/{hand}.xml)"
    )
    parser.add_argument(
        "--config",
        default="",
        help="retarget 配置 yaml (默认 config/wh120_{hand}.yaml)",
    )
    parser.add_argument(
        "--hand", default="right", choices=["left", "right"], help="hand side"
    )
    parser.add_argument(
        "--glove-sn", default="", help="Wuji Glove SN (required if multiple online)"
    )
    parser.add_argument(
        "--kp", type=float, default=3.0, help="WH120 MIT kp (default: 3.0)"
    )
    parser.add_argument(
        "--kd", type=float, default=0.1, help="WH120 MIT kd (default: 0.1)"
    )
    parser.add_argument(
        "--effort",
        type=float,
        default=1.0,
        help="WH120 effort limit amps (default: 1.0)",
    )
    parser.add_argument(
        "--ramp",
        type=float,
        default=1.0,
        dest="ramp_sec",
        help="startup ramp seconds (0 disables, default: 1.0)",
    )
    parser.add_argument(
        "--ramp-rate",
        type=float,
        default=200.0,
        dest="ramp_rate",
        help="startup ramp publish rate Hz (default: 200)",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=200.0,
        help="WH120 command publish rate Hz (default: 200)",
    )
    parser.add_argument(
        "--wrist-udp",
        default="",
        help="把手背 IMU 的小臂 rpy 发到 host:port (空=不发, 例 127.0.0.1:9101)",
    )
    parser.add_argument(
        "--palm-cal-time",
        type=float,
        default=2.0,
        help="手背 IMU 零位标定时长 秒 (默认 2.0)",
    )
    parser.add_argument(
        "--recalibrate-palm", action="store_true", help="忽略已存手背零位, 重新标定"
    )
    parser.add_argument("--palm-order", default="ZYX", help="手背欧拉顺序 (默认 ZYX)")
    args = parser.parse_args()

    mjcf = (
        Path(args.mjcf)
        if args.mjcf
        else PROJECT_ROOT / "wuji_hand_description" / "mjcf" / f"{args.hand}.xml"
    )
    if not mjcf.exists():
        raise SystemExit(f"MuJoCo model file not found: {mjcf}")
    model = mujoco.MjModel.from_xml_path(str(mjcf))
    data = mujoco.MjData(model)
    if model.nq < 20:
        raise SystemExit(f"Model has nq={model.nq}, expected >= 20")

    broken_joint = f"{args.hand}_finger4_joint2"
    broken_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, broken_joint)
    if broken_idx < 0 or broken_idx >= 20:
        raise SystemExit(
            f"Broken joint '{broken_joint}' not found within the 20 SDK joints"
        )

    config_file = (
        Path(args.config)
        if args.config
        else PROJECT_ROOT / "config" / f"wh120_{args.hand}.yaml"
    )
    if not config_file.exists():
        raise SystemExit(f"Retargeting config not found: {config_file}")

    wuji_sdk.set_log_level("warn")
    manager = SdkManager.instance()
    opts = wuji_sdk.ConnectOptions(enable_bridge=False)
    hand = (
        manager.connect(address=args.ip, device_name="wuji_hand_2", options=opts)
        if args.ip
        else _scan_hand2(manager, opts)
    )

    online = hand.online_joints_count().get()
    if online == 0:
        raise SystemExit("WH120: 0 joints online — check device power/network")
    print(f"WH120 connected: {online}/20 joints online")
    if online < 20:
        print("WARN: some joints offline; their targets will be ignored by firmware.")

    glove = WujiGloveDevice(
        hand_side=args.hand,
        device_name="glove",
        sn=args.glove_sn or None,
        palm_euler_order=args.palm_order,
    )
    retargeter = Retargeter.from_yaml(str(config_file), args.hand)
    print(f"Glove + retargeter ready (config={config_file.name})")

    palm_zero_path = PROJECT_ROOT / "config" / f"palm_zero_{args.hand}.json"
    if args.recalibrate_palm or not palm_zero_path.exists():
        print(f"[palm] 标定手背零位: 保持手静止 {args.palm_cal_time:.1f}s ...")
        if glove.set_palm_zero(duration=args.palm_cal_time):
            palm_zero_path.parent.mkdir(parents=True, exist_ok=True)
            palm_zero_path.write_text(
                json.dumps(
                    {
                        "quat_xyzw": list(glove.get_palm_zero()),
                        "hand": args.hand,
                        "euler_order": args.palm_order,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"[palm] 零位已标定并保存 -> {palm_zero_path}")
        else:
            print("[palm] WARN: 标定失败, 退回绝对角 (yaw 会漂移)")
    else:
        try:
            glove.apply_palm_zero(
                json.loads(palm_zero_path.read_text(encoding="utf-8"))["quat_xyzw"]
            )
            print(f"[palm] 已加载手背零位 {palm_zero_path.name}")
        except Exception as e:
            print(f"[palm] 读取零位失败 ({e}), 退回绝对角")

    wrist_sock = None
    wrist_addr = None
    if args.wrist_udp:
        host, _, port = args.wrist_udp.partition(":")
        wrist_addr = (host or "127.0.0.1", int(port))
        wrist_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(
            f"[palm] 小臂 rpy -> UDP {wrist_addr[0]}:{wrist_addr[1]} (hand={args.hand})"
        )

    _enable_hand(hand, args.kp, args.kd, args.effort, broken_idx)
    publisher = hand.joint_command().publisher()
    zeros = [0.0] * 20

    lo = model.jnt_range[:20, 0].copy()
    hi = model.jnt_range[:20, 1].copy()
    qpos_sim = data.qpos[:20]

    viewer = mujoco.viewer.launch_passive(model, data)
    viewer.cam.azimuth = 180
    viewer.cam.elevation = -20
    viewer.cam.distance = 0.5
    viewer.cam.lookat[:] = [0, 0, 0.05]

    fingers_key = f"{args.hand}_fingers"

    def glove_to_target():
        fingers_data = glove.get_fingers_data()
        fingers_pose = fingers_data[fingers_key]
        if fingers_pose is None or np.allclose(fingers_pose, 0):
            return None
        qpos = np.asarray(retargeter.retarget(fingers_pose), dtype=np.float64)[:20]
        target = np.clip(qpos, lo, hi)
        target[broken_idx] = 0.0
        return target

    period = 1.0 / args.rate if args.rate > 0 else 0.0
    frames = 0
    t0 = time.time()
    first = True
    next_pub = perf_counter()
    try:
        while viewer.is_running():
            if wrist_sock is not None:
                rpy = glove.get_palm_rpy(filtered=True)
                if rpy is not None:
                    wrist_sock.sendto(
                        json.dumps(
                            {
                                "ts": perf_counter(),
                                "hand": args.hand,
                                "roll": rpy["roll"],
                                "pitch": rpy["pitch"],
                                "yaw": rpy["yaw"],
                            }
                        ).encode("utf-8"),
                        wrist_addr,
                    )
            target = glove_to_target()
            if target is None:
                sleep(0.005)
                continue

            if first:
                first = False
                if args.ramp_sec > 0:
                    _ramp_to(
                        publisher,
                        hand,
                        target.tolist(),
                        zeros,
                        ramp_sec=args.ramp_sec,
                        ramp_rate=args.ramp_rate,
                    )
                next_pub = perf_counter()

            now_m = perf_counter()
            if now_m < next_pub:
                sleep(next_pub - now_m)
            next_pub += period

            publisher.send(target.tolist(), zeros, zeros)
            qpos_sim[:] = target
            mujoco.mj_forward(model, data)
            viewer.sync()

            frames += 1
            if frames == 200:
                now = time.time()
                print(
                    f"FPS: {frames / (now - t0):.1f}  "
                    f"glove->real[0..3]={np.round(target[:4], 3).tolist()}"
                )
                frames = 0
                t0 = now
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except RuntimeError as e:
        print(f"WH120: send failed, device likely disconnected: {e}")
    finally:
        for cleanup in (
            viewer.close,
            publisher.close,
            hand.disable,
            manager.disconnect_all,
            getattr(glove, "cleanup", lambda: None),
            wrist_sock.close if wrist_sock is not None else (lambda: None),
        ):
            try:
                cleanup()
            except Exception as e:
                name = getattr(cleanup, "__name__", "cleanup")
                print(f"WH120: cleanup {name} warn: {e}")
        os._exit(0)


if __name__ == "__main__":
    main()
