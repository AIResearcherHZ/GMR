import argparse
import json
import math
import os
import select
import socket
import sys
from datetime import datetime
from pathlib import Path

_curr = os.path.abspath(__file__)
while _curr != "/" and os.path.basename(_curr) != "backend":
    _curr = os.path.dirname(_curr)
if _curr not in sys.path:
    sys.path.insert(0, _curr)

PROJECT_ROOT = Path(__file__).resolve().parent
LIB_DIR = PROJECT_ROOT / "lib"
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))

import wuji_sdk
from handctl import WujiGloveDevice
from libs.drivers.rate_limiter import perf_counter, sleep

SPREAD_WARN_DEG = 3.0


def _enter_pressed():
    return bool(select.select([sys.stdin], [], [], 0)[0])


def _zero_path(hand):
    return PROJECT_ROOT / "config" / f"palm_zero_{hand}.json"


class _PalmSim:
    def __init__(self, hand, order, headless):
        import mujoco
        from scipy.spatial.transform import Rotation as R

        self._mj = mujoco
        self._R = R
        self._order = order.upper()
        xml = PROJECT_ROOT / "wuji_hand_description" / "mjcf" / f"{hand}.xml"
        spec = mujoco.MjSpec.from_file(str(xml))
        spec.worldbody.first_body().add_freejoint()
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.data.qpos[:7] = [0, 0, 0, 1, 0, 0, 0]
        mujoco.mj_forward(self.model, self.data)
        self.viewer = None
        if not headless:
            import mujoco.viewer

            self.viewer = mujoco.viewer.launch_passive(
                self.model, self.data, show_left_ui=False, show_right_ui=False
            )

    def update(self, rpy):
        vals = [
            {"X": rpy["roll"], "Y": rpy["pitch"], "Z": rpy["yaw"]}[a]
            for a in self._order
        ]
        qx, qy, qz, qw = self._R.from_euler(self._order, vals).as_quat()
        self.data.qpos[3:7] = [qw, qx, qy, qz]
        self._mj.mj_forward(self.model, self.data)
        if self.viewer is not None:
            self.viewer.sync()

    def alive(self):
        return self.viewer is None or self.viewer.is_running()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()


def _save_zero(path, quat, hand, order):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "quat_xyzw": list(quat),
                "hand": hand,
                "euler_order": order,
                "calibrated_at": datetime.now().isoformat(timespec="seconds"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _calibrate_and_save(glove, path, hand, order, cal_time):
    print(f"[palm_rpy] 标定零点: 保持手静止 {cal_time:.1f}s ...", flush=True)
    if not glove.set_palm_zero(duration=cal_time):
        print("[palm_rpy] WARN: 未取到有效 IMU 帧, 标定失败。")
        return False
    n = glove._palm_zero_nsamples
    spread_deg = math.degrees(glove._palm_zero_spread)
    _save_zero(path, glove.get_palm_zero(), hand, order)
    note = "  [⚠ 标定时手在动, 建议重标]" if spread_deg > SPREAD_WARN_DEG else ""
    print(
        f"[palm_rpy] 零点已标定 (n={n}, 抖动≈{spread_deg:.2f}°) 并保存 -> {path}{note}"
    )
    return True


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="不开 MuJoCo 窗口 (纯命令行)",
    )
    p.add_argument("--glove-sn", default="", help="手套 SN, 单只时可省略")
    p.add_argument("--hand", default="right", choices=["left", "right"], help="手侧")
    p.add_argument("--rate", type=float, default=30.0, help="打印频率 Hz (默认 30)")
    p.add_argument("--order", default="ZYX", help="欧拉顺序 (默认 ZYX)")
    p.add_argument(
        "--remap",
        default="+y,+x,-z",
        help="IMU 安装朝向的轴重映射, 形如 '+y,+x,-z'; 空=不重映射",
    )
    p.add_argument(
        "--cal-time",
        type=float,
        default=2.0,
        dest="cal_time",
        help="标定采样时长 秒 (默认 2.0)",
    )
    p.add_argument(
        "--recalibrate",
        action="store_true",
        default=False,
        help="忽略已存零点, 重新标定",
    )
    p.add_argument(
        "--load-zero",
        action="store_true",
        default=False,
        help="跳过启动标定, 直接加载已存零点",
    )
    p.add_argument(
        "--no-filter", action="store_true", default=False, help="关闭 SLERP 平滑"
    )
    p.add_argument(
        "--absolute", action="store_true", default=False, help="不归零, 输出绝对角"
    )
    p.add_argument(
        "--wrist-udp",
        default="",
        help="把小臂 rpy 发到 host:port (空=不发, 例 127.0.0.1:9102)",
    )
    args = p.parse_args()

    wuji_sdk.set_log_level("warn")
    glove = WujiGloveDevice(
        hand_side=args.hand,
        device_name="glove",
        sn=args.glove_sn or None,
        palm_euler_order=args.order,
        palm_axis_remap=args.remap,
    )
    print(f"[palm_rpy] glove connected (hand={args.hand}, order={args.order})")

    zero_path = _zero_path(args.hand)
    if args.absolute:
        print("[palm_rpy] 绝对角模式 (不归零)。")
    elif not (args.load_zero and zero_path.exists()):
        if not _calibrate_and_save(
            glove, zero_path, args.hand, args.order, args.cal_time
        ):
            print("[palm_rpy] 退回绝对角模式。")
    else:
        try:
            data = json.loads(zero_path.read_text(encoding="utf-8"))
            glove.apply_palm_zero(data["quat_xyzw"])
            print(
                f"[palm_rpy] 已加载零点 {zero_path.name} "
                f"(标定于 {data.get('calibrated_at', '?')})。重标: --recalibrate 或运行中按回车。"
            )
        except Exception as e:
            print(f"[palm_rpy] 读取零点失败 ({e}), 重新标定。")
            _calibrate_and_save(glove, zero_path, args.hand, args.order, args.cal_time)

    interactive = (not args.absolute) and sys.stdin.isatty()
    if interactive:
        print(
            f"[palm_rpy] 运行中 —— 按 回车 原地重标零点 (阻塞 {args.cal_time:.1f}s, 清 yaw 漂移), "
            "Ctrl-C 退出。"
        )

    period = 1.0 / args.rate if args.rate > 0 else 0.0
    filtered = not args.no_filter
    wrist_sock = None
    wrist_addr = None
    if args.wrist_udp:
        host, _, port = args.wrist_udp.partition(":")
        wrist_addr = (host or "127.0.0.1", int(port))
        wrist_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(
            f"[palm_rpy] 小臂 rpy -> UDP {wrist_addr[0]}:{wrist_addr[1]} (hand={args.hand})"
        )
    sim = None
    if not args.headless:
        sim = _PalmSim(args.hand, args.order, headless=False)
        print("[palm_rpy] MuJoCo 可视化 (开窗)。")
    try:
        while True:
            t0 = perf_counter()
            if sim is not None and not sim.alive():
                break
            if interactive and _enter_pressed():
                sys.stdin.readline()
                _calibrate_and_save(
                    glove, zero_path, args.hand, args.order, args.cal_time
                )
            rpy = glove.get_palm_rpy(filtered=filtered)
            if rpy is not None:
                print(
                    f"roll={rpy['roll']:+8.4f}  pitch={rpy['pitch']:+8.4f}  "
                    f"yaw={rpy['yaw']:+8.4f}   (rad)",
                    flush=True,
                )
                if wrist_sock is not None:
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
                if sim is not None:
                    sim.update(rpy)
            wait = period - (perf_counter() - t0)
            if wait > 0:
                sleep(wait)
    except KeyboardInterrupt:
        print("\n[palm_rpy] interrupted.")
    finally:
        if sim is not None:
            sim.close()
        if wrist_sock is not None:
            wrist_sock.close()
        glove.cleanup()
        try:
            wuji_sdk.SdkManager.instance().disconnect_all()
        except Exception:
            pass


if __name__ == "__main__":
    main()
