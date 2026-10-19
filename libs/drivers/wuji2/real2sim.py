import argparse
import os
import sys
from pathlib import Path

_curr = os.path.abspath(__file__)
while _curr != "/" and os.path.basename(_curr) != "backend":
    _curr = os.path.dirname(_curr)
if _curr not in sys.path:
    sys.path.insert(0, _curr)

import mujoco
import mujoco.viewer
import numpy as np
import wuji_sdk
from libs.drivers.rate_limiter import sleep
from wuji_sdk import SdkManager

PROJECT_ROOT = Path(__file__).resolve().parent

try:
    os.getcwd()
except OSError:
    os.chdir(PROJECT_ROOT)

MJCF_PATH = PROJECT_ROOT / "wuji_hand_description" / "mjcf" / "right.xml"

BROKEN_JOINT = "right_finger4_joint2"


def _scan_hand2(manager, options):
    devs = manager.scan()
    cands = [d for d in devs if d.sn.startswith("WH2")]
    if not cands:
        raise SystemExit("No Wuji Hand 2 discovered (no SN starts with 'WH2').")
    return manager.connect(sn=cands[0].sn, device_name="wuji_hand_2", options=options)


def main():
    parser = argparse.ArgumentParser(
        description="Mirror real WH120 joints in MuJoCo (read-only)"
    )
    parser.add_argument(
        "--ip", default="", help="optional override; auto-scan when empty"
    )
    parser.add_argument("--mjcf", default=str(MJCF_PATH), help="MuJoCo model path")
    parser.add_argument("--rate", type=float, default=200.0, help="max loop Hz")
    args = parser.parse_args()

    mjcf = Path(args.mjcf)
    if not mjcf.exists():
        raise SystemExit(f"MuJoCo model file not found: {mjcf}")
    model = mujoco.MjModel.from_xml_path(str(mjcf))
    data = mujoco.MjData(model)
    if model.nq < 20:
        raise SystemExit(f"Model has nq={model.nq}, expected >= 20")

    broken_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, BROKEN_JOINT)
    if broken_idx < 0 or broken_idx >= 20:
        raise SystemExit(
            f"Broken joint '{BROKEN_JOINT}' not found within the 20 SDK joints"
        )

    wuji_sdk.set_log_level("warn")
    manager = SdkManager.instance()
    opts = wuji_sdk.ConnectOptions(enable_bridge=False)
    hand = (
        manager.connect(address=args.ip, device_name="wuji_hand_2", options=opts)
        if args.ip
        else _scan_hand2(manager, opts)
    )

    online = hand.online_joints_count().get()
    print(f"WH120 connected: {online}/20 joints online (read-only, motors NOT enabled)")
    if online < 20:
        print("WARN: some joints offline; their sim joints will hold the last value.")

    latest = {"state": None}
    sub = hand.joint_state().subscribe_with_callback(
        lambda s: latest.__setitem__("state", s)
    )
    viewer = mujoco.viewer.launch_passive(model, data)
    viewer.cam.azimuth = 180
    viewer.cam.elevation = -20
    viewer.cam.distance = 0.5
    viewer.cam.lookat[:] = [0, 0, 0.05]

    nu = min(model.nu, 20)
    lo = model.jnt_range[:20, 0].copy()
    hi = model.jnt_range[:20, 1].copy()
    qpos_sim = data.qpos[:20]
    qvel_sim = data.qvel[:20]
    ctrl_sim = data.ctrl[:nu]

    qpos = np.zeros(20)
    qvel = np.zeros(20)
    frames = 0
    t0 = time.time()
    render_period = 1.0 / 60.0
    next_render = time.time()
    try:
        while viewer.is_running():
            state = latest["state"]
            if not state or state.num_joints < 20:
                continue
            joints = state.joints
            for i in range(20):
                j = joints[i]
                qpos[i] = j.actual_pos
                qvel[i] = j.actual_vel
            qpos[broken_idx] = 0.0
            qvel[broken_idx] = 0.0
            np.clip(qpos, lo, hi, out=qpos_sim)
            qvel_sim[:] = qvel
            ctrl_sim[:] = qpos_sim[:nu]
            mujoco.mj_forward(model, data)
            viewer.sync()

            next_render += render_period
            wait = next_render - time.time()
            if wait > 0:
                sleep(wait)
            else:
                next_render = time.time()

            frames += 1
            if frames == 200:
                now = time.time()
                print(
                    f"FPS: {frames / (now - t0):.1f}  "
                    f"pos[0..3]={np.round(qpos[:4], 3).tolist()}  "
                    f"vel[0]={qvel[0]:.3f}  tau[0]={joints[0].torque:.3f}"
                )
                frames = 0
                t0 = now
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        try:
            viewer.close()
        except Exception:
            pass
        sub.close()
        manager.disconnect_all()
        os._exit(0)


if __name__ == "__main__":
    main()
