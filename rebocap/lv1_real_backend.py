from __future__ import annotations

import argparse
import math
import os
import signal
import sys
import threading
from pathlib import Path

import mujoco
import numpy as np
from rich import box
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
for _p in (HERE, REPO_ROOT, REPO_ROOT / "backend"):
    sp = str(_p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from libs.drivers.rate_limiter import perf_counter, sleep
from pos_vel_tau_plotter import PosVelTauPlotter

console = Console()

UI_HZ = 20.0
CTRL_HZ = 200.0
VIEWER_SYNC_HZ = 30.0
SHUTDOWN_TIMEOUT = 3.0
TAU_SCALE = 0.8


SCENE_XML = HERE.parents[0] / "assets" / "Semi_Taks_LV1" / "scene_Semi_Taks_LV1.xml"
MODEL_XML = HERE.parents[0] / "assets" / "Semi_Taks_LV1" / "Semi_Taks_LV1.xml"

DEFAULT_MODE = "qd"
MODES = ("q", "qd", "qdd", "none")
_MODE_DESC = {
    "q": "只重力 g(q)",
    "qd": "重力+科氏 g+Cq̇",
    "qdd": "全逆动力学 Mq̈+Cq̇+g",
    "none": "不补偿",
}

DEFAULT_RAMP_T = 5.0

SHOULDER = "RP50H"
WRIST = "RP40C"
WAIST = "RP70H"
HEAD = "RP40C"

JOINTS: dict[tuple[str, int], str] = {
    ("can0", 1): "right_shoulder_pitch_joint",
    ("can0", 2): "right_shoulder_roll_joint",
    ("can0", 3): "right_shoulder_yaw_joint",
    ("can0", 4): "right_elbow_joint",
    ("can0", 5): "right_wrist_roll_joint",
    ("can0", 6): "right_arm_long_link_motor_joint",
    ("can0", 7): "right_arm_short_link_motor_joint",
    ("can1", 1): "left_shoulder_pitch_joint",
    ("can1", 2): "left_shoulder_roll_joint",
    ("can1", 3): "left_shoulder_yaw_joint",
    ("can1", 4): "left_elbow_joint",
    ("can1", 5): "left_wrist_roll_joint",
    ("can1", 6): "left_arm_long_link_motor_joint",
    ("can1", 7): "left_arm_short_link_motor_joint",
    ("can2", 1): "waist_yaw_joint",
    ("can2", 2): "waist_right_motor_joint",
    ("can2", 3): "waist_left_motor_joint",
    ("can2", 4): "head_roll_joint",
    ("can2", 5): "head_pitch_joint",
    ("can2", 6): "head_yaw_joint",
}

MOTOR_TYPE: dict[tuple[str, int], str] = {
    **{
        (bus, did): SHOULDER
        for bus, ids in (("can0", range(1, 5)), ("can1", range(1, 5)))
        for did in ids
    },
    **{
        (bus, did): WRIST
        for bus, ids in (("can0", range(5, 8)), ("can1", range(5, 8)))
        for did in ids
    },
    **{("can2", did): WAIST for did in (1, 2, 3)},
    **{("can2", did): HEAD for did in (4, 5, 6)},
}

DEFAULT_CAN_MAP: dict[str, list[int]] = {
    "can0": [1, 2, 3, 4, 5, 6, 7],
    "can1": [1, 2, 3, 4, 5, 6, 7],
    "can2": [1, 2, 3, 4, 5, 6],
}

JOINT_KPKD: dict[str, tuple[float, float]] = {
    **{
        f"{side}_{joint}_joint": (11.943982412897888 * 1, 0.7603775364861449 * 1)
        for side in ("right", "left")
        for joint in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow")
    },
    **{
        f"{side}_wrist_roll_joint": (2.943826643921725 * 1, 0.18740982479430696 * 1)
        for side in ("right", "left")
    },
    **{
        f"{side}_{joint}_motor_joint": (5.88765328784345 * 1, 0.3748196495886139 * 1)
        for side in ("right", "left")
        for joint in ("arm_long_link", "arm_short_link")
    },
    "waist_yaw_joint": (38.89891391234306 * 1, 2.4763817720221986 * 1),
    "waist_left_motor_joint": (77.79782782468612 * 1, 4.952763544044397 * 1),
    "waist_right_motor_joint": (77.79782782468612 * 1, 4.952763544044397 * 1),
    "head_roll_joint": (2.943826643921725 * 1, 0.18740982479430696 * 1),
    "head_pitch_joint": (2.943826643921725 * 1, 0.18740982479430696 * 1),
    "head_yaw_joint": (2.943826643921725 * 1, 0.18740982479430696 * 1),
}

JOINT_TARGET: dict[str, float] = {
    "right_shoulder_pitch_joint": 0.0,
    "right_shoulder_roll_joint": 0.0,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 0.0,
    "right_wrist_roll_joint": 0.0,
    "right_arm_long_link_motor_joint": 0.0,
    "right_arm_short_link_motor_joint": 0.0,
    "left_shoulder_pitch_joint": 0.0,
    "left_shoulder_roll_joint": 0.0,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.0,
    "left_wrist_roll_joint": 0.0,
    "left_arm_long_link_motor_joint": 0.0,
    "left_arm_short_link_motor_joint": 0.0,
    "waist_yaw_joint": 0.0,
    "waist_left_motor_joint": 0.0,
    "waist_right_motor_joint": 0.0,
    "head_roll_joint": 0.0,
    "head_pitch_joint": 0.0,
    "head_yaw_joint": 0.0,
}


def _force_exit_after(seconds: float):
    t = threading.Timer(seconds, os._exit, args=(0,))
    t.daemon = True
    t.start()
    return t


class GracefulExit:
    def __init__(self):
        self.stop = False
        self._prev = {}

    def _handle(self, *_):
        self.stop = True

    def __enter__(self):
        for s in (signal.SIGINT, signal.SIGTERM):
            try:
                self._prev[s] = signal.signal(s, self._handle)
            except (ValueError, OSError):
                pass
        return self

    def __exit__(self, *_):
        for s, h in self._prev.items():
            try:
                signal.signal(s, h)
            except (ValueError, OSError):
                pass
        self._prev.clear()


class RateLimiter:
    __slots__ = ("_next", "period")

    def __init__(self, hz: float):
        self.period = 1.0 / hz
        self._next = perf_counter() + self.period

    def sleep(self):
        now = perf_counter()
        wait = self._next - now
        if wait > 0:
            sleep(wait)
            self._next += self.period
        else:
            self._next = now + self.period


def _expand_ids(ids: str) -> list[int]:
    out: list[int] = []
    for tok in ids.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-")
            out.extend(range(int(a.strip(), 0), int(b.strip(), 0) + 1))
        else:
            out.append(int(tok, 0))
    return out


def _parse_can_map(spec, default):
    if not spec:
        return {bus: list(ids) for bus, ids in default.items()}
    result: dict[str, list[int]] = {}
    for item in spec.split(";"):
        item = item.strip()
        if not item:
            continue
        bus, _, ids = item.partition(":")
        result[bus.strip()] = _expand_ids(ids)
    return result


def _normalize_mode(mode: str) -> str:
    if mode in MODES:
        return mode
    console.print(
        f"[yellow]⚠️  未知补偿模式 {mode!r}，已回退到 {DEFAULT_MODE!r}。"
        f"可选值：{', '.join(MODES)}[/yellow]"
    )
    return DEFAULT_MODE


def _bool_arg(s):
    if s in ("True", "False"):
        return s == "True"
    raise argparse.ArgumentTypeError(f"无法解析布尔值：{s!r}（用 True/False）")


def _resolve_joint_token(tok, names):
    if tok.isdigit():
        i = int(tok)
        if 0 <= i < len(names):
            return [i]
        console.print(f"[yellow]⚠️  关节序号 {i} 越界，已忽略[/yellow]")
        return []
    hit = [i for i, n in enumerate(names) if tok in n]
    if not hit:
        console.print(f"[yellow]⚠️  无匹配关节 {tok!r}，已忽略[/yellow]")
    return hit


def _parse_joint_sel(spec, names):
    if spec is None or str(spec).strip().lower() in ("", "all"):
        return list(range(len(names)))
    idx = []
    for tok in str(spec).replace("，", ",").split(","):
        tok = tok.strip()
        if tok:
            idx.extend(_resolve_joint_token(tok, names))
    return list(dict.fromkeys(idx))


def _parse_joint_targets(spec, names, base=None):
    out = np.zeros(len(names)) if base is None else np.asarray(base, float).copy()
    if not spec or str(spec).strip() == "":
        return out
    for tok in str(spec).replace("，", ",").split(","):
        tok = tok.strip()
        if not tok or "=" not in tok:
            continue
        key, _, val = tok.partition("=")
        try:
            v = float(val)
        except ValueError:
            continue
        for i in _resolve_joint_token(key.strip(), names):
            out[i] = v
    return out


def _load_model(xml) -> mujoco.MjModel:
    if isinstance(xml, mujoco.MjSpec):
        return xml.compile()
    return mujoco.MjModel.from_xml_path(str(xml))


def ramp(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)


class CompModel:
    _EQ = int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)

    def __init__(self, xml_path, tau_scale: float = TAU_SCALE):
        self.m = _load_model(xml_path)
        self.m.opt.jacobian = mujoco.mjtJacobian.mjJAC_DENSE
        self.d = mujoco.MjData(self.m)
        self.tau_scale = float(tau_scale)

        self.nu = int(self.m.nu)
        self.nv = int(self.m.nv)

        self.act_jnt_id = np.array(
            [int(self.m.actuator_trnid[i, 0]) for i in range(self.nu)]
        )
        self.act_dofs = np.array([int(self.m.jnt_dofadr[j]) for j in self.act_jnt_id])
        self.act_qadr = np.array([int(self.m.jnt_qposadr[j]) for j in self.act_jnt_id])
        self.act_names = [
            mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            for i in range(self.nu)
        ]
        gears = np.asarray(self.m.actuator_gear)[:, 0]

        self.B = np.zeros((self.nv, self.nu))
        for i in range(self.nu):
            self.B[self.act_dofs[i], i] = gears[i]

        self.total_mass = float(self.m.body_mass.sum())

    def tau(self, qpos, qvel=None, qacc=None, mode: str = DEFAULT_MODE) -> np.ndarray:
        d = self.d
        d.qpos[:] = qpos
        d.qvel[:] = qvel if (mode in ("qd", "qdd") and qvel is not None) else 0.0
        d.qacc[:] = qacc if (mode == "qdd" and qacc is not None) else 0.0
        qacc_des = np.array(d.qacc, copy=True)

        mujoco.mj_forward(self.m, d)

        rhs = d.qfrc_bias.copy()
        if mode == "qdd":
            Mqacc = np.zeros(self.nv)
            mujoco.mj_mulM(self.m, d, Mqacc, qacc_des)
            rhs = rhs + Mqacc
        rhs *= self.tau_scale

        eqJ = self._equality_jacobian(d)
        A = np.hstack([self.B, eqJ.T])
        sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)
        return sol[: self.nu]

    def _equality_jacobian(self, d) -> np.ndarray:
        nefc = int(d.nefc)
        if nefc == 0:
            return np.zeros((0, self.nv))
        efcJ = np.asarray(d.efc_J).reshape(nefc, self.nv)
        mask = np.asarray(d.efc_type[:nefc]) == self._EQ
        return efcJ[mask]

    def to_qfrc(self, tau) -> np.ndarray:
        return self.B @ np.asarray(tau)


PHASE_START, PHASE_HOLD, PHASE_STOP, PHASE_DONE = "缓启动", "保持", "缓停止", "结束"


class RampScheduler:
    def __init__(self, q_start, nu, t_start, t_stop, hold=None, q_target=None):
        self.q_start = np.asarray(q_start, dtype=float).copy()
        self.q_target = (
            np.zeros(nu) if q_target is None else np.asarray(q_target, float).copy()
        )
        self.nu = int(nu)
        self.t_start = float(t_start)
        self.t_stop = float(t_stop)
        self.hold = hold
        self.phase = PHASE_START
        self._t0 = None
        self._hold_t0 = None
        self._stop_t0 = None
        self._a_stop = 1.0
        self._q_stop = np.zeros(nu)

    def request_stop(self, now, a_now, q_now):
        if self.phase in (PHASE_STOP, PHASE_DONE):
            return
        self.phase = PHASE_STOP
        self._stop_t0 = now
        self._a_stop = float(a_now)
        self._q_stop = np.asarray(q_now, dtype=float).copy()

    def step(self, now):
        if self._t0 is None:
            self._t0 = now

        if self.phase == PHASE_START:
            u = (now - self._t0) / self.t_start if self.t_start > 0 else 1.0
            if u < 1.0:
                return (
                    PHASE_START,
                    ramp(u),
                    self.q_start + ramp(u) * (self.q_target - self.q_start),
                )
            self.phase = PHASE_HOLD
            self._hold_t0 = now
            return PHASE_HOLD, 1.0, self.q_target.copy()

        if self.phase == PHASE_HOLD:
            return PHASE_HOLD, 1.0, self.q_target.copy()

        if self.phase == PHASE_STOP:
            u = (now - self._stop_t0) / self.t_stop if self.t_stop > 0 else 1.0
            if u >= 1.0:
                self.phase = PHASE_DONE
                return PHASE_DONE, 0.0, self._q_stop
            return PHASE_STOP, self._a_stop * (1.0 - ramp(u)), self._q_stop

        return PHASE_DONE, 0.0, self._q_stop


class SimBackend:
    name = "MuJoCo 仿真"
    interactive = True
    tau_lag_ticks = 0.5

    def __init__(self, comp: CompModel, scene_xml, view=True, control_hz=None):
        self.comp = comp
        self.m = _load_model(scene_xml)
        self.m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
        if control_hz is not None and control_hz > 0:
            self.m.opt.timestep = (1.0 / float(control_hz)) / 64.0
            self.m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        self.d = mujoco.MjData(self.m)
        self.dt = float(self.m.opt.timestep)
        self.viewer = None
        self._want_view = view
        self.interactive = view
        self._key_cb = None
        self.frcrange = np.asarray(comp.m.actuator_forcerange, dtype=float).copy()
        self.last_tau = np.zeros(comp.nu)
        self._tau_sum = np.zeros(comp.nu)
        self._tau_n = 0
        self._last_sync = 0.0
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.m, self.d)
        mujoco.mj_forward(self.m, self.d)
        self._tau_sum[:] = 0.0
        self._tau_n = 0

    def enable(self):
        if self._want_view and self.viewer is None:
            import mujoco.viewer

            self.viewer = mujoco.viewer.launch_passive(
                self.m,
                self.d,
                show_left_ui=False,
                show_right_ui=False,
                key_callback=lambda k: self._key_cb and self._key_cb(k),
            )
            mujoco.mjv_defaultFreeCamera(self.m, self.viewer.cam)

    def register_key_callback(self, cb):
        self._key_cb = cb

    def read(self):
        if self._tau_n:
            tau = self._tau_sum / self._tau_n
            self._tau_sum[:] = 0.0
            self._tau_n = 0
        else:
            tau = self.last_tau.copy()
        return (
            self.d.qpos.copy(),
            self.d.qvel.copy(),
            self.d.qacc.copy(),
            tau,
        )

    def motor_state(self):
        c = self.comp
        return (
            self.d.qpos[c.act_qadr].copy(),
            self.d.qvel[c.act_dofs].copy(),
            self.last_tau.copy(),
        )

    def command(self, kp, kd, pos_des, vel_des, tau_ff):
        c = self.comp
        q = self.d.qpos[c.act_qadr]
        qd = self.d.qvel[c.act_dofs]
        tau = np.clip(
            kp * (pos_des - q) + kd * (vel_des - qd) + tau_ff,
            self.frcrange[:, 0],
            self.frcrange[:, 1],
        )
        self.d.qfrc_applied[:] = c.to_qfrc(tau)
        mujoco.mj_step(self.m, self.d)
        self.last_tau = tau
        self._tau_sum += tau
        self._tau_n += 1
        return tau

    def is_running(self):
        return self.viewer.is_running() if self.viewer else True

    def sync(self):
        if not self.viewer:
            return
        now = perf_counter()
        if now - self._last_sync < 1.0 / VIEWER_SYNC_HZ:
            return
        self._last_sync = now
        mujoco.mj_camlight(self.m, self.d)
        self.viewer.sync()

    def disable(self):
        self.d.qfrc_applied[:] = 0.0
        if self.viewer:
            try:
                self.viewer.close()
            except Exception:
                pass
            self.viewer = None


class _CanWorker(threading.Thread):
    def __init__(self, interface, specs, cache, cache_lock, cmd, cmd_lock, cmd_event):
        super().__init__(daemon=True)
        self.interface = interface
        self.specs = specs
        self.cache = cache
        self.cache_lock = cache_lock
        self.cmd = cmd
        self.cmd_lock = cmd_lock
        self.cmd_event = cmd_event
        self._stop_event = threading.Event()
        self._mc = None
        self._motors: dict[int, object] = {}

    @staticmethod
    def _can_up(interface):
        path = f"/sys/class/net/{interface}"
        if not os.path.exists(path):
            return False
        try:
            with open(f"{path}/operstate") as f:
                return f.read().strip() in ("up", "unknown")
        except OSError:
            return True

    def run(self):
        if not self._can_up(self.interface):
            console.print(
                f"[yellow]⚠️  CAN {self.interface} 不在线，跳过 "
                f"{len(self.specs)} 个电机[/yellow]"
            )
            return
        from libs.drivers.EYou_RP_CANFD import (
            EYouRp_Motor_Type,
            EYouRpCanfdControl,
            Motor,
        )

        try:
            motors = []
            for _, device_id, mtype, _ in self.specs:
                motor = Motor(EYouRp_Motor_Type[mtype], device_id)
                self._motors[device_id] = motor
                motors.append(motor)
            self._mc = EYouRpCanfdControl(self.interface, motors, silent=True)
        except Exception as exc:
            console.print(f"[red]CAN {self.interface} 初始化失败: {exc}[/red]")
            return
        try:
            self._mc.enable()
        except Exception as exc:
            console.print(f"[red]CAN {self.interface} 使能失败: {exc}[/red]")
            self._safe_close()
            return
        self._mc.controlMIT(
            [(m, 0.0, 0.0, 0.0, 0.0, 0.0) for m in self._motors.values()]
        )
        self._publish_feedback()

        try:
            while not self._stop_event.is_set():
                self.cmd_event.wait(1.0 / CTRL_HZ)
                self.cmd_event.clear()
                if self._stop_event.is_set():
                    break
                with self.cmd_lock:
                    c = self.cmd.copy()
                commands = []
                for _, device_id, _, ai in self.specs:
                    m = self._motors[device_id]
                    kp, kd, pos, _vel, tau = (float(x) for x in c[ai])
                    commands.append((m, kp, kd, pos, 0, tau))
                self._mc.controlMIT(commands)
                self._publish_feedback()
        finally:
            try:
                self._mc.controlMIT(
                    [(m, 0.0, 0.0, 0.0, 0.0, 0.0) for m in self._motors.values()]
                )
            except Exception:
                pass
            self._safe_close()

    def _publish_feedback(self):
        batch = {
            (self.interface, did): (
                m.getPosition(),
                m.getVelocity(),
                m.getTorque(),
                m.getFeedbackAge() < 0.1,
            )
            for did, m in self._motors.items()
        }
        with self.cache_lock:
            self.cache.update(batch)

    def _safe_close(self):
        if self._mc is None:
            return
        for fn in (
            self._mc.disable,
            self._mc.close,
        ):
            try:
                fn()
            except Exception:
                pass

    def stop(self):
        self._stop_event.set()
        self.cmd_event.set()


class RealBackend:
    name = "EYou RP CAN-FD 真机"
    interactive = False
    tau_lag_ticks = 1.0

    def __init__(
        self,
        comp: CompModel,
        scene_xml,
        can_map,
        view=False,
        joints=None,
        motor_type=None,
        kpkd=None,
    ):
        self.comp = comp
        self.dt = 1.0 / CTRL_HZ
        self._want_view = bool(view)
        self.viewer = None
        self._vd = None
        self._key_cb = None

        joints = JOINTS if joints is None else joints
        motor_type = MOTOR_TYPE if motor_type is None else motor_type
        name_to_act = {name: i for i, name in enumerate(comp.act_names)}
        can_set = {(bus, did) for bus, ids in can_map.items() for did in ids}
        self.specs = []
        for (bus, device_id), jname in joints.items():
            if (bus, device_id) not in can_set or jname not in name_to_act:
                continue
            self.specs.append(
                (
                    bus,
                    device_id,
                    motor_type[(bus, device_id)],
                    name_to_act[jname],
                )
            )

        for bus in can_map:
            device_ids = [spec[1] for spec in self.specs if spec[0] == bus]
            if len(device_ids) > 8 or len(device_ids) != len(set(device_ids)):
                raise ValueError(f"{bus} 的 EYou RP 设备 ID 必须唯一且位于 1..8")

        self._cache: dict[tuple[str, int], tuple] = {}
        self._cache_lock = threading.Lock()
        self._cmd = np.zeros((comp.nu, 5))
        self._cmd_lock = threading.Lock()
        self._cmd_events: list[threading.Event] = []
        self._workers: list[_CanWorker] = []

        self._em = _load_model(scene_xml)
        self._em.opt.gravity[:] = 0.0
        self._em.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
        if kpkd is not None:
            for i, name in enumerate(comp.act_names):
                kp, kd = kpkd[name]
                self._em.actuator_gaintype[i] = mujoco.mjtGain.mjGAIN_FIXED
                self._em.actuator_gainprm[i, :] = 0.0
                self._em.actuator_gainprm[i, 0] = kp
                self._em.actuator_biastype[i] = mujoco.mjtBias.mjBIAS_AFFINE
                self._em.actuator_biasprm[i, :] = 0.0
                self._em.actuator_biasprm[i, 1] = -kp
                self._em.actuator_biasprm[i, 2] = -kd
                self._em.actuator_ctrllimited[i] = 0
        self._ed = mujoco.MjData(self._em)
        mujoco.mj_resetData(self._em, self._ed)
        mujoco.mj_forward(self._em, self._ed)
        self._em_acc = 0.0
        self._em_last = None
        self._last_sync = 0.0

        self._qpos_prev = self._ed.qpos.copy()
        self._qvel_prev = np.zeros(comp.nv)
        self._enabled = False

        self.frcrange = np.asarray(comp.m.actuator_forcerange, dtype=float).copy()
        self.last_tau = np.zeros(comp.nu)

    def enable(self):
        if self._enabled:
            return
        by_bus: dict[str, list] = {}
        for spec in self.specs:
            by_bus.setdefault(spec[0], []).append(spec)
        for bus, group in by_bus.items():
            ev = threading.Event()
            w = _CanWorker(
                bus, group, self._cache, self._cache_lock, self._cmd, self._cmd_lock, ev
            )
            w.start()
            self._workers.append(w)
            self._cmd_events.append(ev)
        self._enabled = True

        if self._want_view and self.viewer is None:
            import mujoco.viewer

            self._vd = mujoco.MjData(self._em)
            self.viewer = mujoco.viewer.launch_passive(
                self._em,
                self._vd,
                show_left_ui=False,
                show_right_ui=False,
                key_callback=lambda k: self._key_cb and self._key_cb(k),
            )
            mujoco.mjv_defaultFreeCamera(self._em, self.viewer.cam)

    def register_key_callback(self, cb):
        self._key_cb = cb

    def _read_motor_state(self):
        with self._cache_lock:
            snap = dict(self._cache)
        pos = np.zeros(self.comp.nu)
        vel = np.zeros(self.comp.nu)
        tau = np.zeros(self.comp.nu)
        online = 0
        for bus, device_id, _, ai in self.specs:
            d = snap.get((bus, device_id))
            if d and d[3]:
                pos[ai] = d[0]
                vel[ai] = d[1]
                tau[ai] = d[2]
                online += 1
        return pos, vel, tau, online

    def current_motor_pos(self):
        return self._read_motor_state()[0]

    def online_count(self):
        return self._read_motor_state()[3]

    def motor_state(self):
        pos, vel, tau, _ = self._read_motor_state()
        return pos, vel, tau

    def seed_embedded(self, steps=400):
        targets = self._read_motor_state()[0]
        self._ed.ctrl[:] = targets
        for _ in range(int(steps)):
            mujoco.mj_step(self._em, self._ed)
        self._qpos_prev = self._ed.qpos.copy()
        self._qvel_prev[:] = 0.0
        self._em_acc = 0.0
        self._em_last = None

    def read(self):
        pos, vel, tau, _ = self._read_motor_state()
        now = perf_counter()
        if self._em_last is None:
            self._em_last = now
        self._em_acc = min(self._em_acc + (now - self._em_last), 0.05)
        self._em_last = now
        ts = float(self._em.opt.timestep)
        n = int(self._em_acc / ts)
        self._ed.ctrl[:] = pos
        if n > 0:
            self._em_acc -= n * ts
            for _ in range(n):
                mujoco.mj_step(self._em, self._ed)
            qpos = self._ed.qpos.copy()
            qvel = np.zeros(self.comp.nv)
            mujoco.mj_differentiatePos(self._em, qvel, n * ts, self._qpos_prev, qpos)
            qacc = (qvel - self._qvel_prev) / (n * ts)
            self._qpos_prev = qpos
            self._qvel_prev = qvel.copy()
        else:
            qpos = self._ed.qpos.copy()
            qvel = self._qvel_prev.copy()
            qacc = np.zeros(self.comp.nv)
        qpos[self.comp.act_qadr] = pos
        qvel[self.comp.act_dofs] = vel
        return qpos, qvel, qacc, tau

    def command(self, kp, kd, pos_des, vel_des, tau_ff):
        tau_ff = np.clip(tau_ff, self.frcrange[:, 0], self.frcrange[:, 1])
        with self._cmd_lock:
            self._cmd[:] = np.column_stack([kp, kd, pos_des, vel_des, tau_ff])
        for ev in self._cmd_events:
            ev.set()
        q = self._ed.qpos[self.comp.act_qadr]
        qd = self._ed.qvel[self.comp.act_dofs]
        self.last_tau = kp * (pos_des - q) + kd * (vel_des - qd) + tau_ff
        return self.last_tau

    def is_running(self):
        if not self._enabled:
            return False
        return self.viewer.is_running() if self.viewer else True

    def sync(self):
        if not self.viewer:
            return
        now = perf_counter()
        if now - self._last_sync < 1.0 / VIEWER_SYNC_HZ:
            return
        self._last_sync = now
        self._vd.qpos[:] = self._ed.qpos
        mujoco.mj_forward(self._em, self._vd)
        self.viewer.sync()

    def disable(self):
        if not self._enabled:
            return
        with self._cmd_lock:
            self._cmd[:] = 0.0
        for ev in self._cmd_events:
            ev.set()
        sleep(max(0.01, 2.0 / CTRL_HZ))
        for w in self._workers:
            w.stop()
        for w in self._workers:
            w.join(0.5)
        self._workers.clear()
        self._cmd_events.clear()
        self._enabled = False
        if self.viewer:
            try:
                self.viewer.close()
            except Exception:
                pass
            self.viewer = None
            self._vd = None


def _bar(frac: float, width: int = 24) -> str:
    n = round(ramp(frac) * width)
    return "█" * n + "░" * (width - n)


def _panel(
    backend,
    comp,
    mode,
    phase,
    a,
    elapsed,
    total,
    comp_on,
    qpos,
    qvel,
    pos_des,
    kp,
    kd,
    tau_ff,
    tau_cmd,
    tau_meas,
    q_target,
):
    t = Table(box=box.SIMPLE_HEAVY, expand=False, pad_edge=False)
    t.add_column("电机/关节", style="cyan", no_wrap=True)
    t.add_column("q", justify="right")
    t.add_column("dq", justify="right")
    t.add_column("目标", justify="right", style="magenta")
    t.add_column("kp", justify="right")
    t.add_column("kd", justify="right")
    t.add_column("τff", justify="right", style="yellow")
    t.add_column("τcmd", justify="right", style="green")
    real = isinstance(backend, RealBackend)
    if real:
        t.add_column("实测τ", justify="right", style="green")

    qa, da = comp.act_qadr, comp.act_dofs
    for i, name in enumerate(comp.act_names):
        short = name.replace("_joint", "").replace("_link", "")
        row = [
            short,
            f"{qpos[qa[i]]:+.3f}",
            f"{qvel[da[i]]:+.3f}",
            f"{q_target[i]:+.3f}",
            f"{kp[i]:7.2f}",
            f"{kd[i]:5.2f}",
            f"{tau_ff[i]:+.2f}",
            f"{tau_cmd[i]:+.2f}",
        ]
        if real:
            row.append(f"{tau_meas[i]:+.2f}")
        t.add_row(*row)

    phase_color = {
        PHASE_START: "bold cyan",
        PHASE_HOLD: "bold green",
        PHASE_STOP: "bold red",
        PHASE_DONE: "dim",
    }[phase]
    on = "[bold green]开 ON[/]" if comp_on else "[bold red]关 OFF[/]"
    head = Text.assemble(
        ("Semi-Taks-LV1 缓启动 / 缓停止   ", "bold white"),
        (f"[{phase}] ", phase_color),
        (f"模式 {mode}（{_MODE_DESC[mode]}）", "bold yellow"),
    )
    total_s = "∞" if (total is None or math.isinf(total)) else f"{total:.1f}"
    prog = f"{_bar(a)} a={a * 100:5.1f}%   {elapsed:5.2f}/{total_s}s"
    info = (
        f"后端 {backend.name}   补偿 {on}   斜坡 {prog}   "
        f"|τ|max {np.abs(tau_cmd).max():.2f}Nm"
    )
    foot = (
        "空格=开/关补偿   Ctrl+右键拖拽连杆   Q=缓停止并退出（关窗亦会补跑缓停止）"
        if backend.interactive
        else "Ctrl+C=缓停止并安全停机（kp/kd/τ 缓慢归零，避免掉落）"
    )
    grp = Table.grid()
    grp.add_row(Text(info, style="dim"))
    grp.add_row(t)
    return Panel(
        grp,
        title=head,
        subtitle=Text(foot, style="dim italic"),
        box=box.HEAVY,
        expand=False,
    )


def run(
    backend,
    comp,
    mode=DEFAULT_MODE,
    comp_on=True,
    t_start=DEFAULT_RAMP_T,
    t_stop=None,
    kp_scale=1.0,
    kd_scale=1.0,
    hold=None,
    targets=None,
    plot=False,
    plot_joints=None,
    plot_window=10.0,
    kpkd=None,
    target=None,
):
    kpkd = JOINT_KPKD if kpkd is None else kpkd
    target = JOINT_TARGET if target is None else target
    mode = _normalize_mode(mode)
    t_stop = t_start if t_stop is None else t_stop
    is_real = isinstance(backend, RealBackend)
    nu = comp.nu
    Z = np.zeros(nu)

    plot_idx = _parse_joint_sel(plot_joints, comp.act_names) if plot else []
    plot_sel = np.asarray(plot_idx, dtype=np.intp)
    vel_des_zero = np.zeros(len(plot_idx))
    plotter = None

    kp_target = np.array([kpkd[n][0] for n in comp.act_names], float) * kp_scale
    kd_target = np.array([kpkd[n][1] for n in comp.act_names], float) * kd_scale
    q_target = _parse_joint_targets(
        targets,
        comp.act_names,
        base=np.array([target[n] for n in comp.act_names], float),
    )

    state = {"comp": comp_on, "stop": False}

    def key_cb(keycode):
        if keycode == 32:
            state["comp"] = not state["comp"]
        elif keycode in (ord("q"), ord("Q")):
            state["stop"] = True

    st = {
        "a": 0.0,
        "qpos": None,
        "qvel": None,
        "tau_meas": Z.copy(),
        "pos_des": Z.copy(),
        "kp": Z.copy(),
        "kd": Z.copy(),
        "tau_ff": Z.copy(),
        "tau_cmd": Z.copy(),
        "phase": PHASE_START,
    }

    sched = None

    def tick(now):
        qpos, qvel, qacc, tau_meas = backend.read()
        q_act = qpos[comp.act_qadr]

        stop_now = ge.stop or state["stop"]
        if (
            not stop_now
            and sched.phase == PHASE_HOLD
            and sched.hold is not None
            and sched._hold_t0 is not None
            and (now - sched._hold_t0) >= sched.hold
        ):
            stop_now = True
        if stop_now and sched.phase in (PHASE_START, PHASE_HOLD):
            sched.request_stop(now, st["a"], q_act.copy())

        phase, a, pos_des = sched.step(now)
        tau_model = (
            comp.tau(qpos, qvel, qacc, mode) if state["comp"] and mode != "none" else Z
        )
        kp = a * kp_target
        kd = a * kd_target
        tau_ff = a * tau_model
        tau_cmd = backend.command(kp, kd, pos_des, Z, tau_ff)

        if plotter is not None and len(plot_sel):
            plotter.push(
                pos_des[plot_sel],
                q_act[plot_sel],
                vel_des_zero,
                qvel[comp.act_dofs][plot_sel],
                tau_cmd[plot_sel],
                tau_meas[plot_sel],
            )

        st.update(
            a=a,
            qpos=qpos,
            qvel=qvel,
            tau_meas=tau_meas,
            pos_des=pos_des,
            kp=kp,
            kd=kd,
            tau_ff=tau_ff,
            tau_cmd=tau_cmd,
            phase=phase,
        )
        return phase

    with GracefulExit() as ge:
        try:
            backend.register_key_callback(key_cb)
            backend.enable()

            dt = backend.dt
            rl = RateLimiter(1.0 / dt if dt > 0 else CTRL_HZ)
            ui_period = 1.0 / UI_HZ
            last_ui = 0.0

            console.print(
                f"[bold]后端：{backend.name}[/]   控制周期 {dt * 1e3:.2f} ms   "
                f"缓启动 {t_start:.1f}s / 缓停止 {t_stop:.1f}s   曲线 smoothstep   "
                f"补偿 {'开' if comp_on else '关'} · 模式 {mode}"
            )
            if is_real:
                console.print(
                    f"[yellow]真机 MIT 目标增益（保守）"
                    f"kp∈[{kp_target.min():.0f},{kp_target.max():.0f}] "
                    f"kd∈[{kd_target.min():.1f},{kd_target.max():.1f}]；"
                    f"如需更稳请 --kp-scale 调小。[/yellow]"
                )
            nz = np.flatnonzero(np.abs(q_target) > 1e-9)
            if nz.size:
                tgt_str = "  ".join(
                    f"{comp.act_names[i].replace('_joint', '')}={q_target[i]:+.3f}"
                    for i in nz
                )
                console.print(f"[magenta]缓启动目标位置（非零关节）：{tgt_str}[/]")
            else:
                console.print("[dim]缓启动目标位置：全部关节 0（默认零位）[/dim]")
            console.print(
                "[dim]缓慢插值 kp/kd/pos/τ：上电软启动→保持→Ctrl+C/Q 软下电。[/dim]\n"
            )

            qpos0, _qv0, _qa0, _tm0 = backend.read()
            if is_real:
                want = len(getattr(backend, "specs", [])) or comp.nu
                deadline = perf_counter() + 3.0
                while backend.online_count() < want and perf_counter() < deadline:
                    backend.read()
                    sleep(0.02)
                n_on = backend.online_count()
                q_start = backend.current_motor_pos()
                backend.seed_embedded()
                console.print(
                    f"[dim]电机在线 {n_on}/{want}，缓启动起点 "
                    f"|q_start|max={np.abs(q_start).max():.3f} rad[/]"
                )
            else:
                q_start = qpos0[comp.act_qadr].copy()

            if plot and plot_idx:
                labels = [comp.act_names[i].replace("_joint", "") for i in plot_idx]
                plotter = PosVelTauPlotter(
                    labels,
                    freq=(1.0 / dt if dt > 0 else CTRL_HZ),
                    window_sec=plot_window,
                )
                console.print(
                    f"[green]绘图已启动：{', '.join(labels)}（目标 vs 真机 pos/vel/τ）[/]"
                )

            sched = RampScheduler(q_start, nu, t_start, t_stop, hold, q_target=q_target)

            with Live(
                console=console, refresh_per_second=UI_HZ, transient=False
            ) as live:
                while backend.is_running():
                    now = perf_counter()
                    phase = tick(now)
                    if phase == PHASE_DONE:
                        break

                    if now - last_ui >= ui_period:
                        last_ui = now
                        if phase == PHASE_STOP:
                            elapsed = now - sched._stop_t0
                            total = t_stop
                        elif phase == PHASE_HOLD:
                            elapsed = now - sched._hold_t0 if sched._hold_t0 else 0.0
                            total = hold if hold is not None else float("inf")
                        else:
                            elapsed = (now - sched._t0) if sched._t0 else 0.0
                            total = t_start
                        live.update(
                            _panel(
                                backend,
                                comp,
                                mode,
                                phase,
                                st["a"],
                                elapsed,
                                total,
                                state["comp"],
                                st["qpos"],
                                st["qvel"],
                                st["pos_des"],
                                st["kp"],
                                st["kd"],
                                st["tau_ff"],
                                st["tau_cmd"],
                                st["tau_meas"],
                                q_target,
                            )
                        )
                    backend.sync()
                    rl.sleep()
        except KeyboardInterrupt:
            ge.stop = True
        finally:
            try:
                if sched is not None and sched.phase != PHASE_DONE:
                    console.print("\n[yellow]补跑缓停止 ramp-down……[/]")
                    rl2 = RateLimiter(1.0 / backend.dt if backend.dt > 0 else CTRL_HZ)
                    if sched.phase in (PHASE_START, PHASE_HOLD):
                        q_now = (
                            st["qpos"][comp.act_qadr].copy()
                            if st["qpos"] is not None
                            else Z.copy()
                        )
                        sched.request_stop(perf_counter(), st["a"], q_now)
                    deadline = perf_counter() + t_stop + 1.0
                    while sched.phase != PHASE_DONE and perf_counter() < deadline:
                        if tick(perf_counter()) == PHASE_DONE:
                            break
                        try:
                            backend.sync()
                        except Exception:
                            pass
                        rl2.sleep()
            except Exception:
                pass

            console.print("[yellow]安全停机中……[/]")
            if plotter is not None:
                try:
                    plotter.close()
                except Exception:
                    pass

            wd = _force_exit_after(SHUTDOWN_TIMEOUT)
            try:
                backend.command(Z, Z, Z, Z, Z)
                backend.disable()
            except Exception:
                pass
            wd.cancel()

    console.print("[green]已安全停机。[/]")


def main():
    ap = argparse.ArgumentParser(
        description="GMR Semi-Taks-LV1 真机控制后端"
    )
    ap.add_argument(
        "--backend",
        choices=["sim", "real"],
        default="real",
        help="后端：sim 纯 MuJoCo 仿真 / real EYou RP CAN-FD 真机（默认 real）",
    )
    ap.add_argument(
        "--no-view", action="store_true", default=False, help="不开 MuJoCo viewer"
    )
    ap.add_argument(
        "--mode",
        choices=MODES,
        default=DEFAULT_MODE,
        help=f"补偿模式 q/qd/qdd/none（none=不补偿；默认 {DEFAULT_MODE}）",
    )
    ap.add_argument(
        "--comp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="缓启动/停时是否加重力补偿前馈（--comp 开 / --no-comp 关；默认关，对照）",
    )
    ap.add_argument(
        "-T",
        "--time",
        type=float,
        default=DEFAULT_RAMP_T,
        help=f"缓启动时长（秒，默认 {DEFAULT_RAMP_T:g}）",
    )
    ap.add_argument(
        "--stop-time",
        type=float,
        default=None,
        help="缓停止时长（秒，默认与缓启动相同）",
    )
    ap.add_argument(
        "--ff-scale", type=float, default=TAU_SCALE, help="重力补偿前馈缩放"
    )
    ap.add_argument("--kp-scale", type=float, default=1.0, help="目标 kp 整体缩放")
    ap.add_argument("--kd-scale", type=float, default=1.0, help="目标 kd 整体缩放")
    ap.add_argument(
        "--hold",
        type=float,
        default=None,
        help="保持 N 秒后自动缓停止退出（默认保持到 Ctrl+C/Q；无人值守可设秒数）",
    )
    ap.add_argument(
        "--targets",
        default=None,
        metavar="SEL",
        help="每关节缓启动目标位置(rad)，覆盖 JOINT_TARGET（默认全 0）；"
        "格式 '关节名子串或序号=值' 逗号分隔，如 'right_elbow=0.6,waist_yaw=-0.2,0=0.1'",
    )
    ap.add_argument(
        "--cans",
        default=None,
        help="真机 CAN 分配，如 'can0:1-7;can1:1-7;can2:1-6'",
    )
    ap.add_argument(
        "--plot",
        type=_bool_arg,
        default=True,
        metavar="BOOL",
        help="实时绘图：目标 vs 真机的 pos/vel/τ 对比（默认 True；--plot False 关闭）",
    )
    ap.add_argument(
        "--plot-joints",
        default="all",
        metavar="SEL",
        help="选择绘图关节：关节名子串或序号，逗号分隔（默认 all；如 'elbow,right_wrist' 或 '0,3,5'）",
    )
    ap.add_argument(
        "--plot-window", type=float, default=10.0, help="绘图时间窗（秒，默认 10）"
    )
    args = ap.parse_args()

    real = args.backend == "real"
    xml = MODEL_XML if not SCENE_XML.exists() else SCENE_XML
    comp = CompModel(xml, tau_scale=args.ff_scale)

    console.print(
        Panel(
            Text.assemble(
                ("模型 ", "dim"),
                (Path(xml).name, "bold"),
                (
                    f"   nu={comp.nu} nv={comp.nv}   总质量 {comp.total_mass:.3f}kg   ",
                    "dim",
                ),
                ("真机" if real else "仿真", "bold magenta"),
            ),
            box=box.MINIMAL,
            expand=False,
        )
    )

    if real:
        try:
            can_map = _parse_can_map(args.cans, DEFAULT_CAN_MAP)
        except ValueError as exc:
            ap.error(f"--cans 解析失败: {exc}")
        backend = RealBackend(comp, xml, can_map, view=not args.no_view)
    else:
        backend = SimBackend(comp, xml, view=not args.no_view)

    run(
        backend,
        comp,
        mode=args.mode,
        comp_on=args.comp,
        t_start=args.time,
        t_stop=args.stop_time,
        kp_scale=args.kp_scale,
        kd_scale=args.kd_scale,
        hold=args.hold,
        targets=args.targets,
        plot=args.plot,
        plot_joints=args.plot_joints,
        plot_window=args.plot_window,
    )


if __name__ == "__main__":
    main()
