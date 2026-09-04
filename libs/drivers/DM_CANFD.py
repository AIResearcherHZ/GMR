import os
from enum import IntEnum

from .libs import (
    Control_Mode as _Control_Mode,
)
from .libs import (
    Control_Mode_Code as _Control_Mode_Code,
)
from .libs import (
    DM_Motor_Type as _DM_Motor_Type,
)
from .libs import (
    DmActData as _DmActData,
)
from .libs import (
    Limit_param as _Limit_param,
)
from .libs import (
    MotorControl as _MotorControl,
)
from .libs import (
    taks_driver as _taks_driver,
)

_CERT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "certs")
_DEFAULT_CA_CERT = os.path.join(_CERT_DIR, "ca-cert.pem")
_DEFAULT_CLIENT_CERT = os.path.join(_CERT_DIR, "client-cert.pem")
_DEFAULT_CLIENT_KEY = os.path.join(_CERT_DIR, "client-key.pem")
_DEFAULT_REMOTE_PORT = 5555


def _parse_address(address):
    if ":" in address:
        host, _, port = address.rpartition(":")
        return host, int(port)
    return address, _DEFAULT_REMOTE_PORT


class DM_Motor_Type(IntEnum):
    DM4310 = 0
    DM4310_48V = 1
    DM4340 = 2
    DM4340_48V = 3
    DM6006 = 4
    DM8006 = 5
    DM8009 = 6
    DM10010L = 7
    DM10010 = 8
    DMH3510 = 9
    DMG6215 = 10
    DMH6220 = 11
    DMJH11 = 12
    DM6248P = 13
    DM3507 = 14
    DM4310P = 15
    DM4340P = 16
    DMS3519 = 17


class Control_Mode(IntEnum):
    MIT_MODE = 0x000
    POS_VEL_MODE = 0x100
    VEL_MODE = 0x200
    POS_FORCE_MODE = 0x300


class Control_Type(IntEnum):
    MIT = 1
    POS_VEL = 2
    VEL = 3
    POS_FORCE = 4


def _load_joint_position_limits():
    from .libs import get_joint_position_limits

    return get_joint_position_limits()


JOINT_POSITION_LIMITS = _load_joint_position_limits()


def _convert_motor_type(mt: DM_Motor_Type):
    return getattr(_DM_Motor_Type, mt.name)


def _convert_control_type(ct: Control_Type):
    return getattr(_Control_Mode_Code, ct.name)


_ABSENT_STATE = (0.0, 0.0, 0.0, float("inf"), 0, 0, 0)


class Motor:
    __slots__ = ("MasterID", "MotorType", "SlaveID", "_motor")

    def __init__(
        self,
        motor_type: DM_Motor_Type,
        slave_id: int,
        master_id: int,
    ):
        self.MotorType = motor_type
        self.SlaveID = slave_id
        self.MasterID = master_id
        self._motor = None

    def _set_internal_motor(self, motor):
        self._motor = motor

    def state(self) -> tuple:
        return self._motor.state() if self._motor else _ABSENT_STATE

    def getPosition(self) -> float:
        return self._motor.getPosition() if self._motor else 0.0

    def getVelocity(self) -> float:
        return self._motor.getVelocity() if self._motor else 0.0

    def getTorque(self) -> float:
        return self._motor.getTorque() if self._motor else 0.0

    def getFeedbackAge(self) -> float:
        return self._motor.getFeedbackAge() if self._motor else float("inf")

    def getError(self) -> int:
        return self._motor.getError() if self._motor else 0

    def getTMos(self) -> int:
        return self._motor.getTMos() if self._motor else 0

    def getTRotor(self) -> int:
        return self._motor.getTRotor() if self._motor else 0

    def get_limit_param(self):
        if self._motor:
            return self._motor.get_limit_param()
        return _Limit_param()


class MotorControl:
    __slots__ = (
        "_address",
        "_ca_cert",
        "_client_cert",
        "_client_key",
        "_data_list",
        "_initialized",
        "_is_remote",
        "_motor_control",
        "_motors",
        "_port",
        "_send_throttle_us",
        "_silent",
        "can_interface",
    )

    def __init__(
        self,
        can_interface: str = "can0",
        silent: bool = False,
        send_throttle_us: int = 0,
        address: str | None = None,
        port: int | None = None,
        ca_cert: str | None = None,
        client_cert: str | None = None,
        client_key: str | None = None,
        **kwargs,
    ):
        self.can_interface = can_interface
        self._silent = silent
        self._send_throttle_us = send_throttle_us
        self._motors: dict[int, Motor] = {}
        self._motor_control: _MotorControl | None = None
        self._data_list = []
        self._initialized = False
        if address is not None:
            self._is_remote = True
            host, default_port = _parse_address(address)
            self._address = host
            self._port = port if port is not None else default_port
            self._ca_cert = ca_cert or _DEFAULT_CA_CERT
            self._client_cert = client_cert or _DEFAULT_CLIENT_CERT
            self._client_key = client_key or _DEFAULT_CLIENT_KEY
        else:
            self._is_remote = False
            self._address = None
            self._port = 0
            self._ca_cert = None
            self._client_cert = None
            self._client_key = None

    def bus_stats(self) -> dict:
        self._ensure_initialized()
        if not self._motor_control:
            return {}
        if self._is_remote:
            resp = self._motor_control.read(
                self._motors[next(iter(self._motors.keys()))]._motor,
                "bus_stats",
                [],
                "",
            )
            if resp.get("ok") and resp.get("text"):
                import json

                try:
                    return json.loads(resp["text"])
                except Exception:
                    return {}
            return {}
        return self._motor_control.bus_stats()

    def _ensure_initialized(self):
        if self._initialized or not self._data_list:
            return
        if self._is_remote:
            remote = _taks_driver.remote
            remote_motors = [
                remote.Motor(
                    _convert_motor_type(m.MotorType),
                    _Control_Mode.MIT_MODE,
                    m.SlaveID,
                    m.MasterID,
                )
                for m in self._motors.values()
            ]
            self._motor_control = remote.MotorControl(
                self.can_interface,
                self._address,
                self._port,
                remote_motors,
                self._ca_cert,
                self._client_cert,
                self._client_key,
            )
        else:
            self._motor_control = _MotorControl(
                self.can_interface, self._data_list, self._silent
            )
        self._initialized = True
        self._data_list = []
        for motor in self._motors.values():
            internal = self._motor_control.getMotor(motor.SlaveID)
            if internal:
                motor._set_internal_motor(internal)

    def addMotor(self, motor: Motor):
        if self._initialized:
            raise RuntimeError(
                "addMotor must be called before any control call initializes the bus"
            )
        self._motors[motor.SlaveID] = motor
        self._data_list.append(
            _DmActData(
                _convert_motor_type(motor.MotorType),
                _Control_Mode.MIT_MODE,
                motor.SlaveID,
                motor.MasterID,
            )
        )

    def getMotor(self, motor_id: int) -> Motor | None:
        return self._motors.get(motor_id)

    def enable(self, motor: Motor):
        self._ensure_initialized()
        internal = motor._motor
        if internal and self._motor_control:
            self._motor_control.enable(internal)

    def disable(self, motor: Motor):
        internal = motor._motor
        if internal and self._motor_control:
            self._motor_control.disable(internal)

    def controlMIT(
        self,
        motor: Motor,
        kp: float,
        kd: float,
        q: float,
        dq: float,
        tau: float,
        debug: bool = False,
    ):
        self._ensure_initialized()
        internal = motor._motor
        if internal and self._motor_control:
            self._motor_control.controlMIT(internal, kp, kd, q, dq, tau)
            if debug and motor.SlaveID in (0x04, 0x34):
                print(
                    f"[DEBUG] MIT: J{motor.SlaveID} (kp={kp}, kd={kd}) -> "
                    f"q={q:.6f}, dq={dq:.6f}, tau={tau:.6f}"
                )

    def control_Pos_Vel(
        self, motor: Motor, pos: float, vel: float, debug: bool = False
    ):
        self._ensure_initialized()
        internal = motor._motor
        if internal and self._motor_control:
            self._motor_control.control_Pos_Vel(internal, pos, vel)
            if debug:
                print(
                    f"[DEBUG] Pos_Vel: J{motor.SlaveID} -> pos={pos:.6f}, vel={vel:.6f}"
                )

    def control_Vel(self, motor: Motor, vel: float, debug: bool = False):
        self._ensure_initialized()
        internal = motor._motor
        if internal and self._motor_control:
            self._motor_control.control_Vel(internal, vel)
            if debug:
                print(f"[DEBUG] Vel: J{motor.SlaveID} -> vel={vel:.6f}")

    def control_Pos_Force(
        self,
        motor: Motor,
        pos: float,
        vel_limit: float,
        i_des: float,
        debug: bool = False,
    ):
        self._ensure_initialized()
        internal = motor._motor
        if internal and self._motor_control:
            self._motor_control.control_Pos_Force(internal, pos, vel_limit, i_des)
            if debug:
                print(
                    f"[DEBUG] Pos_Force: J{motor.SlaveID} -> "
                    f"pos={pos:.6f}, vel_limit={vel_limit:.4f}, i_des={i_des:.6f}"
                )

    def set_zero_position(self, motor: Motor):
        self._ensure_initialized()
        internal = motor._motor
        if internal and self._motor_control:
            self._motor_control.set_zero_position(internal)

    def refresh_motor_status(self, motor: Motor):
        self._ensure_initialized()
        internal = motor._motor
        if not internal or not self._motor_control:
            return
        if self._is_remote:
            self._motor_control.send_cmd(internal, "refresh_motor_status", [], "")
        else:
            self._motor_control.refresh_motor_status(internal)

    def read_motor_param(
        self, motor: Motor, RID: int, timeout: float = 0.2, strict: bool = False
    ) -> float | None:
        self._ensure_initialized()
        internal = motor._motor
        if not internal or not self._motor_control:
            if strict:
                raise RuntimeError(
                    f"电机 0x{motor.SlaveID:02X} 未注册到该 MotorControl 或驱动未初始化"
                )
            return None
        if self._is_remote:
            resp = self._motor_control.read(
                internal,
                "read_motor_param",
                [float(RID), timeout * 1000],
                "",
            )
            if resp.get("ok"):
                return resp.get("value")
            if strict:
                raise RuntimeError(
                    f"电机 0x{motor.SlaveID:02X} 读寄存器 {RID} 失败: {resp.get('text', '')}"
                )
            print(
                f"[DM] 电机 0x{motor.SlaveID:02X} 读寄存器 {RID} 失败: {resp.get('text', '')}"
            )
            return None
        try:
            return self._motor_control.read_motor_param(
                internal, RID, int(timeout * 1000)
            )
        except RuntimeError as exc:
            if strict:
                raise
            print(f"[DM] 电机 0x{motor.SlaveID:02X} 读寄存器 {RID} 失败: {exc}")
            return None

    def switchControlMode(self, motor: Motor, mode: Control_Type) -> bool:
        self._ensure_initialized()
        internal = motor._motor
        if not internal or not self._motor_control:
            return False
        if self._is_remote:
            try:
                resp = self._motor_control.send_cmd(
                    internal,
                    "switch_control_mode",
                    [float(mode.value)],
                    "",
                )
                return bool(resp.get("ok"))
            except RuntimeError as exc:
                print(f"[DM] 电机 0x{motor.SlaveID:02X} 切换模式失败: {exc}")
                return False
        try:
            self._motor_control.switchControlMode(internal, _convert_control_type(mode))
            return True
        except RuntimeError as exc:
            print(f"[DM] 电机 0x{motor.SlaveID:02X} 切换模式失败: {exc}")
            return False

    def change_motor_param(self, motor: Motor, RID: int, data: float) -> bool:
        self._ensure_initialized()
        internal = motor._motor
        if not internal or not self._motor_control:
            return False
        if self._is_remote:
            try:
                self._motor_control.send_cmd(
                    internal,
                    "change_motor_param",
                    [float(RID), float(data)],
                    "",
                )
                return True
            except RuntimeError as exc:
                print(f"[DM] 电机 0x{motor.SlaveID:02X} 写寄存器 {RID} 失败: {exc}")
                return False
        try:
            self._motor_control.change_motor_param(internal, RID, data)
            return True
        except RuntimeError as exc:
            print(f"[DM] 电机 0x{motor.SlaveID:02X} 写寄存器 {RID} 失败: {exc}")
            return False

    def save_motor_param(self, motor: Motor) -> bool:
        self._ensure_initialized()
        internal = motor._motor
        if not internal or not self._motor_control:
            return False
        if self._is_remote:
            try:
                self._motor_control.send_cmd(internal, "save_motor_param", [], "")
                return True
            except RuntimeError as exc:
                print(f"[DM] 电机 0x{motor.SlaveID:02X} 保存参数失败: {exc}")
                return False
        try:
            self._motor_control.save_motor_param(internal)
            return True
        except RuntimeError as exc:
            print(f"[DM] 电机 0x{motor.SlaveID:02X} 保存参数失败: {exc}")
            return False

    def close(self):
        if self._motor_control:
            try:
                for motor in self._motors.values():
                    self._motor_control.disable(motor._motor)
            except Exception:
                pass
        self._initialized = False
        self._motor_control = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
