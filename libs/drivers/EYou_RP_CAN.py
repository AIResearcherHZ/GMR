import os
from enum import IntEnum

from .libs import (
    EYouRp_Limit_param as _EYouRp_Limit_param,
)
from .libs import (
    EYouRp_Motor_Type as _EYouRp_Motor_Type,
)
from .libs import (
    EYouRpControl as _EYouRpControl,
)
from .libs import (
    EYouRpMotor as _EYouRpMotor,
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


class EYouRp_Motor_Type(IntEnum):
    RP40C = 0
    RP50L = 1
    RP50H = 2
    RP70L = 3
    RP70H = 4
    RP90L = 5


def _convert_motor_type(mt: EYouRp_Motor_Type):
    return getattr(_EYouRp_Motor_Type, mt.name)


class Motor:
    __slots__ = ("MotorType", "NodeID", "_motor")

    def __init__(self, motor_type: EYouRp_Motor_Type, node_id: int):
        self.MotorType = motor_type
        self.NodeID = node_id
        self._motor = _EYouRpMotor(_convert_motor_type(motor_type), node_id)

    def getNodeId(self) -> int:
        return self._motor.getNodeId()

    def state(self) -> tuple:
        return self._motor.state()

    def getPosition(self) -> float:
        return self._motor.getPosition()

    def getVelocity(self) -> float:
        return self._motor.getVelocity()

    def getTorque(self) -> float:
        return self._motor.getTorque()

    def getFeedbackAge(self) -> float:
        return self._motor.getFeedbackAge()

    def getStatusWord(self) -> int:
        return self._motor.getStatusWord()

    def getErrorCode(self) -> int:
        return self._motor.getErrorCode()

    def get_limit_param(self) -> _EYouRp_Limit_param:
        return self._motor.get_limit_param()


class EYouRpControl:
    __slots__ = (
        "_address",
        "_ca_cert",
        "_client_cert",
        "_client_key",
        "_initialized",
        "_is_remote",
        "_motor_control",
        "_motors",
        "_port",
        "_silent",
        "_sync_hz",
        "can_interface",
    )

    def __init__(
        self,
        can_interface: str = "can0",
        motors: list | None = None,
        silent: bool = False,
        sync_hz: float = 1000.0,
        address: str | None = None,
        port: int | None = None,
        ca_cert: str | None = None,
        client_cert: str | None = None,
        client_key: str | None = None,
        **kwargs,
    ):
        self.can_interface = can_interface
        self._silent = silent
        self._sync_hz = sync_hz
        self._motors: dict[int, Motor] = {}
        self._motor_control = None
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
        if motors:
            for m in motors:
                self.addMotor(m)

    def addMotor(self, motor: Motor):
        self._motors[motor.NodeID] = motor

    def init(self):
        if self._initialized:
            return
        if self._is_remote:
            remote = _taks_driver.remote
            remote_motors = [
                remote.EYouRpMotor(_convert_motor_type(m.MotorType), m.NodeID)
                for m in self._motors.values()
            ]
            self._motor_control = remote.EYouRpControl(
                self.can_interface,
                self._address,
                self._port,
                remote_motors,
                self._ca_cert,
                self._client_cert,
                self._client_key,
            )
        else:
            rust_motors = [m._motor for m in self._motors.values()]
            self._motor_control = _EYouRpControl(
                self.can_interface, rust_motors, self._silent, self._sync_hz
            )
        self._initialized = True
        for motor in self._motors.values():
            internal = self._motor_control.getMotor(motor.NodeID)
            if internal:
                motor._motor = internal

    def _ensure_initialized(self):
        if not self._initialized:
            self.init()

    def getMotor(self, node_id: int) -> Motor | None:
        return self._motors.get(node_id)

    def enable(self, motor: Motor):
        self._ensure_initialized()
        if self._motor_control:
            self._motor_control.enable(motor._motor)

    def disable(self, motor: Motor):
        self._ensure_initialized()
        if self._motor_control:
            self._motor_control.disable(motor._motor)

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
        if self._motor_control:
            self._motor_control.controlMIT(motor._motor, kp, kd, q, dq, tau)
            if debug:
                print(
                    f"[DEBUG] MIT控制: J{motor.NodeID} (kp={kp}, kd={kd}) -> "
                    f"q={q:.6f}, dq={dq:.6f}, tau={tau:.6f}"
                )

    def send_sync(self):
        self._ensure_initialized()
        if self._motor_control:
            self._motor_control.send_sync()

    def bus_stats(self) -> dict:
        self._ensure_initialized()
        if not self._motor_control:
            return {}
        if self._is_remote:
            m = next(iter(self._motors.values()))
            resp = self._motor_control.read(m._motor, "bus_stats", [], "")
            if resp.get("ok") and resp.get("text"):
                import json

                try:
                    return json.loads(resp["text"])
                except Exception:
                    return {}
            return {}
        return self._motor_control.bus_stats()

    def control_Pos(self, motor: Motor, pos_rad: float, debug: bool = False):
        self._ensure_initialized()
        if self._motor_control:
            self._motor_control.control_Pos(motor._motor, pos_rad)
            if debug:
                print(f"[DEBUG] Pos控制: J{motor.NodeID} -> pos={pos_rad:.6f}")

    def control_Vel(self, motor: Motor, vel_rads: float, debug: bool = False):
        self._ensure_initialized()
        if self._motor_control:
            self._motor_control.control_Vel(motor._motor, vel_rads)
            if debug:
                print(f"[DEBUG] Vel控制: J{motor.NodeID} -> vel={vel_rads:.6f}")

    def control_Torque(self, motor: Motor, torque_raw: int, debug: bool = False):
        self._ensure_initialized()
        if self._motor_control:
            self._motor_control.control_Torque(motor._motor, torque_raw)
            if debug:
                print(f"[DEBUG] Torque控制: J{motor.NodeID} -> raw={torque_raw}")

    def set_profile_velocity(self, motor: Motor, vel_rads: float):
        self._ensure_initialized()
        if self._motor_control:
            self._motor_control.set_profile_velocity(motor._motor, vel_rads)

    def set_profile_acceleration(self, motor: Motor, acc_rads2: float):
        self._ensure_initialized()
        if self._motor_control:
            self._motor_control.set_profile_acceleration(motor._motor, acc_rads2)

    def set_profile_deceleration(self, motor: Motor, dec_rads2: float):
        self._ensure_initialized()
        if self._motor_control:
            self._motor_control.set_profile_deceleration(motor._motor, dec_rads2)

    def switch_mode_mit(self, motor: Motor):
        self._ensure_initialized()
        if self._motor_control:
            self._motor_control.switch_mode_mit(motor._motor)

    def switch_mode_pp(self, motor: Motor):
        self._ensure_initialized()
        if self._motor_control:
            self._motor_control.switch_mode_pp(motor._motor)

    def switch_mode_pv(self, motor: Motor):
        self._ensure_initialized()
        if self._motor_control:
            self._motor_control.switch_mode_pv(motor._motor)

    def switch_mode_pt(self, motor: Motor):
        self._ensure_initialized()
        if self._motor_control:
            self._motor_control.switch_mode_pt(motor._motor)

    def switch_mode_csp(self, motor: Motor):
        self._ensure_initialized()
        if self._motor_control:
            self._motor_control.switch_mode_csp(motor._motor)

    def switch_mode_csv(self, motor: Motor):
        self._ensure_initialized()
        if self._motor_control:
            self._motor_control.switch_mode_csv(motor._motor)

    def switch_mode_cst(self, motor: Motor):
        self._ensure_initialized()
        if self._motor_control:
            self._motor_control.switch_mode_cst(motor._motor)

    def set_zero_position(self, motor: Motor):
        self._ensure_initialized()
        if self._motor_control:
            self._motor_control.set_zero_position(motor._motor)

    def save_params(self, motor: Motor):
        self._ensure_initialized()
        if self._motor_control:
            self._motor_control.save_params(motor._motor)

    def set_node_id_parameter(self, motor: Motor, new_node_id: int):
        self._ensure_initialized()
        if self._motor_control:
            self._motor_control.set_node_id_parameter(motor._motor, new_node_id)

    def fault_reset(self, motor: Motor):
        self._ensure_initialized()
        if self._motor_control:
            self._motor_control.fault_reset(motor._motor)

    def read_status_word(self, motor: Motor) -> int:
        self._ensure_initialized()
        if not self._motor_control:
            return 0
        if self._is_remote:
            resp = self._motor_control.read(motor._motor, "read_status_word", [], "")
            return int(resp.get("value", 0)) if resp.get("ok") else 0
        return self._motor_control.read_status_word(motor._motor)

    def read_mode_display(self, motor: Motor) -> int:
        self._ensure_initialized()
        if not self._motor_control:
            return 0
        if self._is_remote:
            resp = self._motor_control.read(motor._motor, "read_mode_display", [], "")
            return int(resp.get("value", 0)) if resp.get("ok") else 0
        return self._motor_control.read_mode_display(motor._motor)

    def read_error_code(self, motor: Motor) -> int:
        self._ensure_initialized()
        if not self._motor_control:
            return 0
        if self._is_remote:
            resp = self._motor_control.read(motor._motor, "read_error_code", [], "")
            return int(resp.get("value", 0)) if resp.get("ok") else 0
        return self._motor_control.read_error_code(motor._motor)

    def read_actual_position_counts(self, motor: Motor) -> int:
        self._ensure_initialized()
        if not self._motor_control:
            return 0
        if self._is_remote:
            resp = self._motor_control.read(
                motor._motor, "read_actual_position_counts", [], ""
            )
            return int(resp.get("value", 0)) if resp.get("ok") else 0
        return self._motor_control.read_actual_position_counts(motor._motor)

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
        self._ensure_initialized()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
