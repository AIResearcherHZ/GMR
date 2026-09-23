import os

from .libs import (
    Moteus_Limit_param as _Moteus_Limit_param,
)
from .libs import (
    MoteusControl as _MoteusControl,
)
from .libs import (
    MoteusMotor as _MoteusMotor,
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


class Motor:
    __slots__ = ("NodeID", "_motor")

    def __init__(self, node_id: int):
        self.NodeID = node_id
        self._motor = _MoteusMotor(node_id)

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

    def getVoltage(self) -> float:
        return self._motor.getVoltage()

    def getTemperature(self) -> float:
        return self._motor.getTemperature()

    def getMode(self) -> int:
        return self._motor.getMode()

    def getFault(self) -> int:
        return self._motor.getFault()

    def get_limit_param(self) -> _Moteus_Limit_param:
        return self._motor.get_limit_param()

    def set_limit(self, lp: _Moteus_Limit_param):
        self._motor.set_limit(lp)


class MoteusControl:
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
        "can_interface",
    )

    def __init__(
        self,
        can_interface: str = "can0",
        motors: list | None = None,
        silent: bool = False,
        address: str | None = None,
        port: int | None = None,
        ca_cert: str | None = None,
        client_cert: str | None = None,
        client_key: str | None = None,
        **kwargs,
    ):
        self.can_interface = can_interface
        self._silent = silent
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
                remote.MoteusMotor(m.NodeID) for m in self._motors.values()
            ]
            self._motor_control = remote.MoteusControl(
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
            self._motor_control = _MoteusControl(
                self.can_interface, rust_motors, self._silent
            )
        self._initialized = True
        for motor in self._motors.values():
            internal = self._motor_control.getMotor(motor.NodeID)
            if internal:
                motor._motor = internal

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

    def control_Torque(self, motor: Motor, tau_nm: float, debug: bool = False):
        self._ensure_initialized()
        if self._motor_control:
            self._motor_control.control_Torque(motor._motor, tau_nm)
            if debug:
                print(f"[DEBUG] Torque控制: J{motor.NodeID} -> tau={tau_nm:.6f}")

    def set_watchdog_timeout(self, motor: Motor, timeout_s: float):
        self._ensure_initialized()
        if self._motor_control:
            self._motor_control.set_watchdog_timeout(motor._motor, timeout_s)

    def set_velocity_limit(self, motor: Motor, vel_rads: float):
        self._ensure_initialized()
        if self._motor_control:
            self._motor_control.set_velocity_limit(motor._motor, vel_rads)

    def set_accel_limit(self, motor: Motor, acc_rads2: float):
        self._ensure_initialized()
        if self._motor_control:
            self._motor_control.set_accel_limit(motor._motor, acc_rads2)

    def refresh_motor_status(self, motor: Motor):
        self._ensure_initialized()
        if self._motor_control:
            self._motor_control.refresh_motor_status(motor._motor)

    def fault_reset(self, motor: Motor):
        self._ensure_initialized()
        if self._motor_control:
            self._motor_control.fault_reset(motor._motor)

    def set_zero_position(self, motor: Motor):
        self._ensure_initialized()
        if self._motor_control:
            self._motor_control.set_zero_position(motor._motor)

    def save_params(self, motor: Motor):
        self._ensure_initialized()
        if self._motor_control:
            self._motor_control.save_params(motor._motor)

    def read_config(self, motor: Motor, name: str) -> float:
        self._ensure_initialized()
        if not self._motor_control:
            return 0.0
        if self._is_remote:
            resp = self._motor_control.read(motor._motor, "read_config", [], name)
            return resp.get("value", 0.0) if resp.get("ok") else 0.0
        return self._motor_control.read_config(motor._motor, name)

    def write_config(self, motor: Motor, name: str, value: float):
        self._ensure_initialized()
        if self._motor_control:
            self._motor_control.write_config(motor._motor, name, value)

    def changeMotorLimit(
        self, motor: Motor, q_max: float, dq_max: float, tau_max: float
    ):
        self._ensure_initialized()
        if self._motor_control:
            if self._is_remote:
                self._motor_control.changeMotorLimit(
                    motor._motor, q_max, dq_max, tau_max
                )
            else:
                _MoteusControl.changeMotorLimit(motor._motor, q_max, dq_max, tau_max)

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
