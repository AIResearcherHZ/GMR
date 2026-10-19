import os
from enum import IntEnum

from .libs import (
    EYouRp_Limit_param as _EYouRp_Limit_param,
)
from .libs import (
    EYouRp_Motor_Type as _EYouRp_Motor_Type,
)
from .libs import (
    EYouRpCanfdControl as _EYouRpCanfdControl,
)
from .libs import (
    EYouRpCanfdMotor as _EYouRpCanfdMotor,
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
    __slots__ = ("DeviceID", "MotorType", "_motor")

    def __init__(self, motor_type: EYouRp_Motor_Type, device_id: int):
        self.MotorType = motor_type
        self.DeviceID = device_id
        self._motor = _EYouRpCanfdMotor(_convert_motor_type(motor_type), device_id)

    def getDeviceId(self) -> int:
        return self._motor.getDeviceId()

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

    def getStatus(self) -> int:
        return self._motor.getStatus()

    def getMosTemp(self) -> float:
        return self._motor.getMosTemp()

    def getCoilTemp(self) -> float:
        return self._motor.getCoilTemp()

    def getVoltage(self) -> float:
        return self._motor.getVoltage()

    def get_limit_param(self) -> _EYouRp_Limit_param:
        return self._motor.get_limit_param()


class EYouRpCanfdControl:
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
        self._motors[motor.DeviceID] = motor

    def init(self):
        if self._initialized:
            return
        if self._is_remote:
            remote = _taks_driver.remote
            remote_motors = [
                remote.EYouRpCanfdMotor(_convert_motor_type(m.MotorType), m.DeviceID)
                for m in self._motors.values()
            ]
            self._motor_control = remote.EYouRpCanfdControl(
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
            self._motor_control = _EYouRpCanfdControl(
                self.can_interface, rust_motors, self._silent
            )
        self._initialized = True
        for motor in self._motors.values():
            internal = self._motor_control.getMotor(motor.DeviceID)
            if internal:
                motor._motor = internal

    def _ensure_initialized(self):
        if not self._initialized:
            self.init()

    def getMotor(self, device_id: int) -> Motor | None:
        return self._motors.get(device_id)

    def enable(self):
        self._ensure_initialized()
        if self._motor_control:
            self._motor_control.enable()

    def disable(self):
        if self._motor_control:
            self._motor_control.disable()

    def controlMIT(self, commands: list, debug: bool = False):
        self._ensure_initialized()
        if self._motor_control:
            self._motor_control.controlMIT(
                [(m._motor, kp, kd, q, dq, tau) for m, kp, kd, q, dq, tau in commands]
            )
            if debug:
                for m, kp, kd, q, dq, tau in commands:
                    if m.DeviceID in (0x04,):
                        print(
                            f"[DEBUG] MIT控制: J{m.DeviceID} (kp={kp}, kd={kd}) -> "
                            f"q={q:.6f}, dq={dq:.6f}, tau={tau:.6f}"
                        )

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
                print(f"[DEBUG] Pos控制: J{motor.DeviceID} -> pos={pos_rad:.6f}")

    def control_Vel(self, motor: Motor, vel_rads: float, debug: bool = False):
        self._ensure_initialized()
        if self._motor_control:
            self._motor_control.control_Vel(motor._motor, vel_rads)
            if debug:
                print(f"[DEBUG] Vel控制: J{motor.DeviceID} -> vel={vel_rads:.6f}")

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

    def set_zero_position(self, motor: Motor):
        self._ensure_initialized()
        if self._motor_control:
            self._motor_control.set_zero_position(motor._motor)

    def fault_reset(self, motor: Motor):
        self._ensure_initialized()
        if self._motor_control:
            self._motor_control.fault_reset(motor._motor)

    def set_device_id_parameter(self, motor: Motor, new_device_id: int):
        self._ensure_initialized()
        if self._motor_control:
            self._motor_control.set_device_id_parameter(motor._motor, new_device_id)

    def set_profile_acceleration(self, motor: Motor, acc_rads2: float):
        self._ensure_initialized()
        if self._motor_control:
            self._motor_control.set_profile_acceleration(motor._motor, acc_rads2)

    def set_profile_velocity(self, motor: Motor, vel_rads: float):
        self._ensure_initialized()
        if self._motor_control:
            self._motor_control.set_profile_velocity(motor._motor, vel_rads)

    def set_profile_deceleration(self, motor: Motor, dec_rads2: float):
        self._ensure_initialized()
        if self._motor_control:
            self._motor_control.set_profile_deceleration(motor._motor, dec_rads2)

    def set_position_gain(self, motor: Motor, gain_p: float):
        self._ensure_initialized()
        if self._motor_control:
            self._motor_control.set_position_gain(motor._motor, gain_p)

    def set_velocity_gain(self, motor: Motor, gain_p: float, gain_i: float):
        self._ensure_initialized()
        if self._motor_control:
            self._motor_control.set_velocity_gain(motor._motor, gain_p, gain_i)

    def set_velocity_filter_cutoff(self, motor: Motor, cutoff_hz: float):
        self._ensure_initialized()
        if self._motor_control:
            self._motor_control.set_velocity_filter_cutoff(motor._motor, cutoff_hz)

    def set_current_limit(self, motor: Motor, amps: float):
        self._ensure_initialized()
        if self._motor_control:
            self._motor_control.set_current_limit(motor._motor, amps)

    def set_torque_limit(self, motor: Motor, nm: float):
        self._ensure_initialized()
        if self._motor_control:
            self._motor_control.set_torque_limit(motor._motor, nm)

    def set_comm_config(
        self,
        motor: Motor,
        arb_baud: int,
        arb_sample_point: int,
        data_baud: int,
        data_sample_point: int,
    ):
        self._ensure_initialized()
        if self._motor_control:
            if self._is_remote:
                self._motor_control.send_cmd(
                    motor._motor,
                    "set_comm_config",
                    [
                        float(arb_baud),
                        float(arb_sample_point),
                        float(data_baud),
                        float(data_sample_point),
                    ],
                    "",
                )
            else:
                self._motor_control.set_comm_config(
                    motor._motor,
                    arb_baud,
                    arb_sample_point,
                    data_baud,
                    data_sample_point,
                )

    def read_comm_config(self, motor: Motor) -> tuple[int, int, int, int]:
        self._ensure_initialized()
        if not self._motor_control:
            return (0, 0, 0, 0)
        if self._is_remote:
            resp = self._motor_control.read(motor._motor, "read_comm_config", [], "")
            if resp.get("ok"):
                vs = resp.get("values", [])
                if len(vs) >= 4:
                    return (int(vs[0]), int(vs[1]), int(vs[2]), int(vs[3]))
            return (0, 0, 0, 0)
        return self._motor_control.read_comm_config(motor._motor)

    def read_product_model(self, motor: Motor) -> str:
        self._ensure_initialized()
        if not self._motor_control:
            return ""
        if self._is_remote:
            resp = self._motor_control.read(motor._motor, "read_product_model", [], "")
            return resp.get("text", "") if resp.get("ok") else ""
        return self._motor_control.read_product_model(motor._motor)

    def read_serial_number(self, motor: Motor) -> str:
        self._ensure_initialized()
        if not self._motor_control:
            return ""
        if self._is_remote:
            resp = self._motor_control.read(motor._motor, "read_serial_number", [], "")
            return resp.get("text", "") if resp.get("ok") else ""
        return self._motor_control.read_serial_number(motor._motor)

    def read_hardware_version(self, motor: Motor) -> int:
        self._ensure_initialized()
        if not self._motor_control:
            return 0
        if self._is_remote:
            resp = self._motor_control.read(
                motor._motor, "read_hardware_version", [], ""
            )
            return int(resp.get("value", 0)) if resp.get("ok") else 0
        return self._motor_control.read_hardware_version(motor._motor)

    def read_software_version(self, motor: Motor) -> int:
        self._ensure_initialized()
        if not self._motor_control:
            return 0
        if self._is_remote:
            resp = self._motor_control.read(
                motor._motor, "read_software_version", [], ""
            )
            return int(resp.get("value", 0)) if resp.get("ok") else 0
        return self._motor_control.read_software_version(motor._motor)

    def set_product_code(self, motor: Motor, code: str):
        self._ensure_initialized()
        if self._motor_control:
            if self._is_remote:
                self._motor_control.send_cmd(motor._motor, "set_product_code", [], code)
            else:
                self._motor_control.set_product_code(motor._motor, code)

    def set_serial_number(self, motor: Motor, sn: str):
        self._ensure_initialized()
        if self._motor_control:
            if self._is_remote:
                self._motor_control.send_cmd(motor._motor, "set_serial_number", [], sn)
            else:
                self._motor_control.set_serial_number(motor._motor, sn)

    def set_hardware_version(self, motor: Motor, ver: int):
        self._ensure_initialized()
        if self._motor_control:
            if self._is_remote:
                self._motor_control.send_cmd(
                    motor._motor, "set_hardware_version", [float(ver)], ""
                )
            else:
                self._motor_control.set_hardware_version(motor._motor, ver)

    def set_software_version(self, motor: Motor, ver: int):
        self._ensure_initialized()
        if self._motor_control:
            if self._is_remote:
                self._motor_control.send_cmd(
                    motor._motor, "set_software_version", [float(ver)], ""
                )
            else:
                self._motor_control.set_software_version(motor._motor, ver)

    def read_mit_range(self, motor: Motor) -> tuple[float, ...]:
        self._ensure_initialized()
        if not self._motor_control:
            return (0.0,) * 10
        if self._is_remote:
            resp = self._motor_control.read(motor._motor, "read_mit_range", [], "")
            if resp.get("ok"):
                return tuple(resp.get("values", []))
            return (0.0,) * 10
        return self._motor_control.read_mit_range(motor._motor)

    def set_mit_range(
        self,
        motor: Motor,
        p_min: float,
        p_max: float,
        v_min: float,
        v_max: float,
        t_min: float,
        t_max: float,
        kp_min: float,
        kp_max: float,
        kd_min: float,
        kd_max: float,
    ):
        self._ensure_initialized()
        if self._motor_control:
            if self._is_remote:
                self._motor_control.send_cmd(
                    motor._motor,
                    "set_mit_range",
                    [
                        p_min,
                        p_max,
                        v_min,
                        v_max,
                        t_min,
                        t_max,
                        kp_min,
                        kp_max,
                        kd_min,
                        kd_max,
                    ],
                    "",
                )
            else:
                self._motor_control.set_mit_range(
                    motor._motor,
                    p_min,
                    p_max,
                    v_min,
                    v_max,
                    t_min,
                    t_max,
                    kp_min,
                    kp_max,
                    kd_min,
                    kd_max,
                )

    def read_protection_thresholds(
        self, motor: Motor, level: int
    ) -> tuple[float, float, float, float, float, float]:
        self._ensure_initialized()
        if not self._motor_control:
            return (0.0,) * 6
        if self._is_remote:
            resp = self._motor_control.read(
                motor._motor, "read_protection_thresholds", [float(level)], ""
            )
            if resp.get("ok"):
                vs = resp.get("values", [])
                if len(vs) >= 6:
                    return tuple(vs[:6])
            return (0.0,) * 6
        return self._motor_control.read_protection_thresholds(motor._motor, level)

    def set_protection_thresholds(
        self,
        motor: Motor,
        level: int,
        mos_overtemp_warn: float,
        mos_overtemp_recover: float,
        overcurrent_warn: float,
        overcurrent_recover: float,
        overvoltage_warn: float,
        overvoltage_recover: float,
    ):
        self._ensure_initialized()
        if self._motor_control:
            if self._is_remote:
                self._motor_control.send_cmd(
                    motor._motor,
                    "set_protection_thresholds",
                    [
                        float(level),
                        mos_overtemp_warn,
                        mos_overtemp_recover,
                        overcurrent_warn,
                        overcurrent_recover,
                        overvoltage_warn,
                        overvoltage_recover,
                    ],
                    "",
                )
            else:
                self._motor_control.set_protection_thresholds(
                    motor._motor,
                    level,
                    mos_overtemp_warn,
                    mos_overtemp_recover,
                    overcurrent_warn,
                    overcurrent_recover,
                    overvoltage_warn,
                    overvoltage_recover,
                )

    def read_mode(self, motor: Motor) -> int:
        self._ensure_initialized()
        if not self._motor_control:
            return 0
        if self._is_remote:
            resp = self._motor_control.read(motor._motor, "read_mode", [], "")
            return int(resp.get("value", 0)) if resp.get("ok") else 0
        return self._motor_control.read_mode(motor._motor)

    def read_fault(self, motor: Motor) -> int:
        self._ensure_initialized()
        if not self._motor_control:
            return 0
        if self._is_remote:
            resp = self._motor_control.read(motor._motor, "read_fault", [], "")
            return int(resp.get("value", 0)) if resp.get("ok") else 0
        return self._motor_control.read_fault(motor._motor)

    def read_profile_acceleration(self, motor: Motor) -> float:
        self._ensure_initialized()
        if not self._motor_control:
            return 0.0
        if self._is_remote:
            resp = self._motor_control.read(
                motor._motor, "read_profile_acceleration", [], ""
            )
            return resp.get("value", 0.0) if resp.get("ok") else 0.0
        return self._motor_control.read_profile_acceleration(motor._motor)

    def read_profile_velocity(self, motor: Motor) -> float:
        self._ensure_initialized()
        if not self._motor_control:
            return 0.0
        if self._is_remote:
            resp = self._motor_control.read(
                motor._motor, "read_profile_velocity", [], ""
            )
            return resp.get("value", 0.0) if resp.get("ok") else 0.0
        return self._motor_control.read_profile_velocity(motor._motor)

    def read_profile_deceleration(self, motor: Motor) -> float:
        self._ensure_initialized()
        if not self._motor_control:
            return 0.0
        if self._is_remote:
            resp = self._motor_control.read(
                motor._motor, "read_profile_deceleration", [], ""
            )
            return resp.get("value", 0.0) if resp.get("ok") else 0.0
        return self._motor_control.read_profile_deceleration(motor._motor)

    def read_position_gain(self, motor: Motor) -> float:
        self._ensure_initialized()
        if not self._motor_control:
            return 0.0
        if self._is_remote:
            resp = self._motor_control.read(motor._motor, "read_position_gain", [], "")
            return resp.get("value", 0.0) if resp.get("ok") else 0.0
        return self._motor_control.read_position_gain(motor._motor)

    def read_velocity_gain(self, motor: Motor) -> tuple[float, float]:
        self._ensure_initialized()
        if not self._motor_control:
            return (0.0, 0.0)
        if self._is_remote:
            resp = self._motor_control.read(motor._motor, "read_velocity_gain", [], "")
            if resp.get("ok"):
                vs = resp.get("values", [])
                if len(vs) >= 2:
                    return (vs[0], vs[1])
            return (0.0, 0.0)
        return self._motor_control.read_velocity_gain(motor._motor)

    def read_velocity_filter_cutoff(self, motor: Motor) -> float:
        self._ensure_initialized()
        if not self._motor_control:
            return 0.0
        if self._is_remote:
            resp = self._motor_control.read(
                motor._motor, "read_velocity_filter_cutoff", [], ""
            )
            return resp.get("value", 0.0) if resp.get("ok") else 0.0
        return self._motor_control.read_velocity_filter_cutoff(motor._motor)

    def read_current_limit(self, motor: Motor) -> float:
        self._ensure_initialized()
        if not self._motor_control:
            return 0.0
        if self._is_remote:
            resp = self._motor_control.read(motor._motor, "read_current_limit", [], "")
            return resp.get("value", 0.0) if resp.get("ok") else 0.0
        return self._motor_control.read_current_limit(motor._motor)

    def read_torque_limit(self, motor: Motor) -> float:
        self._ensure_initialized()
        if not self._motor_control:
            return 0.0
        if self._is_remote:
            resp = self._motor_control.read(motor._motor, "read_torque_limit", [], "")
            return resp.get("value", 0.0) if resp.get("ok") else 0.0
        return self._motor_control.read_torque_limit(motor._motor)

    def close(self):
        if self._motor_control:
            try:
                self._motor_control.disable()
            except Exception:
                pass
        self._initialized = False
        self._motor_control = None

    def __enter__(self):
        self._ensure_initialized()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
