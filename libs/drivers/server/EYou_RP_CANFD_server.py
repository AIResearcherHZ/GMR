from __future__ import annotations

from enum import IntEnum

from ..libs import taks_driver as _taks_driver

_RustMotorType = _taks_driver.EYouRp_Motor_Type
_RustMotor = _taks_driver.EYouRpCanfdMotor
_RustControl = _taks_driver.EYouRpCanfdControl


class EYouRp_Motor_Type(IntEnum):
    RP40C = 0
    RP50L = 1
    RP50H = 2
    RP70L = 3
    RP70H = 4
    RP90L = 5


def _convert_motor_type(mt: EYouRp_Motor_Type):
    return getattr(_RustMotorType, mt.name)


class Motor:
    __slots__ = ("DeviceID", "MotorType", "_motor")

    def __init__(self, motor_type: EYouRp_Motor_Type, device_id: int):
        if not 0x01 <= device_id <= 0x08:
            raise ValueError("EYou RP CAN-FD 设备 ID 必须位于 0x01..0x08")
        self.MotorType = motor_type
        self.DeviceID = device_id
        self._motor = _RustMotor(_convert_motor_type(motor_type), device_id)

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


class EYouRpCanfdControl:
    __slots__ = ("_ctrl", "_initialized", "_motors", "can_interface")

    def __init__(self, can_interface: str = "can0"):
        self.can_interface = can_interface
        self._motors: dict[int, Motor] = {}
        self._ctrl = None
        self._initialized = False

    def addMotor(self, motor: Motor):
        self._motors[motor.DeviceID] = motor

    def init(self):
        if self._initialized:
            return
        self._ctrl = _RustControl(
            self.can_interface,
            [m._motor for m in self._motors.values()],
            False,
        )
        self._initialized = True

    def getMotor(self, device_id: int) -> Motor | None:
        return self._motors.get(device_id)

    def enable(self):
        if self._ctrl:
            self._ctrl.enable()

    def disable(self):
        if self._ctrl:
            self._ctrl.disable()

    def disable_all(self):
        self.disable()

    def register_to_server(self, server, device: str | None = None):
        if not self._initialized:
            self.init()
        dev = device or self.can_interface
        ids = list(self._motors.keys())
        server.add_eyou_canfd(dev, self._ctrl, ids)
        return dev, ids
