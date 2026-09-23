from __future__ import annotations

from enum import IntEnum

from ..libs import taks_driver as _taks_driver

_RustMotorType = _taks_driver.EYouRp_Motor_Type
_RustMotor = _taks_driver.EYouRpMotor
_RustControl = _taks_driver.EYouRpControl


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
    __slots__ = ("MotorType", "NodeID", "_motor")

    def __init__(self, motor_type: EYouRp_Motor_Type, node_id: int):
        self.MotorType = motor_type
        self.NodeID = node_id
        self._motor = _RustMotor(_convert_motor_type(motor_type), node_id)

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


class EYouRpControl:
    __slots__ = ("_ctrl", "_initialized", "_motors", "_sync_hz", "can_interface")

    def __init__(self, can_interface: str = "can0", sync_hz: float = 1000.0):
        self.can_interface = can_interface
        self._motors: dict[int, Motor] = {}
        self._ctrl = None
        self._initialized = False
        self._sync_hz = sync_hz

    def addMotor(self, motor: Motor):
        self._motors[motor.NodeID] = motor

    def init(self):
        if self._initialized:
            return
        self._ctrl = _RustControl(
            self.can_interface,
            [m._motor for m in self._motors.values()],
            False,
            self._sync_hz,
        )
        self._initialized = True

    def getMotor(self, node_id: int) -> Motor | None:
        return self._motors.get(node_id)

    def enable(self, motor: Motor):
        if self._ctrl:
            self._ctrl.enable(motor._motor)

    def disable(self, motor: Motor):
        if self._ctrl:
            self._ctrl.disable(motor._motor)

    def disable_all(self):
        if self._ctrl:
            for motor in self._motors.values():
                try:
                    self._ctrl.disable(motor._motor)
                except Exception:
                    pass

    def register_to_server(self, server, device: str | None = None):
        if not self._initialized:
            self.init()
        dev = device or self.can_interface
        ids = list(self._motors.keys())
        server.add_eyou_can(dev, self._ctrl, ids)
        return dev, ids
