from __future__ import annotations

from ..libs import taks_driver as _taks_driver

_MoteusMotor = _taks_driver.MoteusMotor
_MoteusControl = _taks_driver.MoteusControl


class Motor:
    __slots__ = ("NodeID", "_motor")

    def __init__(self, node_id: int):
        self.NodeID = node_id
        self._motor = _MoteusMotor(node_id)

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


class MoteusControl:
    __slots__ = ("_ctrl", "_initialized", "_motors", "can_interface")

    def __init__(self, can_interface: str = "can0"):
        self.can_interface = can_interface
        self._motors: dict[int, Motor] = {}
        self._ctrl = None
        self._initialized = False

    def addMotor(self, motor: Motor):
        self._motors[motor.NodeID] = motor

    def init(self):
        if self._initialized:
            return
        self._ctrl = _MoteusControl(
            self.can_interface,
            list(self._motors.values()),
            False,
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
        server.add_moteus(dev, self._ctrl, ids)
        return dev, ids
