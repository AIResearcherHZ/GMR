from __future__ import annotations

from ..libs import taks_driver as _taks_driver

_MotorControl = _taks_driver.MotorControl
_DmActData = _taks_driver.DmActData
_MotorType = _taks_driver.DM_Motor_Type
_ControlMode = _taks_driver.Control_Mode


class Motor:
    __slots__ = ("MasterID", "MotorType", "SlaveID", "_motor")

    def __init__(self, motor_type, slave_id: int, master_id: int):
        self.MotorType = motor_type
        self.SlaveID = slave_id
        self.MasterID = master_id
        self._motor = None

    def _bind(self, motor):
        self._motor = motor

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


class MotorControl:
    __slots__ = ("_act_data", "_ctrl", "_initialized", "_motors", "can_interface")

    def __init__(self, can_interface: str = "can0"):
        self.can_interface = can_interface
        self._motors: dict[int, Motor] = {}
        self._act_data = []
        self._ctrl = None
        self._initialized = False

    def addMotor(self, motor: Motor):
        if self._initialized:
            raise RuntimeError("addMotor must be called before init")
        self._motors[motor.SlaveID] = motor
        self._act_data.append(
            _DmActData(
                motor.MotorType, _ControlMode.MIT_MODE, motor.SlaveID, motor.MasterID
            )
        )

    def init(self):
        if self._initialized:
            return
        self._ctrl = _MotorControl(self.can_interface, self._act_data, False)
        self._act_data = []
        self._initialized = True
        for motor in self._motors.values():
            internal = self._ctrl.getMotor(motor.SlaveID)
            if internal:
                motor._bind(internal)

    def getMotor(self, motor_id: int) -> Motor | None:
        return self._motors.get(motor_id)

    def enable(self, motor: Motor):
        if self._ctrl and motor._motor:
            self._ctrl.enable(motor._motor)

    def disable(self, motor: Motor):
        if self._ctrl and motor._motor:
            self._ctrl.disable(motor._motor)

    def disable_all(self):
        if self._ctrl:
            for motor in self._motors.values():
                if motor._motor:
                    try:
                        self._ctrl.disable(motor._motor)
                    except Exception:
                        pass

    def register_to_server(self, server, device: str | None = None):
        if not self._initialized:
            self.init()
        dev = device or self.can_interface
        ids = list(self._motors.keys())
        server.add_dm(dev, self._ctrl, ids)
        return dev, ids


DM_Motor_Type = _MotorType
Control_Mode = _ControlMode
