from __future__ import annotations

from ..libs import taks_driver as _taks_driver

_DM_IMU = _taks_driver.imu.DM_IMU


class DM_IMU:
    __slots__ = ("_impl", "baudrate", "port")

    def __init__(self, port: str = "/dev/imu", baudrate: int = 921600):
        self.port = port
        self.baudrate = baudrate
        self._impl = _DM_IMU(port, baudrate)

    def start(self):
        self._impl.start()

    def stop(self):
        self._impl.stop()

    def register_to_server(self, server):
        server.set_imu(self._impl)
