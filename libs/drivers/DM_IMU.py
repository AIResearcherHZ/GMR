import os
import sys
from collections.abc import Callable

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(
        0,
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")),
    )
    from libs.drivers.libs import taks_driver as _taks_driver
    from libs.drivers.rate_limiter import sleep
else:
    from .libs import taks_driver as _taks_driver
    from .rate_limiter import sleep

_ImuImpl = _taks_driver.imu.DM_IMU
_RemoteImuImpl = _taks_driver.remote.DM_IMU

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


class DM_IMU:
    __slots__ = (
        "_address",
        "_ca_cert",
        "_client_cert",
        "_client_key",
        "_device",
        "_impl",
        "_is_remote",
        "_port",
        "baudrate",
        "port",
    )

    def __init__(
        self,
        port: str = "/dev/imu",
        baudrate: int = 921600,
        address: str | None = None,
        port_num: int | None = None,
        ca_cert: str | None = None,
        client_cert: str | None = None,
        client_key: str | None = None,
        **kwargs,
    ):
        self.port = port
        self.baudrate = baudrate
        self._impl = None
        if address is not None:
            self._is_remote = True
            host, default_port = _parse_address(address)
            self._address = host
            self._port = port_num if port_num is not None else default_port
            self._ca_cert = ca_cert or _DEFAULT_CA_CERT
            self._client_cert = client_cert or _DEFAULT_CLIENT_CERT
            self._client_key = client_key or _DEFAULT_CLIENT_KEY
            self._device = "imu"
        else:
            self._is_remote = False
            self._address = None
            self._port = 0
            self._ca_cert = None
            self._client_cert = None
            self._client_key = None
            self._device = None
            self._impl = _ImuImpl(port, baudrate)

    def _ensure_connected(self):
        if self._impl is not None:
            return
        if self._is_remote:
            self._impl = _RemoteImuImpl(
                self._device,
                self._address,
                self._port,
                self._ca_cert,
                self._client_cert,
                self._client_key,
            )

    def set_callback(self, callback: Callable[[dict, dict, dict], None]):
        if self._is_remote:
            return
        self._ensure_connected()
        self._impl.set_callback(callback)

    def start(self):
        if not self._is_remote:
            self._ensure_connected()
            self._impl.start()

    def stop(self):
        if not self._is_remote and self._impl is not None:
            self._impl.stop()

    def get_data(self) -> dict:
        self._ensure_connected()
        return self._impl.get_data()

    def get_accel(self) -> dict:
        self._ensure_connected()
        return self._impl.get_accel()

    def get_gyro(self) -> dict:
        self._ensure_connected()
        return self._impl.get_gyro()

    def get_euler(self) -> dict:
        self._ensure_connected()
        return self._impl.get_euler()

    def get_quat(self) -> dict:
        self._ensure_connected()
        return self._impl.get_quat()

    def _remote_cmd(self, op: str, args: list[float] | None = None):
        self._ensure_connected()
        return self._impl.send_cmd(op, args or [])

    def enter_setting_mode(self):
        self._ensure_connected()
        if self._is_remote:
            return self._remote_cmd("enter_setting_mode")
        return self._impl.enter_setting_mode()

    def exit_setting_mode(self):
        self._ensure_connected()
        if self._is_remote:
            return self._remote_cmd("exit_setting_mode")
        return self._impl.exit_setting_mode()

    def save_params(self):
        self._ensure_connected()
        if self._is_remote:
            return self._remote_cmd("save_params")
        return self._impl.save_params()

    def calibrate_zero(self, save: bool = True):
        self._ensure_connected()
        if self._is_remote:
            return self._remote_cmd("calibrate_zero", [1.0 if save else 0.0])
        return self._impl.calibrate_zero(save)

    def calibrate_gyro(self, save: bool = True):
        self._ensure_connected()
        if self._is_remote:
            return self._remote_cmd("calibrate_gyro", [1.0 if save else 0.0])
        return self._impl.calibrate_gyro(save)

    def restart(self):
        self._ensure_connected()
        if self._is_remote:
            return self._remote_cmd("restart")
        return self._impl.restart()

    def set_485_active(self, enable: bool):
        self._ensure_connected()
        if self._is_remote:
            return self._remote_cmd("set_485_active", [1.0 if enable else 0.0])
        return self._impl.set_485_active(enable)

    def set_accel_output(self, enable: bool):
        self._ensure_connected()
        if self._is_remote:
            return self._remote_cmd("set_accel_output", [1.0 if enable else 0.0])
        return self._impl.set_accel_output(enable)

    def set_gyro_output(self, enable: bool):
        self._ensure_connected()
        if self._is_remote:
            return self._remote_cmd("set_gyro_output", [1.0 if enable else 0.0])
        return self._impl.set_gyro_output(enable)

    def set_euler_output(self, enable: bool):
        self._ensure_connected()
        if self._is_remote:
            return self._remote_cmd("set_euler_output", [1.0 if enable else 0.0])
        return self._impl.set_euler_output(enable)

    def set_quat_output(self, enable: bool):
        self._ensure_connected()
        if self._is_remote:
            return self._remote_cmd("set_quat_output", [1.0 if enable else 0.0])
        return self._impl.set_quat_output(enable)

    def set_can_active(self, enable: bool):
        self._ensure_connected()
        if self._is_remote:
            return self._remote_cmd("set_can_active", [1.0 if enable else 0.0])
        return self._impl.set_can_active(enable)

    def calibrate_accel_six_axis(self):
        self._ensure_connected()
        if self._is_remote:
            return self._remote_cmd("calibrate_accel_six_axis")
        return self._impl.calibrate_accel_six_axis()

    def set_temp_control(self, enable: bool):
        self._ensure_connected()
        if self._is_remote:
            return self._remote_cmd("set_temp_control", [1.0 if enable else 0.0])
        return self._impl.set_temp_control(enable)

    def set_target_temp(self, temp: int):
        self._ensure_connected()
        if self._is_remote:
            return self._remote_cmd("set_target_temp", [float(temp)])
        return self._impl.set_target_temp(temp)

    def set_can_id(self, id: int):
        self._ensure_connected()
        if self._is_remote:
            return self._remote_cmd("set_can_id", [float(id)])
        return self._impl.set_can_id(id)

    def set_mst_id(self, id: int):
        self._ensure_connected()
        if self._is_remote:
            return self._remote_cmd("set_mst_id", [float(id)])
        return self._impl.set_mst_id(id)

    def set_output_interface(self, interface_id: int):
        self._ensure_connected()
        if self._is_remote:
            return self._remote_cmd("set_output_interface", [float(interface_id)])
        return self._impl.set_output_interface(interface_id)

    def restore_factory_settings(self):
        self._ensure_connected()
        if self._is_remote:
            return self._remote_cmd("restore_factory_settings")
        return self._impl.restore_factory_settings()

    def set_feedback_rate(self, target_hz: float):
        self._ensure_connected()
        if target_hz <= 0:
            raise ValueError("target_hz must be > 0")
        interval_ms = max(1, min(65535, round(1000.0 / target_hz)))
        if self._is_remote:
            return self._remote_cmd("set_feedback_rate", [float(interval_ms)])
        self._impl.set_feedback_rate(interval_ms)

    def set_can_baudrate(self, baud_index: int):
        self._ensure_connected()
        if self._is_remote:
            return self._remote_cmd("set_can_baudrate", [float(baud_index)])
        return self._impl.set_can_baudrate(baud_index)

    def set_rs485_baudrate(self, baud_index: int):
        self._ensure_connected()
        if self._is_remote:
            return self._remote_cmd("set_rs485_baudrate", [float(baud_index)])
        return self._impl.set_rs485_baudrate(baud_index)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False

    def __del__(self):
        try:
            self.stop()
        except Exception:
            pass


def main():
    imu = DM_IMU(port="/dev/ttyACM0", baudrate=921600)
    try:
        imu.start()
        sleep(0.1)
        imu.enter_setting_mode()
        sleep(0.1)
        imu.calibrate_zero(save=False)
        sleep(0.1)
        imu.set_temp_control(True)
        sleep(0.1)
        imu.set_target_temp(30)
        sleep(0.1)
        imu.set_485_active(False)
        sleep(0.1)
        imu.set_can_active(False)
        sleep(0.1)
        imu.save_params()
        sleep(0.5)
        imu.exit_setting_mode()
        sleep(0.1)

        while True:
            data = imu.get_data()
            accel = data["accel"]
            gyro = data["gyro"]
            euler = data["euler"]
            quat = imu.get_quat()
            print(
                f"\raccel(m/s²)=[{accel['x']:8.3f},{accel['y']:8.3f},{accel['z']:8.3f}] "
                f"gyro(rad/s)=[{gyro['x']:8.4f},{gyro['y']:8.4f},{gyro['z']:8.4f}] "
                f"euler(rad)=[{euler['roll']:7.4f},{euler['pitch']:7.4f},{euler['yaw']:7.4f}] "
                f"quat=[{quat['w']:7.4f},{quat['x']:7.4f},{quat['y']:7.4f},{quat['z']:7.4f}]",
                end="",
                flush=True,
            )
            sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        imu.stop()


if __name__ == "__main__":
    main()
