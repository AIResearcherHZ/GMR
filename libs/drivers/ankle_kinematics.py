import os
import sys

import numpy as np

_current_dir = os.path.dirname(os.path.abspath(__file__))
_lib_path = os.path.join(_current_dir, "libs")
if _lib_path not in sys.path:
    sys.path.insert(0, _lib_path)

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(
        0,
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")),
    )
    from libs.drivers.libs import taks_driver as _taks_driver
else:
    from .libs import taks_driver as _taks_driver

_ankle = _taks_driver.ankle


def ankle_ik(pitch: float, roll: float, left_leg: bool = True) -> tuple[float, float]:
    return _ankle.ankle_ik(pitch, roll, left_leg)


def ankle_fk(
    theta_upper: float, theta_lower: float, left_leg: bool = True
) -> tuple[float, float]:
    return _ankle.ankle_fk(theta_upper, theta_lower, left_leg)


def motor_vel_to_ankle_vel(
    pitch: float,
    roll: float,
    vel_upper: float,
    vel_lower: float,
    left_leg: bool = True,
) -> tuple[float, float]:
    return _ankle.motor_vel_to_ankle_vel(pitch, roll, vel_upper, vel_lower, left_leg)


def ankle_vel_to_motor_vel(
    pitch: float,
    roll: float,
    pitch_vel: float,
    roll_vel: float,
    left_leg: bool = True,
) -> tuple[float, float]:
    return _ankle.ankle_vel_to_motor_vel(pitch, roll, pitch_vel, roll_vel, left_leg)


def motor_tau_to_ankle_tau(
    pitch: float,
    roll: float,
    tau_upper: float,
    tau_lower: float,
    left_leg: bool = True,
) -> tuple[float, float]:
    return _ankle.motor_tau_to_ankle_tau(pitch, roll, tau_upper, tau_lower, left_leg)


def ankle_tau_to_motor_tau(
    pitch: float,
    roll: float,
    tau_pitch: float,
    tau_roll: float,
    left_leg: bool = True,
) -> tuple[float, float]:
    return _ankle.ankle_tau_to_motor_tau(pitch, roll, tau_pitch, tau_roll, left_leg)


if __name__ == "__main__":
    print("=" * 60)
    print("踝关节运动学测试")
    print("=" * 60)

    test_pitch = np.deg2rad(45.0)
    test_roll = np.deg2rad(0.0)

    print(
        f"\n输入: pitch={np.rad2deg(test_pitch):.2f}°, roll={np.rad2deg(test_roll):.2f}°"
    )

    theta_upper, theta_lower = ankle_ik(test_pitch, test_roll, left_leg=True)
    print(
        f"IK结果: theta_upper={np.rad2deg(theta_upper):.4f}°, theta_lower={np.rad2deg(theta_lower):.4f}°"
    )

    pitch_fk, roll_fk = ankle_fk(theta_upper, theta_lower, left_leg=True)
    print(f"FK结果: pitch={np.rad2deg(pitch_fk):.4f}°, roll={np.rad2deg(roll_fk):.4f}°")

    err_pitch = abs(
        np.rad2deg(
            np.arctan2(
                np.sin(pitch_fk - test_pitch),
                np.cos(pitch_fk - test_pitch),
            )
        )
    )
    err_roll = abs(
        np.rad2deg(
            np.arctan2(
                np.sin(roll_fk - test_roll),
                np.cos(roll_fk - test_roll),
            )
        )
    )
    print(f"误差: pitch={err_pitch:.6f}°, roll={err_roll:.6f}°")

    test_vel = (0.1, 0.05)
    motor_vel = ankle_vel_to_motor_vel(test_pitch, test_roll, *test_vel)
    ankle_vel_back = motor_vel_to_ankle_vel(test_pitch, test_roll, *motor_vel)
    vel_error = np.max(np.abs(np.array(ankle_vel_back) - np.array(test_vel)))
    print(f"\n速度变换: {test_vel} -> {motor_vel} -> {ankle_vel_back}")
    print(f"速度往返误差: {vel_error:.2e}")

    test_tau = (1.0, 0.5)
    motor_tau = ankle_tau_to_motor_tau(test_pitch, test_roll, *test_tau)
    ankle_tau_back = motor_tau_to_ankle_tau(test_pitch, test_roll, *motor_tau)
    tau_error = np.max(np.abs(np.array(ankle_tau_back) - np.array(test_tau)))
    print(f"力矩变换: {test_tau} -> {motor_tau} -> {ankle_tau_back}")
    print(f"力矩往返误差: {tau_error:.2e}")

    print("\n" + "=" * 60)
    if err_pitch < 0.01 and err_roll < 0.01 and vel_error < 1e-5 and tau_error < 1e-5:
        print("✓ 测试通过")
    else:
        print("✗ 测试失败")
    print("=" * 60)
