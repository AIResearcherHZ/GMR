from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import threading

_DRIVERS_DIR = os.path.abspath(os.path.dirname(__file__))
_BACKEND_DIR = os.path.abspath(os.path.join(_DRIVERS_DIR, "..", ".."))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from libs.drivers.libs import taks_driver
from libs.drivers.server.DM_CANFD_server import DM_Motor_Type
from libs.drivers.server.DM_CANFD_server import Motor as DmMotor
from libs.drivers.server.DM_CANFD_server import MotorControl as DmControl
from libs.drivers.server.DM_IMU_server import DM_IMU
from libs.drivers.server.EYou_RP_CAN_server import EYouRp_Motor_Type as EYouCanType
from libs.drivers.server.EYou_RP_CAN_server import EYouRpControl
from libs.drivers.server.EYou_RP_CAN_server import Motor as EYouCanMotor
from libs.drivers.server.EYou_RP_CANFD_server import EYouRp_Motor_Type as EYouFdType
from libs.drivers.server.EYou_RP_CANFD_server import EYouRpCanfdControl
from libs.drivers.server.EYou_RP_CANFD_server import Motor as EYouFdMotor
from libs.drivers.server.Moteus_CAN_server import MoteusControl
from libs.drivers.server.Moteus_CAN_server import Motor as MoteusMotor

logger = logging.getLogger("remote-server")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)

_CERT_DIR = os.path.join(_DRIVERS_DIR, "certs")
_DEFAULT_SERVER_CERT = os.path.join(_CERT_DIR, "server-cert.pem")
_DEFAULT_SERVER_KEY = os.path.join(_CERT_DIR, "server-key.pem")
_DEFAULT_CA_CERT = os.path.join(_CERT_DIR, "ca-cert.pem")

_DEFAULT_CONFIG = os.path.join(_DRIVERS_DIR, "server", "dm_semi_t1.json")

_DRIVER_BUILDERS = {
    "dm_canfd": "_build_dm_bus",
    "eyou_rp_can": "_build_eyou_can_bus",
    "eyou_rp_canfd": "_build_eyou_canfd_bus",
    "moteus": "_build_moteus_bus",
}


def _iface_online(iface: str) -> bool:
    try:
        with open(f"/sys/class/net/{iface}/operstate", "r") as f:
            return f.read().strip() == "up"
    except OSError:
        return False


def _build_dm_bus(bus_cfg: dict) -> tuple | None:
    iface = bus_cfg["interface"]
    if not _iface_online(iface):
        logger.warning(f"⚠ CAN接口 {iface} 不在线，跳过")
        return None
    mc = DmControl(iface)
    for m in bus_cfg["motors"]:
        mt = getattr(DM_Motor_Type, m["type"])
        mc.addMotor(DmMotor(mt, m["id"], m.get("master_id", m["id"] + 0x80)))
    logger.info(f"✓ {iface}: {len(bus_cfg['motors'])} 个DM电机")
    return mc


def _build_eyou_can_bus(bus_cfg: dict) -> tuple | None:
    iface = bus_cfg["interface"]
    if not _iface_online(iface):
        logger.warning(f"⚠ CAN接口 {iface} 不在线，跳过")
        return None
    ctrl = EYouRpControl(iface, sync_hz=bus_cfg.get("sync_hz", 1000.0))
    for m in bus_cfg["motors"]:
        mt = getattr(EYouCanType, m["type"])
        ctrl.addMotor(EYouCanMotor(mt, m["id"]))
    logger.info(f"✓ {iface}: {len(bus_cfg['motors'])} 个EYou_RP_CAN电机")
    return ctrl


def _build_eyou_canfd_bus(bus_cfg: dict) -> tuple | None:
    iface = bus_cfg["interface"]
    if not _iface_online(iface):
        logger.warning(f"⚠ CAN接口 {iface} 不在线，跳过")
        return None
    ctrl = EYouRpCanfdControl(iface)
    for m in bus_cfg["motors"]:
        mt = getattr(EYouFdType, m["type"])
        ctrl.addMotor(EYouFdMotor(mt, m["id"]))
    logger.info(f"✓ {iface}: {len(bus_cfg['motors'])} 个EYou_RP_CANFD电机")
    return ctrl


def _build_moteus_bus(bus_cfg: dict) -> tuple | None:
    iface = bus_cfg["interface"]
    if not _iface_online(iface):
        logger.warning(f"⚠ CAN接口 {iface} 不在线，跳过")
        return None
    ctrl = MoteusControl(iface)
    for m in bus_cfg["motors"]:
        ctrl.addMotor(MoteusMotor(m["id"]))
    logger.info(f"✓ {iface}: {len(bus_cfg['motors'])} 个Moteus电机")
    return ctrl


_BUILDERS = {
    "dm_canfd": _build_dm_bus,
    "eyou_rp_can": _build_eyou_can_bus,
    "eyou_rp_canfd": _build_eyou_canfd_bus,
    "moteus": _build_moteus_bus,
}


def main():
    parser = argparse.ArgumentParser(description="Taks 远程 Zenoh 服务端")
    parser.add_argument(
        "--config", type=str, default=_DEFAULT_CONFIG, help="硬件配置JSON文件"
    )
    parser.add_argument("--port", type=int, default=5555, help="监听端口")
    parser.add_argument("--target-hz", type=float, default=1000.0, help="状态发布频率")
    parser.add_argument("--no-imu", action="store_true", help="不启用IMU")
    parser.add_argument("--cert", type=str, default=_DEFAULT_SERVER_CERT)
    parser.add_argument("--key", type=str, default=_DEFAULT_SERVER_KEY)
    parser.add_argument("--ca-cert", type=str, default=_DEFAULT_CA_CERT)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        hw_config = json.load(f)

    logger.info(f"配置文件: {args.config}")
    logger.info(f"监听端口: {args.port}")
    logger.info(f"发布频率: {args.target_hz} Hz")

    controllers = []
    for bus_cfg in hw_config.get("buses", []):
        driver = bus_cfg.get("driver")
        builder = _BUILDERS.get(driver)
        if builder is None:
            logger.error(f"✗ 未知驱动类型: {driver} (接口 {bus_cfg.get('interface')})")
            continue
        ctrl = builder(bus_cfg)
        if ctrl is not None:
            controllers.append(ctrl)

    if not controllers:
        logger.error("✗ 无可用CAN总线，退出")
        return

    imu = None
    imu_cfg = hw_config.get("imu")
    if imu_cfg and not args.no_imu:
        imu_port = imu_cfg.get("port", "/dev/imu")
        if os.path.exists(imu_port):
            imu = DM_IMU(port=imu_port, baudrate=imu_cfg.get("baudrate", 921600))
            imu.start()
            logger.info(f"✓ IMU 已启动: {imu_port}")
        else:
            logger.warning(f"⚠ IMU设备 {imu_port} 不存在，跳过")

    server = taks_driver.remote.ZenohServer(
        port=args.port,
        cert=args.cert,
        key=args.key,
        ca_cert=args.ca_cert,
        target_hz=args.target_hz,
    )

    for ctrl in controllers:
        ctrl.register_to_server(server)

    if imu is not None:
        imu.register_to_server(server)

    logger.info("✓ 服务端已启动")

    stop_event = threading.Event()

    def _signal_handler(signum, frame):
        logger.info(f"收到中断信号 ({signum})")
        stop_event.set()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    try:
        while not stop_event.is_set():
            stop_event.wait(timeout=1.0)
    finally:
        logger.info("正在关闭...")
        for ctrl in controllers:
            ctrl.disable_all()
        if imu is not None:
            imu.stop()
        server.close()
        logger.info("✓ 服务端已关闭")


if __name__ == "__main__":
    main()
