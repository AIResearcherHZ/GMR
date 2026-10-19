from __future__ import annotations

import argparse
import asyncio
import math
import os
import signal
import socket
import sys
import threading
import time

import anyio
import msgspec
import rebocap_ws_sdk

JOINT_NAMES = rebocap_ws_sdk.REBOCAP_JOINT_NAMES
_NUM_JOINTS = len(JOINT_NAMES)
_NAME2ID = {n: i for i, n in enumerate(JOINT_NAMES)}
_HALF_PI = math.pi / 2


class Quaternion(msgspec.Struct, gc=False):
    w: float
    x: float
    y: float
    z: float


class EulerDeg(msgspec.Struct, gc=False):
    x: float
    y: float
    z: float


class JointData(msgspec.Struct, gc=False):
    id: int
    name: str
    quaternion: Quaternion
    euler_deg: EulerDeg


class PosePayload(msgspec.Struct, gc=False):
    timestamp: float
    static_index: int
    root_translation: list[float]
    coordinate_type: str
    rotation_mode: str
    joints: list[JointData]


_encoder = msgspec.json.Encoder()
_decoder = msgspec.json.Decoder(PosePayload)


def quat_to_euler_xyz_deg(
    w: float, x: float, y: float, z: float
) -> tuple[float, float, float]:
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    sp = 2 * (w * y - z * x)
    pitch = math.copysign(_HALF_PI, sp) if abs(sp) >= 1 else math.asin(sp)
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def resolve_joint_ids(sel: str) -> list[int]:
    if not sel or sel.strip().lower() == "all":
        return list(range(_NUM_JOINTS))
    out, seen = [], set()
    for tok in (t.strip() for t in sel.split(",")):
        if not tok:
            continue
        idx = int(tok) if tok.isdigit() else _NAME2ID.get(tok)
        if idx is None or not (0 <= idx < _NUM_JOINTS):
            print(f"[WARN] 未知关节: {tok}")
            continue
        if idx not in seen:
            seen.add(idx)
            out.append(idx)
    return out


def build_payload(
    coord_name: str,
    trans,
    pose24,
    static_index,
    ts: float,
    joint_ids: list[int],
    rotation_mode: str,
) -> PosePayload:
    joints = []
    for i in joint_ids:
        q = pose24[i]
        x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        ex, ey, ez = quat_to_euler_xyz_deg(w, x, y, z)
        joints.append(
            JointData(
                id=i,
                name=JOINT_NAMES[i],
                quaternion=Quaternion(w=w, x=x, y=y, z=z),
                euler_deg=EulerDeg(x=ex, y=ey, z=ez),
            )
        )
    return PosePayload(
        timestamp=ts,
        static_index=int(static_index),
        root_translation=[float(trans[0]), float(trans[1]), float(trans[2])],
        coordinate_type=coord_name,
        rotation_mode=rotation_mode,
        joints=joints,
    )


def print_payload(tag: str, p: PosePayload):
    print(
        f"\n[{tag}] ts={p.timestamp:.3f} coord={p.coordinate_type} "
        f"root={p.root_translation} static_index={p.static_index}"
    )
    for j in p.joints:
        q, e = j.quaternion, j.euler_deg
        print(
            f"  bone[{j.id:>2}] {j.name:<11} "
            f"quat(wxyz)=[{q.w:+.4f},{q.x:+.4f},{q.y:+.4f},{q.z:+.4f}] "
            f"euler_deg(xyz)=[{e.x:+8.2f},{e.y:+8.2f},{e.z:+8.2f}]"
        )
    print(flush=True)


class _LatestSlot:
    __slots__ = ("_closed", "_dropped", "_event", "_item", "_lock", "_loop")

    def __init__(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop
        self._lock = threading.Lock()
        self._event = asyncio.Event()
        self._item: PosePayload | None = None
        self._dropped = 0
        self._closed = False

    def put_from_thread(self, item: PosePayload | None) -> None:
        with self._lock:
            if self._item is not None and item is not None:
                self._dropped += 1
            self._item = item
            if item is None:
                self._closed = True
        self._loop.call_soon_threadsafe(self._event.set)

    @property
    def dropped(self) -> int:
        return self._dropped

    async def get(self) -> PosePayload | None:

        while True:
            await self._event.wait()
            with self._lock:
                item, self._item = self._item, None
                if item is not None or self._closed:
                    self._event.clear()
                    return item

                self._event.clear()


async def sender_task(slot: _LatestSlot, host: str, port: int, print_every: int):
    sock = await anyio.create_connected_udp_socket(host, port)
    sent = fps = 0
    last_log = time.monotonic()
    try:
        while True:
            payload = await slot.get()
            if payload is None:
                break
            data = _encoder.encode(payload)
            try:
                await sock.send(data)
            except OSError as e:
                print(f"[WARN] sendto failed: {e}, size={len(data)}")
                continue
            sent += 1
            fps += 1
            if print_every and sent % print_every == 0:
                print_payload("FWD", payload)
            now = time.monotonic()
            if now - last_log >= 1.0:
                print(
                    f"[FWD] -> {host}:{port}  {fps} fps  "
                    f"payload={len(data)}B  joints={len(payload.joints)}  "
                    f"dropped={slot.dropped}"
                )
                fps, last_log = 0, now
    finally:
        await sock.aclose()


async def _wait_signals():
    if sys.platform == "win32":
        ev = threading.Event()
        prev_int = signal.signal(signal.SIGINT, lambda *_: ev.set())
        try:
            prev_term = signal.signal(signal.SIGTERM, lambda *_: ev.set())
        except (ValueError, OSError):
            prev_term = None
        try:
            await anyio.to_thread.run_sync(ev.wait, abandon_on_cancel=True)
        finally:
            signal.signal(signal.SIGINT, prev_int)
            if prev_term is not None:
                signal.signal(signal.SIGTERM, prev_term)
    else:
        with anyio.open_signal_receiver(signal.SIGINT, signal.SIGTERM) as sigs:
            async for _ in sigs:
                return


async def run_forwarder(
    host: str,
    port: int,
    sdk_port: int,
    coordinate: str,
    use_global: bool,
    joints_sel: str,
    print_every: int,
):
    joint_ids = resolve_joint_ids(joints_sel)
    if not joint_ids:
        print("[ERR] 没有有效关节, 退出")
        return
    print(
        f"[CFG] 发送关节({len(joint_ids)}): "
        + ",".join(JOINT_NAMES[i] for i in joint_ids)
    )

    coord = getattr(rebocap_ws_sdk.CoordinateType, coordinate)
    coord_name = coord.name
    rotation_mode = "global" if use_global else "local"
    sdk = rebocap_ws_sdk.RebocapWsSdk(
        coordinate_type=coord, use_global_rotation=use_global
    )

    loop = asyncio.get_running_loop()
    slot = _LatestSlot(loop)

    def on_pose(_self, tran, pose24, static_index, ts):
        slot.put_from_thread(
            build_payload(
                coord_name, tran, pose24, static_index, ts, joint_ids, rotation_mode
            )
        )

    def on_close(_self):
        print("[SDK] exception close")
        slot.put_from_thread(None)

    sdk.set_pose_msg_callback(on_pose)
    sdk.set_exception_close_callback(on_close)

    ret = sdk.open(sdk_port)
    if ret != 0:
        print(f"[SDK] open failed: code={ret}")
        return
    print(f"[SDK] connected on port {sdk_port}, forwarding to {host}:{port}")

    try:
        async with anyio.create_task_group() as tg:
            tg.start_soon(sender_task, slot, host, port, print_every)
            await _wait_signals()
            print("\n[MAIN] 收到退出信号, 正在关闭...")
            slot.put_from_thread(None)
            tg.cancel_scope.cancel()
    finally:
        try:
            sdk.close()
        except Exception as e:
            print(f"[SDK] close error: {e}")


async def _receiver_loop(sock, print_every: int):
    n = 0
    async for data, addr in sock:
        n += 1
        if not print_every or n % print_every:
            continue
        try:
            print_payload(f"RECV {addr[0]}:{addr[1]}", _decoder.decode(data))
        except msgspec.DecodeError as e:
            print(f"[RECV] decode error: {e}")


async def run_receiver(port: int, print_every: int):
    sock = await anyio.create_udp_socket(
        family=socket.AF_INET, local_host="0.0.0.0", local_port=port
    )
    print(f"[RECV] listening on 0.0.0.0:{port}")
    try:
        async with anyio.create_task_group() as tg:
            tg.start_soon(_receiver_loop, sock, print_every)
            await _wait_signals()
            print("\n[MAIN] 收到退出信号, 正在关闭...")
            tg.cancel_scope.cancel()
    finally:
        await sock.aclose()


def _try_uvloop() -> dict:
    try:
        import uvloop  # noqa: F401

        return {"use_uvloop": True}
    except ImportError:
        return {}


def main():
    ap = argparse.ArgumentParser(description="Rebocap pose UDP forwarder")
    ap.add_argument("--host", default="10.0.41.233", help="目标电脑 IP")
    ap.add_argument("--port", type=int, default=9010, help="目标 UDP 端口")
    ap.add_argument(
        "--sdk-port", type=int, default=7690, help="Rebocap 软件数据输出端口"
    )
    ap.add_argument(
        "--coordinate",
        default="BlenderCoordinate",
        choices=[c.name for c in rebocap_ws_sdk.CoordinateType],
    )
    ap.add_argument(
        "--local-rotation",
        action="store_true",
        default=False,
        help="使用 local 旋转 (默认 global)",
    )
    ap.add_argument(
        "--recv", action="store_true", default=False, help="作为接收端运行 (调试用)"
    )
    ap.add_argument("--joints", default="all", help="转发的关节, 逗号分隔或 'all'")
    ap.add_argument("--print-every", type=int, default=60, help="0=不周期打印")
    ap.add_argument(
        "--list-joints", action="store_true", default=False, help="列出关节名后退出"
    )
    args = ap.parse_args()

    if args.list_joints:
        for i, n in enumerate(JOINT_NAMES):
            print(f"{i:>2}: {n}")
        return

    backend = _try_uvloop()
    try:
        if args.recv:
            anyio.run(
                run_receiver, args.port, args.print_every, backend_options=backend
            )
        else:
            anyio.run(
                run_forwarder,
                args.host,
                args.port,
                args.sdk_port,
                args.coordinate,
                not args.local_rotation,
                args.joints,
                args.print_every,
                backend_options=backend,
            )
    except KeyboardInterrupt:
        pass
    print("[MAIN] stopped")
    os._exit(0)


if __name__ == "__main__":
    main()
