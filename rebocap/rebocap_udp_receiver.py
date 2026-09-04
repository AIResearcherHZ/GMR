from __future__ import annotations

import argparse
import math
import os
import select
import signal
import socket
import sys
import threading
import time

import anyio
import msgspec

_RECV_BUF = 65536
_KERNEL_BUF = 4 * 1024 * 1024


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


_decoder = msgspec.json.Decoder(PosePayload)


class LatestRebocapUdpReceiver:
    __slots__ = (
        "_dropped",
        "_has_payload",
        "_last_update",
        "_lock",
        "_payload",
        "_seq",
        "_sock",
        "_stop",
        "_thread",
        "host",
        "port",
    )

    def __init__(self, host: str = "0.0.0.0", port: int = 9000):
        self.host = host
        self.port = port
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._payload: PosePayload | None = None
        self._last_update = 0.0
        self._seq = 0
        self._dropped = 0
        self._has_payload = threading.Event()

    def start(self) -> None:
        if self._thread is not None:
            return
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, _KERNEL_BUF)
        except OSError:
            pass
        sock.bind((self.host, self.port))
        sock.setblocking(False)
        self._sock = sock
        self._thread = threading.Thread(
            target=self._recv_loop, name="rebocap-udp-rx", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        sock, self._sock = self._sock, None
        if sock is not None:
            sock.close()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    @property
    def seq(self) -> int:
        return self._seq

    @property
    def dropped(self) -> int:
        return self._dropped

    def _snapshot(
        self, timeout: float | None, max_age: float | None
    ) -> PosePayload | None:

        if timeout is not None and not self._has_payload.wait(timeout):
            return None
        with self._lock:
            payload = self._payload
            last_update = self._last_update
        if payload is None:
            return None
        if max_age is not None and time.monotonic() - last_update > max_age:
            return None
        return payload

    def get_latest_payload(
        self,
        timeout: float | None = None,
        max_age: float | None = None,
    ) -> PosePayload | None:
        return self._snapshot(timeout, max_age)

    def get_joint_euler_rad(
        self,
        joint_id: int = 0,
        timeout: float | None = None,
        max_age: float | None = None,
    ) -> dict[str, float] | None:
        p = self._snapshot(timeout, max_age)
        if p is None:
            return None
        joint = next((j for j in p.joints if j.id == joint_id), None)
        if joint is None:
            return None
        e = joint.euler_deg
        return {
            "pitch": math.radians(e.y),
            "roll": math.radians(e.x),
            "yaw": math.radians(e.z),
        }

    def _recv_loop(self) -> None:

        buf = bytearray(_RECV_BUF)
        view = memoryview(buf)
        sock = self._sock
        if sock is None:
            return
        while not self._stop.is_set():
            try:
                ready, _, _ = select.select([sock], [], [], 0.1)
            except (OSError, ValueError):
                return
            if not ready:
                continue

            nbytes = 0
            drained = -1
            while True:
                try:
                    n, _ = sock.recvfrom_into(buf, _RECV_BUF)
                except BlockingIOError:
                    break
                except OSError:
                    return
                nbytes = n
                drained += 1
            if nbytes == 0:
                continue

            try:
                msg = _decoder.decode(view[:nbytes])
            except msgspec.DecodeError:
                continue

            now = time.monotonic()
            with self._lock:
                self._payload = msg
                self._last_update = now
                self._seq += 1
                self._dropped += drained
            self._has_payload.set()


def print_payload(tag: str, p: PosePayload, joint_filter=None):
    joints = (
        p.joints
        if joint_filter is None
        else [
            j for j in p.joints if j.name in joint_filter or str(j.id) in joint_filter
        ]
    )
    print(
        f"\n[{tag}] ts={p.timestamp:.3f} coord={p.coordinate_type} "
        f"root={p.root_translation} static_index={p.static_index}"
    )
    for j in joints:
        q, e = j.quaternion, j.euler_deg
        print(
            f"  bone[{j.id:>2}] {j.name:<11} "
            f"quat(wxyz)=[{q.w:+.4f},{q.x:+.4f},{q.y:+.4f},{q.z:+.4f}] "
            f"euler_deg(xyz)=[{e.x:+8.2f},{e.y:+8.2f},{e.z:+8.2f}]"
        )
    print(flush=True)


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


async def _recv_loop_cli(sock, print_every, joint_filter, jsonl_path):
    fp = open(jsonl_path, "ab", buffering=1024 * 1024) if jsonl_path else None
    n = fps = 0
    last_log = time.monotonic()
    try:
        async for data, addr in sock:
            n += 1
            fps += 1
            try:
                msg = _decoder.decode(data)
            except msgspec.DecodeError as e:
                print(f"[RECV] decode error from {addr}: {e}")
                continue
            if fp is not None:
                fp.write(data)
                fp.write(b"\n")
            if print_every and n % print_every == 0:
                print_payload(f"RECV {addr[0]}:{addr[1]}", msg, joint_filter)
            now = time.monotonic()
            if now - last_log >= 1.0:
                print(
                    f"[RECV] {fps} fps  last={len(data)}B  joints={len(msg.joints)}  total={n}"
                )
                fps, last_log = 0, now
    finally:
        if fp is not None:
            fp.close()


async def run_receiver(
    host: str, port: int, print_every: int, joint_filter, jsonl_path
):
    sock = await anyio.create_udp_socket(
        family=socket.AF_INET, local_host=host, local_port=port
    )
    print(
        f"[RECV] listening on {host}:{port}"
        + (f"  -> {jsonl_path}" if jsonl_path else "")
    )
    try:
        async with anyio.create_task_group() as tg:
            tg.start_soon(_recv_loop_cli, sock, print_every, joint_filter, jsonl_path)
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
    ap = argparse.ArgumentParser(description="Rebocap UDP receiver")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=9010)
    ap.add_argument("--print-every", type=int, default=60, help="0=禁用周期打印")
    ap.add_argument("--joints", default="all", help="过滤关节, 逗号分隔 (名字或id)")
    ap.add_argument("--jsonl", default=None, help="每帧追加到 .jsonl 落盘")
    args = ap.parse_args()

    joints = args.joints.strip()
    joint_filter = (
        None
        if joints.lower() == "all"
        else {t.strip() for t in joints.split(",") if t.strip()}
    )

    try:
        anyio.run(
            run_receiver,
            args.host,
            args.port,
            args.print_every,
            joint_filter,
            args.jsonl,
            backend_options=_try_uvloop(),
        )
    except KeyboardInterrupt:
        pass
    print("[MAIN] stopped")
    os._exit(0)


if __name__ == "__main__":
    main()
