import numpy as np
import wuji_sdk
from libs.drivers.rate_limiter import perf_counter, sleep
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from wuji_sdk import SdkManager


class WujiGloveDevice:
    def __init__(
        self,
        hand_side="right",
        device_name="glove",
        sn=None,
        palm_euler_order="ZYX",
        palm_axis_remap="",
        palm_filter_alpha=0.3,
        palm_imu_resource="imu_data_palm",
        subscribe_skeleton=True,
    ):
        self._hand_side = hand_side.lower() if hand_side else None
        self._device_name = device_name
        self._last_data = {"left_fingers": None, "right_fingers": None}

        self._palm_imu_resource = palm_imu_resource
        self._palm_euler_order = palm_euler_order
        self._palm_frame = (
            R.from_matrix(_axis_remap_matrix(palm_axis_remap))
            if palm_axis_remap
            else None
        )
        self._palm_alpha = float(palm_filter_alpha)
        self._imu_sub = None
        self._palm_q0_inv = None
        self._palm_heading = None
        self._palm_q_filt = None
        self._palm_prev_euler = None
        self._palm_zero_nsamples = 0
        self._palm_zero_spread = 0.0

        manager = SdkManager.instance()
        opts = wuji_sdk.ConnectOptions(enable_bridge=False)
        if sn:
            self._device = manager.connect(sn=sn, device_name=device_name, options=opts)
        else:
            self._device = _scan_and_connect_glove(
                manager, self._hand_side, device_name
            )
        self._sub = (
            self._device.hand_skeleton().subscribe() if subscribe_skeleton else None
        )

    def get_fingers_data(self):
        if self._sub is None:
            raise RuntimeError("glove constructed with subscribe_skeleton=False")
        skeleton = self._sub.recv()
        if skeleton is None:
            return self._last_data
        while True:
            newer = self._sub.recv()
            if newer is None:
                break
            skeleton = newer

        keypoints = np.array(
            [j.pose.position for j in skeleton.joints], dtype=np.float32
        )
        if keypoints.shape != (21, 3):
            print(
                f"Warning: unexpected skeleton shape {keypoints.shape}, skipping frame"
            )
            return self._last_data

        hand_side = self._hand_side or self._detect_hand_side(skeleton)
        result = {"left_fingers": None, "right_fingers": None}
        result[f"{hand_side}_fingers"] = keypoints
        self._last_data = result
        return result

    def _recv_palm_quat(self):
        if self._imu_sub is None:
            self._imu_sub = getattr(self._device, self._palm_imu_resource)().subscribe()
        imu = self._imu_sub.recv()
        if imu is None:
            return None
        while True:
            newer = self._imu_sub.recv()
            if newer is None:
                break
            imu = newer
        cov = getattr(imu, "orientation_covariance", None)
        if cov is not None and len(cov) > 0 and cov[0] == -1.0:
            return None
        o = imu.orientation
        return _canon([o.x, o.y, o.z, o.w])

    def set_palm_zero(self, duration=2.0):
        quats = []
        if duration and duration > 0:
            deadline = perf_counter() + duration
            while perf_counter() < deadline:
                q = self._recv_palm_quat()
                if q is not None:
                    quats.append(q)
                sleep(0.01)
        else:
            q = self._recv_palm_quat()
            if q is not None:
                quats.append(q)
        if not quats:
            self._palm_zero_nsamples = 0
            return False

        q0 = _avg_quat(quats)
        dots = np.clip(np.abs(np.asarray(quats) @ q0), -1.0, 1.0)
        self._palm_zero_spread = float(2.0 * np.arccos(dots.min()))
        self._palm_zero_nsamples = len(quats)

        self._set_zero_quat(q0)
        return True

    def get_palm_zero(self):
        if self._palm_q0_inv is None:
            return None
        return self._palm_q0_inv.inv().as_quat().tolist()

    def apply_palm_zero(self, q0_xyzw):
        self._set_zero_quat(_canon(q0_xyzw))

    def _set_zero_quat(self, q0):
        self._palm_q0_inv = R.from_quat(q0).inv()
        yaw0 = R.from_quat(q0).as_euler("ZYX")[0]
        self._palm_heading = R.from_euler("z", yaw0)
        self._palm_q_filt = None
        self._palm_prev_euler = None

    def get_palm_rpy(self, filtered=True):
        q = self._recv_palm_quat()
        if q is None:
            return None
        q = _canon(q, self._palm_q_filt)
        if filtered and self._palm_q_filt is not None:
            q = Slerp([0.0, 1.0], R.from_quat(np.vstack([self._palm_q_filt, q])))(
                [self._palm_alpha]
            ).as_quat()[0]
        self._palm_q_filt = q

        q0_inv = self._palm_q0_inv if self._palm_q0_inv is not None else R.identity()
        rel = R.from_quat(q) * q0_inv
        if self._palm_heading is not None:
            rel = self._palm_heading.inv() * rel * self._palm_heading
        if self._palm_frame is not None:
            rel = self._palm_frame.inv() * rel * self._palm_frame
        euler = rel.as_euler(self._palm_euler_order)
        if self._palm_prev_euler is not None:
            euler = self._palm_prev_euler + (
                (euler - self._palm_prev_euler + np.pi) % (2.0 * np.pi) - np.pi
            )
        self._palm_prev_euler = euler

        by_axis = {ax: float(a) for ax, a in zip(self._palm_euler_order.upper(), euler)}
        return {
            "roll": by_axis.get("X", 0.0),
            "pitch": by_axis.get("Y", 0.0),
            "yaw": by_axis.get("Z", 0.0),
        }

    def cleanup(self):
        self._sub = None
        self._imu_sub = None
        self._device = None

    @staticmethod
    def _detect_hand_side(skeleton):
        return "left" if skeleton.header.frame_id.startswith("l") else "right"


def _scan_and_connect_glove(manager, want_side, device_name):
    candidates = [d for d in manager.scan() if d.sn.startswith("WG")]
    if not candidates:
        raise RuntimeError("No Wuji Glove discovered (no SN starts with 'WG').")
    opts = wuji_sdk.ConnectOptions(enable_bridge=False)
    if len(candidates) == 1 or not want_side:
        return manager.connect(
            sn=candidates[0].sn, device_name=device_name, options=opts
        )

    want = want_side.lower()
    seen = []
    wuji_sdk.set_log_level("error")
    try:
        for cand in candidates:
            dev = manager.connect(sn=cand.sn, device_name=device_name, options=opts)
            side = dev.hand_side().get().lower()
            if side == want:
                return dev
            seen.append((cand.sn, side))
            manager.disconnect(device_name=device_name)
    finally:
        wuji_sdk.set_log_level("warn")
    raise RuntimeError(
        f"Found {len(candidates)} Wuji Gloves but none matched hand_side='{want_side}': {seen}"
    )


def _axis_remap_matrix(spec):
    basis = {
        "x": np.array([1.0, 0.0, 0.0]),
        "y": np.array([0.0, 1.0, 0.0]),
        "z": np.array([0.0, 0.0, 1.0]),
    }
    cols = []
    for tok in spec.split(","):
        tok = tok.strip().lower()
        if not tok:
            continue
        sign = -1.0 if tok[0] == "-" else 1.0
        tok = tok.lstrip("+-")
        cols.append(sign * basis[tok])
    m = np.column_stack(cols)
    if m.shape != (3, 3) or not np.allclose(m.T @ m, np.eye(3), atol=1e-6):
        raise ValueError(f"palm_axis_remap '{spec}' 不是正交轴置换")
    if np.linalg.det(m) < 0:
        raise ValueError(f"palm_axis_remap '{spec}' 是镜像(det<0), 需翻转一个轴的符号")
    return m


def _canon(q, ref=None):
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n == 0.0:
        return q
    q = q / n
    if ref is not None:
        if np.dot(q, np.asarray(ref, dtype=np.float64)) < 0.0:
            q = -q
    elif q[3] < 0.0:
        q = -q
    return q


def _avg_quat(quats):
    Q = np.asarray(quats, dtype=np.float64)
    eigvals, eigvecs = np.linalg.eigh(Q.T @ Q)
    return _canon(eigvecs[:, int(np.argmax(eigvals))])
