from __future__ import annotations

import enum
import importlib
import random
import sys

__all__ = ["REBOCAP_JOINT_NAMES", "CoordinateType", "RebocapWsSdk"]

_SUPPORTED = ("py36", "py37", "py38", "py39", "py310", "py311", "py312")
_v = sys.version_info
_tag = f"py{_v.major}{_v.minor}"
if _tag not in _SUPPORTED:
    raise ImportError(
        f"rebocap_ws_sdk: 当前 Python {_v.major}.{_v.minor} 无对应扩展, "
        f"已编译版本: {', '.join(_SUPPORTED)}"
    )
_ext = importlib.import_module(f".{_tag}.rebocap_ws_sdk_ext", __package__)

REBOCAP_JOINT_NAMES = [
    "Pelvis",
    "L_Hip",
    "R_Hip",
    "Spine1",
    "L_Knee",
    "R_Knee",
    "Spine2",
    "L_Ankle",
    "R_Ankle",
    "Spine3",
    "L_Foot",
    "R_Foot",
    "Neck",
    "L_Collar",
    "R_Collar",
    "Head",
    "L_Shoulder",
    "R_Shoulder",
    "L_Elbow",
    "R_Elbow",
    "L_Wrist",
    "R_Wrist",
    "L_Hand",
    "R_Hand",
]


class CoordinateType(enum.Enum):
    DefaultCoordinate = 0
    UnityCoordinate = 1
    BlenderCoordinate = 2
    MayaCoordinate = 3
    MaxCoordinate = 4
    UECoordinate = 5


class RebocapWsSdk:
    __slots__ = (
        "coordinate_type",
        "exception_close_callback_f",
        "handle",
        "pose_msg_callback_f",
    )

    def __init__(
        self,
        coordinate_type: CoordinateType = CoordinateType.DefaultCoordinate,
        use_global_rotation: bool = False,
    ):
        self.pose_msg_callback_f = None
        self.exception_close_callback_f = None
        self.coordinate_type = coordinate_type
        self.handle = _ext.rebocap_ws_sdk_new(
            self,
            RebocapWsSdk.pose_msg_callback,
            RebocapWsSdk.exception_close_callback,
            coordinate_type.value,
            1 if use_global_rotation else 0,
        )

    def __del__(self):

        try:
            _ext.rebocap_ws_sdk_release(self.handle)
        except Exception:
            pass

    def set_pose_msg_callback(self, callback):
        self.pose_msg_callback_f = callback

    def set_exception_close_callback(self, callback):
        self.exception_close_callback_f = callback

    def open(self, port: int, name: str = "reborn_app", uid: int | None = None) -> int:
        if uid is None:
            uid = random.randint(0, 2**63 - 1)
        return _ext.rebocap_ws_sdk_open(self.handle, port, name, uid)

    def close(self):
        _ext.rebocap_ws_sdk_close(self.handle)

    def pose_msg_callback(self, trans, pose24, static_index: int, tp: int):

        cb = self.pose_msg_callback_f
        if cb is not None:
            cb(self, trans, pose24, static_index, tp / 1000.0)

    def exception_close_callback(self):
        cb = self.exception_close_callback_f
        if cb is not None:
            cb(self)

    def get_last_msg(self):
        return _ext.rebocap_ws_sdk_get_last_msg(self.handle)
