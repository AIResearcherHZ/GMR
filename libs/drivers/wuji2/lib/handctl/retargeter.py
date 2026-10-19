from pathlib import Path

import numpy as np
import yaml

from .mediapipe import apply_mediapipe_transformations
from .optimizer import Optimizer


class LPFilter:
    def __init__(self, alpha):
        self.alpha = alpha
        self.y = None

    def next(self, x):
        if self.y is None:
            self.y = x.copy()
        else:
            self.y = self.y + self.alpha * (x - self.y)
        return self.y.copy()

    def reset(self):
        self.y = None


class Retargeter:
    def __init__(self, config, hand_side="right"):
        self.config = config
        self.hand_side = hand_side.lower()
        if self.hand_side not in ("left", "right"):
            raise ValueError(f"hand_side must be 'left' or 'right', got {hand_side}")
        config.setdefault("optimizer", {})["hand_side"] = self.hand_side

        self.optimizer = Optimizer(config)

        rc = config.get("retarget", {})
        self.lp_filter = LPFilter(rc.get("lp_alpha", 0.2))
        self.rotation_xyz = rc.get("mediapipe_rotation", {})
        self.wrist_offset_m = (
            np.array(rc.get("wrist_offset_cm", [0.0, 0.0, 0.0]), dtype=np.float64)
            / 100.0
        )
        self.thumb_offset_m = (
            np.array(rc.get("thumb_offset_cm", [0.0, 0.0, 0.0]), dtype=np.float64)
            / 100.0
        )

    def _apply_rotation(self, keypoints):
        x_deg = self.rotation_xyz.get("x", 0.0)
        y_deg = self.rotation_xyz.get("y", 0.0)
        z_deg = self.rotation_xyz.get("z", 0.0)
        if x_deg == 0 and y_deg == 0 and z_deg == 0:
            return keypoints
        from scipy.spatial.transform import Rotation

        rot = Rotation.from_euler("xyz", [x_deg, y_deg, z_deg], degrees=True)
        return keypoints @ rot.as_matrix().T

    def _apply_offset(self, kp):
        kp[5:] = kp[5:] + self.wrist_offset_m
        kp[1:5] = kp[1:5] + self.thumb_offset_m
        return kp

    def retarget(self, raw_keypoints, apply_filter=True):
        kp = apply_mediapipe_transformations(raw_keypoints, self.hand_side)
        if self.rotation_xyz:
            kp = self._apply_rotation(kp)
        kp = self._apply_offset(kp)
        qpos = self.optimizer.solve(kp)
        if apply_filter:
            qpos = self.lp_filter.next(qpos)
        return qpos

    @classmethod
    def from_yaml(cls, yaml_path, hand_side="right"):
        yaml_path = Path(yaml_path).resolve()
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)
        config["__yaml_dir"] = str(yaml_path.parent)
        return cls(config, hand_side)
