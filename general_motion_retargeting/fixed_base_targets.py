import mujoco as mj
import numpy as np


class FixedBaseTargets:
    def __init__(self, root_name, root_pos, root_quat, heading_offset=0.0):
        self.root_name = root_name
        self.root_pos = root_pos.copy()
        self.heading = 2.0 * np.arctan2(root_quat[3], root_quat[0]) + heading_offset
        self._origin = np.empty(3)
        self._position = np.empty(3)
        self._rotated_position = np.empty(3)
        self._orientation = np.empty(4)
        self._rotation = np.zeros(4)

    def apply(self, human_data):
        root_pos, root_quat = human_data[self.root_name]
        self._origin[:] = root_pos
        half_angle = 0.5 * self.heading - np.arctan2(root_quat[3], root_quat[0])
        self._rotation[0] = np.cos(half_angle)
        self._rotation[3] = np.sin(half_angle)
        for pos, quat in human_data.values():
            np.subtract(pos, self._origin, out=self._position)
            mj.mju_rotVecQuat(self._rotated_position, self._position, self._rotation)
            np.add(self._rotated_position, self.root_pos, out=pos)
            mj.mju_mulQuat(self._orientation, self._rotation, quat)
            quat[:] = self._orientation
        return human_data
