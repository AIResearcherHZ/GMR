import numpy as np
import pinocchio as pin


class RobotWrapper:
    def __init__(self, urdf_path, hand_side=None):
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        if self.model.nv != self.model.nq:
            raise NotImplementedError("Cannot handle robot with special joint.")
        self.hand_side = hand_side.lower() if hand_side else None

    @property
    def joint_limits(self):
        return np.stack(
            [self.model.lowerPositionLimit, self.model.upperPositionLimit], axis=1
        )

    def get_link_index(self, name):
        for candidate in (name, f"{self.hand_side}_{name}"):
            idx = self.model.getFrameId(candidate, pin.BODY)
            if idx < self.model.nframes:
                return idx
        raise RuntimeError(f"Frame '{name}' not found.")

    def compute_forward_kinematics(self, qpos):
        pin.forwardKinematics(self.model, self.data, qpos)

    def get_link_pose(self, link_id):
        return pin.updateFramePlacement(self.model, self.data, link_id).homogeneous

    def compute_all_jacobians_batch(self, qpos, link_indices):
        qpos = np.asarray(qpos, dtype=np.float64)
        pin.computeJointJacobians(self.model, self.data, qpos)
        pin.updateFramePlacements(self.model, self.data)
        jacobians = []
        for idx in link_indices:
            J_local = pin.getFrameJacobian(self.model, self.data, idx, pin.LOCAL)
            R = self.data.oMf[idx].rotation
            jacobians.append(R @ J_local[:3, :])
        return np.stack(jacobians, axis=0)
