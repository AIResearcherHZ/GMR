from pathlib import Path

import nlopt
import numpy as np

from .robot import RobotWrapper

M_TO_CM = 100.0


def huber_loss_np(x, delta=2.0):
    abs_x = np.abs(x)
    return np.where(abs_x <= delta, 0.5 * x**2, delta * (abs_x - 0.5 * delta))


def huber_loss_grad_np(x, delta=2.0):
    abs_x = np.abs(x)
    return np.where(abs_x <= delta, x, delta * np.sign(x))


class Optimizer:
    MP_TIP_INDICES = [4, 8, 12, 16, 20]
    MP_PIP_INDICES = [2, 6, 10, 14, 18]
    MP_DIP_INDICES = [3, 7, 11, 15, 19]

    def __init__(self, config):
        self.config = config
        opt_cfg = config.get("optimizer", {})
        self.hand_side = opt_cfg.get("hand_side", "right").lower()
        if self.hand_side not in ("right", "left"):
            raise ValueError(
                f"hand_side must be 'right' or 'left', got {self.hand_side}"
            )

        rc = config.get("retarget", {})
        self.huber_delta = rc.get("huber_delta", 2.0)
        self.norm_delta = rc.get("norm_delta", 0.04)
        self.huber_delta_dir = rc.get("huber_delta_dir", 0.5)
        self.w_pos = rc.get("w_pos", 1.0)
        self.w_dir = rc.get("w_dir", 10.0)
        self.scaling = rc.get("scaling", 1.0)
        self.w_full_hand = rc.get("w_full_hand", 1.0)

        seg = rc.get("segment_scaling", {})
        self.segment_scaling = np.ones((5, 3), dtype=np.float64)
        for i, name in enumerate(("thumb", "index", "middle", "ring", "pinky")):
            if name in seg:
                scales = np.array(seg[name])
                if len(scales) == 4:
                    self.segment_scaling[i] = scales[1:4]
                elif len(scales) == 3:
                    self.segment_scaling[i] = scales

        pc = rc.get("pinch_thresholds", {})
        keys = ("index", "middle", "ring", "pinky")
        self.d1 = np.array(
            [pc.get(k, {}).get("d1", 2.0) for k in keys], dtype=np.float64
        )
        self.d2 = np.array(
            [pc.get(k, {}).get("d2", 4.0) for k in keys], dtype=np.float64
        )

        self.thumb_skip_pip = rc.get("thumb_skip_pip", False)
        self.w_hyper = rc.get("w_hyper", 0.0)
        self.soft_min = rc.get("soft_min", 0.0)
        self.w_couple = rc.get("w_couple", 0.0)
        self.couple_ratio = rc.get("couple_ratio", 0.7)
        self._pip_idx = np.array([2, 6, 10, 14, 18], dtype=np.int64)
        self._dip_idx = np.array([3, 7, 11, 15, 19], dtype=np.int64)
        self._flex_idx = np.array([2, 3, 6, 7, 10, 11, 14, 15, 18, 19], dtype=np.int64)

        urdf_path = opt_cfg.get("urdf_path")
        if not urdf_path:
            raise RuntimeError("optimizer.urdf_path is required")
        urdf_path = Path(urdf_path)
        if not urdf_path.is_absolute():
            yaml_dir = config.get("__yaml_dir")
            if yaml_dir is None:
                raise RuntimeError(
                    "relative urdf_path needs config['__yaml_dir']; use Retargeter.from_yaml"
                )
            urdf_path = (Path(yaml_dir) / urdf_path).resolve()
        if not urdf_path.exists():
            raise FileNotFoundError(f"urdf_path not found: {urdf_path}")

        self.robot = RobotWrapper(str(urdf_path), hand_side=self.hand_side)
        self.num_joints = self.robot.model.nq

        self.opt = nlopt.opt(nlopt.LD_SLSQP, self.num_joints)
        self.opt.set_maxeval(50)
        self.opt.set_ftol_abs(1e-4)
        self.opt.set_lower_bounds(self.robot.joint_limits[:, 0].tolist())
        self.opt.set_upper_bounds(self.robot.joint_limits[:, 1].tolist())

        self.origin_link_name = "palm_link"
        self.task_link_names = [f"finger{i}_tip_link" for i in range(1, 6)]
        self.link3_names = [f"finger{i}_link3" for i in range(1, 6)]
        self.link4_names = [f"finger{i}_link4" for i in range(1, 6)]
        self._build_link_indices()
        self.last_qpos = None

    def _build_link_indices(self):
        names = (
            [self.origin_link_name]
            + self.task_link_names
            + self.link3_names
            + self.link4_names
        )
        self.computed_link_names = list(dict.fromkeys(names))
        self.computed_link_indices = [
            self.robot.get_link_index(n) for n in self.computed_link_names
        ]
        idx = self.computed_link_names.index
        self.origin_indices = [idx(self.origin_link_name)] * 5
        self.task_indices = [idx(n) for n in self.task_link_names]
        self.link3_indices = [idx(n) for n in self.link3_names]
        self.link4_indices = [idx(n) for n in self.link4_names]

    def _compute_pinch_alpha(self, kp):
        thumb_tip = kp[self.MP_TIP_INDICES[0]]
        finger_tips = kp[self.MP_TIP_INDICES[1:]]
        distances = np.linalg.norm(finger_tips - thumb_tip, axis=1) * M_TO_CM
        alphas_4 = np.clip((self.d2 - distances) / (self.d2 - self.d1 + 1e-8), 0.0, 0.7)
        return np.concatenate([[np.max(alphas_4)], alphas_4])

    def _compute_tip_vectors(self, kp, scaling=1.0):
        wrist = kp[0]
        vectors = (
            np.array([kp[i] - wrist for i in self.MP_TIP_INDICES]) * scaling * M_TO_CM
        )
        return vectors.astype(np.float64)

    def _compute_tip_dirs(self, kp):
        tip_dirs = []
        for dip_idx, tip_idx in zip(self.MP_DIP_INDICES, self.MP_TIP_INDICES):
            d = kp[tip_idx] - kp[dip_idx]
            tip_dirs.append(d / (np.linalg.norm(d) + 1e-8))
        return np.array(tip_dirs, dtype=np.float64)

    def _compute_full_hand_vectors(self, kp, scaling):
        wrist = kp[0]
        pip = np.array([kp[i] - wrist for i in self.MP_PIP_INDICES]) * scaling[:, 0:1]
        dip = np.array([kp[i] - wrist for i in self.MP_DIP_INDICES]) * scaling[:, 1:2]
        tip = np.array([kp[i] - wrist for i in self.MP_TIP_INDICES]) * scaling[:, 2:3]
        return (np.vstack([pip, dip, tip]) * M_TO_CM).astype(np.float64)

    def _get_init_qpos(self, last_qpos):
        if last_qpos is not None:
            init = np.asarray(last_qpos, dtype=np.float64)
        elif self.last_qpos is not None:
            init = self.last_qpos
        else:
            init = self.robot.joint_limits.mean(axis=1)
        jl = self.robot.joint_limits
        return np.clip(init, jl[:, 0], jl[:, 1])

    def _get_reg_qpos(self, last_qpos):
        if last_qpos is not None:
            return np.asarray(last_qpos, dtype=np.float64)
        if self.last_qpos is not None:
            return self.last_qpos
        return None

    def solve(self, mediapipe_keypoints, last_qpos=None):
        kp = np.asarray(mediapipe_keypoints, dtype=np.float64)
        if kp.shape != (21, 3):
            raise ValueError(f"Expected shape (21, 3), got {kp.shape}")
        reg_qpos = self._get_reg_qpos(last_qpos)
        init_qpos = self._get_init_qpos(last_qpos)
        alphas = self._compute_pinch_alpha(kp)
        target_tip_vectors = self._compute_tip_vectors(kp, self.scaling)
        target_tip_dirs = self._compute_tip_dirs(kp)
        target_full_hand_vectors = self._compute_full_hand_vectors(
            kp, self.segment_scaling
        )
        objective_fn = self._get_objective(
            target_tip_vectors,
            target_tip_dirs,
            target_full_hand_vectors,
            alphas,
            reg_qpos,
        )
        return self._run_optimization(objective_fn, init_qpos)

    def _get_objective(
        self,
        target_tip_vectors,
        target_tip_dirs,
        target_full_hand_vectors,
        alphas,
        last_qpos,
    ):
        target_tip_vectors = np.asarray(target_tip_vectors, dtype=np.float64)
        target_tip_dirs = np.asarray(target_tip_dirs, dtype=np.float64)
        target_full_hand_vectors = np.asarray(
            target_full_hand_vectors, dtype=np.float64
        )
        alphas = np.asarray(alphas, dtype=np.float64)
        if last_qpos is not None:
            last_qpos = np.asarray(last_qpos, dtype=np.float64)

        def objective(x, grad_out):
            loss, grad = self._loss_and_grad(
                np.asarray(x, dtype=np.float64),
                target_tip_vectors,
                target_tip_dirs,
                target_full_hand_vectors,
                alphas,
                last_qpos,
            )
            if grad_out.size > 0:
                grad_out[:] = grad
            return float(loss)

        return objective

    def _run_optimization(self, objective_fn, init_qpos):
        self.opt.set_min_objective(objective_fn)
        try:
            qpos = np.array(self.opt.optimize(init_qpos.tolist()), dtype=np.float32)
        except RuntimeError as e:
            print(f"[Optimizer] Optimization failed: {e}")
            qpos = np.array(init_qpos, dtype=np.float32)
        self.last_qpos = qpos.astype(np.float64)
        return qpos

    def _loss_and_grad(
        self,
        qpos,
        target_tip_vectors,
        target_tip_dirs,
        target_full_hand_vectors,
        alphas,
        last_qpos,
    ):
        qpos = np.asarray(qpos, dtype=np.float64)
        self.robot.compute_forward_kinematics(qpos)
        positions = (
            np.array(
                [
                    self.robot.get_link_pose(i)[:3, 3]
                    for i in self.computed_link_indices
                ],
                dtype=np.float64,
            )
            * M_TO_CM
        )
        Js = (
            self.robot.compute_all_jacobians_batch(qpos, self.computed_link_indices)
            * M_TO_CM
        )

        origin_pos = positions[self.origin_indices]
        task_pos = positions[self.task_indices]
        link3_pos = positions[self.link3_indices]
        link4_pos = positions[self.link4_indices]
        wrist_pos = positions[self.origin_indices[0]]

        J_origin = Js[self.origin_indices]
        J_task = Js[self.task_indices]
        J_link3 = Js[self.link3_indices]
        J_link4 = Js[self.link4_indices]
        J_wrist = Js[self.origin_indices[0]]

        total_grad = np.zeros(self.num_joints, dtype=np.float64)

        diff_pos = (task_pos - origin_pos) - target_tip_vectors
        dist_pos = np.linalg.norm(diff_pos, axis=1)
        loss_tip_pos = huber_loss_np(dist_pos, self.huber_delta)
        huber_grad_pos = huber_loss_grad_np(dist_pos, self.huber_delta)
        diff_normed_pos = diff_pos / (dist_pos[:, None] + 1e-8)
        for i in range(5):
            grad_coeff = alphas[i] * self.w_pos * huber_grad_pos[i]
            total_grad += grad_coeff * (diff_normed_pos[i] @ (J_task[i] - J_origin[i]))

        robot_tip_dir_vec = task_pos - link4_pos
        robot_tip_dir_norm = np.linalg.norm(robot_tip_dir_vec, axis=1, keepdims=True)
        robot_tip_dirs = robot_tip_dir_vec / (robot_tip_dir_norm + 1e-8)
        diff_dir = robot_tip_dirs - target_tip_dirs
        dist_dir = np.linalg.norm(diff_dir, axis=1)
        loss_tip_dir = huber_loss_np(dist_dir, self.huber_delta_dir)
        huber_grad_dir = huber_loss_grad_np(dist_dir, self.huber_delta_dir)
        diff_normed_dir = diff_dir / (dist_dir[:, None] + 1e-8)
        for i in range(5):
            grad_coeff = alphas[i] * self.w_dir * huber_grad_dir[i]
            u = robot_tip_dirs[i]
            n = robot_tip_dir_norm[i, 0]
            J_norm = (np.eye(3) - np.outer(u, u)) / (n + 1e-8)
            total_grad += grad_coeff * (
                diff_normed_dir[i] @ J_norm @ (J_task[i] - J_link4[i])
            )

        robot_pip_vec = link3_pos - wrist_pos
        robot_dip_vec = link4_pos - wrist_pos
        robot_tip_vec_full = task_pos - wrist_pos
        target_pip = target_full_hand_vectors[:5]
        target_dip = target_full_hand_vectors[5:10]
        target_tip = target_full_hand_vectors[10:15]
        diff_pip = robot_pip_vec - target_pip
        diff_dip = robot_dip_vec - target_dip
        diff_tip = robot_tip_vec_full - target_tip
        dist_pip = np.linalg.norm(diff_pip, axis=1)
        dist_dip = np.linalg.norm(diff_dip, axis=1)
        dist_tip = np.linalg.norm(diff_tip, axis=1)
        loss_pip = huber_loss_np(dist_pip, self.huber_delta)
        loss_dip = huber_loss_np(dist_dip, self.huber_delta)
        loss_tip_full = huber_loss_np(dist_tip, self.huber_delta)

        pip_mask = np.ones(5, dtype=np.float64)
        n_terms = np.full(5, 3.0, dtype=np.float64)
        if self.thumb_skip_pip:
            pip_mask[0] = 0.0
            n_terms[0] = 2.0
        loss_full_hand = (pip_mask * loss_pip + loss_dip + loss_tip_full) / n_terms

        huber_grad_pip = huber_loss_grad_np(dist_pip, self.huber_delta)
        huber_grad_dip = huber_loss_grad_np(dist_dip, self.huber_delta)
        huber_grad_tip = huber_loss_grad_np(dist_tip, self.huber_delta)
        diff_normed_pip = diff_pip / (dist_pip[:, None] + 1e-8)
        diff_normed_dip = diff_dip / (dist_dip[:, None] + 1e-8)
        diff_normed_tip = diff_tip / (dist_tip[:, None] + 1e-8)
        for i in range(5):
            grad_coeff = (1.0 - alphas[i]) * self.w_full_hand / n_terms[i]
            if pip_mask[i] != 0.0:
                total_grad += (
                    grad_coeff
                    * huber_grad_pip[i]
                    * (diff_normed_pip[i] @ (J_link3[i] - J_wrist))
                )
            total_grad += (
                grad_coeff
                * huber_grad_dip[i]
                * (diff_normed_dip[i] @ (J_link4[i] - J_wrist))
            )
            total_grad += (
                grad_coeff
                * huber_grad_tip[i]
                * (diff_normed_tip[i] @ (J_task[i] - J_wrist))
            )

        loss_tip_dir_vec = self.w_pos * loss_tip_pos + self.w_dir * loss_tip_dir
        loss_full = self.w_full_hand * loss_full_hand
        loss_per_finger = alphas * loss_tip_dir_vec + (1.0 - alphas) * loss_full
        total_loss = np.sum(loss_per_finger)

        if last_qpos is not None:
            total_loss += self.norm_delta * np.sum((qpos - last_qpos) ** 2)
            total_grad += 2.0 * self.norm_delta * (qpos - last_qpos)

        if self.w_hyper != 0.0:
            flex_qpos = qpos[self._flex_idx]
            penalty = np.maximum(self.soft_min - flex_qpos, 0.0)
            total_loss += self.w_hyper * np.sum(penalty**2)
            total_grad[self._flex_idx] += self.w_hyper * (-2.0 * penalty)

        if self.w_couple != 0.0:
            diff = qpos[self._dip_idx] - self.couple_ratio * qpos[self._pip_idx]
            total_loss += self.w_couple * np.sum(diff**2)
            total_grad[self._dip_idx] += self.w_couple * (2.0 * diff)
            total_grad[self._pip_idx] += self.w_couple * (
                -2.0 * self.couple_ratio * diff
            )

        return total_loss, total_grad
