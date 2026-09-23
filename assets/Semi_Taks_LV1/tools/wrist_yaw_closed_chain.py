from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

MODEL_XML = Path(__file__).resolve().parents[1] / "Semi_Taks_LV1.xml"


def rz(q: float) -> np.ndarray:
    c, s = np.cos(q), np.sin(q)
    return np.array(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)))


def drz(q: float) -> np.ndarray:
    c, s = np.cos(q), np.sin(q)
    return np.array(((-s, -c, 0.0), (c, -s, 0.0), (0.0, 0.0, 0.0)))


def wrap(q: float) -> float:
    return float((q + np.pi) % (2.0 * np.pi) - np.pi)


def candidates(a: float, b: float, c: float) -> list[float]:
    radius = float(np.hypot(a, b))
    if radius == 0.0 or abs(c) > radius + 1e-12:
        return []
    value = np.clip(-c / radius, -1.0, 1.0)
    phase = np.arctan2(b, a)
    delta = np.arccos(value)
    return [wrap(phase + delta), wrap(phase - delta)]


def choose(values: list[float], limits: np.ndarray, reference: float) -> float:
    valid = [q for q in values if limits[0] - 1e-10 <= q <= limits[1] + 1e-10]
    if not valid:
        raise ValueError("闭链目标不可达或超出关节限位")
    return min(valid, key=lambda q: abs(wrap(q - reference)))


class WristYawClosedChain:
    def __init__(
        self,
        side: str,
        model: mujoco.MjModel | None = None,
        model_xml: Path | str = MODEL_XML,
    ):
        if side not in ("left", "right"):
            raise ValueError("side 必须是 left 或 right")
        self.side = side
        self.model = model or mujoco.MjModel.from_xml_path(str(model_xml))
        self.data = mujoco.MjData(self.model)
        self.motor_name = f"{side}_wrist_yaw_motor_joint"
        self.output_name = f"{side}_wrist_yaw_joint"
        self.motor_joint = self.model.joint(self.motor_name)
        self.output_joint = self.model.joint(self.output_name)
        if not np.allclose(self.model.jnt_axis[self.motor_joint.id], (0, 0, 1)):
            raise ValueError("腕部 yaw 电机轴与求解器不匹配")
        if not np.allclose(self.model.jnt_axis[self.output_joint.id], (0, 0, 1)):
            raise ValueError("腕部 yaw 输出轴与求解器不匹配")
        self.motor_body = self.model.body(f"{side}_wrist_yaw_motor_link")
        self.output_body = self.model.body(f"{side}_wrist_yaw_link")
        self.motor_limits = self.model.jnt_range[self.motor_joint.id].copy()
        self.output_limits = self.model.jnt_range[self.output_joint.id].copy()
        self.loops = []
        for kind in ("outer", "inner"):
            equality = self.model.equality(f"{side}_wrist_yaw_{kind}_loop")
            if self.model.eq_type[equality.id] != mujoco.mjtEq.mjEQ_CONNECT:
                raise ValueError(f"腕部 {side} {kind} 闭链约束类型不匹配")
            rod = self.model.body(f"{side}_wrist_yaw_{kind}_rod_link")
            anchor = self.model.eq_data[equality.id, :6].copy()
            self.loops.append((kind, rod, anchor[:3], anchor[3:], equality.id))
        self.equality_ids = {loop[4] for loop in self.loops}
        self.equality_rows: np.ndarray | None = None
        if any(abs(self._constraint(0.0, 0.0, loop)) > 1e-10 for loop in self.loops):
            raise ValueError(f"腕部 {side} 闭链零位锚点不匹配")

    def parameters(self) -> dict:
        return {
            "side": self.side,
            "motor_origin": self.motor_body.pos.copy(),
            "output_origin": self.output_body.pos.copy(),
            "motor_limits": self.motor_limits.copy(),
            "output_limits": self.output_limits.copy(),
            "loops": [
                {
                    "name": kind,
                    "rod_origin": rod.pos.copy(),
                    "rod_length": float(np.linalg.norm(rod_anchor)),
                    "output_anchor": output_anchor.copy(),
                }
                for kind, rod, rod_anchor, output_anchor, _ in self.loops
            ],
        }

    def _constraint(self, motor: float, output: float, loop: tuple) -> float:
        _, rod, rod_anchor, output_anchor, _ = loop
        pivot = self.motor_body.pos + rz(motor) @ rod.pos
        target = self.output_body.pos + rz(output) @ output_anchor
        return float(
            np.dot(pivot - target, pivot - target) - np.dot(rod_anchor, rod_anchor)
        )

    def _motor_candidates(self, output: float, loop: tuple) -> list[float]:
        _, rod, rod_anchor, output_anchor, _ = loop
        target = self.output_body.pos + rz(output) @ output_anchor
        base = self.motor_body.pos - target
        p = rod.pos
        fixed = np.array((p[0], 0.0, p[2]))
        cosine = np.array((0.0, p[1], 0.0))
        sine = np.array((-p[1], 0.0, 0.0))
        x = base + fixed
        a = 2.0 * np.dot(x, cosine)
        b = 2.0 * np.dot(x, sine)
        c = np.dot(x, x) + np.dot(cosine, cosine) - np.dot(rod_anchor, rod_anchor)
        return candidates(float(a), float(b), float(c))

    def _output_candidates(self, motor: float, loop: tuple) -> list[float]:
        _, rod, rod_anchor, output_anchor, _ = loop
        pivot = self.motor_body.pos + rz(motor) @ rod.pos
        base = self.output_body.pos - pivot
        p = output_anchor
        fixed = np.array((0.0, 0.0, p[2]))
        cosine = np.array((p[0], p[1], 0.0))
        sine = np.array((-p[1], p[0], 0.0))
        x = base + fixed
        a = 2.0 * np.dot(x, cosine)
        b = 2.0 * np.dot(x, sine)
        c = np.dot(x, x) + np.dot(cosine, cosine) - np.dot(rod_anchor, rod_anchor)
        return candidates(float(a), float(b), float(c))

    def ik(self, output: float) -> float:
        if not self.output_limits[0] <= output <= self.output_limits[1]:
            raise ValueError("腕部 yaw 输出角超限")
        roots = [
            choose(self._motor_candidates(output, loop), self.motor_limits, output)
            for loop in self.loops
        ]
        if abs(wrap(roots[0] - roots[1])) > 2e-7:
            raise ValueError(f"双闭环不一致: {roots}")
        return float(np.mean(roots))

    def fk(self, motor: float) -> float:
        if not self.motor_limits[0] <= motor <= self.motor_limits[1]:
            raise ValueError("腕部 yaw 电机角超限")
        roots = [
            choose(self._output_candidates(motor, loop), self.output_limits, motor)
            for loop in self.loops
        ]
        if abs(wrap(roots[0] - roots[1])) > 2e-7:
            raise ValueError(f"双闭环不一致: {roots}")
        return float(np.mean(roots))

    def jacobian(self, output: float) -> float:
        motor = self.ik(output)
        values = []
        for _, rod, _, output_anchor, _ in self.loops:
            pivot = self.motor_body.pos + rz(motor) @ rod.pos
            target = self.output_body.pos + rz(output) @ output_anchor
            error = pivot - target
            fm = 2.0 * error @ (drz(motor) @ rod.pos)
            fo = -2.0 * error @ (drz(output) @ output_anchor)
            if abs(fm) < 1e-10:
                raise ValueError("腕部闭链位于奇异点")
            values.append(float(-fo / fm))
        if abs(values[0] - values[1]) > 2e-7:
            raise ValueError(f"双闭环 Jacobian 不一致: {values}")
        return float(np.mean(values))

    def velocity_ik(self, output: float, output_velocity: float) -> float:
        return self.jacobian(output) * output_velocity

    def velocity_fk(self, output: float, motor_velocity: float) -> float:
        return motor_velocity / self.jacobian(output)

    def torque_fk(self, output: float, motor_torque: float) -> float:
        return self.jacobian(output) * motor_torque

    def torque_ik(self, output: float, output_torque: float) -> float:
        return output_torque / self.jacobian(output)

    def _set_mujoco_state(self, motor: float, output: float) -> None:
        self.data.qpos[:] = self.model.qpos0
        self.data.qpos[self.motor_joint.qposadr[0]] = motor
        self.data.qpos[self.output_joint.qposadr[0]] = output
        for _, rod, _, output_anchor, _ in self.loops:
            pivot = self.motor_body.pos + rz(motor) @ rod.pos
            target = self.output_body.pos + rz(output) @ output_anchor
            angle = np.arctan2(target[1] - pivot[1], target[0] - pivot[0])
            joint_id = self.model.body_jntadr[rod.id]
            address = self.model.jnt_qposadr[joint_id]
            self.data.qpos[address] = wrap(angle - motor)
        mujoco.mj_forward(self.model, self.data)

    def validate(self, samples: int = 41) -> dict:
        position_error = 0.0
        fk_error = 0.0
        jacobian_error = 0.0
        equality_error = 0.0
        velocity_error = 0.0
        torque_error = 0.0
        h = 1e-6
        for output in np.linspace(
            self.output_limits[0] + 1e-4, self.output_limits[1] - 1e-4, samples
        ):
            motor = self.ik(float(output))
            recovered = self.fk(motor)
            jacobian = self.jacobian(float(output))
            numeric = (self.ik(float(output + h)) - self.ik(float(output - h))) / (
                2.0 * h
            )
            self._set_mujoco_state(motor, float(output))
            if self.equality_rows is None:
                self.equality_rows = np.asarray(
                    [
                        i
                        for i in range(self.data.nefc)
                        if self.data.efc_type[i]
                        == mujoco.mjtConstraint.mjCNSTR_EQUALITY
                        and self.data.efc_id[i] in self.equality_ids
                    ],
                    dtype=np.int32,
                )
            equality_error = max(
                equality_error,
                float(np.max(np.abs(self.data.efc_pos[self.equality_rows]))),
            )
            position_error = max(
                position_error,
                max(
                    abs(self._constraint(motor, float(output), loop))
                    for loop in self.loops
                ),
            )
            fk_error = max(fk_error, abs(wrap(recovered - output)))
            jacobian_error = max(jacobian_error, abs(jacobian - numeric))
            velocity_error = max(
                velocity_error,
                abs(
                    self.velocity_fk(
                        float(output), self.velocity_ik(float(output), 0.37)
                    )
                    - 0.37
                ),
            )
            torque_error = max(
                torque_error,
                abs(
                    self.torque_ik(float(output), self.torque_fk(float(output), 1.23))
                    - 1.23
                ),
            )
        result = {
            "samples": samples,
            "position_squared_error_m2": position_error,
            "mujoco_equality_error_m": equality_error,
            "fk_error_rad": fk_error,
            "jacobian_error": jacobian_error,
            "velocity_error_rad_s": velocity_error,
            "torque_error_nm": torque_error,
        }
        limits = {
            "position_squared_error_m2": 1e-14,
            "mujoco_equality_error_m": 1e-9,
            "fk_error_rad": 1e-9,
            "jacobian_error": 1e-6,
            "velocity_error_rad_s": 1e-12,
            "torque_error_nm": 1e-12,
        }
        failed = {
            name: result[name] for name, limit in limits.items() if result[name] > limit
        }
        if failed:
            raise AssertionError(f"腕部验证失败: {failed}")
        return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", choices=("left", "right", "both"), default="both")
    parser.add_argument("--yaw", type=float, default=0.35)
    parser.add_argument("--velocity", type=float, default=0.4)
    parser.add_argument("--torque", type=float, default=1.0)
    parser.add_argument("--samples", type=int, default=41)
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    for side in ("left", "right") if args.side == "both" else (args.side,):
        chain = WristYawClosedChain(side, model)
        motor = chain.ik(args.yaw)
        jacobian = chain.jacobian(args.yaw)
        print(f"\n[{side} wrist yaw]")
        print("XML 参数:", chain.parameters())
        print(f"IK: output={args.yaw:.9f} -> motor={motor:.9f}")
        print(f"FK: motor={motor:.9f} -> output={chain.fk(motor):.9f}")
        print(f"Jacobian dmotor/doutput={jacobian:.12f}")
        print(
            f"速度: output={args.velocity:.9f} -> motor={chain.velocity_ik(args.yaw, args.velocity):.9f}"
        )
        print(
            f"力矩: motor={args.torque:.9f} -> output={chain.torque_fk(args.yaw, args.torque):.9f}"
        )
        print("MuJoCo 验证:", chain.validate(args.samples))


if __name__ == "__main__":
    main()
