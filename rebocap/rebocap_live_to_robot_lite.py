from __future__ import annotations

import argparse
import json
import os
import pickle
import select
import signal
import socket
import time
from pathlib import Path

import mink
import mujoco as mj
import mujoco.viewer
import numpy as np


HERE = Path(__file__).resolve().parent
IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])
AXIS = np.array([2**-0.5, 2**-0.5, 0.0, 0.0])
FIX = np.array([2**-0.5, -2**-0.5, 0.0, 0.0])
FIX_INV = np.array([2**-0.5, 2**-0.5, 0.0, 0.0])
SMPL = (
    "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee", "Spine2",
    "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot", "Neck",
    "L_Collar", "R_Collar", "Head", "L_Shoulder", "R_Shoulder",
    "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist", "L_Hand", "R_Hand",
)
SMPL_PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21)
MIXAMO = dict(zip(
    SMPL,
    ("Hips", "LeftUpLeg", "RightUpLeg", "Spine", "LeftLeg", "RightLeg",
     "Spine1", "LeftFoot", "RightFoot", "Spine2", "LeftToeBase",
     "RightToeBase", "Neck", "LeftShoulder", "RightShoulder", "Head",
     "LeftArm", "RightArm", "LeftForeArm", "RightForeArm", "LeftHand",
     "RightHand", None, None),
))
ALIGN = {
    "L_Collar": (0.5, 0.5, 0.5, -0.5),
    "R_Collar": (0.5, 0.5, -0.5, 0.5),
    "L_Shoulder": (0.5, 0.5, 0.5, -0.5),
    "R_Shoulder": (0.5, 0.5, -0.5, 0.5),
    "L_Elbow": (0.5, 0.5, 0.5, -0.5),
    "R_Elbow": (0.5, 0.5, -0.5, 0.5),
    "L_Wrist": (0.5, 0.5, 0.5, -0.5),
    "R_Wrist": (0.5, 0.5, -0.5, 0.5),
    "L_Hip": (0, 0, 0, 1), "R_Hip": (0, 0, 0, 1),
    "L_Knee": (0, 0, 0, 1), "R_Knee": (0, 0, 0, 1),
    "L_Ankle": (0.0329465681, -0.0156536438, 0.4596812682, 0.8873345585),
    "R_Ankle": (-0.0329512081, 0.0156518976, 0.4596813656, 0.8873343666),
    "L_Foot": (0.0362878733, -0.0037028146, 0.7305516176, 0.6818825510),
    "R_Foot": (-0.0362914591, 0.0036995102, 0.7305516486, 0.6818823450),
}


def qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array((aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by,
                     aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw))


def qrot(q, v):
    t = 2 * np.cross(q[1:], v)
    return v + q[0] * t + np.cross(q[1:], t)


def qmat(q):
    w, x, y, z = q
    return np.array(((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)),
                     (2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)),
                     (2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y))))


def parse_skeleton(path):
    names, parents, offsets, stack = [], [], [], []
    pending = None
    end_site = False
    with open(path, encoding="utf-8") as fp:
        for raw in fp:
            line = raw.strip()
            if line == "MOTION":
                break
            if line.startswith(("ROOT ", "JOINT ")):
                pending = line.split(None, 1)[1]
            elif line.startswith("End Site"):
                end_site = True
            elif line == "{":
                if not end_site:
                    names.append(pending.split(":", 1)[-1])
                    parents.append(stack[-1] if stack else -1)
                    offsets.append(np.zeros(3))
                    stack.append(len(names) - 1)
            elif line == "}":
                if end_site:
                    end_site = False
                elif stack:
                    stack.pop()
            elif line.startswith("OFFSET ") and not end_site and stack:
                offsets[stack[-1]] = np.fromstring(line[7:], sep=" ")
    if not names or len(names) != len(offsets):
        raise ValueError("BVH 模板无有效骨架")
    return names, parents, np.asarray(offsets)


def frame_from_payload(payload, skeleton, rotation_mode, rest_offset):
    names, parents, offsets = skeleton
    mode = payload.get("rotation_mode", "local") if rotation_mode == "auto" else rotation_mode
    if mode not in ("local", "global"):
        raise ValueError(f"无效 rotation_mode: {mode}")
    incoming = {}
    for joint in payload.get("joints", ()):
        if joint.get("name") in SMPL:
            q = joint["quaternion"]
            incoming[joint["name"]] = np.array([q[k] for k in ("w", "x", "y", "z")], dtype=float)
    if mode == "global":
        glob = [incoming.get(name, IDENTITY) for name in SMPL]
        incoming = {name: glob[i] if p < 0 else qmul(glob[p] * (1, -1, -1, -1), glob[i])
                    for i, (name, p) in enumerate(zip(SMPL, SMPL_PARENTS))}
    by_mixamo = {}
    for i, name in enumerate(SMPL):
        if MIXAMO[name] is None or name not in incoming:
            continue
        qa = np.asarray(ALIGN.get(name, IDENTITY)) if rest_offset else IDENTITY
        parent = SMPL_PARENTS[i]
        qp = np.asarray(ALIGN.get(SMPL[parent], IDENTITY)) if rest_offset and parent >= 0 else IDENTITY
        left = qmul(qp * (1, -1, -1, -1), FIX)
        by_mixamo[MIXAMO[name]] = qmul(qmul(left, incoming[name]), qmul(FIX_INV, qa))
    rt = payload.get("root_translation") or (0, 0, 0)
    root = np.array(rt, dtype=float)
    positions = [None] * len(names)
    quats = [None] * len(names)
    frame = {}
    for i, name in enumerate(names):
        local_q = by_mixamo.get(name, IDENTITY)
        if parents[i] < 0:
            positions[i] = root
            quats[i] = local_q
        else:
            parent = parents[i]
            positions[i] = positions[parent] + qrot(quats[parent], offsets[i]) / 100.0
            quats[i] = qmul(quats[parent], local_q)
        pos = np.array((positions[i][0], -positions[i][2], positions[i][1]))
        frame[name] = (pos, qmul(AXIS, quats[i]))
    for side in ("Left", "Right"):
        foot, toe = side + "Foot", side + "ToeBase"
        if foot in frame and toe in frame:
            frame[side + "FootMod"] = (frame[foot][0], frame[toe][1])
    return frame


class LatestUdp:
    def __init__(self, host, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, port))
        self.sock.setblocking(False)
        self.seq = 0
        self.dropped = 0

    def receive(self, timeout):
        ready, _, _ = select.select([self.sock], [], [], timeout)
        if not ready:
            return None
        latest = None
        drained = 0
        while True:
            try:
                latest, _ = self.sock.recvfrom(65536)
                drained += 1
            except BlockingIOError:
                break
        if latest is None:
            return None
        self.dropped += drained - 1
        self.seq += 1
        try:
            return json.loads(latest)
        except (UnicodeError, json.JSONDecodeError):
            return None

    def close(self):
        self.sock.close()


class Retargeter:
    def __init__(self, xml_path, config_path):
        self.model = mj.MjModel.from_xml_path(str(xml_path))
        self.configuration = mink.Configuration(self.model)
        with open(config_path, encoding="utf-8") as fp:
            config = json.load(fp)
        self.root_name = config["human_root_name"]
        self.robot_root = self.model.body(config["robot_root_name"]).pos.copy()
        ratio = 1.75 / config["human_height_assumption"]
        self.scales = {k: v * ratio for k, v in config["human_scale_table"].items()}
        self.src_rotation = None
        if config.get("src_frame_rot") is not None:
            self.src_rotation = np.asarray(config["src_frame_rot"], dtype=float)
        self.passes = []
        for suffix in ("1", "2"):
            if not config["use_ik_match_table" + suffix]:
                continue
            tasks, entries = [], []
            for body, (human, pw, rw, pos, rot) in config["ik_match_table" + suffix].items():
                task = None
                if pw or rw:
                    task = mink.FrameTask(frame_name=body, frame_type="body", position_cost=pw,
                                          orientation_cost=rw, lm_damping=1)
                    tasks.append(task)
                entries.append((human, task, np.asarray(pos), np.asarray(rot)))
            if self.model.neq:
                tasks.append(mink.EqualityConstraintTask(self.model, cost=5e3, lm_damping=1.0))
            tasks.append(mink.DampingTask(self.model, cost=5.0))
            self.passes.append((tasks, entries))
        self.limits = [mink.ConfigurationLimit(self.model)]
        self.scaled_human_data = {}

    def retarget(self, frame):
        if self.src_rotation is not None:
            origin = frame[self.root_name][0]
            frame = {name: (origin + qrot(self.src_rotation, pos - origin), qmul(self.src_rotation, rot))
                     for name, (pos, rot) in frame.items()}
        root_pos = frame[self.root_name][0]
        scaled_root = self.scales[self.root_name] * root_pos
        scaled = {name: (scaled_root + (frame[name][0] - root_pos) * scale, frame[name][1])
                  for name, scale in self.scales.items()}
        for human, _, pos_offset, rot_offset in self.passes[0][1]:
            pos, rot = scaled[human]
            updated_rot = qmul(rot, rot_offset)
            scaled[human] = (pos + qrot(updated_rot, pos_offset), updated_rot)
        delta = self.robot_root - scaled[self.root_name][0]
        scaled = {name: (pos + delta, rot) for name, (pos, rot) in scaled.items()}
        for tasks, entries in self.passes:
            for human, task, _, _ in entries:
                if task is not None:
                    pos, rot = scaled[human]
                    task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(rot), pos))
            error = self._error(tasks)
            for _ in range(11):
                dt = self.model.opt.timestep
                velocity = mink.solve_ik(self.configuration, tasks, dt, "daqp", 0.5, self.limits)
                self.configuration.integrate_inplace(velocity, dt)
                next_error = self._error(tasks)
                if error - next_error <= 0.001:
                    break
                error = next_error
        self.scaled_human_data = scaled
        return self.configuration.data.qpos.copy()

    def _error(self, tasks):
        return np.linalg.norm(np.concatenate([task.compute_error(self.configuration) for task in tasks]))


class QposStream:
    def __init__(self, path, dim):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.tmp = Path(str(path) + ".qpos.tmp")
        self.fp = open(self.tmp, "wb", buffering=1024 * 1024)
        self.dim = dim
        self.count = 0

    def write(self, qpos):
        self.fp.write(np.ascontiguousarray(qpos, dtype=np.float64).tobytes())
        self.count += 1

    def save(self, model, fps):
        self.fp.close()
        if self.count == 0:
            self.tmp.unlink(missing_ok=True)
            return
        qpos = np.memmap(self.tmp, dtype=np.float64, mode="r", shape=(self.count, self.dim))
        free = model.njnt > 0 and model.jnt_type[0] == mj.mjtJoint.mjJNT_FREE
        split = 7 if free else 0
        dof = np.asarray(qpos[:, split:])
        if free:
            root_pos = np.asarray(qpos[:, :3]).copy()
            root_rot = np.asarray(qpos[:, 3:7])[:, [1, 2, 3, 0]]
        else:
            root_pos = np.repeat(model.body_pos[1:2], self.count, axis=0)
            root_rot = np.repeat(model.body_quat[1:2, [1, 2, 3, 0]], self.count, axis=0)
        root_pos[:, :2] -= root_pos[0, :2]
        ids = [1]
        for i in range(2, model.nbody):
            if int(model.body_parentid[i]) in ids:
                ids.append(i)
        local = np.empty((self.count, len(ids), 3), dtype=np.float32)
        data = mj.MjData(model)
        for i, row in enumerate(qpos):
            data.qpos[:] = row
            mj.mj_forward(model, data)
            local[i] = data.xpos[ids] - data.xpos[ids[0]]
        result = {"fps": fps, "root_pos": root_pos, "root_rot": root_rot,
                  "dof_pos": dof, "local_body_pos": local,
                  "link_body_list": [model.body(i).name for i in ids]}
        with open(self.path, "wb") as fp:
            pickle.dump(result, fp, protocol=pickle.HIGHEST_PROTOCOL)
        del qpos
        self.tmp.unlink()
        print(f"[save] {self.count} frames -> {self.path} ({self.path.stat().st_size}B)")


def draw_human_axes(viewer, human_data):
    scene = viewer.user_scn
    scene.ngeom = 0
    colors = ((1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1))
    for pos, quat in human_data.values():
        matrix = qmat(quat)
        for axis, color in enumerate(colors):
            if scene.ngeom >= scene.maxgeom:
                return
            mj.mjv_initGeom(scene.geoms[scene.ngeom], mj.mjtGeom.mjGEOM_ARROW,
                            (0.01, 0.01, 0.01), pos, matrix.ravel(), color)
            mj.mjv_connector(scene.geoms[scene.ngeom], mj.mjtGeom.mjGEOM_ARROW,
                             0.005, pos, pos + 0.1 * matrix[:, axis])
            scene.ngeom += 1


def parse_args():
    parser = argparse.ArgumentParser(description="Rebocap UDP -> semi_taks_lv1")
    parser.add_argument("--robot", choices=("semi_taks_lv1",), default="semi_taks_lv1")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9010)
    parser.add_argument("--motion_fps", type=int, default=30)
    parser.add_argument("--max-stale", type=float, default=0.5)
    parser.add_argument("--rotation-mode", choices=("auto", "local", "global"), default="auto")
    parser.add_argument("--no-rest-offset", action="store_true")
    parser.add_argument("--rate_limit", action="store_true")
    parser.add_argument("--save_path")
    parser.add_argument("--template", default=str(HERE / "template.txt"))
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--video_path", default="videos/rebocap_live.mp4")
    parser.add_argument("--no-viewer", action="store_true")
    args = parser.parse_args()
    if args.motion_fps <= 0 or not 0 < args.port < 65536 or args.max_stale < 0:
        parser.error("motion_fps、port 或 max-stale 无效")
    return args


def main():
    args = parse_args()
    if args.record_video and args.no_viewer:
        raise ValueError("录制视频需要启用 viewer")
    skeleton = parse_skeleton(args.template)
    xml = HERE / "assets" / "scene_Semi_Taks_LV1.xml"
    retargeter = Retargeter(xml, HERE / "bvh_mixamo_to_semi_taks_lv1.json")
    receiver = LatestUdp(args.host, args.port)
    stream = QposStream(args.save_path, retargeter.model.nq) if args.save_path else None
    data = mj.MjData(retargeter.model)
    viewer = None
    renderer = None
    writer = None
    stop = False

    def stop_signal(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, stop_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, stop_signal)
    last_log = time.monotonic()
    frames = 0
    next_frame = last_log
    try:
        if not args.no_viewer:
            viewer = mj.viewer.launch_passive(retargeter.model, data, show_left_ui=False, show_right_ui=False)
            viewer.cam.distance = 2.0
            viewer.cam.elevation = -10
        if args.record_video:
            import imageio
            Path(args.video_path).parent.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(args.video_path, fps=args.motion_fps)
            renderer = mj.Renderer(retargeter.model, height=480, width=640)
        print(f"[udp] {args.host}:{args.port}, robot={args.robot}, fps={args.motion_fps}", flush=True)
        while not stop and (viewer is None or viewer.is_running()):
            payload = receiver.receive(0.2)
            if payload is None:
                continue
            frame = frame_from_payload(payload, skeleton, args.rotation_mode, not args.no_rest_offset)
            qpos = retargeter.retarget(frame)
            data.qpos[:] = qpos
            mj.mj_forward(retargeter.model, data)
            if viewer is not None:
                viewer.cam.lookat = data.xpos[retargeter.model.body("waist_roll_link").id]
                draw_human_axes(viewer, retargeter.scaled_human_data)
                viewer.sync()
            if renderer is not None:
                renderer.update_scene(data, camera=viewer.cam)
                writer.append_data(renderer.render())
            if stream is not None:
                stream.write(qpos)
            frames += 1
            if args.rate_limit:
                next_frame = max(next_frame + 1 / args.motion_fps, time.monotonic())
                time.sleep(max(0, next_frame - time.monotonic()))
            now = time.monotonic()
            if now - last_log >= 1:
                print(f"[live] frames={frames}/s udp_seq={receiver.seq} dropped={receiver.dropped}", flush=True)
                frames = 0
                last_log = now
    finally:
        receiver.close()
        if viewer is not None:
            viewer.close()
        if renderer is not None:
            renderer.close()
        if writer is not None:
            writer.close()
        if stream is not None:
            stream.save(retargeter.model, args.motion_fps)


if __name__ == "__main__":
    main()
