import json
import os

import numpy as np

LINKS = json.load(
    open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "links.json"))
)
OUT = "/home/xhz/taks-controller-web/taks_level1/assets/Semi_Taks_LV1"

O_CAD = np.array([0.0, -4.8, -383.3])
ELBOW_Y, ELBOW_Z = -8.0, -208.92
R_ELBOW = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)
M_ROB = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float)


def to_robot(p, repose=False, mirror=False):
    p = np.array(p, float)
    if mirror:
        p = p * np.array([-1, 1, 1])
    if repose:
        a = np.array([p[0], ELBOW_Y, ELBOW_Z])
        p = a + R_ELBOW @ (p - a)
    return (M_ROB @ (p - O_CAD)) / 1000.0


def O(link):
    return np.array(LINKS[link]["origin"])


def fmt(v, nd=8):
    out = []
    for x in np.atleast_1d(v):
        s = f"{x:.{nd}g}"
        if s == "-0":
            s = "0"
        out.append(s)
    return " ".join(out)


BALL_WR = to_robot([-10.5, -38.9, -232.0])
BALL_WL = to_robot([10.5, -38.9, -232.0])
PIN_LONG = to_robot([-173.9, -14.2, -433.35], repose=True)
PIN_SHORT = to_robot([-149.9, 48.2, -433.35], repose=True)
PIN_LONG_L = to_robot([-173.9, -14.2, -433.35], repose=True, mirror=True)
PIN_SHORT_L = to_robot([-149.9, 48.2, -433.35], repose=True, mirror=True)
PIN_HF = to_robot([10.95, -29.2, 131.1])
PIN_HR = to_robot([-10.95, 29.2, 131.1])
TCP_R = to_robot([-180.58, 18.58, -504.03], repose=True)
TCP_L = to_robot([-180.58, 18.58, -504.03], repose=True, mirror=True)


def mirror_left(pt_cad, repose=False):
    return to_robot(pt_cad, repose=repose, mirror=True)


BODY_TREE = {
    "base_link": (None, None),
    "waist_yaw_link": (
        "base_link",
        {"axis": "0 0 1", "range": "-2.618 2.618", "cls": "torso_motor"},
    ),
    "waist_right_link_driven_link": (
        "waist_yaw_link",
        {"ball": True, "cls": "torso_motor"},
    ),
    "waist_left_link_driven_link": (
        "waist_yaw_link",
        {"ball": True, "cls": "torso_motor"},
    ),
    "waist_roll_link": (
        "waist_yaw_link",
        {"axis": "1 0 0", "range": "-0.7854 0.7854", "cls": "torso_motor"},
    ),
    "waist_pitch_link": (
        "waist_roll_link",
        {"axis": "0 1 0", "range": "-0.7854 0.7854", "cls": "torso_motor"},
    ),
    "waist_right_motor_link": (
        "waist_pitch_link",
        {"axis": "1 0 0", "range": "-1.0908 1.0908", "cls": "torso_motor"},
    ),
    "waist_left_motor_link": (
        "waist_pitch_link",
        {"axis": "1 0 0", "range": "-1.0908 1.0908", "cls": "torso_motor"},
    ),
}

for side, sp_axis, roll_range in [
    ("right", "0 0.94832 -0.3173", "-2.2515 1.5882"),
    ("left", "0 0.94832 0.3173", "-1.5882 2.2515"),
]:
    s = side
    BODY_TREE.update(
        {
            f"{s}_shoulder_pitch_link": (
                "waist_pitch_link",
                {"axis": sp_axis, "range": "-3.1415 1.0472", "cls": "shoulder_motor"},
            ),
            f"{s}_shoulder_roll_link": (
                f"{s}_shoulder_pitch_link",
                {"axis": "1 0 0", "range": roll_range, "cls": "shoulder_motor"},
            ),
            f"{s}_shoulder_yaw_link": (
                f"{s}_shoulder_roll_link",
                {"axis": "0 0 1", "range": "-2.618 2.618", "cls": "shoulder_motor"},
            ),
            f"{s}_elbow_link": (
                f"{s}_shoulder_yaw_link",
                {"axis": "0 1 0", "range": "-1.0472 1.35", "cls": "shoulder_motor"},
            ),
            f"{s}_wrist_roll_link": (
                f"{s}_elbow_link",
                {"axis": "1 0 0", "range": "-2.67 2.67", "cls": "wrist_motor"},
            ),
            f"{s}_arm_long_link_motor_link": (
                f"{s}_wrist_roll_link",
                {"axis": "0 0 1", "range": "-0.9885 0.9885", "cls": "wrist_motor"},
            ),
            f"{s}_arm_long_link_active_link": (
                f"{s}_arm_long_link_motor_link",
                {"axis": "0 0 1", "free": True, "cls": "wrist_motor"},
            ),
            f"{s}_arm_short_link_motor_link": (
                f"{s}_wrist_roll_link",
                {"axis": "0 0 1", "range": "-0.9885 0.9885", "cls": "wrist_motor"},
            ),
            f"{s}_arm_short_link_active_link": (
                f"{s}_arm_short_link_motor_link",
                {"axis": "0 0 1", "free": True, "cls": "wrist_motor"},
            ),
            f"{s}_wrist_yaw_link": (
                f"{s}_wrist_roll_link",
                {"axis": "0 0 1", "free": True, "cls": "wrist_motor"},
            ),
            f"{s}_wrist_pitch_link": (
                f"{s}_wrist_yaw_link",
                {"axis": "0 1 0", "range": "-1.57 1.57", "cls": "wrist_motor"},
            ),
            f"{s}_arm_long_link_bevel_gear_link": (
                f"{s}_wrist_yaw_link",
                {"axis": "0 0 1", "free": True, "cls": "wrist_motor"},
            ),
            f"{s}_arm_short_link_bevel_gear_link": (
                f"{s}_wrist_yaw_link",
                {"axis": "0 0 1", "free": True, "cls": "wrist_motor"},
            ),
        }
    )

BODY_TREE.update(
    {
        "head_front_link_motor_link": (
            "waist_pitch_link",
            {"axis": "1 0 0", "range": "-0.9885 0.9885", "cls": "wrist_motor"},
        ),
        "head_front_link_active_link": (
            "head_front_link_motor_link",
            {"axis": "1 0 0", "free": True, "cls": "wrist_motor"},
        ),
        "head_rear_link_motor_link": (
            "waist_pitch_link",
            {"axis": "1 0 0", "range": "-0.9885 0.9885", "cls": "wrist_motor"},
        ),
        "head_rear_link_active_link": (
            "head_rear_link_motor_link",
            {"axis": "1 0 0", "free": True, "cls": "wrist_motor"},
        ),
        "head_roll_link": (
            "waist_pitch_link",
            {"axis": "1 0 0", "free": True, "cls": "wrist_motor"},
        ),
        "head_pitch_link": (
            "head_roll_link",
            {"axis": "0 1 0", "range": "-1.57 1.57", "cls": "wrist_motor"},
        ),
        "head_yaw_link": (
            "head_pitch_link",
            {"axis": "0 0 1", "range": "-2.618 2.618", "cls": "wrist_motor"},
        ),
        "head_front_link_bevel_gear_link": (
            "head_roll_link",
            {"axis": "1 0 0", "free": True, "cls": "wrist_motor"},
        ),
        "head_rear_link_bevel_gear_link": (
            "head_roll_link",
            {"axis": "1 0 0", "free": True, "cls": "wrist_motor"},
        ),
    }
)

JOINT_NAME = {
    link: link.replace("_link", "_joint") if not link.endswith("_link") else None
    for link in BODY_TREE
}
for link in BODY_TREE:
    if link == "base_link":
        continue
    if link.endswith("_motor_link"):
        JOINT_NAME[link] = link[: -len("_link")] + "_joint"
    else:
        JOINT_NAME[link] = link[: -len("_link")] + "_joint"

CHILDREN = {}
for link, (parent, _) in BODY_TREE.items():
    if parent:
        CHILDREN.setdefault(parent, []).append(link)

ORDER = [
    "waist_yaw_link",
    "waist_right_link_driven_link",
    "waist_left_link_driven_link",
    "waist_roll_link",
    "waist_pitch_link",
    "waist_right_motor_link",
    "waist_left_motor_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_arm_long_link_motor_link",
    "right_arm_long_link_active_link",
    "right_arm_short_link_motor_link",
    "right_arm_short_link_active_link",
    "right_wrist_yaw_link",
    "right_wrist_pitch_link",
    "right_arm_long_link_bevel_gear_link",
    "right_arm_short_link_bevel_gear_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_arm_long_link_motor_link",
    "left_arm_long_link_active_link",
    "left_arm_short_link_motor_link",
    "left_arm_short_link_active_link",
    "left_wrist_yaw_link",
    "left_wrist_pitch_link",
    "left_arm_long_link_bevel_gear_link",
    "left_arm_short_link_bevel_gear_link",
    "head_front_link_motor_link",
    "head_front_link_active_link",
    "head_rear_link_motor_link",
    "head_rear_link_active_link",
    "head_roll_link",
    "head_pitch_link",
    "head_yaw_link",
    "head_front_link_bevel_gear_link",
    "head_rear_link_bevel_gear_link",
]
for link in CHILDREN:
    CHILDREN[link].sort(key=lambda l: ORDER.index(l))

SITES = {
    "right_elbow_link": [("right_elbow", "0 0 0", 'size="0.006"')],
    "left_elbow_link": [("left_elbow", "0 0 0", 'size="0.006"')],
    "right_wrist_pitch_link": [
        (
            "right_tcp",
            fmt(TCP_R - O("right_wrist_pitch_link"), 6),
            'size="0.004" rgba="1 0 1 1"',
        )
    ],
    "left_wrist_pitch_link": [
        (
            "left_tcp",
            fmt(TCP_L - O("left_wrist_pitch_link"), 6),
            'size="0.004" rgba="1 0 1 1"',
        )
    ],
}

CAMERAS = {
    "head_yaw_link": [
        ("head_left_camera", "0.08538 0.02451 0.05648"),
        ("head_right_camera", "0.08538 -0.02347 0.05648"),
    ],
}


def inertial_xml(link, indent):
    d = LINKS[link]
    I = np.array(d["inertia_com"])
    com = np.array(d["com_link"])
    full = [I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]]
    return (
        f'{indent}<inertial pos="{fmt(com, 7)}" mass="{d["mass"]:.6g}" '
        f'fullinertia="{fmt(full, 7)}" />'
    )


def body_xml(link, parent, lines, indent):
    pos = O(link) - (O(parent) if parent != "world" else 0)
    jn = JOINT_NAME[link]
    spec = BODY_TREE[link][1]
    lines.append(f'{indent}<body name="{link}" pos="{fmt(pos, 7)}">')
    lines.append(inertial_xml(link, indent + "  "))
    if spec is not None:
        cls = spec["cls"]
        if spec.get("ball"):
            lines.append(
                f'{indent}  <joint name="{jn}" class="{cls}" type="ball" limited="false" />'
            )
        elif spec.get("free"):
            lines.append(
                f'{indent}  <joint name="{jn}" class="{cls}" type="hinge" axis="{spec["axis"]}" limited="false" />'
            )
        else:
            lines.append(
                f'{indent}  <joint name="{jn}" class="{cls}" type="hinge" axis="{spec["axis"]}" range="{spec["range"]}" />'
            )
    for name, pos_s, extra in SITES.get(link, []):
        lines.append(f'{indent}  <site name="{name}" pos="{pos_s}" {extra} />')
    for name, pos_s in CAMERAS.get(link, []):
        lines.append(
            f'{indent}  <camera name="{name}" pos="{pos_s}" xyaxes="0 -1 0 0.17365 0 0.98481" fovy="45" resolution="1920 1080" />'
        )
    lines.append(f'{indent}  <geom type="mesh" mesh="{link}" />')
    lines.append(
        f'{indent}  <geom type="mesh" mesh="{link}" contype="1" conaffinity="2" group="3" />'
    )
    if link == "waist_pitch_link":
        lines.append(
            f'{indent}  <geom name="waist_pitch_collision" type="ellipsoid" pos="0.005 0 0.214" size="0.07 0.105 0.135" contype="1" conaffinity="2" group="3" rgba="1 0.4 0 0.3" />'
        )
    for ch in CHILDREN.get(link, []):
        body_xml(ch, link, lines, indent + "  ")
    lines.append(f"{indent}</body>")


lines = []
lines.append("<?xml version='1.0' encoding='utf-8'?>")
lines.append('<mujoco model="Semi_Taks_LV1">')
lines.append(
    '  <compiler angle="radian" meshdir="meshes/" autolimits="true" balanceinertia="true" />'
)
lines.append(
    '  <option timestep="0.002" iterations="150" solver="Newton" tolerance="1e-10" integrator="implicit" gravity="0 0 -9.81" />'
)
lines.append("  <default>")
lines.append('    <joint limited="true" />')
lines.append(
    '    <geom contype="0" conaffinity="0" condim="1" group="1" density="0" friction="1 0.005 0.0001" />'
)
lines.append('    <equality solref="0.01 1" solimp="0.95 0.99 0.001 0.5 2" />')
lines.append('    <default class="torso_motor">')
lines.append(
    '      <joint armature="1.492992e-01" damping="0.01" frictionloss="0.1" />'
)
lines.append("    </default>")
lines.append('    <default class="shoulder_motor">')
lines.append(
    '      <joint armature="2.848000e-02" damping="0.01" frictionloss="0.1" />'
)
lines.append("    </default>")
lines.append('    <default class="wrist_motor">')
lines.append(
    '      <joint armature="1.680000e-03" damping="0.01" frictionloss="0.01" />'
)
lines.append("    </default>")
lines.append("  </default>")
lines.append("  <asset>")
for link in sorted(LINKS):
    lines.append(f'    <mesh name="{link}" file="{link}.STL" />')
lines.append("  </asset>")
lines.append("  <worldbody>")
base_lines = []
lines.append('    <body name="base_link" pos="0 0 0">')
lines.append(inertial_xml("base_link", "      "))
lines.append('      <geom type="mesh" mesh="base_link" />')
lines.append(
    '      <geom type="mesh" mesh="base_link" contype="1" conaffinity="2" group="3" />'
)
sub = []
body_xml("waist_yaw_link", "base_link", sub, "      ")
lines += sub
lines.append("    </body>")
lines.append("  </worldbody>")

eq = []
eq.append(
    f'    <connect name="waist_right_parallel_loop" body1="waist_right_motor_link" body2="waist_right_link_driven_link" anchor="{fmt(BALL_WR - O("waist_right_motor_link"), 7)}" />'
)
eq.append(
    f'    <connect name="waist_left_parallel_loop" body1="waist_left_motor_link" body2="waist_left_link_driven_link" anchor="{fmt(BALL_WL - O("waist_left_motor_link"), 7)}" />'
)
eq.append(
    f'    <connect name="right_arm_long_parallel_loop" body1="right_arm_long_link_active_link" body2="right_arm_long_link_bevel_gear_link" anchor="{fmt(PIN_LONG - O("right_arm_long_link_active_link"), 7)}" />'
)
eq.append(
    f'    <connect name="right_arm_short_parallel_loop" body1="right_arm_short_link_active_link" body2="right_arm_short_link_bevel_gear_link" anchor="{fmt(PIN_SHORT - O("right_arm_short_link_active_link"), 7)}" />'
)
eq.append(
    f'    <connect name="left_arm_long_parallel_loop" body1="left_arm_long_link_active_link" body2="left_arm_long_link_bevel_gear_link" anchor="{fmt(PIN_LONG_L - O("left_arm_long_link_active_link"), 7)}" />'
)
eq.append(
    f'    <connect name="left_arm_short_parallel_loop" body1="left_arm_short_link_active_link" body2="left_arm_short_link_bevel_gear_link" anchor="{fmt(PIN_SHORT_L - O("left_arm_short_link_active_link"), 7)}" />'
)
eq.append(
    f'    <connect name="head_front_parallel_loop" body1="head_front_link_active_link" body2="head_front_link_bevel_gear_link" anchor="{fmt(PIN_HF - O("head_front_link_active_link"), 7)}" />'
)
eq.append(
    f'    <connect name="head_rear_parallel_loop" body1="head_rear_link_active_link" body2="head_rear_link_bevel_gear_link" anchor="{fmt(PIN_HR - O("head_rear_link_active_link"), 7)}" />'
)
eq.append(
    '    <joint name="right_arm_long_bevel_mesh" joint1="right_arm_long_link_bevel_gear_joint" joint2="right_wrist_pitch_joint" polycoef="0 0.629630 0 0 0" />'
)
eq.append(
    '    <joint name="right_arm_short_bevel_mesh" joint1="right_arm_short_link_bevel_gear_joint" joint2="right_wrist_pitch_joint" polycoef="0 -0.629630 0 0 0" />'
)
eq.append(
    '    <joint name="left_arm_long_bevel_mesh" joint1="left_arm_long_link_bevel_gear_joint" joint2="left_wrist_pitch_joint" polycoef="0 -0.629630 0 0 0" />'
)
eq.append(
    '    <joint name="left_arm_short_bevel_mesh" joint1="left_arm_short_link_bevel_gear_joint" joint2="left_wrist_pitch_joint" polycoef="0 0.629630 0 0 0" />'
)
eq.append(
    '    <joint name="head_front_bevel_mesh" joint1="head_front_link_bevel_gear_joint" joint2="head_pitch_joint" polycoef="0 -0.629630 0 0 0" />'
)
eq.append(
    '    <joint name="head_rear_bevel_mesh" joint1="head_rear_link_bevel_gear_joint" joint2="head_pitch_joint" polycoef="0 0.629630 0 0 0" />'
)
lines.append("  <equality>")
lines += eq
lines.append("  </equality>")

ACT = [
    ("waist_yaw_joint", "-2.618 2.618", "589.41", "37.52", "-97 97"),
    ("waist_right_motor_joint", "-1.0908 1.0908", "1178.82", "75.04", "-97 97"),
    ("waist_left_motor_joint", "-1.0908 1.0908", "1178.82", "75.04", "-97 97"),
    (
        "right_shoulder_pitch_joint",
        "-3.1415 1.0472",
        "112.434533",
        "7.157805",
        "-27 27",
    ),
    ("right_shoulder_roll_joint", "-2.2515 1.5882", "112.434533", "7.157805", "-27 27"),
    ("right_shoulder_yaw_joint", "-2.618 2.618", "112.434533", "7.157805", "-27 27"),
    ("right_elbow_joint", "-0.78 1.35", "112.434533", "7.157805", "-27 27"),
    ("right_wrist_roll_joint", "-2.67 2.67", "6.632374", "0.422230", "-7 7"),
    (
        "right_arm_long_link_motor_joint",
        "-0.9885 0.9885",
        "13.264748",
        "0.84446",
        "-7 7",
    ),
    (
        "right_arm_short_link_motor_joint",
        "-0.9885 0.9885",
        "13.264748",
        "0.84446",
        "-7 7",
    ),
    ("left_shoulder_pitch_joint", "-3.1415 1.0472", "112.434533", "7.157805", "-27 27"),
    ("left_shoulder_roll_joint", "-1.5882 2.2515", "112.434533", "7.157805", "-27 27"),
    ("left_shoulder_yaw_joint", "-2.618 2.618", "112.434533", "7.157805", "-27 27"),
    ("left_elbow_joint", "-0.78 1.35", "112.434533", "7.157805", "-27 27"),
    ("left_wrist_roll_joint", "-2.67 2.67", "6.632374", "0.422230", "-7 7"),
    (
        "left_arm_long_link_motor_joint",
        "-0.9885 0.9885",
        "13.264748",
        "0.84446",
        "-7 7",
    ),
    (
        "left_arm_short_link_motor_joint",
        "-0.9885 0.9885",
        "13.264748",
        "0.84446",
        "-7 7",
    ),
    ("head_front_link_motor_joint", "-0.9885 0.9885", "13.264748", "0.84446", "-7 7"),
    ("head_rear_link_motor_joint", "-0.9885 0.9885", "13.264748", "0.84446", "-7 7"),
    ("head_yaw_joint", "-2.618 2.618", "6.632374", "0.422230", "-7 7"),
]
lines.append("  <actuator>")
for name, cr, kp, kv, fr in ACT:
    lines.append(
        f'    <position name="{name}" joint="{name}" ctrlrange="{cr}" kp="{kp}" kv="{kv}" forcerange="{fr}" />'
    )
lines.append("  </actuator>")
lines.append("</mujoco>")

with open(f"{OUT}/Semi_Taks_LV1.xml", "w") as f:
    f.write("\n".join(lines))

URDF_LIMITS = {
    "torso_motor": ("97", "4.19"),
    "shoulder_motor": ("27", "3.77"),
    "wrist_motor": ("7", "12.57"),
}
ARMATURE = {
    "torso_motor": "1.492992e-01",
    "shoulder_motor": "2.848000e-02",
    "wrist_motor": "1.680000e-03",
}

u = []
u.append("<?xml version='1.0' encoding='utf-8'?>")
u.append('<robot name="Semi_Taks_LV1">')
u.append("  <mujoco>")
u.append(
    '    <compiler meshdir="." balanceinertia="true" discardvisual="false" strippath="false"/>'
)
u.append("  </mujoco>")
u.append('  <link name="world" />')
u.append('  <joint name="world_to_base" type="fixed">')
u.append('    <origin xyz="0 0 0" rpy="0 0 0" />')
u.append('    <parent link="world" />')
u.append('    <child link="base_link" />')
u.append("  </joint>")


def urdf_link(link):
    d = LINKS[link]
    I = np.array(d["inertia_com"])
    com = np.array(d["com_link"])
    u.append(f'  <link name="{link}">')
    u.append("    <inertial>")
    u.append(f'      <origin xyz="{fmt(com, 7)}" rpy="0 0 0" />')
    u.append(f'      <mass value="{d["mass"]:.6g}" />')
    u.append(
        f'      <inertia ixx="{I[0, 0]:.6g}" ixy="{I[0, 1]:.6g}" ixz="{I[0, 2]:.6g}" iyy="{I[1, 1]:.6g}" iyz="{I[1, 2]:.6g}" izz="{I[2, 2]:.6g}" />'
    )
    u.append("    </inertial>")
    u.append("    <visual>")
    u.append('      <origin xyz="0 0 0" rpy="0 0 0" />')
    u.append("      <geometry>")
    u.append(f'        <mesh filename="meshes/{link}.STL" />')
    u.append("      </geometry>")
    u.append("    </visual>")
    u.append("    <collision>")
    u.append('      <origin xyz="0 0 0" rpy="0 0 0" />')
    u.append("      <geometry>")
    u.append(f'        <mesh filename="meshes/{link}.STL" />')
    u.append("      </geometry>")
    u.append("    </collision>")
    u.append("  </link>")


def urdf_joint(link):
    parent, spec = BODY_TREE[link]
    jn = JOINT_NAME[link]
    pos = O(link) - O(parent)
    cls = spec["cls"]
    axis = "1 0 0" if spec.get("ball") else spec["axis"]
    if spec.get("ball") or spec.get("free"):
        lower, upper, eff, vel = "-3.1415", "3.1415", "0", "0"
    else:
        lo, hi = spec["range"].split()
        lower, upper = lo, hi
        eff, vel = URDF_LIMITS[cls]
    if link in ("waist_roll_link", "waist_pitch_link", "waist_yaw_link"):
        eff, vel = URDF_LIMITS["torso_motor"]
    if link == "head_yaw_link":
        eff, vel = URDF_LIMITS["wrist_motor"]
    u.append(f'  <joint name="{jn}" type="revolute">')
    u.append(f'    <origin xyz="{fmt(pos, 7)}" rpy="0 0 0" />')
    u.append(f'    <parent link="{parent}" />')
    u.append(f'    <child link="{link}" />')
    u.append(f'    <axis xyz="{axis}" />')
    u.append(
        f'    <limit lower="{lower}" upper="{upper}" effort="{eff}" velocity="{vel}" />'
    )
    u.append(f'    <dynamics armature="{ARMATURE[cls]}" />')
    u.append("  </joint>")


urdf_link("base_link")


def urdf_walk(link):
    urdf_joint(link)
    urdf_link(link)
    for ch in CHILDREN.get(link, []):
        urdf_walk(ch)


urdf_walk("waist_yaw_link")
for parent, cams in CAMERAS.items():
    for name, pos_s in cams:
        u.append(f'  <joint name="{name}_joint" type="fixed">')
        u.append(f'    <origin xyz="{pos_s}" rpy="0 0.17453 0" />')
        u.append(f'    <parent link="{parent}" />')
        u.append(f'    <child link="{name}_link" />')
        u.append("  </joint>")
        u.append(f'  <link name="{name}_link" />')
u.append("</robot>")

with open(f"{OUT}/Semi_Taks_LV1.urdf", "w") as f:
    f.write("\n".join(u))

print("joint origins (robot frame):")
for link in ORDER:
    parent = BODY_TREE[link][0]
    print(f"  {JOINT_NAME[link]:44s} pos={fmt(O(link) - O(parent), 6)}")
print(
    "anchors:",
    fmt(BALL_WR - O("waist_right_motor_link"), 6),
    "|",
    fmt(PIN_LONG - O("right_arm_long_link_active_link"), 6),
    "|",
    fmt(PIN_HF - O("head_front_link_active_link"), 6),
)
print("written:", OUT)
