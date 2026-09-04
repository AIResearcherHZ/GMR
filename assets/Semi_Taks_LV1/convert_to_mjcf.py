#!/usr/bin/env python3
from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np

SRC, OUT = "Semi_Taks_LV1.urdf", "Semi_Taks_LV1.xml"
K_BEVEL = 0.629630

root = ET.parse(SRC).getroot()
JOINT = {j.find("child").get("link"): j for j in root.findall("joint")}
PARENT = {
    j.find("child").get("link"): j.find("parent").get("link")
    for j in root.findall("joint")
}
LINK = {l.get("name"): l for l in root.findall("link")}

TORSO, SHO, WRI = "torso_motor", "shoulder_motor", "wrist_motor"


def vec(s):
    return np.array([float(x) for x in s.split()], dtype=float)


def rel(link):
    j = JOINT.get(link)
    if j is None:
        return np.zeros(3)
    o = j.find("origin")
    return vec(o.get("xyz")) if o is not None and o.get("xyz") else np.zeros(3)


def world(link):
    if link not in PARENT:
        return np.zeros(3)
    return world(PARENT[link]) + rel(link)


def jname(link):
    return JOINT[link].get("name")


def jaxis(link):
    ax = JOINT[link].find("axis")
    return ax.get("xyz") if ax is not None else "0 0 1"


def jlimit(link):
    lim = JOINT[link].find("limit")
    return f"{lim.get('lower')} {lim.get('upper')}"


def fmt(a, p=9):
    a = np.atleast_1d(a)
    a = np.where(np.abs(a) < 1e-12, 0.0, a)
    return " ".join(f"{float(x):.{p}g}" for x in a)


N = []


def node(name, mjcf):
    N.append({"name": name, "mjcf": mjcf})


def M(cls, mtype, rng="urdf"):
    return {"cls": cls, "type": mtype, "range": rng}


node("waist_yaw_link", M(TORSO, "hinge", "urdf"))
for s in ("right", "left"):
    node(f"waist_{s}_link_driven_link", M(TORSO, "ball", None))
node("waist_roll_link", M(TORSO, "hinge", "-0.4363 0.4363"))
node("waist_pitch_link", M(TORSO, "hinge", "-1.0 1.0"))
for s in ("right", "left"):
    node(f"waist_{s}_motor_link", M(TORSO, "hinge", "-0.6 0.6"))


def arm(side):
    node(f"{side}_shoulder_pitch_link", M(SHO, "hinge", "urdf"))
    node(f"{side}_shoulder_roll_link", M(SHO, "hinge", "urdf"))
    node(f"{side}_shoulder_yaw_link", M(SHO, "hinge", "urdf"))
    node(f"{side}_elbow_link", M(SHO, "hinge", "urdf"))
    node(f"{side}_wrist_roll_link", M(WRI, "hinge", "urdf"))
    for lk in ("long", "short"):
        node(f"{side}_arm_{lk}_link_motor_link", M(WRI, "hinge", "-0.9667 0.9667"))
        node(f"{side}_arm_{lk}_link_active_link", M(WRI, "hinge", None))
    node(f"{side}_wrist_yaw_link", M(WRI, "hinge", "urdf"))
    node(f"{side}_wrist_pitch_link", M(WRI, "hinge", "urdf"))
    for lk in ("long", "short"):
        node(f"{side}_arm_{lk}_link_bevel_gear_link", M(WRI, "hinge", None))
        node(f"{side}_arm_{lk}_link_driven_link", M(WRI, None, None))


arm("right")
arm("left")
SPEC = {n["name"]: n["mjcf"] for n in N}

ACT = []
CONNECT = []
GEAR = []
MESH = {}


def add_body(parent_el, link):
    b = ET.SubElement(parent_el, "body", name=link, pos=fmt(rel(link)))
    ine = LINK[link].find("inertial")
    o = ine.find("origin")
    com = vec(o.get("xyz")) if o is not None and o.get("xyz") else np.zeros(3)
    I = ine.find("inertia")
    fi = [I.get(k) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")]
    ET.SubElement(
        b,
        "inertial",
        pos=fmt(com),
        mass=f"{float(ine.find('mass').get('value')):.7g}",
        fullinertia=" ".join(f"{float(x):.7g}" for x in fi),
    )
    m = SPEC.get(link)
    if m and m["type"] is not None:
        ja = {"name": jname(link), "class": m["cls"], "type": m["type"]}
        if m["type"] == "hinge":
            ja["axis"] = jaxis(link)
        if m["range"] is None:
            ja["limited"] = "false"
        else:
            ja["range"] = jlimit(link) if m["range"] == "urdf" else m["range"]
        ET.SubElement(b, "joint", **ja)
    g = LINK[link].find("visual/geometry/mesh")
    if g is not None:
        MESH[link] = g.get("filename").split("/")[-1]
        ET.SubElement(b, "geom", type="mesh", mesh=link)
    return b


CHILD = {}
for n in N:
    CHILD.setdefault(PARENT[n["name"]], []).append(n["name"])


def recurse(parent_el, link):
    for c in CHILD.get(link, []):
        recurse(add_body(parent_el, c), c)


mj = ET.Element("mujoco", model="Semi_Taks_LV1")
ET.SubElement(
    mj,
    "compiler",
    angle="radian",
    meshdir="meshes/",
    autolimits="true",
    balanceinertia="true",
)
opt = ET.SubElement(
    mj,
    "option",
    timestep="0.002",
    iterations="150",
    solver="Newton",
    tolerance="1e-10",
    integrator="implicit",
    gravity="0 0 -9.81",
)
ET.SubElement(opt, "flag", contact="disable")
dft = ET.SubElement(mj, "default")
ET.SubElement(dft, "joint", limited="true")
ET.SubElement(
    dft, "geom", contype="0", conaffinity="0", condim="1", group="1", density="0"
)
ET.SubElement(dft, "equality", solref="0.01 1", solimp="0.95 0.99 0.001 0.5 2")
for cls, av, fl in (
    ("torso_motor", "1.492992e-01", "0.1"),
    ("shoulder_motor", "2.848000e-02", "0.1"),
    ("wrist_motor", "1.680000e-03", "0.01"),
):
    sub = ET.SubElement(dft, "default", **{"class": cls})
    ET.SubElement(sub, "joint", armature=av, damping="0.01", frictionloss=fl)

asset = ET.SubElement(mj, "asset")
wbody = ET.SubElement(mj, "worldbody")
base = ET.SubElement(wbody, "body", name="base_link", pos="0 0 0")
bine = LINK["base_link"].find("inertial")
bo = bine.find("origin")
bcom = vec(bo.get("xyz")) if bo is not None and bo.get("xyz") else np.zeros(3)
bI = bine.find("inertia")
bfi = [bI.get(k) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")]
ET.SubElement(
    base,
    "inertial",
    pos=fmt(bcom),
    mass=f"{float(bine.find('mass').get('value')):.7g}",
    fullinertia=" ".join(f"{float(x):.7g}" for x in bfi),
)
MESH["base_link"] = (
    LINK["base_link"].find("visual/geometry/mesh").get("filename").split("/")[-1]
)
ET.SubElement(base, "geom", type="mesh", mesh="base_link")
recurse(base, "base_link")

for name in sorted(MESH):
    ET.SubElement(asset, "mesh", name=name, file=MESH[name])

for s in ("right", "left"):
    dn = f"waist_{s}_link_driven_link"
    mo = f"waist_{s}_motor_link"
    rod = abs(float(LINK[dn].find("inertial/origin").get("xyz").split()[2])) * 2.0
    tip = world(dn) + np.array([0.0, 0.0, rod])
    CONNECT.append((f"waist_{s}_parallel_loop", mo, dn, tip - world(mo)))
for s in ("right", "left"):
    for lk, sgn in (("long", +1.0), ("short", -1.0)):
        ac = f"{s}_arm_{lk}_link_active_link"
        dv = f"{s}_arm_{lk}_link_driven_link"
        CONNECT.append((f"{s}_arm_{lk}_parallel_loop", ac, dv, world(dv) - world(ac)))
        GEAR.append(
            (
                f"{s}_arm_{lk}_bevel_mesh",
                f"{s}_arm_{lk}_link_bevel_gear_joint",
                f"{s}_wrist_pitch_joint",
                f"0 {sgn * K_BEVEL:.6f} 0 0 0",
            )
        )

eq = ET.SubElement(mj, "equality")
for nm, b1, b2, a in CONNECT:
    ET.SubElement(eq, "connect", name=nm, body1=b1, body2=b2, anchor=fmt(a))
for nm, j1, j2, pc in GEAR:
    ET.SubElement(eq, "joint", name=nm, joint1=j1, joint2=j2, polycoef=pc)

FORCE = {
    "waist_yaw": "97",
    "waist_motor": "97",
    "wrist": "7",
    "wrist_roll": "7",
    "arm": "27",
}
POS = {
    "waist_yaw": ("589.41", "37.52"),
    "waist_motor": ("1178.82", "75.04"),
    "wrist": ("13.264748", "0.84446"),
    "wrist_roll": ("6.632374", "0.422230"),
    "arm": ("112.434533", "7.157805"),
}
ACT.append(("waist_yaw_joint", "-2.618 2.618", "waist_yaw"))
for s in ("right", "left"):
    ACT.append((f"waist_{s}_motor_joint", "-0.6 0.6", "waist_motor"))
for s in ("right", "left"):
    ACT.append((f"{s}_shoulder_pitch_joint", jlimit(f"{s}_shoulder_pitch_link"), "arm"))
    ACT.append((f"{s}_shoulder_roll_joint", jlimit(f"{s}_shoulder_roll_link"), "arm"))
    ACT.append((f"{s}_shoulder_yaw_joint", jlimit(f"{s}_shoulder_yaw_link"), "arm"))
    ACT.append((f"{s}_elbow_joint", jlimit(f"{s}_elbow_link"), "arm"))
    ACT.append((f"{s}_wrist_roll_joint", jlimit(f"{s}_wrist_roll_link"), "wrist_roll"))
    ACT.append((f"{s}_arm_long_link_motor_joint", "-0.9667 0.9667", "wrist"))
    ACT.append((f"{s}_arm_short_link_motor_joint", "-0.9667 0.9667", "wrist"))

act = ET.SubElement(mj, "actuator")
for jn, cr, grp in ACT:
    kp, kv = POS[grp]
    fr = FORCE[grp]
    ET.SubElement(
        act,
        "position",
        name=jn,
        joint=jn,
        ctrlrange=cr,
        kp=kp,
        kv=kv,
        forcerange=f"-{fr} {fr}",
    )

ET.indent(mj, space="  ")
ET.ElementTree(mj).write(OUT, encoding="unicode", xml_declaration=True)
print(
    f"wrote {OUT}: bodies={sum(1 for _ in mj.iter('body'))} "
    f"connects={len(CONNECT)} gears={len(GEAR)} actuators={len(ACT)}"
)
