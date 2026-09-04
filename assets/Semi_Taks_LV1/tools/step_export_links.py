import json
import math
import os

import numpy as np
import trimesh
from OCP.Bnd import Bnd_Box
from OCP.BRep import BRep_Builder, BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCP.BRepGProp import BRepGProp, BRepGProp_Face
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCP.BRepTools import BRepTools
from OCP.GeomAbs import GeomAbs_Cone, GeomAbs_Cylinder, GeomAbs_Plane, GeomAbs_Sphere
from OCP.gp import gp_Ax1, gp_Ax2, gp_Dir, gp_Pnt, gp_Trsf, gp_Vec
from OCP.GProp import GProp_GProps
from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.TCollection import TCollection_ExtendedString
from OCP.TDataStd import TDataStd_Name
from OCP.TDF import TDF_Label, TDF_LabelSequence
from OCP.TDocStd import TDocStd_Document
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_REVERSED, TopAbs_WIRE
from OCP.TopExp import TopExp, TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS
from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape
from OCP.XCAFDoc import XCAFDoc_DocumentTool

STEP = (
    "/home/xhz/taks-controller-web/taks_level1/assets/Semi_Taks_LV1/骨架参考 v91.step"
)
OUT = "/home/xhz/taks-controller-web/taks_level1/assets/Semi_Taks_LV1"
MESH_DIR = os.path.join(OUT, "meshes")
os.makedirs(MESH_DIR, exist_ok=True)

O_CAD = np.array([0.0, -4.8, -383.3])
ELBOW_Y, ELBOW_Z = -8.0, -208.92

props_test = GProp_GProps()
BRepGProp.VolumeProperties_s(
    BRepPrimAPI_MakeBox(gp_Pnt(100, 50, 20), 10.0, 20.0, 30.0).Shape(), props_test
)
mI = props_test.MatrixOfInertia()
Ixx = mI.Value(1, 1)
expected_com = (20.0**2 + 30.0**2) / 12.0 * (10 * 20 * 30)
assert abs(Ixx - expected_com) / expected_com < 1e-6, (
    f"OCC inertia not about COM: {Ixx} vs {expected_com}"
)

doc = TDocStd_Document(TCollection_ExtendedString("doc"))
reader = STEPCAFControl_Reader()
reader.SetNameMode(True)
assert reader.ReadFile(STEP) == IFSelect_RetDone
reader.Transfer(doc)
st = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())


def get_name(label):
    n = TDataStd_Name()
    if label.FindAttribute(TDataStd_Name.GetID_s(), n):
        try:
            return n.Get().ToExtString()
        except Exception:
            s = n.Get()
            return "".join(chr(s.Value(i)) for i in range(1, s.Length() + 1))
    return "?"


leaves = []


def walk(label, loc, path):
    name = get_name(label)
    acc = loc.Multiplied(st.GetLocation_s(label))
    target = label
    if st.IsReference_s(label):
        ref = TDF_Label()
        st.GetReferredShape_s(label, ref)
        target = ref
    disp = name if name != "?" else get_name(target)
    p = path + "/" + disp
    if st.IsAssembly_s(target):
        comps = TDF_LabelSequence()
        st.GetComponents_s(target, comps)
        for i in range(1, comps.Length() + 1):
            walk(comps.Value(i), acc, p)
    else:
        shape = st.GetShape_s(target)
        leaves.append((p, shape.Located(acc.Multiplied(shape.Location()))))


roots = TDF_LabelSequence()
st.GetFreeShapes(roots)
for i in range(1, roots.Length() + 1):
    walk(roots.Value(i), TopLoc_Location(), "")

repose = gp_Trsf()
repose.SetRotation(gp_Ax1(gp_Pnt(0, ELBOW_Y, ELBOW_Z), gp_Dir(1, 0, 0)), -math.pi / 2)
mirror = gp_Trsf()
mirror.SetMirror(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(1, 0, 0)))

REPOSE_MARKS = ["/J4J5连接件（双边固定）:1/", "/eyou手腕（90度旋转） v13:1/"]

W = "/骨架参考 v91"
RULES = [
    (["/底座 v28:1/快拆部分总装:1/"], [], "base_link"),
    (["/yaw轴与十字轴连接装配:1/鱼眼轴承 v1:4/"], [], "waist_right_link_driven_link"),
    (["/yaw轴与十字轴连接装配:1/鱼眼轴承 v1:5/"], [], "waist_left_link_driven_link"),
    (["/yaw轴与十字轴连接装配:1/"], [], "waist_yaw_link"),
    (["/新十字轴组件:1/"], [], "waist_roll_link"),
    (["/雷霆大改电机输出:1/鱼眼轴承 v1:1/"], [], "waist_right_link_driven_link"),
    (["/雷霆大改电机输出:2/鱼眼轴承 v1:1/"], [], "waist_left_link_driven_link"),
    (["/雷霆大改真连杆:1/"], [], "waist_right_link_driven_link"),
    (["/雷霆大改真连杆(镜像):1/"], [], "waist_left_link_driven_link"),
    (["/70H faker v14:1/fake70H电机转子:1/"], [], "waist_right_motor_link"),
    (["/雷霆大改电机输出:1/"], [], "waist_right_motor_link"),
    (["/70H faker v14:2/fake70H电机转子:1/"], [], "waist_left_motor_link"),
    (["/雷霆大改电机输出:2/"], [], "waist_left_motor_link"),
    (["/躯干:1/"], [], "waist_pitch_link"),
    (["/雷霆大改前端固定铝件:1/"], [], "waist_pitch_link"),
    (["/雷霆大改后端固定铝件 :1/"], [], "waist_pitch_link"),
    (["eyou腰雷霆大改改上改（保守方案） v27:1/70H faker"], [], "waist_pitch_link"),
    (["/eyou锥齿轮脖子 v36:1/脖子固定架:1/"], [], "waist_pitch_link"),
    (["/eyou锥齿轮脖子 v36:1/上段电机架:1/"], [], "waist_pitch_link"),
    (["/eyou锥齿轮脖子 v36:1/上段电机架:2/"], [], "waist_pitch_link"),
    (
        ["/eyou锥齿轮脖子 v36:1/RP40S行星关节模组_A01 2026.03.24 v1:1/"],
        [],
        "waist_pitch_link",
    ),
    (
        ["/eyou锥齿轮脖子 v36:1/RP40S行星关节模组_A01 2026.03.24 v1:2/"],
        [],
        "waist_pitch_link",
    ),
    (
        ["/eyou锥齿轮脖子 v36:1/脖子电机输出（打薄）:2/"],
        [],
        "head_front_link_motor_link",
    ),
    (["/eyou锥齿轮脖子 v36:1/m3轴承螺钉:2/"], [], "head_front_link_motor_link"),
    (["/eyou锥齿轮脖子 v36:1/脖子连杆 (1):1/"], [], "head_front_link_active_link"),
    (
        ["/给脖子用的锥齿轮架 v2 v1:1/m1.3000 mm / z27 (1):2/"],
        [],
        "head_front_link_bevel_gear_link",
    ),
    (
        ["/给脖子用的锥齿轮架 v2 v1:1/脖子连杆 与锥齿轮连接:1/"],
        [],
        "head_front_link_bevel_gear_link",
    ),
    (["/eyou锥齿轮脖子 v36:1/m3轴承螺钉:1/"], [], "head_front_link_bevel_gear_link"),
    (
        ["/eyou锥齿轮脖子 v36:1/脖子电机输出（打薄）:1/"],
        [],
        "head_rear_link_motor_link",
    ),
    (["/eyou锥齿轮脖子 v36:1/m3轴承螺钉:4/"], [], "head_rear_link_motor_link"),
    (["/eyou锥齿轮脖子 v36:1/脖子连杆 (1):3/"], [], "head_rear_link_active_link"),
    (
        ["/给脖子用的锥齿轮架 v2 v1:1/m1.3000 mm / z27 (1):1/"],
        [],
        "head_rear_link_bevel_gear_link",
    ),
    (
        ["/给脖子用的锥齿轮架 v2 v1:1/脖子连杆 与锥齿轮连接:2/"],
        [],
        "head_rear_link_bevel_gear_link",
    ),
    (["/eyou锥齿轮脖子 v36:1/m3轴承螺钉:3/"], [], "head_rear_link_bevel_gear_link"),
    (["/给脖子用的锥齿轮架 v2 v1:1/m1.3000 mm / z17 (1):1/"], [], "head_pitch_link"),
    (["/给脖子用的锥齿轮架 v2 v1:1/"], [], "head_roll_link"),
    (["/eyou锥齿轮脖子 v36:1/eyou头部yaw轴:1/"], [], "head_yaw_link"),
    (["/头部设计 v6 v1:1/"], [], "head_yaw_link"),
    (["/相机架子:1/"], [], "head_yaw_link"),
    (["/eyou锥齿轮脖子 v36:1/"], [], "waist_pitch_link"),
    (["/J1电机架 v9:1/50H faker v14:1/零部件3:1/"], [], "right_shoulder_pitch_link"),
    (["/J1电机架 v9:1/"], [], "waist_pitch_link"),
    (["/J1J2连接件:1/"], [], "right_shoulder_pitch_link"),
    (["/J2电机架 v26:1/50H faker v14:1/零部件34:1/"], [], "right_shoulder_roll_link"),
    (["/J2电机架 v26:1/"], [], "right_shoulder_pitch_link"),
    (["/J2J3连接件:1/"], [], "right_shoulder_roll_link"),
    (["/J3电机架 v6:1/50H faker v14:1/零部件3:1/"], [], "right_shoulder_yaw_link"),
    (["/J3电机架 v6:1/"], [], "right_shoulder_roll_link"),
    (["/J3J4连接总装:1/"], [], "right_shoulder_yaw_link"),
    (["/J3电机架 v6:2/50H faker v14:1/零部件3:1/"], [], "right_elbow_link"),
    (["/J3电机架 v6:2/"], [], "right_shoulder_yaw_link"),
    (["/J4J5连接件（双边固定）:1/"], [], "right_elbow_link"),
    (["/eyou手腕（90度旋转） v13:1/J5电机架：40s抱箍:1/"], [], "right_elbow_link"),
    (
        ["/eyou手腕（90度旋转） v13:1/RP40S行星关节模组_A01 2026.03.24 v1:3/"],
        [],
        "right_elbow_link",
    ),
    (
        ["/eyou手腕（90度旋转） v13:1/RP40S行星关节模组_A01 2026.03.24 v1:1/"],
        [],
        "right_wrist_roll_link",
    ),
    (
        ["/eyou手腕（90度旋转） v13:1/RP40S行星关节模组_A01 2026.03.24 v1:2/"],
        [],
        "right_wrist_roll_link",
    ),
    (["/eyou手腕（90度旋转） v13:1/j5手腕连接块:1/"], [], "right_wrist_roll_link"),
    (["/eyou手腕（90度旋转） v13:1/手臂抱死件:1/"], [], "right_wrist_roll_link"),
    (["/eyou手腕（90度旋转） v13:1/手臂抱死件:2/"], [], "right_wrist_roll_link"),
    (["/eyou手腕（90度旋转） v13:1/键压板:1/"], [], "right_wrist_roll_link"),
    (
        ["/eyou手腕（90度旋转） v13:1/电机输出 v1:1/"],
        [],
        "right_arm_long_link_motor_link",
    ),
    (
        ["/eyou手腕（90度旋转） v13:1/m3轴承螺钉 v1:2/"],
        [],
        "right_arm_long_link_motor_link",
    ),
    (["/eyou手腕（90度旋转） v13:1/长连杆:1/"], [], "right_arm_long_link_active_link"),
    (
        ["/独立锥齿轮架 v2:1/m1.3000 mm / z27:2/"],
        [],
        "right_arm_long_link_bevel_gear_link",
    ),
    (["/独立锥齿轮架 v2:1/手腕连杆:1/"], [], "right_arm_long_link_bevel_gear_link"),
    (
        ["/eyou手腕（90度旋转） v13:1/m3轴承螺钉 v1:1/"],
        [],
        "right_arm_long_link_bevel_gear_link",
    ),
    (
        ["/eyou手腕（90度旋转） v13:1/电机输出 v1:2/"],
        [],
        "right_arm_short_link_motor_link",
    ),
    (
        ["/eyou手腕（90度旋转） v13:1/m3轴承螺钉 v1:4/"],
        [],
        "right_arm_short_link_motor_link",
    ),
    (["/eyou手腕（90度旋转） v13:1/短连杆:1/"], [], "right_arm_short_link_active_link"),
    (
        ["/独立锥齿轮架 v2:1/m1.3000 mm / z27:1/"],
        [],
        "right_arm_short_link_bevel_gear_link",
    ),
    (["/独立锥齿轮架 v2:1/手腕连杆:2/"], [], "right_arm_short_link_bevel_gear_link"),
    (
        ["/eyou手腕（90度旋转） v13:1/m3轴承螺钉 v1:3/"],
        [],
        "right_arm_short_link_bevel_gear_link",
    ),
    (["/独立锥齿轮架 v2:1/m1.3000 mm / z17:1/"], [], "right_wrist_pitch_link"),
    (["/独立锥齿轮架 v2:1/"], [], "right_wrist_yaw_link"),
    (["/Wuji Hand 2 右手手掌模型 复制 v15:1/"], [], "right_wrist_pitch_link"),
    (["/eyou手腕（90度旋转） v13:1/手腕连接器-横装"], [], "right_wrist_pitch_link"),
]


def classify(path):
    pn = path + "/"
    for subs, excl, link in RULES:
        if all(s in pn for s in subs) and not any(e in pn for e in excl):
            return link
    return None


STEEL = [
    "轴承",
    "螺钉",
    "螺母",
    "钢柱",
    "垫片",
    "垫块",
    "传动轴",
    "十字轴",
    "销",
    "光轴",
    "丝杆",
    "键压板",
    "Gear",
]
MOTOR = ["RP40S行星关节模组", "RP70H外形图", "fake50H", "fake70H", "手拧丝杆 v1faker"]
PLASTIC = [
    "手掌",
    "LOGO",
    "透光PC",
    "FPC",
    "线束",
    "摄像头",
    "面部盖板",
    "信号板",
    "手套",
]
PLASTIC_LEAF = ["右边", "左边", "面部盖板"]


def density(path):
    leaf = path.rsplit("/", 1)[-1].split(":")[0].strip()
    if any(k in leaf for k in PLASTIC) or leaf in PLASTIC_LEAF:
        return 1200.0
    if "m1.3000" in path and ("SOLID" in leaf or "=>" in path.rsplit("/", 1)[-1]):
        return 7850.0
    if any(k in path for k in MOTOR):
        return 5000.0
    if any(k in leaf for k in STEEL):
        return 7850.0
    return 2700.0


groups = {}
unmatched = []
for p, sh in leaves:
    link = classify(p)
    if link is None:
        unmatched.append(p)
        continue
    if any(m in p for m in REPOSE_MARKS):
        sh = BRepBuilderAPI_Transform(sh, repose, True).Shape()
    groups.setdefault(link, []).append((p, sh))

for right in [l for l in list(groups) if l.startswith("right_")]:
    left = "left_" + right[len("right_") :]
    groups[left] = [
        (p + "(auto镜像)", BRepBuilderAPI_Transform(sh, mirror, True).Shape())
        for p, sh in groups[right]
    ]
groups["waist_pitch_link"] += [
    (p + "(auto镜像)", BRepBuilderAPI_Transform(sh, mirror, True).Shape())
    for p, sh in groups["waist_pitch_link"]
    if "/J1电机架 v9:1/" in p
]

R_ELBOW = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)


def to_robot(p_cad):
    v = np.array(p_cad, float) - O_CAD
    return np.array([-v[1], v[0], v[2]]) / 1000.0


ORIGINS_CAD = {
    "base_link": ([0, -4.8, -383.3], False),
    "waist_yaw_link": ([0, -4.8, -319.5], False),
    "waist_right_link_driven_link": ([-22.0, -38.9, -300.0], False),
    "waist_left_link_driven_link": ([22.0, -38.9, -300.0], False),
    "waist_roll_link": ([0, -4.8, -284.0], False),
    "waist_pitch_link": ([0, -4.8, -284.0], False),
    "waist_right_motor_link": ([-42.5, -25.65, -232.0], False),
    "waist_left_motor_link": ([42.5, -25.65, -232.0], False),
    "right_shoulder_pitch_link": ([-97.855, 0.0, 32.745], False),
    "right_shoulder_roll_link": ([-146.9, -32.5, 11.08], False),
    "right_shoulder_yaw_link": ([-161.9, -5.0, -150.225], False),
    "right_elbow_link": ([-192.299, -8.0, -208.92], False),
    "right_wrist_roll_link": ([-161.9, 17.0, -306.35], True),
    "right_arm_long_link_motor_link": ([-161.9, -9.2, -332.35], True),
    "right_arm_long_link_active_link": ([-173.9, -14.2, -332.35], True),
    "right_arm_long_link_bevel_gear_link": ([-161.9, 6.47, -433.35], True),
    "right_arm_short_link_motor_link": ([-161.9, 43.2, -379.35], True),
    "right_arm_short_link_active_link": ([-149.9, 48.2, -379.35], True),
    "right_arm_short_link_bevel_gear_link": ([-161.9, 27.53, -433.35], True),
    "right_wrist_yaw_link": ([-161.9, 17.0, -433.35], True),
    "right_wrist_pitch_link": ([-161.9, 17.0, -433.35], True),
    "head_front_link_motor_link": ([-25.51, -25.1, 79.0], False),
    "head_front_link_active_link": ([-14.56, -29.2, 74.1], False),
    "head_front_link_bevel_gear_link": ([0.0, -10.53, 136.0], False),
    "head_rear_link_motor_link": ([25.51, 25.1, 79.0], False),
    "head_rear_link_active_link": ([14.56, 29.2, 74.1], False),
    "head_rear_link_bevel_gear_link": ([0.0, 10.53, 136.0], False),
    "head_roll_link": ([0.0, -0.4, 136.0], False),
    "head_pitch_link": ([0.0, -0.4, 136.0], False),
    "head_yaw_link": ([0.0, -0.4, 171.89], False),
}


def link_origin_robot(link):
    if link.startswith("left_"):
        p, rep = ORIGINS_CAD["right_" + link[len("left_") :]]
        p = np.array(p, float) * np.array([-1, 1, 1])
        if rep:
            p = np.array([p[0], ELBOW_Y, ELBOW_Z]) + R_ELBOW @ (
                p - np.array([p[0], ELBOW_Y, ELBOW_Z])
            )
        return to_robot(p)
    p, rep = ORIGINS_CAD[link]
    p = np.array(p, float)
    if rep:
        p = np.array([p[0], ELBOW_Y, ELBOW_Z]) + R_ELBOW @ (
            p - np.array([p[0], ELBOW_Y, ELBOW_Z])
        )
    return to_robot(p)


M_ROB = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float)

HOLE_R_MAX = 2.6


def _face_diag(f):
    b = Bnd_Box()
    try:
        BRepBndLib.Add_s(f, b)
        x0, y0, z0, x1, y1, z1 = b.Get()
        return ((x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2) ** 0.5
    except Exception:
        return 1e9


def _inward(f, ad, A, D):
    u = (ad.FirstUParameter() + ad.LastUParameter()) / 2
    v = (ad.FirstVParameter() + ad.LastVParameter()) / 2
    gf = BRepGProp_Face(f)
    pt, nv = gp_Pnt(), gp_Vec()
    gf.Normal(u, v, pt, nv)
    P = np.array([pt.X(), pt.Y(), pt.Z()])
    radial = P - A - np.dot(P - A, D) * D
    n = np.array([nv.X(), nv.Y(), nv.Z()])
    return np.dot(n, radial) < 0


def _hole_skip_faces(shape):
    walls = []
    ex = TopExp_Explorer(shape, TopAbs_FACE)
    while ex.More():
        f = TopoDS.Face_s(ex.Current())
        try:
            ad = BRepAdaptor_Surface(f)
            t = ad.GetType()
            uspan = ad.LastUParameter() - ad.FirstUParameter()
            if (
                t == GeomAbs_Cylinder
                and ad.Cylinder().Radius() <= HOLE_R_MAX
                and uspan >= 2.8
            ):
                ax = ad.Cylinder().Axis()
                A = np.array([ax.Location().X(), ax.Location().Y(), ax.Location().Z()])
                D = np.array(
                    [ax.Direction().X(), ax.Direction().Y(), ax.Direction().Z()]
                )
                if _inward(f, ad, A, D):
                    walls.append(f)
            elif t == GeomAbs_Cone and uspan >= 5.5 and _face_diag(f) <= 9.0:
                ax = ad.Cone().Axis()
                A = np.array([ax.Location().X(), ax.Location().Y(), ax.Location().Z()])
                D = np.array(
                    [ax.Direction().X(), ax.Direction().Y(), ax.Direction().Z()]
                )
                if _inward(f, ad, A, D):
                    walls.append(f)
        except Exception:
            pass
        ex.Next()
    skipped = {hash(w): w for w in walls}
    if not walls:
        return skipped, set()
    amap = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(shape, TopAbs_EDGE, TopAbs_FACE, amap)
    for w in walls:
        eex = TopExp_Explorer(w, TopAbs_EDGE)
        while eex.More():
            e = eex.Current()
            try:
                for nb in amap.FindFromKey(e):
                    fnb = TopoDS.Face_s(nb)
                    if hash(fnb) in skipped:
                        continue
                    try:
                        tnb = BRepAdaptor_Surface(fnb).GetType()
                    except Exception:
                        tnb = None
                    if (
                        tnb in (GeomAbs_Plane, GeomAbs_Cone, GeomAbs_Sphere)
                        and _face_diag(fnb) <= 6.5
                    ):
                        skipped[hash(fnb)] = fnb
            except Exception:
                pass
            eex.Next()
    skipped_edges = set()
    for f in skipped.values():
        eex = TopExp_Explorer(f, TopAbs_EDGE)
        while eex.More():
            skipped_edges.add(hash(eex.Current()))
            eex.Next()
    return skipped, skipped_edges


def _append_tri(f, verts, faces, origin_r, lin=0.3, ang=0.5):
    loc = TopLoc_Location()
    tri = BRep_Tool.Triangulation_s(f, loc)
    if tri is None:
        BRepMesh_IncrementalMesh(f, lin, False, ang, False)
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(f, loc)
        if tri is None:
            return
    t = loc.Transformation()
    base = len(verts)
    for i in range(1, tri.NbNodes() + 1):
        pt = tri.Node(i).Transformed(t)
        pc = np.array([pt.X(), pt.Y(), pt.Z()])
        verts.append((M_ROB @ (pc - O_CAD)) / 1000.0 - origin_r)
    rev = f.Orientation() == TopAbs_REVERSED
    for i in range(1, tri.NbTriangles() + 1):
        a, b, c = tri.Triangle(i).Get()
        idx = (
            (base + a - 1, base + c - 1, base + b - 1)
            if rev
            else (base + a - 1, base + b - 1, base + c - 1)
        )
        faces.append(idx)


def _rebuild_without_wires(f, drop_wires, lin, ang):
    nf = TopoDS.Face_s(f.EmptyCopied())
    b = BRep_Builder()
    wex = TopExp_Explorer(f, TopAbs_WIRE)
    while wex.More():
        w = TopoDS.Wire_s(wex.Current())
        if hash(w) not in drop_wires:
            b.Add(nf, w)
        wex.Next()
    BRepMesh_IncrementalMesh(nf, lin, False, ang, False)
    return nf


def shape_mesh_robot(shape, origin_r):
    try:
        box = Bnd_Box()
        BRepBndLib.Add_s(shape, box)
        xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
        diag = ((xmax - xmin) ** 2 + (ymax - ymin) ** 2 + (zmax - zmin) ** 2) ** 0.5
        lin = min(1.2, max(0.15, 0.004 * diag))
        ang = 0.8 if diag < 60.0 else 0.5
    except Exception:
        lin, ang = 0.3, 0.5
    skipped, skipped_edges = _hole_skip_faces(shape)
    BRepMesh_IncrementalMesh(shape, lin, False, ang, True)
    verts, faces = [], []
    ex = TopExp_Explorer(shape, TopAbs_FACE)
    while ex.More():
        f = TopoDS.Face_s(ex.Current())
        ex.Next()
        if hash(f) in skipped:
            continue
        drop = set()
        if skipped_edges:
            try:
                ow_h = hash(BRepTools.OuterWire_s(f))
                wex = TopExp_Explorer(f, TopAbs_WIRE)
                while wex.More():
                    w = TopoDS.Wire_s(wex.Current())
                    wex.Next()
                    if hash(w) == ow_h or _face_diag(w) > 8.0:
                        continue
                    eex = TopExp_Explorer(w, TopAbs_EDGE)
                    all_in = True
                    any_e = False
                    while eex.More():
                        any_e = True
                        if hash(eex.Current()) not in skipped_edges:
                            all_in = False
                            break
                        eex.Next()
                    if any_e and all_in:
                        drop.add(hash(w))
            except Exception:
                drop = set()
        if not drop:
            _append_tri(f, verts, faces, origin_r, lin, ang)
        else:
            try:
                nf = _rebuild_without_wires(f, drop, lin, ang)
                _append_tri(nf, verts, faces, origin_r, lin, ang)
            except Exception:
                _append_tri(f, verts, faces, origin_r, lin, ang)
    return np.array(verts), np.array(faces)


MESH_SKIP = ["螺母"]


def mesh_skip(path):
    leaf = path.rsplit("/", 1)[-1]
    return any(k in leaf for k in MESH_SKIP)


report = {}
links_json = {}
for link in sorted(groups):
    origin_r = link_origin_robot(link)
    all_v, all_f = [], []
    total_m = 0.0
    msum = np.zeros(3)
    parts_mass = []
    for p, sh in groups[link]:
        rho = density(p)
        pr = GProp_GProps()
        BRepGProp.VolumeProperties_s(sh, pr)
        vol = pr.Mass() * 1e-9
        if vol <= 0:
            continue
        m = vol * rho
        c = pr.CentreOfMass()
        com_c = np.array([c.X(), c.Y(), c.Z()])
        com_r = (M_ROB @ (com_c - O_CAD)) / 1000.0
        I = np.zeros((3, 3))
        mm = pr.MatrixOfInertia()
        for r in range(3):
            for cc in range(3):
                I[r, cc] = mm.Value(r + 1, cc + 1)
        I = I * rho * 1e-9 * 1e-6
        I_r = M_ROB @ I @ M_ROB.T
        parts_mass.append((m, com_r, I_r))
        total_m += m
        msum += m * com_r
        if mesh_skip(p):
            continue
        v, f = shape_mesh_robot(sh, origin_r)
        if len(v):
            all_v.append(v)
            all_f.append(f)
    com = msum / total_m
    Itot = np.zeros((3, 3))
    for m, c_r, I_r in parts_mass:
        d = c_r - com
        Itot += I_r + m * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
    off = 0
    fv, ff = [], []
    for v, f in zip(all_v, all_f):
        fv.append(v)
        ff.append(f + off)
        off += len(v)
    mesh = trimesh.Trimesh(np.vstack(fv), np.vstack(ff), process=False)
    mesh.export(os.path.join(MESH_DIR, f"{link}.STL"))
    links_json[link] = {
        "origin": origin_r.tolist(),
        "mass": total_m,
        "com_link": (com - origin_r).tolist(),
        "inertia_com": Itot.tolist(),
        "nparts": len(groups[link]),
    }
    report[link] = (total_m, len(groups[link]))

with open(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "links.json"), "w"
) as f:
    json.dump(links_json, f, ensure_ascii=False, indent=1)

for link, (m, n) in sorted(report.items()):
    print(f"{link:42s} m={m:8.4f} kg  parts={n}")
print("TOTAL", sum(m for m, n in report.values()))
print("unmatched (should be only 底座支架/顶层镜像):", len(unmatched))
for u in unmatched[:8]:
    print("  ", u.split("/", 2)[-1][:60])
