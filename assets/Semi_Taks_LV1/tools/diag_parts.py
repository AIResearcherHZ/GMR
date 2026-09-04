import math
import os

import numpy as np
import trimesh
from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCP.BRepGProp import BRepGProp
from OCP.gp import gp_Ax1, gp_Dir, gp_Pnt, gp_Trsf
from OCP.GProp import GProp_GProps
from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.TCollection import TCollection_ExtendedString
from OCP.TDataStd import TDataStd_Name
from OCP.TDF import TDF_Label, TDF_LabelSequence
from OCP.TDocStd import TDocStd_Document
from OCP.TopLoc import TopLoc_Location
from OCP.XCAFDoc import XCAFDoc_DocumentTool

O_CAD = np.array([0.0, -4.8, -383.3])
ELBOW_Y, ELBOW_Z = -8.0, -208.92
M_ROB = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float)

src = open(os.path.join(os.path.dirname(__file__), "step_export_links.py")).read()
seg = src[src.index("HOLE_R_MAX") : src.index("MESH_SKIP")]
exec(seg, globals())
rules_seg = src[src.index("RULES = [") : src.index("def classify")]
exec(rules_seg, globals())


def classify(path):
    pn = path + "/"
    for subs, excl, link in RULES:
        if all(s in pn for s in subs) and not any(e in pn for e in excl):
            return link
    return None


STEP = (
    "/home/xhz/taks-controller-web/taks_level1/assets/Semi_Taks_LV1/骨架参考 v91.step"
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
REPOSE_MARKS = ["/J4J5连接件（双边固定）:1/", "/eyou手腕（90度旋转） v13:1/"]

groups = {}
for p, sh in leaves:
    link = classify(p)
    if link is None:
        continue
    if any(m in p for m in REPOSE_MARKS):
        sh = BRepBuilderAPI_Transform(sh, repose, True).Shape()
    groups.setdefault(link, []).append((p, sh))


def surf_area(sh):
    pr = GProp_GProps()
    BRepGProp.SurfaceProperties_s(sh, pr)
    return pr.Mass()


MESH_SKIP = ["螺母"]


def mesh_skip(path):
    leaf = path.rsplit("/", 1)[-1]
    return any(k in leaf for k in MESH_SKIP)


print(f"{'link/part':70s} {'vol(cm3)':>9s} {'areaCAD':>9s} {'areaSTL':>9s} ratio  note")
for link in sorted(groups):
    if link.startswith("left_"):
        continue
    for p, sh in groups[link]:
        pr = GProp_GProps()
        try:
            BRepGProp.VolumeProperties_s(sh, pr)
            v = pr.Mass() * 1e-9
        except Exception:
            v = float("nan")
        note = []
        if not (v > 0):
            note.append("DROPPED(vol<=0)")
        if mesh_skip(p):
            note.append("MESH_SKIP")
        try:
            a_cad = surf_area(sh) * 1e-6
        except Exception:
            a_cad = float("nan")
        a_stl = float("nan")
        if v > 0 and not mesh_skip(p):
            try:
                verts, faces = shape_mesh_robot(sh, np.zeros(3))
                if len(verts):
                    mm = trimesh.Trimesh(verts, faces, process=False)
                    a_stl = mm.area
                else:
                    note.append("EMPTY_MESH")
            except Exception as e:
                note.append(f"MESH_FAIL:{type(e).__name__}")
        ratio = (
            a_stl / a_cad if a_cad and a_cad > 0 and a_stl == a_stl else float("nan")
        )
        flag = note or (ratio == ratio and ratio < 0.85) or not (a_cad > 0)
        if flag:
            leafname = "/".join(p.split("/")[-2:])
            print(
                f"{link + ' :: ' + leafname:70s} {v * 1e6:9.2f} {a_cad * 1e4:9.2f} {(a_stl * 1e4 if a_stl == a_stl else float('nan')):9.2f} {ratio:5.2f}  {','.join(note)}"
            )
