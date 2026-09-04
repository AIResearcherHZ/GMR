import importlib.util
import json
import math
import os

import numpy as np
import trimesh
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib
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

spec = importlib.util.spec_from_file_location(
    "exp", os.path.join(os.path.dirname(__file__), "step_export_links.py")
)

STEP = (
    "/home/xhz/taks-controller-web/taks_level1/assets/Semi_Taks_LV1/骨架参考 v91.step"
)
OUT = "/home/xhz/taks-controller-web/taks_level1/assets/Semi_Taks_LV1"

O_CAD = np.array([0.0, -4.8, -383.3])
ELBOW_Y, ELBOW_Z = -8.0, -208.92
M_ROB = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float)

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

exec(
    open(os.path.join(os.path.dirname(__file__), "step_export_links.py"))
    .read()
    .split("groups = {}")[0]
    .split("RULES = [")[1]
    .join(["RULES = [", ""])
    .split("def classify")[0],
    globals(),
)


def classify(path):
    pn = path + "/"
    for subs, excl, link in RULES:
        if all(s in pn for s in subs) and not any(e in pn for e in excl):
            return link
    return None


repose = gp_Trsf()
repose.SetRotation(gp_Ax1(gp_Pnt(0, ELBOW_Y, ELBOW_Z), gp_Dir(1, 0, 0)), -math.pi / 2)
REPOSE_MARKS = ["/J4J5连接件（双边固定）:1/", "/eyou手腕（90度旋转） v13:1/"]

groups = {}
unmatched = []
for p, sh in leaves:
    link = classify(p)
    if link is None:
        unmatched.append((p, sh))
        continue
    if any(m in p for m in REPOSE_MARKS):
        sh = BRepBuilderAPI_Transform(sh, repose, True).Shape()
    groups.setdefault(link, []).append((p, sh))


def bbox_robot(sh):
    b = Bnd_Box()
    try:
        BRepBndLib.Add_s(sh, b)
    except Exception:
        return None
    if b.IsVoid():
        return None
    x0, y0, z0, x1, y1, z1 = b.Get()
    lo = (M_ROB @ (np.array([x0, y0, z0]) - O_CAD)) / 1000.0
    hi = (M_ROB @ (np.array([x1, y1, z1]) - O_CAD)) / 1000.0
    return np.minimum(lo, hi), np.maximum(lo, hi)


def vol(sh):
    pr = GProp_GProps()
    BRepGProp.VolumeProperties_s(sh, pr)
    return pr.Mass() * 1e-9


links_json = json.load(open(os.path.join(os.path.dirname(__file__), "links.json")))

print(
    "link                                        STEPvol(L)   STLvol(L)  STLwatertight  bboxDiff(mm)"
)
for link in sorted(set(groups) | set(links_json)):
    if link.startswith("left_"):
        continue
    parts = groups.get(link, [])
    v_step = sum(vol(sh) for _, sh in parts)
    stl_path = os.path.join(OUT, "meshes", f"{link}.STL")
    if not os.path.exists(stl_path):
        print(f"{link:42s}  NO STL")
        continue
    mesh = trimesh.load(stl_path, process=False)
    origin = np.array(links_json[link]["origin"])
    lo_s = np.full(3, 1e9)
    hi_s = np.full(3, -1e9)
    for pp, sh in parts:
        bb = bbox_robot(sh)
        if bb is None:
            print(f"    !! empty/invalid shape: {pp}")
            continue
        lo, hi = bb
        lo_s = np.minimum(lo_s, lo)
        hi_s = np.maximum(hi_s, hi)
    lo_m = mesh.vertices.min(0) + origin
    hi_m = mesh.vertices.max(0) + origin
    d = max(np.abs(lo_s - lo_m).max(), np.abs(hi_s - hi_m).max()) * 1000
    print(
        f"{link:42s}  {v_step * 1000:9.4f}  {mesh.volume * 1000:9.4f}  {mesh.is_watertight!s:5s}  {d:8.2f}"
    )

print()
print("UNMATCHED parts (dropped from export):", len(unmatched))
agg = {}
for p, sh in unmatched:
    key = p.split("/")[2] if len(p.split("/")) > 2 else p
    v = vol(sh)
    a = agg.setdefault(key, [0, 0.0, None, None])
    a[0] += 1
    a[1] += v
    bb = bbox_robot(sh)
    if bb is None:
        continue
    lo, hi = bb
    a[2] = lo if a[2] is None else np.minimum(a[2], lo)
    a[3] = hi if a[3] is None else np.maximum(a[3], hi)
for k, (n, v, lo, hi) in sorted(agg.items(), key=lambda x: -x[1][1]):
    print(
        f"  n={n:3d} vol={v * 1e6:10.2f}cm3  z[{lo[2]:.3f},{hi[2]:.3f}] y[{lo[1]:.3f},{hi[1]:.3f}] x[{lo[0]:.3f},{hi[0]:.3f}]  {k}"
    )
