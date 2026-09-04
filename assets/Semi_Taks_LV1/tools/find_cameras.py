import numpy as np
from OCP.Bnd import Bnd_Box
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepGProp import BRepGProp
from OCP.GeomAbs import GeomAbs_Cylinder
from OCP.GProp import GProp_GProps
from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.TCollection import TCollection_ExtendedString
from OCP.TDataStd import TDataStd_Name
from OCP.TDF import TDF_Label, TDF_LabelSequence
from OCP.TDocStd import TDocStd_Document
from OCP.TopAbs import TopAbs_FACE
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS
from OCP.XCAFDoc import XCAFDoc_DocumentTool

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


hits = []


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
    if "摄像头" in disp:
        hits.append((p, target, acc))
        return
    if st.IsAssembly_s(target):
        comps = TDF_LabelSequence()
        st.GetComponents_s(target, comps)
        for i in range(1, comps.Length() + 1):
            walk(comps.Value(i), acc, p)


roots = TDF_LabelSequence()
st.GetFreeShapes(roots)
for i in range(1, roots.Length() + 1):
    walk(roots.Value(i), TopLoc_Location(), "")

O_CAD = np.array([0.0, -4.8, -383.3])
HEAD_YAW_CAD = np.array([0.0, -0.4, 171.89])
M_ROB = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float)

for p, target, acc in hits:
    print("=" * 80)
    print("path:", p)
    t = acc.Transformation()
    tr = t.TranslationPart()
    R = np.array([[t.Value(i, j) for j in range(1, 4)] for i in range(1, 4)])
    T = np.array([tr.X(), tr.Y(), tr.Z()])
    print("frame origin CAD mm:", np.round(T, 3))
    print("frame R:\n", np.round(R, 4))
    shape = st.GetShape_s(target).Located(
        acc.Multiplied(st.GetShape_s(target).Location())
    )
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props)
    c = props.CentreOfMass()
    com = np.array([c.X(), c.Y(), c.Z()])
    print("COM CAD mm:", np.round(com, 3))
    b = Bnd_Box()
    BRepBndLib.Add_s(shape, b)
    x0, y0, z0, x1, y1, z1 = b.Get()
    print("bbox CAD mm:", np.round([x0, y0, z0], 2), "->", np.round([x1, y1, z1], 2))
    cyls = {}
    ex = TopExp_Explorer(shape, TopAbs_FACE)
    while ex.More():
        f = TopoDS.Face_s(ex.Current())
        try:
            ad = BRepAdaptor_Surface(f)
            if ad.GetType() == GeomAbs_Cylinder:
                cy = ad.Cylinder()
                ax = cy.Axis()
                A = np.array([ax.Location().X(), ax.Location().Y(), ax.Location().Z()])
                D = np.array(
                    [ax.Direction().X(), ax.Direction().Y(), ax.Direction().Z()]
                )
                key = (round(cy.Radius(), 2), tuple(np.round(D, 3)))
                cyls.setdefault(key, []).append(np.round(A, 2))
        except Exception:
            pass
        ex.Next()
    for (r, d), locs in sorted(cyls.items()):
        print(f"  cyl r={r} dir={d} n={len(locs)} loc0={locs[0]}")
    rel = com - HEAD_YAW_CAD
    print("COM rel head_yaw CAD mm:", np.round(rel, 3))
    print("COM rel head_yaw ROBOT m:", np.round(M_ROB @ rel / 1000.0, 5))
    print("R in ROBOT frame:\n", np.round(M_ROB @ R, 4))
