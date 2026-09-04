import argparse
import os

for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_v, "1")

import re
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import trimesh
from gpytoolbox import fast_winding_number
from scipy.spatial.transform import Rotation

warnings.filterwarnings("ignore")

VOX_PAD = 4  # 体素网格在包围盒外每个方向留的空体素层数, 防止边界被截断
VOX_MIN_PER_THIN = (
    40  # 最薄方向至少多少个体素; 精度 ∝ (h/L_min)², 薄轴决定误差, 故按最薄轴定网格
)
VOX_RES_MAX = 320  # 每轴体素数上限 (与 max_cells 内存夹紧共同限制网格规模)
VOX_CONV_TOL = 0.01  # 体素加密收敛阈: 相邻分辨率的主惯量最大相对变化 <1% 即停止细化
VOX_ROT_DEG = 1.5  # 采样前把几何绕 xyz 各旋转的角度, 错开轴对齐平面以消除阶梯误差 (O(h)→O(h²)), 结果再用 Rᵀ 转回 link 系
VOX_VOL_TOL = 0.10  # trimesh 体积与体素并集体积的允许相对差; 超出即判为重叠多壳(解析会重复计数)→ 改走体素
CROSS_COM_TOL = 0.01  # 双法互校质心阈: 体素与补洞解析的质心差 / 包围盒对角线
CROSS_I_TOL = 0.02  # 双法互校惯量阈: 体素与补洞解析的主惯量(I/mass)最大相对差
TWIN_VOL_TOL = 0.03  # 左右镜像件体积相对差阈, 超出疑似几何不一致
TWIN_I_TOL = 0.05  # 左右镜像件主惯量(I/mass)相对差阈, 超出疑似几何不一致
TWIN_FACE_RATIO = (
    2.0  # 镜像件面数比阈: 一侧面数≤另一侧的 1/2 判为被抽面 → LOW, 需CAD重导
)

LINK_WEIGHT_KG = {
    "base_link": 1.1004,
    "waist_yaw_link": 0.13,
    "waist_right_link_driven_link": 0.0123,
    "waist_left_link_driven_link": 0.0123,
    "waist_roll_link": 0.0311,
    "waist_pitch_link": 2.668,
    "waist_right_motor_link": 0.0562,
    "waist_left_motor_link": 0.0562,
    "right_shoulder_pitch_link": 0.4184,
    "right_shoulder_roll_link": 0.4502,
    "right_shoulder_yaw_link": 0.420,
    "right_elbow_link": 0.0227,
    "right_wrist_roll_link": 1.0138,
    "right_arm_long_link_motor_link": 0.009,
    "right_arm_long_link_active_link": 0.007,
    "right_arm_short_link_motor_link": 0.009,
    "right_arm_short_link_active_link": 0.006,
    "right_wrist_yaw_link": 0.0184,
    "right_wrist_pitch_link": 0.0984,
    "right_arm_long_link_bevel_gear_link": 0.008,
    "right_arm_long_link_driven_link": 0.008,
    "right_arm_short_link_bevel_gear_link": 0.008,
    "right_arm_short_link_driven_link": 0.008,
    "left_shoulder_pitch_link": 0.4184,
    "left_shoulder_roll_link": 0.4502,
    "left_shoulder_yaw_link": 0.420,
    "left_elbow_link": 0.0227,
    "left_wrist_roll_link": 1.0138,
    "left_arm_long_link_motor_link": 0.009,
    "left_arm_long_link_active_link": 0.007,
    "left_arm_short_link_motor_link": 0.009,
    "left_arm_short_link_active_link": 0.006,
    "left_wrist_yaw_link": 0.0184,
    "left_wrist_pitch_link": 0.0984,
    "left_arm_long_link_bevel_gear_link": 0.008,
    "left_arm_long_link_driven_link": 0.008,
    "left_arm_short_link_bevel_gear_link": 0.008,
    "left_arm_short_link_driven_link": 0.008,
}


def cpu_count():
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except Exception:
        return os.cpu_count() or 1


def physical_cores():
    try:
        import psutil

        p = psutil.cpu_count(logical=False)
        if p:
            return max(1, p)
    except Exception:
        pass
    return max(1, cpu_count() // 2)


def available_memory():
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable"):
                        return int(line.split()[1]) * 1024
        except Exception:
            pass
    return 4 * 1024**3


def plan_workers(n_items):
    return max(1, min(physical_cores(), n_items))


def max_cells(workers):
    return max(2_000_000, int(available_memory() * 0.4 / workers / 32))


def bbox_diag(mesh):
    return float(np.linalg.norm(mesh.extents))


def principal_reldiff(ia, ib, ma=1.0, mb=1.0):
    ea = np.sort(np.linalg.eigvalsh(ia)) / ma
    eb = np.sort(np.linalg.eigvalsh(ib)) / mb
    return float(np.max(np.abs(ea - eb) / np.maximum(np.abs(eb), 1e-18)))


def analytic_inertia(mesh, mass):
    md = mesh.copy()
    if md.volume < 0:
        md.invert()
    md.density = mass / md.volume
    return np.asarray(md.center_mass), np.asarray(md.moment_inertia)


def repair_watertight(mesh):
    md = mesh.copy()
    md.merge_vertices()
    trimesh.repair.fill_holes(md)
    md.update_faces(md.nondegenerate_faces())
    md.update_faces(md.unique_faces())
    md.remove_unreferenced_vertices()
    trimesh.repair.fix_normals(md)
    if md.volume < 0:
        md.invert()
    return md


def grid_centers(origin, h, n):
    nx, ny, nz = (int(v) for v in n)
    ax = origin[0] + (np.arange(nx) + 0.5) * h[0]
    ay = origin[1] + (np.arange(ny) + 0.5) * h[1]
    az = origin[2] + (np.arange(nz) + 0.5) * h[2]
    Q = np.empty((nx * ny * nz, 3), dtype=np.float64)
    Q[:, 0] = np.repeat(ax, ny * nz)
    Q[:, 1] = np.tile(np.repeat(ay, nz), nx)
    Q[:, 2] = np.tile(az, nx * ny)
    return Q


def voxel_inertia_at_h(V, F, lo, hi, h, mass):
    origin = lo - VOX_PAD * h
    n = np.ceil((hi + VOX_PAD * h - origin) / h).astype(int) + 1
    Q = grid_centers(origin, h, n)
    w = fast_winding_number(Q, V, F)
    inside = np.rint(w) >= 1
    n_in = int(inside.sum())
    if n_in == 0:
        return None
    P = Q[inside]
    del Q, w
    com = P.mean(axis=0)
    d = P - com
    mv = mass / n_in
    ixx = mv * np.sum(d[:, 1] ** 2 + d[:, 2] ** 2)
    iyy = mv * np.sum(d[:, 0] ** 2 + d[:, 2] ** 2)
    izz = mv * np.sum(d[:, 0] ** 2 + d[:, 1] ** 2)
    ixy = -mv * np.sum(d[:, 0] * d[:, 1])
    ixz = -mv * np.sum(d[:, 0] * d[:, 2])
    iyz = -mv * np.sum(d[:, 1] * d[:, 2])
    sx = mass * (h[1] ** 2 + h[2] ** 2) / 12.0
    sy = mass * (h[0] ** 2 + h[2] ** 2) / 12.0
    sz = mass * (h[0] ** 2 + h[1] ** 2) / 12.0
    inertia = np.array(
        [[ixx + sx, ixy, ixz], [ixy, iyy + sy, iyz], [ixz, iyz, izz + sz]]
    )
    return com, inertia, n_in * float(np.prod(h))


def voxel_inertia(mesh, mass, cap):
    rot = Rotation.from_euler("xyz", [VOX_ROT_DEG] * 3, degrees=True).as_matrix()
    V = np.asarray(mesh.vertices, dtype=np.float64) @ rot.T
    F = np.asarray(mesh.faces, dtype=np.int32)
    lo, hi = V.min(axis=0), V.max(axis=0)
    ext = hi - lo
    thin = max(float(ext.min()), 1e-9)
    res = np.clip(np.ceil(ext / (thin / VOX_MIN_PER_THIN)), 1, VOX_RES_MAX).astype(int)
    last, prev, conv = None, None, float("inf")
    for _ in range(3):
        if int(np.prod(res)) > cap:
            res = np.maximum((res * (cap / np.prod(res)) ** (1 / 3)).astype(int), 1)
        out = voxel_inertia_at_h(V, F, lo, hi, ext / res, mass)
        if out is None:
            return None
        com_r, inertia_r, vol = out
        eig = np.sort(np.linalg.eigvalsh(inertia_r))
        last = (com_r, inertia_r, vol)
        if prev is not None:
            conv = float(np.max(np.abs(eig - prev) / np.maximum(eig, 1e-18)))
            if conv < VOX_CONV_TOL:
                break
        prev = eig
        nxt = np.clip(res * 2, 1, VOX_RES_MAX).astype(int)
        if np.array_equal(nxt, res):
            break
        res = nxt
    com_r, inertia_r, vol = last
    return rot.T @ com_r, rot.T @ inertia_r @ rot, conv, vol


def compute_inertia_from_mesh(mesh, mass, cap):
    diag = {"conv": None, "cross_com": None, "cross_i": None}
    clean = mesh.is_watertight and mesh.is_winding_consistent and mesh.volume > 0
    vox = voxel_inertia(mesh, mass, cap)
    if vox is None:
        if clean:
            return (*analytic_inertia(mesh, mass), "解析", diag)
        rep = repair_watertight(mesh)
        if rep.is_watertight and rep.is_winding_consistent and rep.volume > 0:
            return (*analytic_inertia(rep, mass), "补洞解析", diag)
        return (*analytic_inertia(mesh.convex_hull, mass), "凸包*", diag)
    vcom, vinertia, conv, vvol = vox
    diag["conv"] = conv
    if clean and abs(mesh.volume - vvol) / vvol < VOX_VOL_TOL:
        acom, ainertia = analytic_inertia(mesh, mass)
        diag["cross_com"] = float(np.linalg.norm(vcom - acom)) / bbox_diag(mesh)
        diag["cross_i"] = principal_reldiff(vinertia, ainertia)
        return acom, ainertia, "解析", diag
    rep = repair_watertight(mesh)
    if rep.is_watertight and rep.is_winding_consistent and rep.volume > 0:
        rcom, rinertia = analytic_inertia(rep, mass)
        diag["cross_com"] = float(np.linalg.norm(vcom - rcom)) / bbox_diag(mesh)
        diag["cross_i"] = principal_reldiff(vinertia, rinertia)
        if (
            abs(rep.volume - vvol) / vvol < VOX_VOL_TOL
            and diag["cross_com"] < CROSS_COM_TOL
            and diag["cross_i"] < CROSS_I_TOL
        ):
            return rcom, rinertia, "补洞解析", diag
    return vcom, vinertia, "WN-体素", diag


def validate_inertia(name, inertia):
    eig = np.sort(np.linalg.eigvalsh(inertia))
    if eig[0] <= 0:
        raise ValueError(f"{name}: 惯量张量非正定 eig={eig}")
    if eig[0] + eig[1] < eig[2] * (1 - 1e-6):
        raise ValueError(f"{name}: 主惯量不满足三角不等式 eig={eig}")


def link_inertia(task):
    name, mass, meshes_dir, cap = task
    mesh = trimesh.load(os.path.join(meshes_dir, name + ".STL"), process=True)
    com, inertia, track, diag = compute_inertia_from_mesh(mesh, mass, cap)
    validate_inertia(name, inertia)
    return {
        "name": name,
        "mass": mass,
        "com": com,
        "inertia_full": inertia,
        "track": track,
        "faces": len(mesh.faces),
        "volume": float(abs(mesh.volume)),
        "conv": diag["conv"],
        "cross_com": diag["cross_com"],
        "cross_i": diag["cross_i"],
    }


def twin_name(name):
    if name.startswith("waist_left_"):
        return "waist_right_" + name[len("waist_left_") :]
    if name.startswith("waist_right_"):
        return "waist_left_" + name[len("waist_right_") :]
    if name.startswith("left_"):
        return "right_" + name[len("left_") :]
    if name.startswith("right_"):
        return "left_" + name[len("right_") :]
    return None


def grade_one(d, results):
    if d["track"] == "凸包*":
        return "LOW", "凸包兜底,几何高估,需人工核对"
    note = ""
    twin = d.get("twin")
    if twin:
        o = results[twin]
        vbad = d["twin_vol"] is not None and d["twin_vol"] > TWIN_VOL_TOL
        ibad = d["twin_i"] is not None and d["twin_i"] > TWIN_I_TOL
        fbad = d["faces"] * TWIN_FACE_RATIO <= o["faces"]
        if vbad or ibad or fbad:
            worse = fbad or (
                d["track"] != "解析"
                and o["track"] == "解析"
                and d["faces"] <= o["faces"]
            )
            if worse:
                return "LOW", (
                    f"与镜像件{twin}不一致(面数{d['faces']}vs{o['faces']},"
                    f"体积差{(d['twin_vol'] or 0) * 100:.0f}%,I差{(d['twin_i'] or 0) * 100:.0f}%),"
                    "疑似抽面/几何缺失,需CAD重导"
                )
            note = "(注:镜像件几何疑似缺失)"
    if d["track"] == "解析":
        return "HIGH", "水密网格解析(精确)" + note
    conv_ok = (
        d["conv"] is not None and np.isfinite(d["conv"]) and d["conv"] < VOX_CONV_TOL
    )
    cross_ok = (
        d["cross_com"] is not None
        and d["cross_com"] < CROSS_COM_TOL
        and d["cross_i"] is not None
        and d["cross_i"] < CROSS_I_TOL
    )
    if d["track"] == "补洞解析" and cross_ok:
        return "HIGH", "补洞解析,体素独立互校一致" + note
    if conv_ok and cross_ok:
        return "HIGH", "体素收敛+补洞解析互校一致" + note
    if conv_ok or cross_ok:
        return "MEDIUM", "体素收敛/互校其一,无双重确认" + note
    return "MEDIUM", "体素未充分收敛且无独立互校" + note


def grade_confidence(results):
    for name, d in results.items():
        t = twin_name(name)
        d["twin"] = t if t in results else None
    for name, d in results.items():
        tv = ti = fr = None
        t = d["twin"]
        if t:
            o = results[t]
            tv = abs(d["volume"] - o["volume"]) / max(o["volume"], 1e-12)
            ti = principal_reldiff(
                d["inertia_full"], o["inertia_full"], d["mass"], o["mass"]
            )
            fr = max(d["faces"], o["faces"]) / max(min(d["faces"], o["faces"]), 1)
        d["twin_vol"], d["twin_i"], d["face_ratio"] = tv, ti, fr
    for name, d in results.items():
        d["grade"], d["reason"] = grade_one(d, results)


def inertia_components(inertia):
    return (
        inertia[0, 0],
        inertia[1, 1],
        inertia[2, 2],
        inertia[0, 1],
        inertia[0, 2],
        inertia[1, 2],
    )


def num(x):
    return f"{float(x):.6g}"


def diagonalize_inertia(inertia):
    eigvals, eigvecs = np.linalg.eigh(inertia)
    if np.linalg.det(eigvecs) < 0:
        eigvecs = eigvecs.copy()
        eigvecs[:, 0] = -eigvecs[:, 0]
    qx, qy, qz, qw = Rotation.from_matrix(eigvecs).as_quat()
    return eigvals, np.array([qw, qx, qy, qz])


def replace_mjcf_inertial(text, link, inertia):
    pat = re.compile(
        r'(<body\s+name="' + re.escape(link) + r'"[^>]*>\s*)(<inertial\b[^>]*?/>)',
        re.DOTALL,
    )

    def sub(m):
        prefix, old = m.group(1), m.group(2)
        pos_m = re.search(r'\bpos="([^"]+)"', old)
        mass_m = re.search(r'\bmass="([^"]+)"', old)
        if pos_m is None or mass_m is None:
            raise ValueError(f"{link}: MJCF inertial 缺少 pos/mass, 无法从文件取质心")
        diag, quat = diagonalize_inertia(inertia)
        indent = prefix[prefix.rfind("\n") + 1 :]
        quat_str = " ".join(f"{v:.6f}" for v in quat)
        diag_str = " ".join(num(v) for v in diag)
        return (
            f'{prefix}<inertial pos="{pos_m.group(1)}" quat="{quat_str}" '
            f'mass="{mass_m.group(1)}"\n{indent}diaginertia="{diag_str}" />'
        )

    return pat.subn(sub, text)


def replace_urdf_inertial(text, link, inertia):
    pat = re.compile(
        r'(<link\b[^>]*?\bname="'
        + re.escape(link)
        + r'"[^>]*?>\s*)(<inertial>.*?</inertial>)',
        re.DOTALL,
    )

    def sub(m):
        prefix, old = m.group(1), m.group(2)
        xyz_m = re.search(r'\bxyz="([^"]+)"', old)
        mass_m = re.search(r'\bvalue="([^"]+)"', old)
        if xyz_m is None or mass_m is None:
            raise ValueError(f"{link}: URDF inertial 缺少 xyz/mass, 无法从文件取质心")
        ixx, iyy, izz, ixy, ixz, iyz = inertia_components(inertia)
        return prefix + (
            "<inertial>\n"
            "      <origin\n"
            f'        xyz="{xyz_m.group(1)}"\n'
            '        rpy="0 0 0" />\n'
            "      <mass\n"
            f'        value="{mass_m.group(1)}" />\n'
            "      <inertia\n"
            f'        ixx="{num(ixx)}"\n'
            f'        ixy="{num(ixy)}"\n'
            f'        ixz="{num(ixz)}"\n'
            f'        iyy="{num(iyy)}"\n'
            f'        iyz="{num(iyz)}"\n'
            f'        izz="{num(izz)}" />\n'
            "    </inertial>"
        )

    return pat.subn(sub, text)


def write_back_files(results, script_dir):
    for fname, mjcf in (
        ("Semi_Taks_LV1.xml", True),
        ("scene_Semi_Taks_LV1.xml", True),
        ("Semi_Taks_LV1.urdf", False),
    ):
        path = os.path.join(script_dir, fname)
        if not os.path.exists(path):
            print(f"找不到 {path}, 跳过")
            continue
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        updated, absent, dup = 0, 0, []
        for name, d in results.items():
            fn = replace_mjcf_inertial if mjcf else replace_urdf_inertial
            text, n = fn(text, name, d["inertia_full"])
            if n == 1:
                updated += 1
            elif n == 0:
                absent += 1
            else:
                dup.append(name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"已写回 {path}: 更新 {updated} 个link, 该文件不含 {absent} 个")
        for name in dup:
            print(f"  跳过 {name}: 匹配多处")


def format_row(d):
    eig = np.sort(np.linalg.eigvalsh(d["inertia_full"]))
    extra = []
    if d["conv"] is not None and np.isfinite(d["conv"]):
        extra.append(f"收敛{d['conv'] * 100:.1f}%")
    if d["cross_com"] is not None:
        extra.append(
            f"互校{d['cross_com'] * 100:.1f}%/{(d['cross_i'] or 0) * 100:.1f}%"
        )
    if d.get("twin_vol") is not None:
        extra.append(f"镜像{d['twin_vol'] * 100:.0f}%/{d['face_ratio']:.1f}x")
    tail = ("  " + " ".join(extra)) if extra else ""
    return (
        f"[{d['grade']:<6}] {d['name']:36s} m={d['mass']:.4g}kg "
        f"com=[{d['com'][0]:.4f} {d['com'][1]:.4f} {d['com'][2]:.4f}] "
        f"I=[{eig[0]:.4g} {eig[1]:.4g} {eig[2]:.4g}] ({d['track']}){tail}"
    )


def report_lines(results):
    lines, low = [], []
    for name in LINK_WEIGHT_KG:
        d = results.get(name)
        if d is None:
            continue
        lines.append(format_row(d))
        if d["grade"] != "HIGH":
            lines.append(f"    -> {d['grade']}: {d['reason']}")
        if d["grade"] == "LOW" and "重导" in d["reason"]:
            low.append(name)
    return lines, low


def parse_args():
    p = argparse.ArgumentParser(
        description="Semi_Taks_LV1 link 惯量: 几何 + 实测质量 -> URDF/MJCF, 带逐link精度评级"
    )
    p.add_argument(
        "--no-write",
        dest="write",
        action="store_false",
        help="只计算不写回 (默认写回 .urdf 与 .xml 并生成 inertia_report.txt)",
    )
    p.set_defaults(write=True)
    return p.parse_args()


def main():
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    meshes_dir = os.path.join(script_dir, "meshes")

    items = [(n, m) for n, m in LINK_WEIGHT_KG.items() if m is not None]
    missing = [n for n, m in LINK_WEIGHT_KG.items() if m is None]
    workers = plan_workers(len(items))
    cap = max_cells(workers)
    tasks = [(n, m, meshes_dir, cap) for n, m in items]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            computed = list(ex.map(link_inertia, tasks))
    else:
        computed = [link_inertia(t) for t in tasks]
    results = {d["name"]: d for d in computed}
    grade_confidence(results)

    lines, low = report_lines(results)
    grades = {
        g: sum(1 for d in results.values() if d["grade"] == g)
        for g in ("HIGH", "MEDIUM", "LOW")
    }
    header = [
        "=" * 78,
        f"几何(解析/补洞/WN-体素)+实测质量 -> inertial  [workers={workers}]",
        (
            f"评级 HIGH={grades['HIGH']} MEDIUM={grades['MEDIUM']} LOW={grades['LOW']}; "
            f"已算{len(results)}个; 待称重{len(missing)}个"
        ),
        "=" * 78,
    ]
    for line in header + lines:
        print(line)
    if low:
        print("\n需CAD按右侧/全分辨率重导(几何已抽面/缺失, 任何惯量方法都无法恢复):")
        for n in low:
            print(f"  - {n}")
    if missing:
        print("\n待填 LINK_WEIGHT_KG:")
        for n in missing:
            print(f"  {n}")

    if args.write and results:
        write_back_files(results, script_dir)
        with open(
            os.path.join(script_dir, "inertia_report.txt"), "w", encoding="utf-8"
        ) as f:
            f.write("\n".join(header + lines) + "\n")
            if low:
                f.write("\n需CAD重导:\n" + "\n".join(f"  - {n}" for n in low) + "\n")
        print("已写 inertia_report.txt")
    return results


if __name__ == "__main__":
    main()
