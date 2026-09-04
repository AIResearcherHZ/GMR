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

import glob
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import trimesh
from gpytoolbox import fast_winding_number
from scipy import ndimage
from skimage.measure import marching_cubes
from skimage.morphology import ball

warnings.filterwarnings("ignore")

FEATURE_MM_DEFAULT = (
    3.0  # 默认抹平的最大特征直径(mm): M3 螺纹/通孔/沉头; 转米后驱动闭运算半径
)
PITCH_DIV = (
    4  # 体素边长 h = feature_m / PITCH_DIV; 特征直径至少跨 4 体素才能稳定被桥接填平
)
VOX_PAD = 6  # 包围盒外留空体素层(>= 最大球半径), 防 dilation 越界且保证 erosion border_value=1 不啃外壁
VOX_RES_MAX = 384  # 每轴体素上限; 大件据此变粗 h
CLOSE_R_MIN = 1  # 闭运算球半径下限(体素)
CLOSE_R_MAX = 12  # 闭运算球半径上限(体素), 防小件被强行细分出超大结构元
TAUBIN_LAMB = 0.5  # Taubin 平滑收缩步
TAUBIN_NU = 0.53  # Taubin 平滑膨胀步; 满足 0 < 1/lamb - 1/nu < 0.1 => 体积保持去阶梯
TAUBIN_ITER = 12  # 默认 Taubin 迭代次数 (0=关闭)
VOL_TOL = 0.06  # 体积相对偏差红线(闭运算只增材, 正偏差为主); 超出 => LOW
COM_TOL = 0.01  # 质心偏移 / 包围盒对角线 红线; 超出 => LOW
INERTIA_TOL = 0.06  # 主惯量(I/质量)相对偏差红线; 超出 => LOW
SHAPE_TOL_FAC = 1.0  # 外形 Hausdorff(脱敏面->原始面) 容忍 = SHAPE_TOL_FAC * feature_m, 特征级以下不算破坏外形
KEEP_OPEN_MM = 8.0  # 脱敏面到原始面的最大距离若超过此值的一半, 判为封堵了大开口 => 告警并降级 MEDIUM
HAUSDORFF_SAMPLES = 60000  # 外形比对在脱敏面上的采样点数


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
    return max(2_000_000, int(available_memory() * 0.4 / workers / 48))


def bbox_diag(mesh):
    return float(np.linalg.norm(mesh.extents))


def principal_reldiff(ia, ib, ma=1.0, mb=1.0):
    ea = np.sort(np.linalg.eigvalsh(ia)) / ma
    eb = np.sort(np.linalg.eigvalsh(ib)) / mb
    return float(np.max(np.abs(ea - eb) / np.maximum(np.abs(eb), 1e-18)))


def grid_centers(origin, h, n):
    nx, ny, nz = (int(v) for v in n)
    ax = origin[0] + (np.arange(nx) + 0.5) * h
    ay = origin[1] + (np.arange(ny) + 0.5) * h
    az = origin[2] + (np.arange(nz) + 0.5) * h
    Q = np.empty((nx * ny * nz, 3), dtype=np.float64)
    Q[:, 0] = np.repeat(ax, ny * nz)
    Q[:, 1] = np.tile(np.repeat(ay, nz), nx)
    Q[:, 2] = np.tile(az, nx * ny)
    return Q


def choose_pitch(mesh, feature_m, pitch_override, cap):
    ext = np.asarray(mesh.extents, dtype=np.float64)
    if pitch_override > 0:
        h = pitch_override
    else:
        h = feature_m / PITCH_DIV
    h = max(
        h, float(ext.max()) / VOX_RES_MAX, (float(np.prod(ext)) / cap) ** (1.0 / 3.0)
    )
    return h


def close_radius(h, feature_m):
    return max(CLOSE_R_MIN, min(CLOSE_R_MAX, round((feature_m / 2.0) / h)))


def build_occupancy(mesh, h, pad):
    lo = np.asarray(mesh.bounds[0], dtype=np.float64) - pad * h
    hi = np.asarray(mesh.bounds[1], dtype=np.float64) + pad * h
    n = np.ceil((hi - lo) / h).astype(int) + 1
    Q = grid_centers(lo, h, n)
    w = fast_winding_number(
        Q, np.asarray(mesh.vertices), np.asarray(mesh.faces, dtype=np.int32)
    )
    occ = (np.rint(w) >= 1).reshape(tuple(int(v) for v in n))
    return occ, lo


def defeature(occ, r):
    se = ball(r)
    dil = ndimage.binary_dilation(occ, structure=se)
    ero = ndimage.binary_erosion(dil, structure=se, border_value=1)
    return ndimage.binary_fill_holes(ero)


def metrics_from_occ(occ, lo, h):
    idx = np.argwhere(occ)
    centers = lo + (idx + 0.5) * h
    com = centers.mean(axis=0)
    d = centers - com
    n = len(idx)
    mv = 1.0 / n
    s = h * h / 6.0
    ixx = mv * np.sum(d[:, 1] ** 2 + d[:, 2] ** 2) + s
    iyy = mv * np.sum(d[:, 0] ** 2 + d[:, 2] ** 2) + s
    izz = mv * np.sum(d[:, 0] ** 2 + d[:, 1] ** 2) + s
    ixy = -mv * np.sum(d[:, 0] * d[:, 1])
    ixz = -mv * np.sum(d[:, 0] * d[:, 2])
    iyz = -mv * np.sum(d[:, 1] * d[:, 2])
    inertia = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
    return com, inertia, n * h**3


def occ_to_mesh(occ, lo, h, taubin_iter):
    verts, faces, _, _ = marching_cubes(
        occ.astype(np.float32), level=0.5, spacing=(h, h, h)
    )
    mesh = trimesh.Trimesh(vertices=verts + lo + 0.5 * h, faces=faces, process=True)
    if taubin_iter > 0:
        trimesh.smoothing.filter_taubin(
            mesh, lamb=TAUBIN_LAMB, nu=TAUBIN_NU, iterations=taubin_iter
        )
    trimesh.repair.fix_normals(mesh)
    if mesh.volume < 0:
        mesh.invert()
    return mesh


def decimate_mesh(mesh, target):
    if target <= 0 or len(mesh.faces) <= target:
        return mesh
    try:
        import open3d as o3d

        om = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(np.asarray(mesh.vertices)),
            o3d.utility.Vector3iVector(np.asarray(mesh.faces)),
        )
        dec = om.simplify_quadric_decimation(target_number_of_triangles=int(target))
        m = trimesh.Trimesh(
            np.asarray(dec.vertices), np.asarray(dec.triangles), process=True
        )
        trimesh.repair.fix_normals(m)
        if m.volume < 0:
            m.invert()
        return m
    except Exception:
        return mesh


def genus(mesh):
    if mesh is None or not mesh.is_watertight:
        return None
    return round(mesh.body_count - mesh.euler_number / 2.0)


def shape_hausdorff(defeat_mesh, orig_mesh, nsamp):
    pts, _ = trimesh.sample.sample_surface(defeat_mesh, nsamp)
    _, dist, _ = orig_mesh.nearest.on_surface(pts)
    return float(dist.max())


def grade(d):
    d["feature_m"]
    if d["dvol"] > VOL_TOL or d["dcom"] > COM_TOL or d["di"] > INERTIA_TOL:
        worst = max(
            ("体积差%.1f%%" % (d["dvol"] * 100), d["dvol"] / VOL_TOL),
            ("质心偏移%.1f%%bbox" % (d["dcom"] * 100), d["dcom"] / COM_TOL),
            ("主惯量差%.1f%%" % (d["di"] * 100), d["di"] / INERTIA_TOL),
            key=lambda t: t[1],
        )[0]
        return "LOW", "质量属性已失真(" + worst + " 超红线), 勿用于惯量计算"
    if d["sealed_warn"]:
        return "MEDIUM", "疑似封堵了 D≈%.1fmm 的大开口(非螺纹孔), 请核对" % (
            d["shape"] * 2000
        )
    removed = (
        d["genus_before"] is not None
        and d["genus_after"] is not None
        and d["genus_after"] < d["genus_before"]
    ) or d["dvol"] > 0.002
    if not removed:
        return (
            "MEDIUM",
            "未检出特征被填平(--feature-mm 可能过小), 外形与质量保持但脱敏可能无效",
        )
    return "HIGH", "螺纹孔/小孔已填平, 外形与质量属性均在容忍内"


def desensitize_one(task):
    (
        name,
        src_path,
        out_path,
        feature_mm,
        pitch_override,
        taubin_iter,
        decimate,
        cap,
        dry,
    ) = task
    feature_m = feature_mm / 1000.0
    mesh = trimesh.load(src_path, force="mesh", process=True)
    h = choose_pitch(
        mesh, feature_m, pitch_override / 1000.0 if pitch_override else 0.0, cap
    )
    r_vox = close_radius(h, feature_m)
    occ_base, lo = build_occupancy(mesh, h, max(VOX_PAD, r_vox + 2))
    occ_def = defeature(occ_base, r_vox)
    com_b, I_b, vol_b = metrics_from_occ(occ_base, lo, h)
    com_d, I_d, vol_d = metrics_from_occ(occ_def, lo, h)
    defeat_mesh = occ_to_mesh(occ_def, lo, h, taubin_iter)
    shape = shape_hausdorff(defeat_mesh, mesh, HAUSDORFF_SAMPLES)
    genus_after = genus(defeat_mesh)
    export_mesh = decimate_mesh(defeat_mesh, decimate)
    d = {
        "name": name,
        "feature_m": feature_m,
        "h": h,
        "r_vox": r_vox,
        "dvol": (vol_d - vol_b) / vol_b,
        "dcom": float(np.linalg.norm(com_d - com_b)) / bbox_diag(mesh),
        "dcom_mm": float(np.linalg.norm(com_d - com_b)) * 1000.0,
        "di": principal_reldiff(I_d, I_b),
        "shape": shape,
        "sealed_warn": shape > (KEEP_OPEN_MM / 1000.0) / 2.0,
        "genus_before": genus(mesh),
        "genus_after": genus_after,
        "faces_before": len(mesh.faces),
        "faces_after": len(export_mesh.faces),
        "wt_after": bool(export_mesh.is_watertight),
    }
    d["grade"], d["reason"] = grade(d)
    if not dry:
        export_mesh.export(out_path)
    return d


def format_row(d):
    gb = d["genus_before"] if d["genus_before"] is not None else "?"
    ga = d["genus_after"] if d["genus_after"] is not None else "?"
    return (
        f"[{d['grade']:<6}] {d['name']:36s} feat={d['feature_m'] * 1000:.1f}mm "
        f"h={d['h'] * 1000:.2f}mm r={d['r_vox']}vox  dVol={d['dvol'] * 100:+.1f}% "
        f"dCom={d['dcom_mm']:.2f}mm dI={d['di'] * 100:.1f}% shape={d['shape'] * 1000:.2f}mm  "
        f"genus {gb}->{ga}  faces {d['faces_before']}->{d['faces_after']} wt={'Y' if d['wt_after'] else 'N'}"
    )


def parse_args():
    p = argparse.ArgumentParser(
        description="STL 脱敏: 抹平螺纹孔/小孔等敏感特征, 保外形与质量属性, 输出去敏 STL 副本(不覆盖原件)"
    )
    p.add_argument(
        "path", nargs="?", default=None, help="单个 .STL 或目录(默认: 脚本同级 meshes/)"
    )
    p.add_argument(
        "--out", default=None, help="输出目录(默认: 源目录同级 <名>_desens/)"
    )
    p.add_argument(
        "--feature-mm",
        type=float,
        default=FEATURE_MM_DEFAULT,
        help="要抹平的最大特征直径(mm), 默认 3.0",
    )
    p.add_argument(
        "--pitch-mm",
        type=float,
        default=0.0,
        help="体素边长(mm); 0=自适应 feature/4(变粗以适配大件)",
    )
    p.add_argument(
        "--res-max", type=int, default=VOX_RES_MAX, help="每轴体素上限, 默认 384"
    )
    p.add_argument(
        "--taubin-iter",
        type=int,
        default=TAUBIN_ITER,
        help="Taubin 平滑迭代, 默认 12 (0=关闭)",
    )
    p.add_argument(
        "--decimate-faces",
        type=int,
        default=0,
        help="目标面数, 0=不抽面; >0 用二次误差抽面",
    )
    p.add_argument(
        "--workers", type=int, default=0, help="并行进程数, 0=自适应(physical_cores)"
    )
    p.add_argument(
        "--no-report",
        dest="report",
        action="store_false",
        help="不写 desensitize_report.txt",
    )
    p.add_argument("--dry-run", action="store_true", help="只算指标与评级, 不写出 STL")
    p.set_defaults(report=True)
    return p.parse_args()


def main():
    global VOX_RES_MAX
    args = parse_args()
    VOX_RES_MAX = args.res_max
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = args.path or os.path.join(script_dir, "meshes")

    if os.path.isdir(path):
        src_dir = os.path.abspath(path.rstrip("/"))
        files = sorted(
            glob.glob(os.path.join(src_dir, "*.STL"))
            + glob.glob(os.path.join(src_dir, "*.stl"))
        )
    else:
        src_dir = os.path.dirname(os.path.abspath(path))
        files = [os.path.abspath(path)]
    if not files:
        print(f"未找到 STL: {path}")
        return {}

    out_dir = args.out or (src_dir + "_desens")
    if not args.dry_run:
        os.makedirs(out_dir, exist_ok=True)

    workers = args.workers or plan_workers(len(files))
    cap = max_cells(workers)
    tasks = [
        (
            os.path.splitext(os.path.basename(f))[0],
            f,
            os.path.join(out_dir, os.path.basename(f)),
            args.feature_mm,
            args.pitch_mm,
            args.taubin_iter,
            args.decimate_faces,
            cap,
            args.dry_run,
        )
        for f in files
    ]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(desensitize_one, tasks))
    else:
        results = [desensitize_one(t) for t in tasks]

    grades = {
        g: sum(1 for d in results if d["grade"] == g) for g in ("HIGH", "MEDIUM", "LOW")
    }
    header = [
        "=" * 78,
        f"STL 脱敏(绕组数体素->形态学闭运算->补洞->MC->Taubin) feature={args.feature_mm}mm  [workers={workers}]",
        f"评级 HIGH={grades['HIGH']} MEDIUM={grades['MEDIUM']} LOW={grades['LOW']}; 共 {len(results)} 件; 输出 {out_dir}",
        "注: 盲孔内螺纹将被填成实心圆柱, 此网格用于分享/脱敏, 不可用于复刻加工或紧固件规划",
        "=" * 78,
    ]
    lines = []
    for d in results:
        lines.append(format_row(d))
        if d["grade"] != "HIGH":
            lines.append(f"    -> {d['grade']}: {d['reason']}")
    for line in header + lines:
        print(line)
    bad = [d["name"] for d in results if d["grade"] == "LOW"]
    if bad:
        print(
            "\n质量属性失真(勿用于惯量; 小件可调小 --feature-mm, 或仅作分享用途接受该偏差):"
        )
        for n in bad:
            print(f"  - {n}")

    if args.report and not args.dry_run:
        with open(
            os.path.join(out_dir, "desensitize_report.txt"), "w", encoding="utf-8"
        ) as f:
            f.write("\n".join(header + lines) + "\n")
        print(f"\n已写 {os.path.join(out_dir, 'desensitize_report.txt')}")
    return results


if __name__ == "__main__":
    main()
