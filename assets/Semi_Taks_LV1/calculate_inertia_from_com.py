import argparse
import os
import re
from concurrent.futures import ProcessPoolExecutor

import calculate_inertia_from_stl as geom
import numpy as np
import trimesh

LINKS = [
    "base_link",
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
    "right_arm_long_link_driven_link",
    "right_arm_short_link_bevel_gear_link",
    "right_arm_short_link_driven_link",
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
    "left_arm_long_link_driven_link",
    "left_arm_short_link_bevel_gear_link",
    "left_arm_short_link_driven_link",
]

num = geom.num


def parse_vec(s):
    v = np.array([float(t) for t in s.split()], dtype=np.float64)
    if v.shape != (3,):
        raise ValueError(f"质心字段不是三维向量: {s!r}")
    return v


def mesh_unit_inertia(task):
    name, meshes_dir, cap = task
    mesh = trimesh.load(os.path.join(meshes_dir, name + ".STL"), process=True)
    com_mesh, inertia_pm, track, _diag = geom.compute_inertia_from_mesh(mesh, 1.0, cap)
    return {
        "name": name,
        "com_mesh": np.asarray(com_mesh, dtype=np.float64),
        "inertia_pm": np.asarray(inertia_pm, dtype=np.float64),
        "track": track,
    }


def inertia_at_file_com(link, unit, mass, com_file):
    d = com_file - unit["com_mesh"]
    shift = float(d @ d) * np.eye(3) - np.outer(d, d)
    inertia = mass * (unit["inertia_pm"] + shift)
    geom.validate_inertia(link, inertia)
    return inertia, float(np.linalg.norm(d))


def replace_mjcf_inertial(text, link, unit, rows):
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
        mass = float(mass_m.group(1))
        com = parse_vec(pos_m.group(1))
        inertia, dist = inertia_at_file_com(link, unit, mass, com)
        rows.setdefault(link, (mass, com, dist, inertia))
        diag, quat = geom.diagonalize_inertia(inertia)
        indent = prefix[prefix.rfind("\n") + 1 :]
        quat_str = " ".join(f"{v:.6f}" for v in quat)
        diag_str = " ".join(num(v) for v in diag)
        return (
            f'{prefix}<inertial pos="{pos_m.group(1)}" quat="{quat_str}" '
            f'mass="{mass_m.group(1)}"\n{indent}diaginertia="{diag_str}" />'
        )

    return pat.subn(sub, text)


def replace_urdf_inertial(text, link, unit, rows):
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
        mass = float(mass_m.group(1))
        com = parse_vec(xyz_m.group(1))
        inertia, dist = inertia_at_file_com(link, unit, mass, com)
        rows.setdefault(link, (mass, com, dist, inertia))
        ixx, iyy, izz, ixy, ixz, iyz = geom.inertia_components(inertia)
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


def process_file(path, mjcf, units, rows, write):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    updated, absent, dup = 0, 0, []
    fn = replace_mjcf_inertial if mjcf else replace_urdf_inertial
    for name in LINKS:
        text, n = fn(text, name, units[name], rows)
        if n == 1:
            updated += 1
        elif n == 0:
            absent += 1
        else:
            dup.append(name)
    if dup:
        raise ValueError(f"{path}: {', '.join(dup)} 匹配多处, 拒绝写回")
    if write:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"已写回 {path}: 更新 {updated} 个link, 该文件不含 {absent} 个")
    else:
        print(f"[dry-run] {path}: 可更新 {updated} 个link, 该文件不含 {absent} 个")
    return updated


def parse_args():
    p = argparse.ArgumentParser(
        description="Semi_Taks_LV1 link 惯量: 保持文件质心/质量不变, "
        "由 STL 几何算单位质量惯量, 平行轴定理搬到文件质心后写回 URDF/MJCF"
    )
    p.add_argument(
        "--no-write",
        dest="write",
        action="store_false",
        help="只计算不写回 (默认写回 .urdf 与 .xml 并生成 inertia_com_report.txt)",
    )
    p.set_defaults(write=True)
    return p.parse_args()


def main():
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    meshes_dir = os.path.join(script_dir, "meshes")

    workers = geom.plan_workers(len(LINKS))
    cap = geom.max_cells(workers)
    tasks = [(n, meshes_dir, cap) for n in LINKS]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            computed = list(ex.map(mesh_unit_inertia, tasks))
    else:
        computed = [mesh_unit_inertia(t) for t in tasks]
    units = {d["name"]: d for d in computed}

    rows = {}
    total = 0
    for fname, mjcf in (
        ("Semi_Taks_LV1.xml", True),
        ("scene_Semi_Taks_LV1.xml", True),
        ("Semi_Taks_LV1.urdf", False),
    ):
        path = os.path.join(script_dir, fname)
        if not os.path.exists(path):
            print(f"找不到 {path}, 跳过")
            continue
        total += process_file(path, mjcf, units, rows, args.write)
    if total == 0:
        raise ValueError("所有文件均未匹配到任何 link inertial, 请检查文件内容")

    header = [
        "=" * 78,
        "STL几何惯量 + 文件质心/质量(pos/mass 保持原值), 平行轴定理搬移到文件质心",
        f"已算 {len(rows)} 个link; 名单中缺席 {len(LINKS) - len(rows)} 个",
        "=" * 78,
    ]
    lines = []
    for name in LINKS:
        if name not in rows:
            continue
        mass, com, dist, inertia = rows[name]
        eig = np.sort(np.linalg.eigvalsh(inertia))
        lines.append(
            f"{name:36s} m={mass:.4g}kg "
            f"com=[{com[0]:.4f} {com[1]:.4f} {com[2]:.4f}] "
            f"|Δcom|={dist:.4f}m I=[{eig[0]:.4g} {eig[1]:.4g} {eig[2]:.4g}] "
            f"({units[name]['track']})"
        )
    for line in header + lines:
        print(line)

    if args.write:
        with open(
            os.path.join(script_dir, "inertia_com_report.txt"), "w", encoding="utf-8"
        ) as f:
            f.write("\n".join(header + lines) + "\n")
        print("已写 inertia_com_report.txt")
    return rows


if __name__ == "__main__":
    main()
