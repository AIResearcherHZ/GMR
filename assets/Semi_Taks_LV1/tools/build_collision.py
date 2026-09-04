#!/usr/bin/env python3
import argparse
import json
import math
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import trimesh
from scipy.spatial import ConvexHull, HalfspaceIntersection

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_XML = os.path.join(ROOT, "Semi_Taks_LV1.xml")
COLLISION_SUBDIR = "collision"
CONTAINMENT_TOL = 1e-6
SMOOTH_DIRS = 64
FLAT_FACE_FRAC = 0.02
MAX_FLAT_DIRS = 24
COPLANAR_DECIMALS = 3
MAX_DIRS = 128
PRIM_PREFERENCE = (
    ("sphere", 1.15),
    ("capsule", 1.30),
    ("cylinder", 1.15),
    ("box", 1.05),
)
PRIM_POKE_TOL = 1e-3
SDF_PITCH = 0.005
SDF_CLAIM = 0.85
SDF_RESIDUAL = 0.02
INSCRIBED_SEEDS = 14
INSCRIBED_MAX_PARTS = 6
INSCRIBED_MIN_GAIN = 0.04
INSCRIBED_TARGET = 0.95
SOLID_HULL = {"waist_pitch_link"}
MANUAL = {}
SPLIT_LINKS = {}
_CURRENT_LINK = [None]
MIN_HALF_AXIS = 5e-4
COACD_MAX_BLOCKS = 16
MERGE_NEIGHBORS = 4
PLANE_CHUNK = 8192
MC_SAMPLES = 200000
COINCIDENT_TOL = 1e-6
DEGENERATE_SV_TOL = 1e-9
ADJACENCY_HOPS = 2
CLIP_ROUNDS = 6
CLIP_TOL = 1e-9
SWEEP_GRID = 7
SWEEP_STEPS = 1500
MECHANISMS = (
    ("left_arm_long_link_motor_joint", "left_arm_short_link_motor_joint"),
    ("right_arm_long_link_motor_joint", "right_arm_short_link_motor_joint"),
    ("waist_left_motor_joint", "waist_right_motor_joint"),
    ("head_front_link_motor_joint", "head_rear_link_motor_joint"),
)

STRATEGY = {
    "base_link": ("coacd", {"parts": 4, "threshold": 0.008}),
    "waist_yaw_link": ("coacd", {"parts": 10, "threshold": 0.006}),
    "waist_right_link_driven_link": ("none", {}),
    "waist_left_link_driven_link": ("none", {}),
    "waist_roll_link": ("coacd", {"parts": 4, "threshold": 0.008}),
    "waist_pitch_link": ("coacd", {"parts": 14, "threshold": 0.006}),
    "waist_right_motor_link": ("none", {}),
    "waist_left_motor_link": ("none", {}),
    "right_shoulder_pitch_link": ("coacd", {"parts": 12, "threshold": 0.008}),
    "right_shoulder_roll_link": ("coacd", {"parts": 12, "threshold": 0.006}),
    "right_shoulder_yaw_link": ("coacd", {"parts": 12, "threshold": 0.006}),
    "right_elbow_link": ("coacd", {"parts": 16, "threshold": 0.006}),
    "right_wrist_roll_link": ("coacd", {"parts": 8, "threshold": 0.006}),
    "right_arm_long_link_motor_link": ("none", {}),
    "right_arm_long_link_active_link": ("none", {}),
    "right_arm_short_link_motor_link": ("none", {}),
    "right_arm_short_link_active_link": ("none", {}),
    "right_arm_long_link_bevel_gear_link": ("none", {}),
    "right_arm_short_link_bevel_gear_link": ("none", {}),
    "right_wrist_yaw_link": ("coacd", {"parts": 4, "threshold": 0.008}),
    "right_wrist_pitch_link": ("coacd", {"parts": 12, "threshold": 0.005}),
    "left_shoulder_pitch_link": ("coacd", {"parts": 12, "threshold": 0.008}),
    "left_shoulder_roll_link": ("coacd", {"parts": 12, "threshold": 0.006}),
    "left_shoulder_yaw_link": ("coacd", {"parts": 12, "threshold": 0.006}),
    "left_elbow_link": ("coacd", {"parts": 16, "threshold": 0.006}),
    "left_wrist_roll_link": ("coacd", {"parts": 8, "threshold": 0.006}),
    "left_arm_long_link_motor_link": ("none", {}),
    "left_arm_long_link_active_link": ("none", {}),
    "left_arm_short_link_motor_link": ("none", {}),
    "left_arm_short_link_active_link": ("none", {}),
    "left_arm_long_link_bevel_gear_link": ("none", {}),
    "left_arm_short_link_bevel_gear_link": ("none", {}),
    "left_wrist_yaw_link": ("coacd", {"parts": 4, "threshold": 0.008}),
    "left_wrist_pitch_link": ("coacd", {"parts": 12, "threshold": 0.005}),
    "head_front_link_motor_link": ("none", {}),
    "head_front_link_active_link": ("none", {}),
    "head_rear_link_motor_link": ("none", {}),
    "head_rear_link_active_link": ("none", {}),
    "head_front_link_bevel_gear_link": ("none", {}),
    "head_rear_link_bevel_gear_link": ("none", {}),
    "head_roll_link": ("coacd", {"parts": 4, "threshold": 0.008}),
    "head_pitch_link": ("coacd", {"parts": 4, "threshold": 0.008}),
    "head_yaw_link": ("coacd", {"parts": 10, "threshold": 0.012}),
}

PROBE_CANDIDATES = [
    ("sphere", {}),
    ("capsule", {"axis": "pca"}),
    ("cylinder", {"axis": "x"}),
    ("cylinder", {"axis": "y"}),
    ("cylinder", {"axis": "z"}),
    ("box", {}),
]

AXES = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}


def load_vertices(mesh_path):
    m = trimesh.load(mesh_path, force="mesh", process=False)
    return np.asarray(m.vertices, dtype=np.float64), np.asarray(m.faces, dtype=np.int32)


def _frame(v, axis):
    c = v.mean(axis=0)
    if axis == "pca":
        _, _, vt = np.linalg.svd(v - c, full_matrices=False)
        u = vt[0]
    else:
        u = np.asarray(AXES[axis], dtype=np.float64)
    return c, u / np.linalg.norm(u)


def _decompose(v, c, u):
    d = v - c
    t = d @ u
    perp = np.linalg.norm(d - np.outer(t, u), axis=1)
    return t, perp


def fit_capsule(v, axis="pca"):
    c, u = _frame(v, axis)
    t, perp = _decompose(v, c, u)
    r = float(perp.max())
    slack = np.sqrt(np.maximum(r * r - perp * perp, 0.0))
    ta = float((t + slack).min())
    tb = float((t - slack).max())
    if tb <= ta:
        mid = 0.5 * (t.min() + t.max())
        r = float(np.sqrt(((t - mid) ** 2 + perp * perp).max()))
        ta, tb = mid - MIN_HALF_AXIS, mid + MIN_HALF_AXIS
    return {
        "type": "capsule",
        "fromto": np.stack([c + ta * u, c + tb * u]),
        "size": (r,),
    }


def fit_cylinder(v, axis="z"):
    c, u = _frame(v, axis)
    t, perp = _decompose(v, c, u)
    r = float(perp.max())
    ta, tb = float(t.min()), float(t.max())
    if tb - ta < 1e-4:
        mid = 0.5 * (ta + tb)
        ta, tb = mid - MIN_HALF_AXIS, mid + MIN_HALF_AXIS
    return {
        "type": "cylinder",
        "fromto": np.stack([c + ta * u, c + tb * u]),
        "size": (r,),
    }


def fit_sphere(v):
    try:
        c, r = trimesh.nsphere.minimum_nsphere(np.asarray(v))
        c = np.asarray(c, dtype=np.float64).ravel()
        r = float(r)
    except Exception:
        c = 0.5 * (v.min(axis=0) + v.max(axis=0))
        r = 0.0
    slack = float(np.linalg.norm(v - c, axis=1).max())
    return {"type": "sphere", "pos": c, "size": (max(r, slack),)}


def _mat_to_quat(rot):
    if np.linalg.det(rot) < 0.0:
        rot = rot.copy()
        rot[:, 0] = -rot[:, 0]
    tr = rot[0, 0] + rot[1, 1] + rot[2, 2]
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        q = [
            0.25 * s,
            (rot[2, 1] - rot[1, 2]) / s,
            (rot[0, 2] - rot[2, 0]) / s,
            (rot[1, 0] - rot[0, 1]) / s,
        ]
    else:
        i = int(np.argmax(np.diag(rot)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(1.0 + rot[i, i] - rot[j, j] - rot[k, k]) * 2.0
        q = [0.0, 0.0, 0.0, 0.0]
        q[0] = (rot[k, j] - rot[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (rot[j, i] + rot[i, j]) / s
        q[1 + k] = (rot[k, i] + rot[i, k]) / s
    q = np.asarray(q, dtype=np.float64)
    return q / np.linalg.norm(q)


def fit_box(v):
    to_origin, extents = trimesh.bounds.oriented_bounds(np.asarray(v))
    world = np.linalg.inv(to_origin)
    rot = world[:3, :3]
    return {
        "type": "box",
        "pos": world[:3, 3].copy(),
        "quat": _mat_to_quat(rot),
        "size": tuple(float(e) * 0.5 for e in extents),
        "rot": rot,
    }


def _hull_planes(points):
    h = ConvexHull(points)
    return h.equations[:, :3], h.equations[:, 3]


def _max_plane_dist(points, a, b, chunk=PLANE_CHUNK):
    out = np.empty(len(points), dtype=np.float64)
    for i in range(0, len(points), chunk):
        s = points[i : i + chunk]
        out[i : i + chunk] = np.max(s @ a.T + b, axis=1)
    return out


def _fibonacci_dirs(n):
    i = np.arange(n, dtype=np.float64) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = math.pi * (1.0 + 5.0**0.5) * i
    return np.stack(
        [np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)],
        axis=1,
    )


def _dedup_dirs(dirs):
    dirs = np.asarray(dirs, dtype=np.float64)
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    _, uniq = np.unique(np.round(dirs, 9), axis=0, return_index=True)
    return dirs[np.sort(uniq)]


def _flat_dirs(points, hull):
    tri = points[hull.simplices]
    areas = 0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1
    )
    normals = hull.equations[:, :3]
    _, inv = np.unique(
        np.round(normals, COPLANAR_DECIMALS), axis=0, return_inverse=True
    )
    nfacet = int(inv.max()) + 1
    acc = np.zeros((nfacet, 3))
    np.add.at(acc, inv, normals * areas[:, None])
    w = np.zeros(nfacet)
    np.add.at(w, inv, areas)
    keep = np.flatnonzero(w > FLAT_FACE_FRAC * areas.sum())
    keep = keep[np.argsort(-w[keep])][:MAX_FLAT_DIRS]
    if keep.size == 0:
        return np.zeros((0, 3))
    return acc[keep] / np.linalg.norm(acc[keep], axis=1, keepdims=True)


def link_dirs(points, n_uniform=SMOOTH_DIRS):
    hull = ConvexHull(points)
    return _dedup_dirs(
        np.vstack(
            [
                _fibonacci_dirs(n_uniform),
                _flat_dirs(points, hull),
                np.eye(3),
                -np.eye(3),
            ]
        )
    )


def _is_degenerate(points):
    if len(points) < 4:
        return True
    d = points - points.mean(axis=0)
    s = np.linalg.svd(d, compute_uv=False)
    return len(s) < 3 or float(s[2]) < DEGENERATE_SV_TOL


def _absorb_degenerate(v, groups):
    solid, thin = [], []
    for idx in groups:
        (thin if _is_degenerate(v[idx]) else solid).append(idx)
    if not solid:
        raise RuntimeError("全部分量退化, 无法构造凸包")
    if not thin:
        return solid
    cents = np.stack([v[idx].mean(axis=0) for idx in solid])
    merged = [list(idx) for idx in solid]
    for idx in thin:
        j = int(np.argmin(np.linalg.norm(cents - v[idx].mean(axis=0), axis=1)))
        merged[j].extend(idx)
    return [np.unique(np.asarray(g, dtype=np.int64)) for g in merged]


def outer_polytope(points, dirs, clip=None):
    hull = ConvexHull(points)
    keep = points[hull.vertices]
    if len(hull.equations) <= len(dirs):
        return keep
    interior = keep.mean(axis=0)
    verts = keep
    for _ in range(CLIP_ROUNDS):
        offs = (points @ dirs.T).max(axis=0)
        if np.min(offs - dirs @ interior) <= 1e-12:
            return keep
        verts = np.asarray(
            HalfspaceIntersection(
                np.column_stack([dirs, -offs]), interior
            ).intersections
        )
        if clip is None:
            return verts
        ca, cb = clip
        over = np.max(verts @ ca.T + cb, axis=0)
        viol = np.flatnonzero(over > CLIP_TOL)
        if viol.size == 0:
            return verts
        budget = MAX_DIRS - len(dirs)
        if budget <= 0:
            return verts
        viol = viol[np.argsort(-over[viol])[:budget]]
        dirs = _dedup_dirs(np.vstack([dirs, ca[viol]]))
    return verts


def support(g, dirs):
    if g["type"] == "sphere":
        return dirs @ g["pos"] + g["size"][0]
    if g["type"] in ("capsule", "cylinder"):
        p0, p1 = g["fromto"]
        r = g["size"][0]
        base = np.maximum(dirs @ p0, dirs @ p1)
        if g["type"] == "capsule":
            return base + r
        u = p1 - p0
        u = u / np.linalg.norm(u)
        return base + r * np.linalg.norm(dirs - np.outer(dirs @ u, u), axis=1)
    if g["type"] == "box":
        return dirs @ g["pos"] + np.abs(dirs @ g["rot"]) @ np.asarray(g["size"])
    return np.max(dirs @ g["verts"].T, axis=1)


def protrusion(g, clip):
    if clip is None:
        return 0.0
    ca, cb = clip
    return float(np.max(support(g, ca) + cb))


def choose_geom(points, dirs, clip):
    ref = _hull_volume(points)
    if ref > 0.0:
        for kind, limit in PRIM_PREFERENCE:
            if kind == "sphere":
                g = fit_sphere(points)
            elif kind == "capsule":
                g = fit_capsule(points, "pca")
            elif kind == "cylinder":
                g = fit_cylinder(points, "pca")
            else:
                g = fit_box(points)
            if geom_volume(g) > limit * ref:
                continue
            if protrusion(g, clip) > PRIM_POKE_TOL:
                continue
            return g
    hull = trimesh.convex.convex_hull(outer_polytope(points, dirs, clip))
    return {
        "type": "mesh",
        "verts": np.asarray(hull.vertices, dtype=np.float64),
        "faces": np.asarray(hull.faces, dtype=np.int64),
    }


def _hull_pts(p):
    try:
        return p[ConvexHull(p).vertices]
    except Exception:
        return p


def _prim_dev(g, hull_pts):
    try:
        a, b = _hull_planes(hull_pts)
    except Exception:
        return float(np.max(outside_distance(g, hull_pts)))
    return float(np.max(support(g, a) + b))


def _axis_len(g):
    return float(np.linalg.norm(g["fromto"][1] - g["fromto"][0]))


def best_round(hull_pts):
    cands = [fit_sphere(hull_pts)]
    for fit in (fit_capsule, fit_cylinder):
        g = fit(hull_pts, "pca")
        if _axis_len(g) > 4.0 * MIN_HALF_AXIS:
            cands.append(g)
    return min(cands, key=lambda g: _prim_dev(g, hull_pts))


def fit_manual_capsule(v, q):
    hv = _hull_pts(v)
    c, u = _frame(hv, "pca")
    t, perp = _decompose(v, c, u)
    r = float(np.percentile(perp, q))
    ta, tb = float(t.min()) + r, float(t.max()) - r
    if tb <= ta:
        mid = 0.5 * (float(t.min()) + float(t.max()))
        ta, tb = mid - MIN_HALF_AXIS, mid + MIN_HALF_AXIS
    return {
        "type": "capsule",
        "fromto": np.stack([c + ta * u, c + tb * u]),
        "size": (r,),
    }


def fit_minimal(v, parts=1):
    hv = _hull_pts(v)
    ref = _hull_planes(hv)
    if parts <= 1:
        groups = [hv]
    else:
        from scipy.cluster.vq import kmeans2

        _c, lab = kmeans2(v, parts, minit="++", seed=0, iter=40)
        groups = [_hull_pts(v[lab == i]) for i in range(parts) if (lab == i).sum() >= 4]
        if not groups:
            groups = [hv]
    out = []
    for g in groups:
        cands = [fit_box(g), fit_capsule(g, "pca"), fit_sphere(g)]
        cands += [fit_cylinder(g, a) for a in ("x", "y", "z", "pca")]
        cands = [
            c for c in cands if "fromto" not in c or _axis_len(c) > 4.0 * MIN_HALF_AXIS
        ]
        out.append(min(cands, key=lambda c: protrusion(c, ref)))
    return out


def _seg_dist(pts, a, b):
    ab = b - a
    L2 = float(ab @ ab)
    if L2 < 1e-18:
        return np.linalg.norm(pts - a, axis=1)
    t = np.clip((pts - a) @ ab / L2, 0.0, 1.0)
    return np.linalg.norm(pts - (a + np.outer(t, ab)), axis=1)


def _inscribed_radius(mesh, a, b, samples=7):
    mid = a + np.outer(np.linspace(0.0, 1.0, samples), b - a)
    return float(trimesh.proximity.signed_distance(mesh, mid).min())


def inscribed_cover(v, f, pitch=SDF_PITCH, max_parts=INSCRIBED_MAX_PARTS):
    from scipy.spatial import cKDTree

    mesh = trimesh.Trimesh(v, f, process=False)
    if not mesh.is_watertight:
        mesh = trimesh.PointCloud(np.asarray(v)).convex_hull
    span = float(np.min(v.max(axis=0) - v.min(axis=0)))
    pitch = min(pitch, max(span / 8.0, 1e-3))
    lo = v.min(axis=0) - pitch
    hi = v.max(axis=0) + pitch
    grid = np.mgrid[lo[0] : hi[0] : pitch, lo[1] : hi[1] : pitch, lo[2] : hi[2] : pitch]
    pts = grid.reshape(3, -1).T
    dist = trimesh.proximity.signed_distance(mesh, pts)
    keep = dist > pitch * 0.5
    if keep.sum() < 2:
        return [best_round(_hull_pts(v))]
    pts, rad = pts[keep], dist[keep]

    tree = cKDTree(pts)
    live = np.ones(len(pts), dtype=bool)
    seeds = []
    while len(seeds) < INSCRIBED_SEEDS:
        j = int(np.argmax(np.where(live, rad, -1.0)))
        if rad[j] <= pitch:
            break
        live[tree.query_ball_point(pts[j], rad[j] * SDF_CLAIM)] = False
        seeds.append((pts[j], float(rad[j])))
        if live.mean() < SDF_RESIDUAL:
            break
    if not seeds:
        return [best_round(_hull_pts(v))]

    cands = [
        (
            {"type": "sphere", "pos": c, "size": (r,)},
            np.linalg.norm(pts - c, axis=1) <= r,
        )
        for c, r in seeds
    ]
    for i in range(len(seeds)):
        for k in range(i + 1, len(seeds)):
            a, b = seeds[i][0], seeds[k][0]
            r = _inscribed_radius(mesh, a, b)
            if r <= pitch:
                continue
            cands.append(
                (
                    {"type": "capsule", "fromto": np.stack([a, b]), "size": (r,)},
                    _seg_dist(pts, a, b) <= r,
                )
            )

    covered = np.zeros(len(pts), dtype=bool)
    floor = INSCRIBED_MIN_GAIN * len(pts)
    out = []
    while len(out) < max_parts:
        gains = [int((m & ~covered).sum()) for _g, m in cands]
        j = int(np.argmax(gains))
        if gains[j] < floor:
            break
        out.append(cands[j][0])
        covered |= cands[j][1]
        cands.pop(j)
        if covered.mean() >= INSCRIBED_TARGET or not cands:
            break
    return out or [best_round(_hull_pts(v))]


def fit_coacd(v, f, parts, threshold, seed=0, n_dirs=SMOOTH_DIRS):
    import coacd

    mesh = trimesh.Trimesh(v, f, process=False)
    mesh.merge_vertices()
    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.faces, dtype=np.int32)
    coacd.set_log_level("error")
    blocks = coacd.run_coacd(
        coacd.Mesh(v, f),
        threshold=threshold,
        max_convex_hull=COACD_MAX_BLOCKS,
        preprocess_mode="auto",
        merge=False,
        real_metric=True,
        seed=seed,
    )
    planes = [
        _hull_planes(np.asarray(bv, dtype=np.float64))
        for bv, _bf in blocks
        if len(bv) >= 4
    ]
    if not planes:
        raise RuntimeError("CoACD 未产生有效凸块")
    centers = v[f].mean(axis=1)
    labels = np.argmin(
        np.stack([_max_plane_dist(centers, a, b) for a, b in planes], axis=1), axis=1
    )
    groups = _absorb_degenerate(
        v,
        _merge_to_budget(v, _split_components(mesh, f, labels, len(planes)), parts),
    )
    return [hull_geom(v[idx]) for idx in groups]


def hull_geom(pts):
    hull = trimesh.convex.convex_hull(pts)
    return {
        "type": "mesh",
        "verts": np.asarray(hull.vertices, dtype=np.float64),
        "faces": np.asarray(hull.faces, dtype=np.int64),
    }


def _split_components(mesh, f, labels, nblocks):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    adjacency = mesh.face_adjacency
    same = labels[adjacency[:, 0]] == labels[adjacency[:, 1]]
    pair = adjacency[same]
    graph = coo_matrix(
        (np.ones(len(pair)), (pair[:, 0], pair[:, 1])), shape=(len(f), len(f))
    )
    _ncc, comp = connected_components(graph, directed=False)
    order = np.argsort(comp, kind="stable")
    bounds = np.flatnonzero(np.diff(comp[order])) + 1
    groups = [np.unique(f[part]) for part in np.split(order, bounds) if part.size]
    if not groups:
        raise RuntimeError("连通分量拆分为空")
    return groups


def _hull_volume(points):
    if len(points) < 4:
        return 0.0
    try:
        return float(ConvexHull(points).volume)
    except Exception:
        return 0.0


def _merge_to_budget(v, groups, budget):
    import heapq

    from scipy.spatial import cKDTree

    keys = [
        g[ConvexHull(v[g]).vertices] if _hull_volume(v[g]) > 0.0 else g for g in groups
    ]
    vols = [_hull_volume(v[k]) for k in keys]
    alive = set(range(len(keys)))
    if len(alive) <= budget:
        return [keys[i] for i in sorted(alive)]

    centers = np.stack([v[k].mean(axis=0) for k in keys])
    stamp = [0] * len(keys)
    heap = []

    def offer(a, b):
        merged = np.union1d(keys[a], keys[b])
        vol = _hull_volume(v[merged])
        heapq.heappush(heap, (vol - vols[a] - vols[b], vol, a, b, stamp[a], stamp[b]))

    tree = cKDTree(centers)
    knn = min(MERGE_NEIGHBORS + 1, len(keys))
    for a, nbrs in enumerate(tree.query(centers, k=knn)[1]):
        for b in np.atleast_1d(nbrs):
            if int(b) > a:
                offer(a, int(b))

    while len(alive) > budget:
        if not heap:
            live = sorted(alive)
            centers_live = np.stack([v[keys[i]].mean(axis=0) for i in live])
            tree = cKDTree(centers_live)
            knn = min(MERGE_NEIGHBORS + 1, len(live))
            for n, nbrs in enumerate(tree.query(centers_live, k=knn)[1]):
                for m in np.atleast_1d(nbrs):
                    if int(m) > n:
                        offer(live[n], live[int(m)])
            if not heap:
                break
        _cost, vol, a, b, sa, sb = heapq.heappop(heap)
        if a not in alive or b not in alive or stamp[a] != sa or stamp[b] != sb:
            continue
        merged = np.union1d(keys[a], keys[b])
        keys[a] = merged[ConvexHull(v[merged]).vertices] if vol > 0.0 else merged
        vols[a] = vol
        alive.discard(b)
        stamp[a] += 1
        center = v[keys[a]].mean(axis=0)
        live = [i for i in alive if i != a]
        if live:
            near = np.argsort(
                np.linalg.norm(
                    np.stack([v[keys[i]].mean(axis=0) for i in live]) - center, axis=1
                )
            )[:MERGE_NEIGHBORS]
            for n in near:
                offer(a, live[int(n)])
    return [keys[i] for i in sorted(alive)]


def build_geoms(kind, params, v, f, mode="inscribed"):
    if kind == "none":
        return []
    if _CURRENT_LINK[0] in SOLID_HULL:
        return [hull_geom(v)]
    manual = MANUAL.get(_CURRENT_LINK[0])
    if manual is not None:
        return [fit_manual_capsule(v, manual["q"])]
    if mode == "inscribed":
        return fit_minimal(v, SPLIT_LINKS.get(_CURRENT_LINK[0], 1))
    if kind == "coacd":
        return fit_coacd(v, f, params["parts"], params["threshold"])
    return [hull_geom(v)]
    raise ValueError(f"未知策略 {kind}")


def geom_volume(g):
    if g["type"] == "sphere":
        return 4.0 / 3.0 * math.pi * g["size"][0] ** 3
    if g["type"] == "capsule":
        r = g["size"][0]
        length = float(np.linalg.norm(g["fromto"][1] - g["fromto"][0]))
        return math.pi * r * r * length + 4.0 / 3.0 * math.pi * r**3
    if g["type"] == "cylinder":
        r = g["size"][0]
        length = float(np.linalg.norm(g["fromto"][1] - g["fromto"][0]))
        return math.pi * r * r * length
    if g["type"] == "box":
        return 8.0 * g["size"][0] * g["size"][1] * g["size"][2]
    return float(ConvexHull(g["verts"]).volume)


def outside_distance(g, v):
    if g["type"] == "sphere":
        return np.linalg.norm(v - g["pos"], axis=1) - g["size"][0]
    if g["type"] in ("capsule", "cylinder"):
        a, b = g["fromto"]
        ab = b - a
        length = float(np.linalg.norm(ab))
        u = ab / length
        t = (v - a) @ u
        perp = np.linalg.norm(v - a - np.outer(t, u), axis=1)
        r = g["size"][0]
        if g["type"] == "capsule":
            tc = np.clip(t, 0.0, length)
            d = np.linalg.norm(v - a - np.outer(tc, u), axis=1)
            return d - r
        over_r = np.maximum(perp - r, 0.0)
        over_t = np.maximum(np.maximum(-t, t - length), 0.0)
        return np.sqrt(over_r**2 + over_t**2)
    if g["type"] == "box":
        local = (v - g["pos"]) @ g["rot"]
        over = np.maximum(np.abs(local) - np.asarray(g["size"]), 0.0)
        return np.linalg.norm(over, axis=1)
    a, b = _hull_planes(g["verts"])
    return _max_plane_dist(v, a, b)


def containment_error(geoms, v):
    d = np.min(np.stack([outside_distance(g, v) for g in geoms], axis=1), axis=1)
    return float(np.max(d))


def coverage_stats(geoms, v, samples=MC_SAMPLES, seed=0):
    lo = v.min(axis=0) - 0.005
    hi = v.max(axis=0) + 0.005
    rng = np.random.default_rng(seed)
    p = rng.uniform(lo, hi, size=(samples, 3))
    box = float(np.prod(hi - lo))
    inside = np.zeros(samples, dtype=bool)
    for g in geoms:
        inside |= outside_distance(g, p) <= 0.0
    ha, hb = _hull_planes(v)
    hull_in = _max_plane_dist(p, ha, hb) <= 0.0
    return float(inside.mean() * box), float(hull_in.mean() * box)


def candidate_frames(model, name, mujoco):
    out = []
    b = int(model.body(name).id)

    def climb(start):
        q = int(start)
        while q != 0:
            qname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, q)
            if STRATEGY.get(qname, ("none", {}))[0] != "none":
                return qname
            q = int(model.body_parentid[q])
        return None

    first = climb(int(model.body_parentid[b]))
    if first:
        out.append(first)
    for e in range(model.neq):
        if model.eq_type[e] != mujoco.mjtEq.mjEQ_CONNECT:
            continue
        ids = (int(model.eq_obj1id[e]), int(model.eq_obj2id[e]))
        if b not in ids:
            continue
        other = ids[0] if ids[1] == b else ids[1]
        got = climb(other)
        if got:
            out.append(got)
    return list(dict.fromkeys(out))


_SWEEP_CACHE = {}


def _sweep_cached(xml_path, mesh_dir):
    key = (xml_path, mesh_dir)
    if key not in _SWEEP_CACHE:
        _SWEEP_CACHE[key] = sweep_internal(xml_path, mesh_dir)
    return _SWEEP_CACHE[key]


def sweep_internal(xml_path, mesh_dir):
    import mujoco

    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    act = {
        mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i for i in range(m.nu)
    }
    dropped = [k for k, (kind, _) in STRATEGY.items() if kind == "none"]
    if not dropped:
        return {}
    cands = {n: candidate_frames(m, n, mujoco) for n in dropped}
    verts = {n: load_vertices(os.path.join(mesh_dir, n + ".STL"))[0] for n in dropped}
    grids = []
    for pair in MECHANISMS:
        ids = [act[j] for j in pair]
        rngs = [m.actuator_ctrlrange[i] for i in ids]
        grids.append((ids, [np.linspace(lo, hi, SWEEP_GRID) for lo, hi in rngs]))
    acc = {n: {c: [] for c in cands[n]} for n in dropped}
    for ia in range(SWEEP_GRID):
        for ib in range(SWEEP_GRID):
            mujoco.mj_resetData(m, d)
            for ids, gs in grids:
                d.ctrl[ids[0]] = gs[0][ia]
                d.ctrl[ids[1]] = gs[1][ib]
            for _ in range(SWEEP_STEPS):
                mujoco.mj_step(m, d)
            for n in dropped:
                bi = int(m.body(n).id)
                rw = d.xmat[bi].reshape(3, 3)
                world = verts[n] @ rw.T + d.xpos[bi]
                for c in cands[n]:
                    ai = int(m.body(c).id)
                    ra = d.xmat[ai].reshape(3, 3)
                    acc[n][c].append((world - d.xpos[ai]) @ ra)
    out = {}
    for n in dropped:
        best, best_vol, best_hull = None, np.inf, None
        for c in cands[n]:
            hull = trimesh.convex.convex_hull(np.vstack(acc[n][c]))
            if hull.volume < best_vol:
                best, best_vol, best_hull = c, float(hull.volume), hull
        print(
            f"  {n:44s} -> {best:24s} 扫掠体 {best_vol * 1e6:8.1f} cm3"
            + (f"  (候选 {len(cands[n])} 选 1)" if len(cands[n]) > 1 else ""),
            flush=True,
        )
        out.setdefault(best, []).append(
            (
                np.asarray(best_hull.vertices, dtype=np.float64),
                np.asarray(best_hull.faces, dtype=np.int32),
            )
        )
    return out


def _job_cost(mesh_dir, name):
    kind, params = STRATEGY[name]
    if kind != "coacd":
        return 0.0
    path = os.path.join(mesh_dir, f"{name}.STL")
    size = os.path.getsize(path) if os.path.exists(path) else 0
    return size * float(params.get("parts", 1))


def fit_link(job):
    mesh_dir, mesh_name, extra, mode = job
    kind, params = STRATEGY[mesh_name]
    _CURRENT_LINK[0] = mesh_name
    v, f = load_vertices(os.path.join(mesh_dir, f"{mesh_name}.STL"))
    for ev, ef in extra:
        f = np.vstack([f, np.asarray(ef, dtype=np.int32) + len(v)])
        v = np.vstack([v, ev])
    t0 = time.perf_counter()
    geoms = build_geoms(kind, params, v, f, mode)
    seconds = time.perf_counter() - t0
    if not geoms:
        return {
            "mesh": mesh_name,
            "strategy": kind,
            "params": params,
            "geoms": [],
            "max_dev_mm": 0.0,
            "max_outside_mm": 0.0,
            "union_cm3": 0.0,
            "hull_cm3": 0.0,
            "inflation": 0.0,
            "src_faces": len(f),
            "hull_vertices": 0,
            "seconds": seconds,
        }
    union, hull_union = coverage_stats(geoms, v)
    link_hull = _hull_planes(_hull_pts(v))
    outside = 0.0 if mode == "inscribed" else containment_error(geoms, v) * 1e3
    return {
        "mesh": mesh_name,
        "strategy": kind,
        "max_dev_mm": max(protrusion(g, link_hull) for g in geoms) * 1e3,
        "params": params,
        "geoms": geoms,
        "max_outside_mm": outside,
        "union_cm3": union * 1e6,
        "hull_cm3": hull_union * 1e6,
        "inflation": union / hull_union,
        "src_faces": len(f),
        "hull_vertices": sum(
            len(g["verts"]) if g["type"] == "mesh" else 0 for g in geoms
        ),
        "seconds": seconds,
    }


def _fmt(x):
    return " ".join(f"{float(c):.6g}" for c in np.ravel(x))


def geom_to_attrib(g, name, mesh_name=None, mode="inscribed"):
    at = {"name": name, "type": g["type"]}
    if g["type"] in ("capsule", "cylinder"):
        at["fromto"] = _fmt(g["fromto"])
        at["size"] = _fmt(g["size"])
    elif g["type"] == "sphere":
        at["pos"] = _fmt(g["pos"])
        at["size"] = _fmt(g["size"])
    elif g["type"] == "box":
        at["pos"] = _fmt(g["pos"])
        at["quat"] = _fmt(g["quat"])
        at["size"] = _fmt(g["size"])
    else:
        at["mesh"] = mesh_name
    at["contype"] = "1"
    at["conaffinity"] = "1"
    at["group"] = "3"
    return at


def collect_targets(root):
    out = []
    for body in root.iter("body"):
        for i, geom in enumerate(list(body)):
            if geom.tag != "geom":
                continue
            if geom.get("group") != "3" or geom.get("type") != "mesh":
                continue
            name = geom.get("name")
            if not name:
                raise RuntimeError(f"body {body.get('name')} 的碰撞 geom 缺少 name")
            out.append((body, i, geom, name, geom.get("mesh")))
    return out


def probe(mesh_dir):
    print(f"{'mesh':46s} {'primitive':16s} {'inflate':>8s} {'maxout(mm)':>11s}")
    for link in sorted(STRATEGY):
        path = os.path.join(mesh_dir, f"{link}.STL")
        v, f = load_vertices(path)
        base = float(trimesh.convex.convex_hull(v).volume)
        for kind, params in PROBE_CANDIDATES:
            g = build_geoms(kind, params, v, f)
            vol = sum(geom_volume(x) for x in g)
            err = containment_error(g, v)
            tag = kind + ("/" + params["axis"] if "axis" in params else "")
            print(f"{link:46s} {tag:16s} {vol / base:8.2f} {err * 1e3:11.4f}")
        print()


def run(args, mode="inscribed"):
    xml_path = os.path.abspath(args.xml)
    root_dir = os.path.dirname(xml_path)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    compiler = root.find("compiler")
    mesh_dir = os.path.join(root_dir, compiler.get("meshdir", ""))
    asset = root.find("asset")
    targets = collect_targets(root)
    if args.only:
        keep = set(args.only)
        targets = [t for t in targets if t[4] in keep]
        if not targets:
            raise RuntimeError(f"--only 未匹配任何 link: {sorted(keep)}")

    missing = [n for _, _, _, _, n in targets if n not in STRATEGY]
    if missing:
        raise RuntimeError(f"策略表缺少 link: {sorted(set(missing))}")

    out_dir = os.path.join(mesh_dir, COLLISION_SUBDIR)
    if not args.audit:
        os.makedirs(out_dir, exist_ok=True)
    written = set()

    report = []
    new_assets = []
    edits = []
    total_parts = 0
    failures = []

    swept = {} if args.audit else _sweep_cached(xml_path, mesh_dir)
    if swept:
        print(
            "扫掠体归属: "
            + ", ".join(f"{k}({len(v)})" for k, v in sorted(swept.items())),
            flush=True,
        )
    jobs = [(mesh_dir, name, swept.get(name, []), mode) for _, _, _, _, name in targets]
    workers = max(1, min(args.jobs, len(jobs)))
    slow = sum(1 for j in jobs if STRATEGY[j[1]][0] == "coacd" and mode == "hull")
    print(
        f"[{mode}] {len(jobs)} 个 link ({slow} 个走 CoACD), {workers} 进程并行; "
        f"CoACD 单线程锁定, 大件需 1~2 分钟",
        flush=True,
    )
    results = [None] * len(jobs)
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        order = sorted(range(len(jobs)), key=lambda i: -_job_cost(mesh_dir, jobs[i][1]))
        futures = {pool.submit(fit_link, jobs[i]): i for i in order}
        for fut in as_completed(futures):
            i = futures[fut]
            res = fut.result()
            results[i] = res
            done += 1
            print(
                f"[{done:2d}/{len(jobs)}] {res['mesh']:46s} {res['strategy']:8s} "
                f"parts={len(res['geoms']):2d} "
                f"union/hull={res['inflation']:5.2f} "
                f"out={res['max_outside_mm']:8.4f}mm "
                f"dev={res['max_dev_mm']:6.1f}mm "
                f"verts={res['hull_vertices']:4d} {res['seconds']:6.2f}s",
                flush=True,
            )
    print()

    for (body, index, geom, geom_name, mesh_name), res in zip(targets, results):
        geoms = res["geoms"]
        if res["max_outside_mm"] > CONTAINMENT_TOL * 1e3:
            failures.append((mesh_name, res["max_outside_mm"]))
        total_parts += len(geoms)

        attribs = []
        for k, g in enumerate(geoms):
            name = geom_name if k == 0 else f"{geom_name}_{k}"
            asset_name = None
            if g["type"] == "mesh":
                asset_name = f"{mesh_name}_col{k}"
                rel = f"{COLLISION_SUBDIR}/{asset_name}.STL"
                new_assets.append((asset_name, rel))
                if not args.audit:
                    trimesh.Trimesh(
                        vertices=g["verts"], faces=g["faces"], process=False
                    ).export(os.path.join(mesh_dir, rel))
                    written.add(f"{asset_name}.STL")
            attribs.append(geom_to_attrib(g, name, asset_name, mode))
        edits.append((body, index, geom, attribs))

        entry = {k: val for k, val in res.items() if k != "geoms"}
        entry["geom"] = geom_name
        entry["parts"] = len(geoms)
        entry["types"] = sorted({g["type"] for g in geoms})
        entry["names"] = [a["name"] for a in attribs]
        report.append(entry)

    print(f"碰撞体总数 {total_parts} (原 {len(targets)} 个 mesh 碰撞体)")
    src_faces = sum(r["src_faces"] for r in report)
    print(f"原碰撞网格三角形总数 {src_faces}")
    print(f"新增凸包顶点总数 {sum(r['hull_vertices'] for r in report)}")
    if failures:
        for name, err in failures:
            print(f"包含性失败 {name}: 最大外露 {err:.4f} mm", file=sys.stderr)
        raise RuntimeError("碰撞体未完全包含原网格")

    devs = sorted(((r["max_dev_mm"], r["mesh"]) for r in report), reverse=True)
    if devs:
        label = "鼓出零件最多的 link" if mode == "inscribed" else "外扩最大的 link"
        print(f"{label}: " + ", ".join(f"{n} {d:+.1f}mm" for d, n in devs[:5]))

    if args.report:
        stem, ext = os.path.splitext(args.report)
        with open(f"{stem}_{mode}{ext}", "w") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
        print(f"报告 -> {stem}_{mode}{ext}")

    if args.audit:
        return

    for name, rel in new_assets:
        ET.SubElement(asset, "mesh", {"name": name, "file": rel})

    for body, _index, geom, attribs in edits:
        pos = list(body).index(geom)
        body.remove(geom)
        for offset, at in enumerate(attribs):
            body.insert(pos + offset, ET.Element("geom", at))

    if mode == "hull":
        for stale in sorted(set(os.listdir(out_dir)) - written):
            os.remove(os.path.join(out_dir, stale))
    add_exclusions(root, xml_path)

    ET.indent(tree, space="  ")
    out_xml = xml_path if args.in_place else os.path.join(root_dir, args.out)
    with open(out_xml, "w") as fh:
        fh.write("<?xml version='1.0' encoding='utf-8'?>\n")
        fh.write(ET.tostring(root, encoding="unicode"))
        fh.write("\n")
    print(f"XML -> {out_xml}")
    verify(out_xml)
    if selfcheck(out_xml) and mode == "hull":
        raise RuntimeError("生成的碰撞体在零位存在非相邻自干涉")


def add_exclusions(root, src_xml):
    import mujoco

    model = mujoco.MjModel.from_xml_path(src_xml)
    near = near_bodies(model, mujoco)
    name = {
        b: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)
        for b in range(model.nbody)
    }
    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    pairs = sorted(
        {
            (min(b, o), max(b, o))
            for b, group in near.items()
            for o in group
            if o != b and 0 not in (b, o)
        }
    )
    for b, o in pairs:
        if name[b] and name[o]:
            ET.SubElement(contact, "exclude", {"body1": name[b], "body2": name[o]})
    print(f"写入 {len(pairs)} 组相邻 body exclude")


def near_bodies(model, mujoco):
    adj = {b: [] for b in range(model.nbody)}
    for b in range(1, model.nbody):
        p = int(model.body_parentid[b])
        w = 0 if float(np.linalg.norm(model.body_pos[b])) < COINCIDENT_TOL else 1
        adj[b].append((p, w))
        adj[p].append((b, w))
    for e in range(model.neq):
        if model.eq_type[e] == mujoco.mjtEq.mjEQ_CONNECT:
            b1, b2 = int(model.eq_obj1id[e]), int(model.eq_obj2id[e])
            adj[b1].append((b2, 1))
            adj[b2].append((b1, 1))
    near = {}
    for b in range(model.nbody):
        dist = {b: 0}
        queue = [b]
        while queue:
            cur = queue.pop()
            for nxt, w in adj[cur]:
                nd = dist[cur] + w
                if nd <= ADJACENCY_HOPS and nd < dist.get(nxt, 99):
                    dist[nxt] = nd
                    queue.append(nxt)
        near[b] = set(dist)
    return near


def selfcheck(model_path):
    import mujoco

    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    gids = [i for i in range(model.ngeom) if model.geom_group[i] == 3]
    near = near_bodies(model, mujoco)
    ft = np.zeros(6)
    hits = []
    for n, i in enumerate(gids):
        bi = model.geom_bodyid[i]
        for j in gids[n + 1 :]:
            bj = model.geom_bodyid[j]
            if bj in near[bi]:
                continue
            dist = mujoco.mj_geomDistance(model, data, i, j, 1.0, ft)
            if dist < 0.0:
                hits.append(
                    (
                        dist,
                        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i),
                        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, j),
                    )
                )
    hits.sort()
    print(f"零位自干涉检查: 非相邻碰撞对互穿 {len(hits)} 处")
    for dist, a, b in hits[:12]:
        print(f"  {dist * 1e3:8.1f} mm  {a} <-> {b}")
    return hits


def verify(xml_path):
    import mujoco

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    group3 = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        for i in range(model.ngeom)
        if model.geom_group[i] == 3
    ]
    print(f"MuJoCo 编译通过: ngeom={model.ngeom} group3={len(group3)} nq={model.nq}")
    for legacy in ("torso_collision", "pelvis_collision", "head_collision"):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, legacy)
        if gid < 0:
            raise RuntimeError(f"缺少下游依赖的 geom {legacy}")
    print("下游依赖 geom 名齐全")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default=DEFAULT_XML)
    ap.add_argument("--out", default="Semi_Taks_LV1_collision.xml")
    ap.add_argument("--mode", choices=("hull", "inscribed"), default="hull")
    ap.add_argument("--in-place", action="store_true")
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--report", default=None)
    args = ap.parse_args()
    if args.probe:
        compiler = ET.parse(args.xml).getroot().find("compiler")
        probe(
            os.path.join(
                os.path.dirname(os.path.abspath(args.xml)),
                compiler.get("meshdir", ""),
            )
        )
        return
    run(args, args.mode)


if __name__ == "__main__":
    main()
