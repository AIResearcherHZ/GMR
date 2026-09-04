#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import struct
from pathlib import Path

NEUTRAL_HEADER = (b"stripped by strip_mesh_color" + b"\x00" * 80)[:80]
DEFAULT_DIR = "taks_level1/assets/Semi_Taks_LV1/meshes"


def stl_is_binary(data):
    if len(data) < 84:
        return False
    n = struct.unpack_from("<I", data, 80)[0]
    return 84 + n * 50 == len(data)


def strip_stl(data):
    if not stl_is_binary(data):
        return False, data
    buf = bytearray(data)
    changed = False
    header = bytes(buf[0:80])
    if b"COLOR=" in header or b"MATERIAL=" in header or header[:5].lower() == b"solid":
        buf[0:80] = NEUTRAL_HEADER
        changed = True
    n = struct.unpack_from("<I", buf, 80)[0]
    for i in range(n):
        off = 84 + i * 50 + 48
        if buf[off] != 0 or buf[off + 1] != 0:
            buf[off] = 0
            buf[off + 1] = 0
            changed = True
    return changed, bytes(buf)


def strip_obj(text):
    out = []
    changed = False
    for ln in text.splitlines():
        s = ln.split()
        if s and s[0].lower() in ("mtllib", "usemtl"):
            changed = True
            continue
        if s and s[0] == "v" and len(s) == 7:
            out.append("v " + " ".join(s[1:4]))
            changed = True
            continue
        out.append(ln)
    return changed, "\n".join(out) + "\n"


def backup(src, root, backup_dir):
    dst = backup_dir / src.relative_to(root)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        dst.write_bytes(src.read_bytes())


def main():
    ap = argparse.ArgumentParser(description="删除文件夹内所有 STL/OBJ 的颜色信息")
    ap.add_argument("folder", nargs="?", default=str(DEFAULT_DIR))
    ap.add_argument("--backup-dir", default=None)
    ap.add_argument("--no-backup", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = Path(args.folder).resolve()
    if not root.is_dir():
        raise SystemExit(f"不是文件夹: {root}")
    backup_dir = (
        Path(args.backup_dir).resolve() if args.backup_dir else root / "_color_backup"
    )

    stl_n = obj_n = skip = 0
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if Path(dp, d).resolve() != backup_dir]
        for fn in fns:
            p = Path(dp, fn)
            ext = p.suffix.lower()
            if ext not in (".stl", ".obj"):
                continue
            if ext == ".stl":
                changed, new = strip_stl(p.read_bytes())
            else:
                changed, new = strip_obj(p.read_text(errors="ignore"))
            if not changed:
                skip += 1
                continue
            tag = "STL" if ext == ".stl" else "OBJ"
            if args.dry_run:
                print(f"[dry] {tag} 去色: {p.relative_to(root)}")
            else:
                if not args.no_backup:
                    backup(p, root, backup_dir)
                if ext == ".stl":
                    p.write_bytes(new)
                else:
                    p.write_text(new)
                print(f"{tag} 去色: {p.relative_to(root)}")
            if ext == ".stl":
                stl_n += 1
            else:
                obj_n += 1

    where = (
        "(dry-run, 未写入)"
        if args.dry_run
        else ("(未备份)" if args.no_backup else f"(原件已备份到 {backup_dir})")
    )
    print(f"\n完成: STL 去色 {stl_n}, OBJ 去色 {obj_n}, 无颜色跳过 {skip} {where}")


if __name__ == "__main__":
    main()
