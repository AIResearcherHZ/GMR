import json
import copy
import pathlib

HERE = pathlib.Path(__file__).resolve().parent.parent / "general_motion_retargeting" / "ik_configs"

SOURCES = ["bvh_lafan1", "bvh_mixamo", "smplx"]

UPPER_BODIES = [
    "waist_pitch_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "left_wrist_pitch_link",
    "right_wrist_pitch_link",
    "head_yaw_link",
]


def sync(src):
    semi_p = HERE / f"{src}_to_semi_taks_lv1.json"
    ch_p = HERE / f"{src}_to_semi_taks_lv1_chassis.json"
    if not semi_p.exists() or not ch_p.exists():
        print(f"[SKIP] {src}: missing config")
        return
    semi = json.load(open(semi_p))
    ch = json.load(open(ch_p))
    semi_st = semi["human_scale_table"]
    for k, v in semi_st.items():
        ch["human_scale_table"][k] = v
    spine1_src = "Spine2" if "Spine2" in semi_st else ("spine3" if "spine3" in semi_st else None)
    if spine1_src is not None:
        ch["human_scale_table"]["Spine1"] = semi_st[spine1_src]
    n = 0
    for t in ["ik_match_table1", "ik_match_table2"]:
        for body in UPPER_BODIES:
            if body in ch[t] and body in semi[t] and ch[t][body][0] == semi[t][body][0]:
                ch[t][body][3] = copy.deepcopy(semi[t][body][3])
                ch[t][body][4] = copy.deepcopy(semi[t][body][4])
                n += 1
    json.dump(ch, open(ch_p, "w"), indent=2, ensure_ascii=False)
    print(f"[OK] {src}: synced {n} upper-body offsets + scale from semi_taks_lv1 -> {ch_p.name}")


if __name__ == "__main__":
    for src in SOURCES:
        sync(src)