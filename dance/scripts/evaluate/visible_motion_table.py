"""Visible-motion table from play_dance.py --dump-states files.

    uv run python dance/scripts/evaluate/visible_motion_table.py out/motion_v3/*.npz --json out/motion_v3/table.json

Each dump is named <policy>_<cmdset>_<song>.npz. The beat file is
dance/data/beats/songs/<song>.json. Prints a markdown table (whole clip) and writes
all statistics to --json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from microduck_dance.groove import load_beats
from microduck_dance.visible_motion import CHANNELS, visible_motion

SONGS = ("swag", "funky_magic", "fallin_for_you", "redwing_beats", "pastel_house")


def parse_name(stem: str) -> tuple[str, str, str]:
    for song in SONGS:
        if stem.endswith("_" + song):
            head = stem[: -len(song) - 1]
            pol, _, cs = head.rpartition("_")
            return pol, cs, song
    raise ValueError(f"cannot parse {stem}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dumps", type=Path, nargs="+")
    ap.add_argument("--beats-dir", type=Path, default=Path("dance/data/beats/songs"))
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()

    rows = []
    for p in sorted(a.dumps):
        pol, cs, song = parse_name(p.stem)
        d = np.load(p)
        s = visible_motion(d["cmd"], d["meas"], d["pos"], d["yaw"], float(d["dt"]),
                           beats=load_beats(a.beats_dir / f"{song}.json"))
        s.update(policy=pol, commands=cs, song=song, falls=int(np.sum(d["term"] & ~np.r_[False, d["term"][:-1]])))
        rows.append(s)

    print("| song | policy | cmds | falls | path m (smooth) / cmd | max disp m / cmd | sum abs yaw rad (smooth) / cmd "
          "| height p5-p95 mm / cmd | roll std | pitch std | head pitch std | head yaw std | nod lag s (cmd) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for s in rows:
        nl, nc = s["nod_lag_realized"], s["nod_lag_command"]
        print(f"| {s['song']} | {s['policy']} | {s['commands']} | {s['falls']} "
              f"| {s['path_m']:.2f} ({s['path_smooth_m']:.2f}) / {s['cmd_path_m']:.2f} "
              f"| {s['max_disp_m']:.2f} / {s['cmd_max_disp_m']:.2f} "
              f"| {s['yaw_abs_change_rad']:.1f} ({s['yaw_abs_change_smooth_rad']:.1f}) / {s['cmd_yaw_abs_change_rad']:.1f} "
              f"| {1000 * s['height_p5_p95_m']:.1f} / {1000 * s['cmd_height_p5_p95_m']:.1f} "
              f"| {s['roll_std_rad']:.3f} | {s['pitch_std_rad']:.3f} | {s['head_pitch_std_rad']:.3f} "
              f"| {s['head_yaw_std_rad']:.3f} | {nl['median_s']:+.2f} ({nc['median_s']:+.2f}) |")
    print()
    print("Realized std / commanded std per channel")
    print("| song | policy | cmds | " + " | ".join(CHANNELS) + " |")
    print("|---|---|---|" + "---|" * len(CHANNELS))
    for s in rows:
        print(f"| {s['song']} | {s['policy']} | {s['commands']} | "
              + " | ".join(f"{s['std_ratio'][c]:.2f}" for c in CHANNELS) + " |")
    print()
    print("Zero-lag correlation of realized with commanded per channel")
    print("| song | policy | cmds | " + " | ".join(CHANNELS) + " |")
    print("|---|---|---|" + "---|" * len(CHANNELS))
    for s in rows:
        print(f"| {s['song']} | {s['policy']} | {s['commands']} | "
              + " | ".join(f"{s['corr'][c]:.2f}" for c in CHANNELS) + " |")
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(rows, indent=2) + "\n")
        print(f"wrote {a.json}")


if __name__ == "__main__":
    main()
