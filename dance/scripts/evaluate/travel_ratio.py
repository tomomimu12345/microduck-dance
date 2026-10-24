"""Travel in body heights, human vs realized, raw 50 Hz path and 0.5 s moving-average path.

Each cell is human/robot=ratio for the raw path, then the same for the smoothed path. The human path is
in figure heights, the robot path in robot heights (0.25 m). The ratio is the measure behind the vel_gain rule.
    uv run python dance/scripts/evaluate/travel_ratio.py base vg0.0 vg0.3 --sim-dir out/fid001/sim
Tags and dump names are the same as in fidelity_eval.py.
"""
import argparse
from pathlib import Path

import numpy as np

from fidelity_eval import REPO, ROBOT_H, SONGS, dump_path, human_channels, path_length


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tags", nargs="+", help="dump tags; 'base' is the Task 15 baseline")
    ap.add_argument("--sim-dir", type=Path, default=REPO / "out/fid001/sim")
    ap.add_argument("--base-dir", type=Path, default=REPO / "out/rec_v4")
    ap.add_argument("--motion-dir", type=Path, default=REPO / "dance/data/motions/songs")
    a = ap.parse_args(argv)
    human = {s: human_channels(a.motion_dir / f"{s}.pkl") for s in SONGS}
    print("tag | " + " | ".join(SONGS) + "   (raw h/m=ratio smoothed h/m=ratio)")
    for tag in a.tags:
        cells = []
        for s in SONGS:
            p = dump_path(tag, s, a.sim_dir, a.base_dir)
            if not p.exists():
                cells.append("-")
                continue
            hch, fig_h = human[s]
            pos = np.load(p)["pos"].astype(float)
            parts = []
            for smooth in (False, True):
                h = path_length(hch["x"], hch["y"], smooth) / fig_h
                m = path_length(pos[:, 0], pos[:, 1], smooth) / ROBOT_H
                parts.append(f"{h:.1f}/{m:.1f}={m / h:.2f}")
            cells.append(" ".join(parts))
        print(tag, " | ".join(cells))


if __name__ == "__main__":
    main()
