"""Per-dimension statistics of command files, to compare distributions.

Usage:
    uv run dance/scripts/retarget/library_stats.py dance/data/library.npz
    uv run dance/scripts/retarget/library_stats.py dance/data/commands/songs/swag.npz dance/data/commands/songs/funky_magic.npz ...

Several files are concatenated before the statistics. Columns are the mean of
|x|, the 95th percentile of |x| and the fraction of frames at a range limit.
"""

from __future__ import annotations

import sys

import numpy as np

from microduck_dance.command_file import load_command_file
from microduck_dance.ranges import ALL_RANGES

NAMES = ["vx", "vy", "wz", "neck_pitch", "head_pitch", "head_yaw", "head_roll",
         "body_x", "body_y", "body_z", "body_roll", "body_pitch", "body_yaw"]


def stats(commands: np.ndarray) -> np.ndarray:
    rows = []
    for d, (lo, hi) in enumerate(ALL_RANGES):
        x = commands[:, d]
        a = np.abs(x)
        if hi == lo:
            clip = float("nan")
        else:
            tol = 1e-3 * (hi - lo)
            clip = float(np.mean((x <= lo + tol) | (x >= hi - tol)))
        rows.append((a.mean(), np.percentile(a, 95), clip))
    return np.array(rows)


def main(paths: list[str]) -> None:
    cmds = np.concatenate([load_command_file(p)[0] for p in paths])
    s = stats(cmds)
    print(f"frames={len(cmds)}")
    print(f"{'dim':12s} {'mean|x|':>8s} {'p95|x|':>8s} {'clip':>7s}")
    for n, (m, p, c) in zip(NAMES, s):
        print(f"{n:12s} {m:8.4f} {p:8.4f} {c:7.3f}")


if __name__ == "__main__":
    main(sys.argv[1:])
