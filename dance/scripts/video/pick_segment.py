"""Pick the liveliest window of a song from its retargeted command file.

Energy of a frame is the sum over the 13 command columns of (value / half
range)^2, with the training ranges of ranges.py. Columns whose range is 0
(body x, y, yaw) are skipped. The liveliest window is the one with the
largest mean energy, searched on a coarse grid of start times (0.5 s by
default) so the choice does not hinge on single frames.

With --need-phrase-change the window must contain a change from a groove
phrase (standing) to a travel phrase, at least --change-margin seconds inside
it. The phrases come from groove_info in the command file meta.

    uv run python dance/scripts/video/pick_segment.py dance/data/commands/songs/swag.npz --length 30
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from microduck_dance.command_file import load_command_file
from microduck_dance.ranges import ALL_RANGES


def command_energy(commands: np.ndarray, ranges=ALL_RANGES) -> np.ndarray:
    """(T,) energy per frame: sum of squared commands, each scaled by its half range."""
    commands = np.asarray(commands, dtype=float)
    half = np.array([(hi - lo) / 2.0 for lo, hi in ranges])
    cols = half > 0
    return np.sum((commands[:, cols] / half[cols]) ** 2, axis=1)


def groove_to_travel_times(meta: dict) -> list[float]:
    """Song times (s) where a groove phrase is followed by a travel phrase, from meta["groove_info"]."""
    ph = (meta.get("groove_info") or {}).get("phrases")
    if not ph:
        raise ValueError("command file has no groove_info phrases")
    travel, starts = ph["travel"], ph["start_times_s"]
    return [float(starts[i]) for i in range(1, len(travel)) if travel[i] and not travel[i - 1]]


def liveliest_window(energy: np.ndarray, dt: float, length_s: float, min_start_s: float = 0.0,
                     max_end_s: float | None = None, step_s: float = 0.5, must_contain=None,
                     margin_s: float = 3.0) -> tuple[float, float]:
    """(start_s, mean energy) of the length_s window with the highest mean energy.

    Starts are on a step_s grid within [min_start_s, max_end_s - length_s].
    If must_contain (a list of times) is given, only windows with one of those
    times in [start + margin_s, start + length_s - margin_s] count.
    """
    energy = np.asarray(energy, dtype=float)
    total_s = len(energy) * dt
    max_end_s = total_s if max_end_s is None else min(max_end_s, total_s)
    last_start = max_end_s - length_s
    if last_start < min_start_s - 1e-9:
        raise ValueError(f"no {length_s} s window fits in [{min_start_s}, {max_end_s}] s")
    n = int(round(length_s / dt))
    csum = np.concatenate([[0.0], np.cumsum(energy)])
    best = None
    for start in np.arange(min_start_s, last_start + 1e-9, step_s):
        if must_contain is not None and not any(
                start + margin_s - 1e-9 <= t <= start + length_s - margin_s + 1e-9 for t in must_contain):
            continue
        i = int(round(start / dt))
        mean = (csum[i + n] - csum[i]) / n
        if best is None or mean > best[1] + 1e-12:
            best = (round(float(start), 3), float(mean))
    if best is None:
        raise ValueError(f"no {length_s} s window in [{min_start_s}, {max_end_s}] s contains one of {must_contain}")
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("commands", type=Path, nargs="+")
    ap.add_argument("--length", type=float, default=30.0)
    ap.add_argument("--min-start", type=float, default=1.0, help="skip the command fade-in")
    ap.add_argument("--max-end", type=float, default=58.5, help="skip the song's fade-out")
    ap.add_argument("--step", type=float, default=0.5)
    ap.add_argument("--need-phrase-change", action="store_true",
                    help="the window must contain a groove to travel phrase change")
    ap.add_argument("--change-margin", type=float, default=3.0)
    a = ap.parse_args()
    out = {}
    for path in a.commands:
        commands, dt, meta = load_command_file(path)
        e = command_energy(commands)
        need = groove_to_travel_times(meta) if a.need_phrase_change else None
        start, mean = liveliest_window(e, dt, a.length, a.min_start, a.max_end, a.step, need, a.change_margin)
        out[path.stem] = {"start_s": start, "length_s": a.length, "mean_energy": round(mean, 4),
                          "song_mean_energy": round(float(e.mean()), 4)}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
