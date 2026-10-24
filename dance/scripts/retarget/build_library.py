"""Concatenate retargeted command files into one library npz.

Usage:
    uv run dance/scripts/retarget/build_library.py dance/data/commands/training/*.npz \
        --out dance/data/library.npz

All inputs must share dt. Each clip fades in from and out to zero command
(see retarget.py), so the joins between clips stay smooth.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from microduck_dance.command_file import load_command_file, save_command_file


def build_library(npz_paths: list[Path], out: Path) -> int:
    """Concatenate the commands of all files in time and save one npz.

    Returns the total number of frames.
    """
    if not npz_paths:
        raise ValueError("no input files")
    parts, dts, lengths = [], [], []
    for p in npz_paths:
        commands, dt, _ = load_command_file(p)
        parts.append(commands)
        dts.append(dt)
        lengths.append(len(commands))
    if max(dts) - min(dts) > 1e-9:
        raise ValueError(f"inputs have different dt: {sorted(set(dts))}")
    commands = np.concatenate(parts, axis=0)
    meta = {
        "kind": "library",
        "sources": [Path(p).name for p in npz_paths],
        "clip_lengths": lengths,
        "num_clips": len(parts),
    }
    save_command_file(out, commands, dts[0], meta)
    return int(len(commands))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    n = build_library(sorted(args.npz), args.out)
    _, dt, _ = load_command_file(args.out)
    print(f"wrote {args.out}: {len(args.npz)} clips, {n} frames, {n * dt / 60:.1f} min")


if __name__ == "__main__":
    main()
