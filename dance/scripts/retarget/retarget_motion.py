"""Retarget a generated SMPL dance (pkl) to a microduck command file (npz).

Usage:
    uv run dance/scripts/retarget/retarget_motion.py dance/data/motions/songs/<song>.pkl \
        dance/data/commands/songs/<song>.npz [--beats dance/data/beats/songs/<song>.json] [--vel-gain 0.5 ...]

The pkl must hold smpl_poses (F, 72) or (F, 24, 3) and smpl_trans (F, 3) at
30 fps. When the pkl also has full_pose (F, 24, 3) joint positions, as
FlowerDance writes, they give the up axis, pelvis and floor. Every RetargetConfig field is a flag. signs and max_rate take 13
comma separated numbers, head_gain 4 or 1. --groove takes 0 or 1.

--beats is the json from dance/scripts/generate/compute_beats.py. With it (and
--groove 1, the default) the beat groove layer is added. Without it the
output has head_gain but no groove layer. The json also carries the music
features, which turn on the phrase gate and the per-song groove pattern
(--phrase-gate, --groove-pattern). A json without them is an error unless
both are 0; then the shared groove of Task 11 is used. The chosen pattern
and phrases go into meta_json (groove_info).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pickle
from pathlib import Path

import numpy as np

from microduck_dance.command_file import save_command_file
from microduck_dance.groove import check_beats
from microduck_dance.retarget import RetargetConfig, detect_up_axis, smpl_to_commands

FPS = 30.0


def _floats(text: str) -> tuple[float, ...]:
    return tuple(float(v) for v in text.split(","))


def _bool(text: str) -> bool:
    v = text.strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected 0 or 1, got {text!r}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("motion_pkl", type=Path)
    ap.add_argument("out_npz", type=Path)
    ap.add_argument("--beats", type=Path, default=None, help="beat json from compute_beats.py")
    for f in dataclasses.fields(RetargetConfig):
        flag = "--" + f.name.replace("_", "-")
        if f.name in ("signs", "max_rate"):
            ap.add_argument(flag, type=_floats, default=None, help="13 comma separated numbers")
        elif f.name == "head_gain":
            ap.add_argument(flag, type=_floats, default=None, help="4 comma separated numbers or one number")
        elif f.type in (bool, "bool"):
            ap.add_argument(flag, type=_bool, default=None, help=f"0 or 1, default {int(f.default)}")
        else:
            ap.add_argument(flag, type=float, default=None, help=f"default {f.default}")
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    overrides = {
        f.name: getattr(args, f.name)
        for f in dataclasses.fields(RetargetConfig)
        if getattr(args, f.name) is not None
    }
    cfg = RetargetConfig(**overrides)

    with args.motion_pkl.open("rb") as fh:
        data = pickle.load(fh)
    poses = np.asarray(data["smpl_poses"], dtype=float)
    poses = poses.reshape(len(poses), 72)
    trans = np.asarray(data["smpl_trans"], dtype=float)
    joints = np.asarray(data["full_pose"], dtype=float) if "full_pose" in data else None

    beats, music = None, None
    if args.beats is not None:
        music = json.loads(args.beats.read_text())
        beats = check_beats(music["beat_times_s"])
        if not (cfg.phrase_gate or cfg.groove_pattern):
            music = None
    commands, info = smpl_to_commands(poses, trans, fps=FPS, cfg=cfg, joints=joints, beats=beats,
                                      music=music, return_info=True)
    meta = {
        "song": args.motion_pkl.stem,
        "source": str(args.motion_pkl),
        "fps": FPS,
        "num_frames": int(len(poses)),
        "up_axis": detect_up_axis(trans, joints),
        "pelvis_source": "full_pose" if joints is not None else "smpl_trans + SMPL_PELVIS_OFFSET",
        "retarget_config": dataclasses.asdict(cfg),
        "beats": str(args.beats) if args.beats is not None else None,
        "groove_applied": bool(beats is not None and cfg.groove),
        "groove_info": info,
    }
    save_command_file(args.out_npz, commands, cfg.out_dt, meta)
    print(f"wrote {args.out_npz}: {commands.shape[0]} steps, {commands.shape[0] * cfg.out_dt:.2f} s")


if __name__ == "__main__":
    main()
