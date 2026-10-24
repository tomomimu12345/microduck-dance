"""Command file format for dance playback.

A command file is an npz with three keys.
- commands: float32 array (T, 13) in the dance command layout (see ranges.py)
- dt: scalar float, seconds between rows
- meta_json: JSON string with provenance (song, source file, retarget config)

meta_json is optional on load, so older files without it still play.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from microduck_dance.ranges import CMD_DIM


def save_command_file(path: str | Path, commands: np.ndarray, dt: float, meta: dict) -> None:
    commands = np.asarray(commands, dtype=np.float32)
    if commands.ndim != 2 or commands.shape[1] != CMD_DIM:
        raise ValueError(f"commands must have shape (T, {CMD_DIM}), got {commands.shape}")
    if not np.isfinite(commands).all():
        raise ValueError("commands contain non-finite values")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        np.savez(f, commands=commands, dt=np.float64(dt), meta_json=np.array(json.dumps(meta)))


def load_command_file(path: str | Path) -> tuple[np.ndarray, float, dict]:
    with np.load(path, allow_pickle=False) as data:
        commands = np.asarray(data["commands"])
        dt = float(data["dt"])
        meta = json.loads(str(data["meta_json"])) if "meta_json" in data.files else {}
    if commands.ndim != 2 or commands.shape[1] != CMD_DIM:
        raise ValueError(f"{path}: commands must have shape (T, {CMD_DIM}), got {commands.shape}")
    return commands, dt, meta
