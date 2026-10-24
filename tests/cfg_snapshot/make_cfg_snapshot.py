"""Write a JSON snapshot of the dance task cfgs, for tests/test_cfg_snapshot.py.

The snapshot was made once from the fork the dance tasks came from (the pai-dance
branch of microduck-playground, where they lived in mjlab_microduck) and is
compared with the cfgs this repository builds. The same script runs in both
trees: tasks are looked up by id in the mjlab registry, module names of our code
are mapped to "OURS", and file paths are written relative to the repository root
or to the upstream package.

    uv run python tests/cfg_snapshot/make_cfg_snapshot.py --out x.json --obs-dims   # also builds each env on CPU

The committed snapshot.json was written with --obs-dims by this script in the old
tree (pai-dance at 3dccc96, run from its repository root).
"""

from __future__ import annotations

import argparse
import dataclasses
import enum
import functools
import json
import math
import types
from pathlib import Path

import mjlab  # noqa: F401  (imports the task packages registered under the mjlab.tasks entry point)
import mjlab_microduck
import numpy as np
import torch
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls

TASKS = (
    "Mjlab-Dance-Flat-MicroDuck",
    "Mjlab-DanceLib-Flat-MicroDuck",
    "Mjlab-DanceV3-Flat-MicroDuck",
    "Mjlab-DanceV3-Rough-MicroDuck",
    "Mjlab-DanceV3-RoughMix-MicroDuck",
    "Mjlab-BallDance-MicroDuck",
    "Mjlab-RollerDance-MicroDuck",
)
# Tasks whose env is built on CPU to record the obs term dims.
OBS_DIM_TASKS = (
    "Mjlab-DanceV3-Flat-MicroDuck",
    "Mjlab-DanceV3-Rough-MicroDuck",
    "Mjlab-DanceV3-RoughMix-MicroDuck",
    "Mjlab-BallDance-MicroDuck",
    "Mjlab-RollerDance-MicroDuck",
)

# Our modules in the old tree and in this repository, mapped to one name.
_OLD_TASK_MODULES = (
    "dance_commands",
    "dance_rollers",
    "dance_v3",
    "microduck_ball_dance_env_cfg",
    "microduck_dance_env_cfg",
    "microduck_dance_v3_env_cfg",
    "microduck_roller_dance_env_cfg",
)


def canon_module(name: str) -> str:
    for m in _OLD_TASK_MODULES:
        for prefix in (f"mjlab_microduck.tasks.{m}", f"microduck_dance.tasks.{m}"):
            if name == prefix or name.startswith(prefix + "."):
                return f"OURS.tasks.{m}" + name[len(prefix):]
    for prefix in ("mjlab_microduck.dance", "microduck_dance"):
        if name == prefix or name.startswith(prefix + "."):
            return "OURS" + name[len(prefix):]
    return name


class Canon:
    def __init__(self, repo: Path):
        self.repo = str(repo.resolve())
        self.pkg = str(Path(mjlab_microduck.__file__).resolve().parent)

    def path(self, s: str) -> str:
        s = s.replace(self.pkg, "<UPSTREAM_PKG>")
        return s.replace(self.repo, "<REPO>")

    def name(self, obj) -> str:
        mod = canon_module(getattr(obj, "__module__", None) or "?")
        return f"{mod}:{getattr(obj, '__qualname__', repr(obj))}"

    def __call__(self, x, depth=0):
        if depth > 40:
            return "<deep>"
        if x is None or isinstance(x, (bool, int, str)):
            return self.path(x) if isinstance(x, str) else x
        if isinstance(x, float):
            return x if math.isfinite(x) else repr(x)
        if isinstance(x, enum.Enum):
            return f"{type(x).__name__}.{x.name}"
        if isinstance(x, Path):
            return self.path(str(x))
        if isinstance(x, (np.ndarray, np.generic)):
            return self(np.asarray(x).tolist(), depth + 1)
        if isinstance(x, torch.Tensor):
            return self(x.detach().cpu().tolist(), depth + 1)
        if isinstance(x, slice):
            return {"__slice__": [x.start, x.stop, x.step]}
        if isinstance(x, (list, tuple)):
            return [self(v, depth + 1) for v in x]
        if isinstance(x, (set, frozenset)):
            return sorted(str(self(v, depth + 1)) for v in x)
        if isinstance(x, dict):
            return {str(k): self(v, depth + 1) for k, v in x.items()}
        if isinstance(x, functools.partial):
            return {"__partial__": self.name(x.func), "args": self(x.args, depth + 1),
                    "keywords": self(x.keywords, depth + 1)}
        if isinstance(x, type):
            return "<class " + self.name(x) + ">"
        if dataclasses.is_dataclass(x):
            out = {"__type__": self.name(type(x))}
            for f in dataclasses.fields(x):
                out[f.name] = self(getattr(x, f.name), depth + 1)
            return out
        if isinstance(x, (types.FunctionType, types.BuiltinFunctionType, types.MethodType)):
            return "<fn " + self.name(x) + ">"
        if callable(x):
            return "<callable " + self.name(type(x)) + ">"
        return "<" + self.name(type(x)) + ">"


def obs_dims(task: str, small_terrain: bool = False) -> dict:
    """Obs term names and dims per group, from the env built on CPU with one env.

    small_terrain builds a 2 x 2 tile grid instead of the task's grid. The obs dims do
    not depend on the grid size, and the env builds in a fraction of the time. The
    snapshot was made with the full grid.
    """
    from mjlab.envs import ManagerBasedRlEnv

    cfg = load_env_cfg(task)
    cfg.scene.num_envs = 1
    gen = cfg.scene.terrain.terrain_generator
    if small_terrain and gen is not None:
        gen.num_rows = gen.num_cols = 2
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
        om = env.observation_manager
        return {g: [[n, list(d)] for n, d in zip(om.active_terms[g], om.group_obs_term_dim[g])]
                for g in om.active_terms}
    finally:
        env.close()


def snapshot(repo: Path, with_obs_dims: bool) -> dict:
    canon = Canon(repo)
    registered = set(list_tasks())
    out = {}
    for task in TASKS:
        if task not in registered:
            raise SystemExit(f"{task} is not registered")
        entry = {
            "env": canon(load_env_cfg(task)),
            "play_env": canon(load_env_cfg(task, play=True)),
            "rl": canon(load_rl_cfg(task)),
            "runner_cls": canon(load_runner_cls(task)),
        }
        if with_obs_dims and task in OBS_DIM_TASKS:
            entry["obs_dims"] = obs_dims(task)
        out[task] = entry
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repo", type=Path, default=Path.cwd(), help="repository root (paths are written relative to it)")
    ap.add_argument("--obs-dims", action="store_true")
    a = ap.parse_args()
    data = snapshot(a.repo, a.obs_dims)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(data, indent=1) + "\n")
    print("wrote", a.out, "tasks", len(data))


if __name__ == "__main__":
    main()
