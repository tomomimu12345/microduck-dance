"""The dance task cfgs built in this repository match the ones of the fork they came from (CPU).

tests/cfg_snapshot/snapshot.json was written once by make_cfg_snapshot.py in the
old tree, where the dance code lived inside mjlab_microduck and four upstream
cfgs were patched. Here the code is a separate package on top of the unpatched
submodule. The whole env cfg (train and play), the runner cfg and the runner
class are compared, with module names of our code mapped to "OURS" and paths
written relative to the repository root. The obs term names and dims come from
the env built on CPU.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_DIR = Path(__file__).resolve().parent / "cfg_snapshot"
_spec = importlib.util.spec_from_file_location("make_cfg_snapshot", _DIR / "make_cfg_snapshot.py")
snap = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(snap)

SNAPSHOT = json.loads((_DIR / "snapshot.json").read_text())


def _diff(a, b, path="", out=None, limit=20):
    """Paths where two canonical cfgs differ (at most limit)."""
    out = [] if out is None else out
    if len(out) >= limit:
        return out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}/{k}: only in {'snapshot' if k in a else 'new'}")
            else:
                _diff(a[k], b[k], f"{path}/{k}", out, limit)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            _diff(x, y, f"{path}[{i}]", out, limit)
    elif a != b:
        out.append(f"{path}: snapshot {a!r} new {b!r}"[:300])
    return out


@pytest.fixture(scope="module")
def current():
    return snap.snapshot(ROOT, with_obs_dims=False)


def test_snapshot_covers_the_dance_tasks():
    assert tuple(SNAPSHOT) == snap.TASKS
    for task in snap.OBS_DIM_TASKS:
        assert "obs_dims" in SNAPSHOT[task], task


@pytest.mark.parametrize("task", snap.TASKS)
@pytest.mark.parametrize("part", ["env", "play_env", "rl", "runner_cls"])
def test_cfg_matches_the_snapshot(current, task, part):
    old, new = SNAPSHOT[task][part], current[task][part]
    assert new == old, "\n".join(_diff(old, new))


@pytest.mark.parametrize("task", snap.TASKS)
def test_key_fields_match_the_snapshot(current, task):
    """The fields the brief names, compared one by one so a failure says which one moved."""
    old, new = SNAPSHOT[task]["env"], current[task]["env"]
    for group in ("actor", "critic"):
        o, n = old["observations"][group]["terms"], new["observations"][group]["terms"]
        assert list(n) == list(o), group
    assert {k: v["weight"] for k, v in new["rewards"].items()} == {k: v["weight"] for k, v in old["rewards"].items()}
    assert {k: v["func"] for k, v in new["rewards"].items()} == {k: v["func"] for k, v in old["rewards"].items()}
    assert new["commands"] == old["commands"]
    assert new["scene"]["terrain"] == old["scene"]["terrain"]
    assert current[task]["rl"]["experiment_name"] == SNAPSHOT[task]["rl"]["experiment_name"]


@pytest.mark.parametrize("task", snap.OBS_DIM_TASKS)
def test_obs_term_dims_match_the_snapshot(task):
    dims = snap.obs_dims(task, small_terrain=True)
    assert dims == SNAPSHOT[task]["obs_dims"]
    assert sum(d[0] for _, d in dims["actor"]) == 61
