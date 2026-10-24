"""Heading and path x correlation per env from play_dance --eval --dump-eval rollouts, with falls and head RMSE.

The correlations are those of fidelity_eval.py (human vs realized, lag 0), computed for every env of an
eval run instead of one record-mode env. The realized heading is the unwrapped trunk yaw relative to its
first sample. The realized path is the trunk xy relative to its first sample, rotated by minus the first
yaw, so it is in the robot's starting frame as the human path is in the dancer's. Each env is scored up to
its first fall (the respawn restarts the path), and an env with less than MIN_WINDOW_S of data before it
is left out.

Run from the repo root with the microduck venv:
    uv run python dance/scripts/evaluate/rough_metrics.py out/rough/t28/<name>_<song>.npz ... --json out.json

For each npz the eval JSON of the same stem (play_dance --json) is read for falls and head RMSE if it exists.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
MIN_WINDOW_S = 20.0


def realized_channels(pos: np.ndarray, yaw: np.ndarray) -> dict:
    """heading, x, y of one env (pos (T, 3), yaw (T,)) in its starting frame."""
    head = np.unwrap(np.asarray(yaw, float))
    head = head - head[0]
    xy = np.asarray(pos, float)[:, :2] - np.asarray(pos, float)[0, :2]
    c, s = np.cos(-yaw[0]), np.sin(-yaw[0])
    x = c * xy[:, 0] - s * xy[:, 1]
    y = s * xy[:, 0] + c * xy[:, 1]
    return {"heading": head, "x": x, "y": y}


def first_fall(term: np.ndarray) -> int:
    """Index of the first terminated step, len(term) if none."""
    term = np.asarray(term, bool)
    return int(np.argmax(term)) if term.any() else len(term)


def env_correlations(human: dict, pos, yaw, term, dt: float, min_window_s: float = MIN_WINDOW_S) -> dict:
    """heading r and path x r of one env against the human channels, up to its first fall."""
    from fidelity_eval import ncc

    k = first_fall(term)
    if k * dt < min_window_s:
        return {"heading": float("nan"), "x": float("nan"), "window_s": k * dt}
    r = realized_channels(pos[:k], yaw[:k])
    return {"heading": ncc(human["heading"][:k], r["heading"]), "x": ncc(human["x"][:k], r["x"]),
            "window_s": k * dt}


def by_terrain_type(types, falls, heading, path_x, minutes: float) -> dict:
    """Per tile type: env count, falls per env per minute, mean heading r and path x r (finite envs)."""
    out = {}
    types, falls = np.asarray(types), np.asarray(falls, float)
    heading, path_x = np.asarray(heading, float), np.asarray(path_x, float)
    for t in sorted(set(types.tolist())):
        m = types == t
        h, x = heading[m], path_x[m]
        out[t] = {"n_envs": int(m.sum()), "falls_per_min": float(falls[m].mean() / minutes),
                  "heading_r": float(np.nanmean(h)) if np.isfinite(h).any() else float("nan"),
                  "path_x_r": float(np.nanmean(x)) if np.isfinite(x).any() else float("nan")}
    return out


def score(npz: Path, human_cache: dict, motion_dir: Path) -> dict:
    from fidelity_eval import human_channels

    d = np.load(npz)
    dt = float(d["dt"])
    song = Path(str(d["commands"])).stem
    if song not in human_cache:
        human_cache[song] = human_channels(motion_dir / f"{song}.pkl")[0]
    human = human_cache[song]
    pos, yaw, term = d["pos"], d["yaw"], d["term"]
    rows = [env_correlations(human, pos[:, i], yaw[:, i], term[:, i], dt) for i in range(pos.shape[1])]
    h = np.array([r["heading"] for r in rows])
    x = np.array([r["x"] for r in rows])
    ok = np.isfinite(h)
    out = {"npz": str(npz), "song": song, "n_envs": int(pos.shape[1]), "n_scored": int(ok.sum()),
           "heading_r": float(np.mean(h[ok])) if ok.any() else float("nan"),
           "heading_r_se": float(np.std(h[ok], ddof=1) / np.sqrt(ok.sum())) if ok.sum() > 1 else float("nan"),
           "path_x_r": float(np.mean(x[ok])) if ok.any() else float("nan"),
           "path_x_r_se": float(np.std(x[ok], ddof=1) / np.sqrt(ok.sum())) if ok.sum() > 1 else float("nan")}
    js = npz.with_suffix(".json")
    if js.exists():
        res = json.loads(js.read_text())
        summ = res["summary"]
        minutes = res["steps"] * res["dt"] / 60.0
        out["falls_per_min"] = summ["falls"] / minutes
        out["head_rmse"] = summ["head_rmse"]
        out["body_rmse"] = summ["body_rmse"]
        terrain = res["per_seed"][0].get("terrain")
        if terrain is not None:
            out["by_type"] = by_terrain_type([e["type"] for e in terrain["envs"]],
                                             [e["falls"] for e in terrain["envs"]], h, x, minutes)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz", nargs="+", type=Path)
    ap.add_argument("--motion-dir", type=Path, default=REPO / "dance/data/motions/songs")
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args(argv)
    cache: dict = {}
    rows = [score(p, cache, a.motion_dir) for p in a.npz]
    for r in rows:
        extra = ""
        if "falls_per_min" in r:
            extra = f" falls/min {r['falls_per_min']:.3f} head RMSE {r['head_rmse']:.3f}"
        print(f"{Path(r['npz']).stem:40s} heading r {r['heading_r']:.3f} ± {r['heading_r_se']:.3f} "
              f"path x r {r['path_x_r']:.3f} ± {r['path_x_r_se']:.3f} ({r['n_scored']}/{r['n_envs']} envs){extra}")
        for t, v in r.get("by_type", {}).items():
            print(f"    {t:15s} n {v['n_envs']:2d} falls/min {v['falls_per_min']:.3f} heading r {v['heading_r']:.3f} "
                  f"path x r {v['path_x_r']:.3f}")
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
