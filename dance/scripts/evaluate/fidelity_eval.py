"""Fidelity of the realized dance to the human dance: per-channel correlation, travel, heading, distinctness.

Reads play_dance.py state dumps and the human pkl of each song. Definitions follow the match report
(human vs realized channels, normalized correlation) and the similarity report (realized features for
the distinctness between songs). Used for the vel_gain sweep and the before/after table of fid001.

Run from the repo root with the microduck venv:
    uv run python dance/scripts/evaluate/fidelity_eval.py base vg0.3 --sim-dir out/fid001/sim

A tag names one set of dumps. The dump of song S and tag T is <sim-dir>/S_T_states.npz, written by
sweep_vel_gain.sh. The tag "base" is read from <base-dir>/S_v4_states.npz (the Task 15 commands).
Output: out/fid001/fidelity_<tags>.json and a summary on stdout.
"""
import argparse
import itertools
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.signal import welch
from scipy.spatial.transform import Rotation as R

from microduck_dance import retarget as rt

REPO = Path(__file__).resolve().parents[3]
DT = 0.02
FPS = 30.0
SONGS = ["swag", "funky_magic", "fallin_for_you"]
LAGS = np.arange(-50, 51)  # +-1 s
ROBOT_H = 0.25  # m, standing robot height, the unit of the robot travel
N_STEPS = 3000  # 60 s at 50 Hz
CHANS = ["heading", "x", "y", "height", "roll", "pitch", "head_yaw", "head_pitch"]
CH = {"vx": 0, "vy": 1, "wz": 2, "neck_pitch": 3, "head_pitch": 4, "head_yaw": 5, "head_roll": 6,
      "body_z": 9, "body_roll": 10, "body_pitch": 11}
ROB = ["vx", "vy", "wz", "head_pitch", "head_yaw", "neck_pitch", "head_roll", "body_z", "body_roll", "body_pitch"]

# Distinctness between songs is the RMS over these realized features of the difference in units of the
# population SD. The SD values are those of the similarity report's roll analysis
# (the realized features of the 8 rolled variants of the 3 songs).
POP_SD = {
    "std_vx": 0.017675068222111876, "std_wz": 0.1266570677007858, "std_head_pitch": 0.032449633356343384,
    "std_head_yaw": 0.0894223499021142, "std_neck_pitch": 0.028342783985275895,
    "std_head_roll": 0.018810834706696162, "std_body_z": 0.002495949169412717,
    "std_body_roll": 0.014042165215370847, "std_body_pitch": 0.024379294025112224,
    "cen_head_pitch": 0.2552808030225078, "cen_head_yaw": 0.113989431686478,
    "cen_body_z": 0.6148297430949624, "cen_body_roll": 0.2748509388154052,
    "cen_body_pitch": 0.15539922608664097, "cen_wz": 0.5844550026245557,
    "path_smooth_per_min": 0.9572585518566342, "yaw_smooth_per_min": 6.598855850959989,
    "step_rate_hz": 0.062373284995015335, "stance_w_std_mm": 0.6610043107698951,
    "stance_l_std_mm": 2.4311762932902194, "lift_p95_mm": 0.8752752310222944,
    "gait_f_feet": 0.036860489038724284,
}
KEYS = list(POP_SD)


def ncc(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else float("nan")


def best_lag(a, b):
    """(r, lag in s) maximizing corr(a[t], b[t+lag]); a positive lag means b is later."""
    best = (-2, 0)
    for L in LAGS:
        r = ncc(a[: len(a) - L], b[L:]) if L >= 0 else ncc(a[-L:], b[: len(b) + L])
        if r > best[0]:
            best = (r, L)
    return best[0], best[1] * DT


def ma(x, n):
    """Moving average over n samples, edge padded."""
    k = np.ones(n) / n
    return np.convolve(np.pad(x, (n // 2, n - 1 - n // 2), mode="edge"), k, mode="valid")


def path_length(x, y, smooth=False, n=25):
    """Path length of (x, y); smooth uses the 0.5 s (25 sample) moving average."""
    if smooth:
        x, y = ma(x, n), ma(y, n)
    return float(np.hypot(np.diff(x), np.diff(y)).sum())


def human_channels(pkl):
    """Human channels on the 50 Hz command grid (N_STEPS) in the common frame, and the figure height in m."""
    d = pickle.load(open(pkl, "rb"))
    poses = np.asarray(d["smpl_poses"], float).reshape(-1, 24, 3)
    joints = np.asarray(d["full_pose"], float)
    F = len(poses)
    up = rt.detect_up_axis(np.asarray(d["smpl_trans"], float), joints)
    if up != 2:
        raise ValueError(f"{pkl}: up axis {up}, expected 2")
    Wr = R.from_matrix(rt.WORLD_TO_ROBOT[up])
    Pr = R.from_matrix(rt.P)
    pelvis_rot = Wr * R.from_rotvec(poses[:, 0]) * Pr.inv()
    psi = rt._heading(pelvis_rot.apply([1.0, 0, 0]))
    pos = joints[:, 0] @ rt.WORLD_TO_ROBOT[up].T
    floor = float(np.percentile(joints[..., up].min(axis=1), rt.FLOOR_PERCENTILE))
    height = pos[:, 2] - floor
    to_heading = R.from_euler("z", -psi[:, None])
    trunk = (joints[:, rt.NECK] - joints[:, 0]) @ rt.WORLD_TO_ROBOT[up].T
    roll, pitch = rt._trunk_roll_pitch(to_heading.apply(trunk))
    head = Pr * rt._chain(poses, rt.NECK_HEAD_CHAIN) * Pr.inv()
    he = rt._euler_zyx(head)
    p0 = psi[0]
    c, s = np.cos(-p0), np.sin(-p0)
    xy = pos[:, :2] - pos[0, :2]
    xy = np.stack([c * xy[:, 0] - s * xy[:, 1], s * xy[:, 0] + c * xy[:, 1]], 1)
    fig_h = float(np.median(joints[:, 15, 2] - floor)) + 0.175  # head joint above the floor plus the head
    t_in = np.arange(F) / FPS
    t = np.arange(N_STEPS) * DT

    def grid(x):
        return np.interp(t, t_in, x)

    ch = {"heading": grid(psi - p0), "x": grid(xy[:, 0]), "y": grid(xy[:, 1]),
          "height": grid(height - np.median(height)), "roll": grid(roll), "pitch": grid(pitch),
          "head_yaw": grid(he[:, 0]), "head_pitch": grid(he[:, 1])}
    return ch, fig_h


def cmd_channels(cmd, pos=None, yaw=None):
    """Channels in the human convention, from the commands, or from the realized pos and yaw when given."""
    if yaw is None:
        head = np.cumsum(cmd[:, 2]) * DT
        head = np.concatenate([[0], head[:-1]])
        vx, vy = cmd[:, 0], cmd[:, 1]
        c, s = np.cos(head), np.sin(head)
        vxw, vyw = c * vx - s * vy, s * vx + c * vy
        x = np.concatenate([[0], np.cumsum(vxw)[:-1] * DT])
        y = np.concatenate([[0], np.cumsum(vyw)[:-1] * DT])
    else:
        head = np.unwrap(yaw.astype(float))
        head = head - head[0]
        x = pos[:, 0] - pos[0, 0]
        y = pos[:, 1] - pos[0, 1]
    return {"heading": head, "x": x, "y": y, "height": cmd[:, 9], "roll": cmd[:, 10], "pitch": cmd[:, 11],
            "head_yaw": cmd[:, 5], "head_pitch": -cmd[:, 3] + cmd[:, 4]}


def centroid(x, fs, fmin=0.1, fmax=10.0):
    """Spectral centroid (Hz) of x between fmin and fmax."""
    x = np.asarray(x, float) - np.mean(x)
    f, p = welch(x, fs=fs, nperseg=min(len(x), int(fs * 8)))
    m = (f >= fmin) & (f <= fmax)
    if p[m].sum() <= 0:
        return float("nan")
    return float((f[m] * p[m]).sum() / p[m].sum())


def peak(x, fs, lo, hi):
    f, p = welch(x - x.mean(), fs=fs, nperseg=min(len(x), int(fs * 8)))
    m = (f >= lo) & (f <= hi)
    return float(f[m][np.argmax(p[m])])


def realized_features(d, skip_s=2.0):
    """The KEYS features of one state dump (the first and last skip_s seconds are dropped)."""
    dt = float(d["dt"])
    fs = 1 / dt
    k = int(skip_s / dt)
    meas = d["meas"][k:-k].astype(float)
    pos = d["pos"][k:-k].astype(float)
    yaw = np.unwrap(d["yaw"][k:-k].astype(float))
    dur = len(meas) * dt
    out = {}
    for n in ROB:
        out[f"std_{n}"] = float(meas[:, CH[n]].std())
        out[f"cen_{n}"] = centroid(meas[:, CH[n]], fs)
    n = int(0.5 / dt)
    xs, ys = ma(pos[:, 0], n), ma(pos[:, 1], n)
    out["path_smooth_per_min"] = float(np.hypot(np.diff(xs), np.diff(ys)).sum() / dur * 60)
    out["yaw_smooth_per_min"] = float(np.abs(np.diff(ma(yaw, n))).sum() / dur * 60)
    feet = d["feet"][k:-k].astype(float)  # (T, 2, 3) world
    lift = feet[..., 2] - np.percentile(feet[..., 2], 5, axis=0)
    swing = lift > 0.004
    steps = sum(int(np.sum(np.diff(swing[:, j].astype(int)) == 1)) for j in range(2))
    out["step_rate_hz"] = float(steps / dur)
    out["lift_p95_mm"] = float(1000 * np.percentile(lift, 95))
    out["gait_f_feet"] = peak(lift[:, 0] - lift[:, 1], fs, 0.5, 8.0)
    c, s = np.cos(yaw), np.sin(yaw)
    dx, dy = feet[:, 0, 0] - feet[:, 1, 0], feet[:, 0, 1] - feet[:, 1, 1]
    out["stance_w_std_mm"] = float(1000 * np.std(-s * dx + c * dy))
    out["stance_l_std_mm"] = float(1000 * np.std(c * dx + s * dy))
    return out


def dump_path(tag, song, sim_dir, base_dir):
    if tag == "base":
        return base_dir / f"{song}_v4_states.npz"
    return sim_dir / f"{song}_{tag}_states.npz"


def evaluate(tags, sim_dir, base_dir, motion_dir):
    res = {}
    human = {s: human_channels(motion_dir / f"{s}.pkl") for s in SONGS}
    for tag in tags:
        res[tag] = {}
        feats = {}
        for song in SONGS:
            p = dump_path(tag, song, sim_dir, base_dir)
            if not p.exists():
                continue
            hch, fig_h = human[song]
            st = np.load(p)
            cch = cmd_channels(st["cmd"].astype(float))
            rch = cmd_channels(st["meas"].astype(float), st["pos"].astype(float), st["yaw"])
            row = {"falls": int(np.sum(st["term"] & ~np.r_[False, st["term"][:-1]]))}
            for k in CHANS:
                row[k] = {"hc": ncc(hch[k], cch[k]), "cm": ncc(cch[k], rch[k]), "hm": ncc(hch[k], rch[k]),
                          "lag_hm": best_lag(hch[k], rch[k])[1]}
            row["path_h_bodyh"] = path_length(hch["x"], hch["y"]) / fig_h
            row["path_m_bodyh"] = path_length(rch["x"], rch["y"]) / ROBOT_H
            row["path_c_bodyh"] = path_length(cch["x"], cch["y"]) / ROBOT_H
            row["travel_ratio"] = row["path_m_bodyh"] / row["path_h_bodyh"]
            row["heading_final_err_deg"] = float(np.degrees(rch["heading"][-1] - hch["heading"][-1]))
            row["heading_rms_err_deg"] = float(np.degrees(np.sqrt(np.mean((rch["heading"] - hch["heading"]) ** 2))))
            row["heading_final_deg"] = {"h": float(np.degrees(hch["heading"][-1])),
                                        "c": float(np.degrees(cch["heading"][-1])),
                                        "m": float(np.degrees(rch["heading"][-1]))}
            res[tag][song] = row
            feats[song] = realized_features(st)
        sd = np.array([POP_SD[k] for k in KEYS])
        vec = {s: np.array([f[k] for k in KEYS]) for s, f in feats.items()}
        res[tag]["_dist"] = {f"{a}-{b}": float(np.sqrt(np.mean(((vec[a] - vec[b]) / sd) ** 2)))
                             for a, b in itertools.combinations(SONGS, 2) if a in vec and b in vec}
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tags", nargs="+", help="dump tags, e.g. vg0.3; 'base' is the Task 15 baseline")
    ap.add_argument("--sim-dir", type=Path, default=REPO / "out/fid001/sim",
                    help="folder of <song>_<tag>_states.npz (sweep_vel_gain.sh output)")
    ap.add_argument("--base-dir", type=Path, default=REPO / "out/rec_v4",
                    help="folder of <song>_v4_states.npz, the dumps behind the tag 'base'")
    ap.add_argument("--motion-dir", type=Path, default=REPO / "dance/data/motions/songs",
                    help="folder of the human <song>.pkl")
    ap.add_argument("--out-dir", type=Path, default=REPO / "out/fid001", help="where the json goes")
    a = ap.parse_args(argv)
    res = evaluate(a.tags, a.sim_dir, a.base_dir, a.motion_dir)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    out = a.out_dir / ("fidelity_" + "_".join(a.tags) + ".json")
    out.write_text(json.dumps(res, indent=1))
    for tag, r in res.items():
        print("==", tag, "dist", {k: round(v, 2) for k, v in r["_dist"].items()},
              "min", round(min(r["_dist"].values()), 2) if r["_dist"] else None)
        for song in SONGS:
            if song not in r:
                continue
            row = r[song]
            print(f"  {song:15s} falls {row['falls']} travel h {row['path_h_bodyh']:.1f} m {row['path_m_bodyh']:.1f} "
                  f"(c {row['path_c_bodyh']:.1f}) ratio {row['travel_ratio']:.2f} "
                  f"headRMS {row['heading_rms_err_deg']:.0f} final {row['heading_final_err_deg']:.0f}")
            print("    hm", " ".join(f"{k[:7]}:{row[k]['hm']:.2f}" for k in CHANS))
    print("wrote", out)


if __name__ == "__main__":
    sys.exit(main())
