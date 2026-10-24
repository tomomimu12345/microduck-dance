"""How much the robot visibly moves in a played dance (pure numpy).

Input is the per-step state dump of dance/scripts/evaluate/play_dance.py --dump-states:
cmd and meas (T, 13) in the dance command layout, pos (T, 3) trunk position
relative to the env origin, yaw (T,) trunk yaw, dt. Statistics cover the
whole clip unless skip_s is given.

Smoothed quantities use a centered moving average of smooth_s seconds. The
raw path and the raw summed |yaw change| include the stepping jitter of the
gait, so both raw and smoothed values are reported.
"""

from __future__ import annotations

import numpy as np

from microduck_dance.ranges import DANCE_STAND_HEIGHT

# Channels of the 13D layout that carry motion (body x, y and yaw are always 0).
CHANNELS = {
    "vx": 0, "vy": 1, "wz": 2,
    "neck_pitch": 3, "head_pitch": 4, "head_yaw": 5, "head_roll": 6,
    "body_z": 9, "body_roll": 10, "body_pitch": 11,
}


def moving_average(x: np.ndarray, n: int) -> np.ndarray:
    """Centered moving average over n samples along axis 0, edges padded with the edge value."""
    x = np.asarray(x, dtype=float)
    if n <= 1:
        return x.copy()
    left = n // 2
    right = n - 1 - left
    pad = [(left, right)] + [(0, 0)] * (x.ndim - 1)
    xp = np.pad(x, pad, mode="edge")
    c = np.cumsum(np.concatenate([np.zeros((1,) + x.shape[1:]), xp]), axis=0)
    return (c[n:] - c[:-n]) / n


def path_length(xy: np.ndarray) -> float:
    xy = np.asarray(xy, dtype=float)
    return float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))


def commanded_path(cmd: np.ndarray, dt: float, yaw0: float = 0.0) -> np.ndarray:
    """(T, 2) planar path from integrating the commanded body-frame vx, vy and wz."""
    cmd = np.asarray(cmd, dtype=float)
    heading = yaw0 + np.concatenate([[0.0], np.cumsum(cmd[:-1, 2]) * dt])
    c, s = np.cos(heading), np.sin(heading)
    vxw = c * cmd[:, 0] - s * cmd[:, 1]
    vyw = s * cmd[:, 0] + c * cmd[:, 1]
    return np.concatenate([[[0.0, 0.0]], np.cumsum(np.stack([vxw, vyw], 1)[:-1], axis=0) * dt])


def nod_lag(t: np.ndarray, head_pitch: np.ndarray, beats, dt: float, detrend_s: float = 1.0,
            edge_s: float = 1.5) -> dict:
    """Median time from each beat to the nearest head_pitch peak (positive is after the beat).

    head_pitch is high-passed by subtracting a detrend_s moving average. For
    each beat b with local period P, the peak is the argmax inside
    [b - P/2, b + P/2). Beats closer than edge_s to either end are skipped.
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(head_pitch, dtype=float)
    hp = x - moving_average(x, max(1, int(round(detrend_s / dt))))
    b = np.asarray(beats, dtype=float)
    per = np.append(np.diff(b), np.diff(b)[-1]) if len(b) > 1 else np.array([0.5])
    lags = []
    for bi, p in zip(b, per):
        if bi < t[0] + edge_s or bi > t[-1] - edge_s:
            continue
        m = (t >= bi - p / 2) & (t < bi + p / 2)
        if not m.any():
            continue
        k = np.flatnonzero(m)[np.argmax(hp[m])]
        lags.append(t[k] - bi)
    lags = np.asarray(lags)
    if not len(lags):
        return {"median_s": float("nan"), "iqr_s": float("nan"), "n": 0}
    q1, q3 = np.percentile(lags, [25, 75])
    return {"median_s": float(np.median(lags)), "iqr_s": float(q3 - q1), "n": int(len(lags))}


def visible_motion(cmd: np.ndarray, meas: np.ndarray, pos: np.ndarray, yaw: np.ndarray, dt: float,
                   beats=None, smooth_s: float = 0.5, skip_s: float = 0.0) -> dict:
    """Visible-motion statistics of one env (see the module docstring)."""
    cmd = np.asarray(cmd, dtype=float)
    meas = np.asarray(meas, dtype=float)
    pos = np.asarray(pos, dtype=float)
    yaw = np.asarray(yaw, dtype=float)
    T = len(cmd)
    if meas.shape != cmd.shape or pos.shape != (T, 3) or yaw.shape != (T,):
        raise ValueError(f"shape mismatch: cmd {cmd.shape} meas {meas.shape} pos {pos.shape} yaw {yaw.shape}")
    k0 = int(round(skip_s / dt))
    cmd, meas, pos, yaw = cmd[k0:], meas[k0:], pos[k0:], yaw[k0:]
    t = (k0 + np.arange(len(cmd))) * dt
    n = max(1, int(round(smooth_s / dt)))

    xy = pos[:, :2]
    xy_s = moving_average(xy, n)
    yaw_u = np.unwrap(yaw)
    yaw_s = moving_average(yaw_u, n)
    cpath = commanded_path(cmd, dt, yaw0=float(yaw_u[0]))
    z = pos[:, 2]
    p5, p95 = np.percentile(z, [5, 95])
    cz = cmd[:, 9] + DANCE_STAND_HEIGHT
    cp5, cp95 = np.percentile(cz, [5, 95])

    out = {
        "seconds": float(len(cmd) * dt),
        "path_m": path_length(xy),
        "path_smooth_m": path_length(xy_s),
        "cmd_path_m": path_length(cpath),
        "max_disp_m": float(np.max(np.linalg.norm(xy - xy[0], axis=1))),
        "cmd_max_disp_m": float(np.max(np.linalg.norm(cpath - cpath[0], axis=1))),
        "yaw_abs_change_rad": float(np.sum(np.abs(np.diff(yaw_u)))),
        "yaw_abs_change_smooth_rad": float(np.sum(np.abs(np.diff(yaw_s)))),
        "cmd_yaw_abs_change_rad": float(np.sum(np.abs(cmd[:-1, 2])) * dt),
        "yaw_net_rad": float(yaw_u[-1] - yaw_u[0]),
        "height_p5_p95_m": float(p95 - p5),
        "cmd_height_p5_p95_m": float(cp95 - cp5),
        "roll_std_rad": float(np.std(meas[:, 10])),
        "pitch_std_rad": float(np.std(meas[:, 11])),
        "head_pitch_std_rad": float(np.std(meas[:, 4])),
        "head_yaw_std_rad": float(np.std(meas[:, 5])),
        "neck_pitch_std_rad": float(np.std(meas[:, 3])),
        "std_ratio": {},
        "cmd_std": {},
        "corr": {},
    }
    for name, c in CHANNELS.items():
        cs = float(np.std(cmd[:, c]))
        ms = float(np.std(meas[:, c]))
        out["cmd_std"][name] = cs
        out["std_ratio"][name] = ms / cs if cs > 1e-9 else float("nan")
        # Zero-lag correlation: a std ratio near 1 can also come from motion unrelated to the command.
        out["corr"][name] = float(np.corrcoef(cmd[:, c], meas[:, c])[0, 1]) if cs > 1e-9 and ms > 1e-12 \
            else float("nan")
    if beats is not None:
        out["nod_lag_realized"] = nod_lag(t, meas[:, 4], beats, dt)
        out["nod_lag_command"] = nod_lag(t, cmd[:, 4], beats, dt)
    return out
