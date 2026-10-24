"""Beat groove layer: head nod, body bounce and head sway locked to the music beats.

The FlowerDance motions carry almost no energy at the beat frequency in pelvis
height or head (motion-amp report, 2026-10-20), so the retarget alone does not
look on the beat. This layer adds it on top of the retargeted commands.

Beats are a json file written by dance/scripts/generate/compute_beats.py:
{"beat_times_s": [...], "tempo_bpm": float, ...}. Times are seconds from the
start of the wav, which is also t = 0 of the motion and of the command file.

Shapes, for beat i at time b_i with local period T_i (the gap to the next beat,
the previous gap for the last beat):
- pulse(t) = 0.5 (1 + cos(2 pi (t - c_i) / w_i)) for |t - c_i| < w_i / 2, else 0,
  with centre c_i = b_i + lag and width w_i = width_frac * T_i. Pulses are
  combined with max, so the result never exceeds 1 even if two overlap.
- nod: head_pitch += nod * pulse (positive is look down).
- bounce: body_z += bounce * pulse (negative is a dip).
- sway: head_yaw += sway * cos(pi * phi(t - lag)), where phi is the continuous
  beat count (phi(b_i) = i, linear in between and extrapolated with the edge
  periods). One full left-right cycle every two beats, at a side on each beat.

Per-song pattern (Task 15, groove_pattern and pattern_layer). The shared
groove made 57 to 78 % of the realized head yaw in every song (similarity
report, 2026-10-20). choose_pattern picks the pattern from the music by fixed
rules, the same for every track:
- accent: the beat position in the bar (0 to 3, bars of 4 beats from the first
  tracked beat, which is taken as the first downbeat) with the highest mean
  onset strength at the beat. The nod falls on that position in every bar and
  also on position 0 when they differ, so 1 or 2 nods per bar.
- bounce: the body_z dip, linear in the track's mean onset strength between
  BOUNCE_SHALLOW at ONSET_MEAN_LO and BOUNCE_DEEP at ONSET_MEAN_HI, clipped to
  that range. The bounce stays on every beat.
- sway: below SWAY_TEMPO_SPLIT_BPM the trunk sways (body_roll, TRUNK_SWAY rad)
  and the head does not. Otherwise the head yaw sways as before. Both use
  the same half-beat-rate cosine.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from microduck_dance.ranges import CMD_DIM

HEAD_PITCH, HEAD_YAW, BODY_Z, BODY_ROLL = 4, 5, 9, 10

BEATS_PER_BAR = 4
# Range of the mean librosa onset strength over the 5 songs and the 33 library
# tracks (dance/data/beats/{songs,training}/*.json, onset_mean, 2026-10-20). A new track outside
# it gets the end value of the bounce.
ONSET_MEAN_LO = 1.4286
ONSET_MEAN_HI = 3.3430
BOUNCE_SHALLOW = -0.010
BOUNCE_DEEP = -0.025
SWAY_TEMPO_SPLIT_BPM = 105.0
TRUNK_SWAY = 0.15


def load_beats(path: str | Path) -> np.ndarray:
    with Path(path).open() as fh:
        data = json.load(fh)
    return check_beats(data["beat_times_s"])


def check_beats(beats) -> np.ndarray:
    b = np.asarray(beats, dtype=float).reshape(-1)
    if len(b) < 2:
        raise ValueError(f"need at least 2 beats, got {len(b)}")
    if not np.all(np.diff(b) > 0):
        raise ValueError("beat times must be strictly increasing")
    return b


def _periods(beats: np.ndarray) -> np.ndarray:
    d = np.diff(beats)
    return np.append(d, d[-1])


def beat_pulse(t: np.ndarray, beats, width_frac: float, lag_s: float) -> np.ndarray:
    """Raised-cosine pulse train in [0, 1], peak 1 at each beat + lag_s."""
    b = check_beats(beats)
    t = np.asarray(t, dtype=float)
    out = np.zeros_like(t)
    for c, w in zip(b + lag_s, width_frac * _periods(b)):
        m = np.abs(t - c) < 0.5 * w
        out[m] = np.maximum(out[m], 0.5 * (1.0 + np.cos(2.0 * np.pi * (t[m] - c) / w)))
    return out


def beat_phase(t: np.ndarray, beats) -> np.ndarray:
    """Continuous beat count: i at beat i, linear between beats and beyond the ends."""
    b = check_beats(beats)
    t = np.asarray(t, dtype=float)
    idx = np.arange(len(b), dtype=float)
    phi = np.interp(t, b, idx)
    first, last = b[1] - b[0], b[-1] - b[-2]
    phi = np.where(t < b[0], (t - b[0]) / first, phi)
    phi = np.where(t > b[-1], idx[-1] + (t - b[-1]) / last, phi)
    return phi


def groove_layer(t: np.ndarray, beats, nod: float, bounce: float, sway: float,
                 width_frac: float, lag_s: float) -> np.ndarray:
    """(T, 13) additive commands for times t (seconds)."""
    t = np.asarray(t, dtype=float)
    pulse = beat_pulse(t, beats, width_frac, lag_s)
    out = np.zeros((len(t), CMD_DIM))
    out[:, HEAD_PITCH] = nod * pulse
    out[:, BODY_Z] = bounce * pulse
    out[:, HEAD_YAW] = sway * np.cos(np.pi * beat_phase(t - lag_s, beats))
    return out


def accent_position(onset_at_beat, beats_per_bar: int = BEATS_PER_BAR) -> int:
    """Bar position (0 is the first tracked beat) with the highest mean onset strength."""
    o = np.asarray(onset_at_beat, float)
    if len(o) < beats_per_bar:
        raise ValueError(f"need at least {beats_per_bar} beats")
    return int(np.argmax([o[p::beats_per_bar].mean() for p in range(beats_per_bar)]))


def bounce_depth(onset_mean: float, lo: float = ONSET_MEAN_LO, hi: float = ONSET_MEAN_HI) -> float:
    """body_z dip (m, negative), linear in the mean onset strength and clipped to the range."""
    x = float(np.clip((onset_mean - lo) / (hi - lo), 0.0, 1.0))
    return BOUNCE_SHALLOW + x * (BOUNCE_DEEP - BOUNCE_SHALLOW)


def choose_pattern(music: dict, nod: float, head_sway: float) -> dict:
    """Groove pattern of a track from its beats json (see the module docstring)."""
    accent = accent_position(music["onset_at_beat"])
    trunk = float(music["tempo_bpm"]) < SWAY_TEMPO_SPLIT_BPM
    return {
        "accent_position": accent,
        "nod_positions": sorted({0, accent}),
        "nod": float(nod),
        "bounce": bounce_depth(float(music["onset_mean"])),
        "sway_type": "trunk_roll" if trunk else "head_yaw",
        "sway": TRUNK_SWAY if trunk else float(head_sway),
        "tempo_bpm": float(music["tempo_bpm"]),
        "onset_mean": float(music["onset_mean"]),
    }


def pattern_layer(t: np.ndarray, beats, pattern: dict, width_frac: float, lag_s: float,
                  gain=1.0) -> np.ndarray:
    """(T, 13) additive commands for a per-song pattern.

    gain is a scalar or a (T,) array that scales the whole layer (1.3 in the
    groove phrases, see retarget.py).
    """
    b = check_beats(beats)
    t = np.asarray(t, dtype=float)
    pos = np.arange(len(b)) % BEATS_PER_BAR
    nod_beats = b[np.isin(pos, pattern["nod_positions"])]
    out = np.zeros((len(t), CMD_DIM))
    if len(nod_beats):
        # Widths come from the full beat grid, so a nod is as wide as a beat pulse.
        per = dict(zip(b, _periods(b)))
        nod_pulse = np.zeros_like(t)
        for c0 in nod_beats:
            c, w = c0 + lag_s, width_frac * per[c0]
            m = np.abs(t - c) < 0.5 * w
            nod_pulse[m] = np.maximum(nod_pulse[m], 0.5 * (1.0 + np.cos(2.0 * np.pi * (t[m] - c) / w)))
        out[:, HEAD_PITCH] = pattern["nod"] * nod_pulse
    out[:, BODY_Z] = pattern["bounce"] * beat_pulse(t, b, width_frac, lag_s)
    sway = np.cos(np.pi * beat_phase(t - lag_s, b))
    out[:, BODY_ROLL if pattern["sway_type"] == "trunk_roll" else HEAD_YAW] = pattern["sway"] * sway
    gain = np.asarray(gain, dtype=float)
    return out * (gain.reshape(-1, 1) if gain.ndim else gain)
