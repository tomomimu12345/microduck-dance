"""Music-triggered tricks (Task 23). Pure functions, no simulator imports.

Two parts.

1. schedule_tricks picks trick moments from a song's beats json (the keys written
   by dance/scripts/generate/compute_beats.py: beat_times_s, beat_onset, beat_rms,
   duration_s). A moment is a downbeat where the music is loud and has just risen
   compared with the two bars before it.
2. TrickHandover is the per-env state machine for dance -> trick -> hold -> dance.
   The caller feeds time, up-axis alignment and trunk height each control step and
   gets back which policy should act.

Policies (Pollen microduck-policies, Apache-2.0, unmodified) are listed in
dance/data/CREDITS.md. The chaining numbers are in dance/docs/EXPERIMENTS.md tricks002.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

DT = 0.02

# Trick durations in seconds (roulade: manifest chain time, sitstand: sit then stand).
ROULADE_S = 1.0
SIT_S = 3.0
STAND_S = 3.0
TRICKS = ("roulade", "sitstand")
DEFAULT_BRIDGE_S = {"roulade": 1.0, "sitstand": 0.0}

# Schedule constraints.
MIN_START_S = 8.0
MIN_GAP_S = 8.0
END_MARGIN_S = 6.0
BEATS_PER_BAR = 4
PRIOR_BEATS = 2 * BEATS_PER_BAR

# Upright check, the "stable" rule of the tricks study: tilt under 20 degrees and
# height over 0.10 m for 0.5 s.
UPRIGHT_TILT_DEG = 20.0
UPRIGHT_Z_M = 0.10
UPRIGHT_HOLD_S = 0.5
HOLD_TIMEOUT_S = 4.0


def trick_duration(trick: str, roulade_s: float = ROULADE_S, sit_s: float = SIT_S, stand_s: float = STAND_S) -> float:
    if trick == "roulade":
        return roulade_s
    if trick == "sitstand":
        return sit_s + stand_s
    raise ValueError(f"unknown trick {trick!r}")


def trick_command(trick: str, t: float, roulade_s: float = ROULADE_S, sit_s: float = SIT_S, stand_s: float = STAND_S):
    """(13D command for the trick policy, done) at time t since the trick started.

    Roulade starts on an all-zero command. Sitstand reads twist.vx as a posture
    flag, 1 while sitting and 0 while standing up.
    """
    c = np.zeros(13, np.float32)
    if trick == "roulade":
        return c, t >= roulade_s
    if trick == "sitstand":
        if t < sit_s:
            c[0] = 1.0
        return c, t >= sit_s + stand_s
    raise ValueError(f"unknown trick {trick!r}")


# ---------------------------------------------------------------- scheduler


def _z(x: np.ndarray) -> np.ndarray:
    sd = float(x.std())
    return (x - float(x.mean())) / (sd if sd > 1e-9 else 1.0)


def downbeat_phase(beat_onset: Sequence[float]) -> int:
    """Which of the 4 beat positions carries the most onset strength (bar start guess)."""
    o = np.asarray(beat_onset, float)
    sums = [float(o[p::BEATS_PER_BAR].sum()) for p in range(BEATS_PER_BAR)]
    return int(np.argmax(sums))


def climax_scores(beat_onset: Sequence[float], beat_rms: Sequence[float]) -> np.ndarray:
    """Score per beat index. NaN where there are not enough beats before or after.

    level = mean over the next bar of (z(onset) + z(rms)), rise = level minus the
    same quantity averaged over the two bars before. score = level + rise.
    """
    o, r = np.asarray(beat_onset, float), np.asarray(beat_rms, float)
    if o.shape != r.shape:
        raise ValueError("beat_onset and beat_rms must have the same length")
    e = _z(o) + _z(r)
    n = len(e)
    out = np.full(n, np.nan)
    for i in range(PRIOR_BEATS, n - BEATS_PER_BAR + 1):
        level = e[i : i + BEATS_PER_BAR].mean()
        prior = e[i - PRIOR_BEATS : i].mean()
        out[i] = level + (level - prior)
    return out


@dataclass(frozen=True)
class TrickEvent:
    t_start: float
    trick: str
    bridge_s: float
    score: float

    def as_dict(self) -> dict:
        return {"t_start": round(self.t_start, 3), "trick": self.trick, "bridge_s": self.bridge_s, "score": round(self.score, 3)}


def schedule_tricks(
    beat_times_s: Sequence[float],
    beat_onset: Sequence[float],
    beat_rms: Sequence[float],
    duration_s: float,
    n_tricks: int = 2,
    order: Sequence[str] = TRICKS,
    min_start_s: float = MIN_START_S,
    min_gap_s: float = MIN_GAP_S,
    end_margin_s: float = END_MARGIN_S,
    bridge_s: dict | None = None,
) -> list[TrickEvent]:
    """Pick up to n_tricks moments, best score first, assigned to tricks in `order`.

    A candidate is a guessed downbeat with t >= min_start_s whose trick (including
    its bridge) ends at least end_margin_s before the end of the song. Chosen
    moments are at least min_gap_s apart, counted from the end of the earlier
    trick to the start of the later one. The result is sorted by time. The trick
    assigned to the best moment is order[0], to the next best order[1].
    """
    bridges = {**DEFAULT_BRIDGE_S, **(bridge_s or {})}
    t = np.asarray(beat_times_s, float)
    scores = climax_scores(beat_onset, beat_rms)
    if len(t) != len(scores):
        raise ValueError("beat_times_s, beat_onset and beat_rms must have the same length")
    phase = downbeat_phase(beat_onset)
    cand = [i for i in range(len(t)) if i % BEATS_PER_BAR == phase and not math.isnan(scores[i]) and t[i] >= min_start_s]
    cand.sort(key=lambda i: -scores[i])
    chosen: list[TrickEvent] = []
    for i in cand:
        if len(chosen) >= min(n_tricks, len(order)):
            break
        trick = order[len(chosen)]
        length = trick_duration(trick) + bridges[trick]
        if t[i] + length > duration_s - end_margin_s:
            continue
        ok = True
        for c in chosen:
            c_end = c.t_start + trick_duration(c.trick) + c.bridge_s
            if not (t[i] >= c_end + min_gap_s or c.t_start >= t[i] + length + min_gap_s):
                ok = False
                break
        if ok:
            chosen.append(TrickEvent(float(t[i]), trick, float(bridges[trick]), float(scores[i])))
    return sorted(chosen, key=lambda e: e.t_start)


def schedule_from_beats(beats: dict, **kw) -> list[TrickEvent]:
    """schedule_tricks on a loaded beats json (needs beat_onset and beat_rms)."""
    for k in ("beat_times_s", "beat_onset", "beat_rms", "duration_s"):
        if k not in beats:
            raise KeyError(f"beats json lacks {k!r}, rerun compute_beats.py")
    return schedule_tricks(beats["beat_times_s"], beats["beat_onset"], beats["beat_rms"], beats["duration_s"], **kw)


# ------------------------------------------------------------ state machine

DANCE, TRICK, HOLD, DANCE_AFTER = "dance", "trick", "hold", "dance_after"


def is_upright(up_z: float, height: float, tilt_deg: float = UPRIGHT_TILT_DEG, z_min: float = UPRIGHT_Z_M) -> bool:
    """up_z is the world-up component of the trunk up axis (minus projected gravity z)."""
    return up_z >= math.cos(math.radians(tilt_deg)) and height >= z_min


class TrickHandover:
    """dance -> trick -> hold -> dance for one env, one event.

    step(t, up_z, height) returns the state that acts at time t.
      dance        the dance policy (before the trick)
      trick        the trick policy with trick_command(trick, t - t_start)
      hold         a standing policy. The caller uses velstand when bridge_s > 0
                   and the trick policy at its final command otherwise.
      dance_after  the dance policy again
    Hold lasts at least bridge_s, then ends once the upright check has held for
    UPRIGHT_HOLD_S. If it has not passed hold_timeout_s after hold began, the
    machine hands back anyway and sets timed_out, so the caller can count the
    event as failed. Call step with non-decreasing t.
    """

    def __init__(
        self,
        t_start: float,
        trick: str,
        bridge_s: float,
        hold_timeout_s: float = HOLD_TIMEOUT_S,
        upright_hold_s: float = UPRIGHT_HOLD_S,
        **trick_kw,
    ):
        if trick not in TRICKS:
            raise ValueError(f"unknown trick {trick!r}")
        self.t_start, self.trick, self.bridge_s = t_start, trick, bridge_s
        self.hold_timeout_s, self.upright_hold_s = hold_timeout_s, upright_hold_s
        self.trick_kw = trick_kw
        self.state = DANCE
        self.t_hold: float | None = None
        self.t_back: float | None = None
        self.upright_since: float | None = None
        self.timed_out = False

    def command(self, t: float) -> np.ndarray:
        return trick_command(self.trick, t - self.t_start, **self.trick_kw)[0]

    def step(self, t: float, up_z: float, height: float) -> str:
        if self.state == DANCE and t >= self.t_start:
            self.state = TRICK
        if self.state == TRICK and trick_command(self.trick, t - self.t_start, **self.trick_kw)[1]:
            self.state = HOLD
            self.t_hold = t
        if self.state == HOLD:
            if is_upright(up_z, height):
                if self.upright_since is None:
                    self.upright_since = t
            else:
                self.upright_since = None
            held_long_enough = t - self.t_hold >= self.bridge_s
            passed = self.upright_since is not None and t - self.upright_since >= self.upright_hold_s - 1e-9
            if held_long_enough and passed:
                self.state, self.t_back = DANCE_AFTER, t
            elif t - self.t_hold >= self.hold_timeout_s:
                self.state, self.t_back, self.timed_out = DANCE_AFTER, t, True
        return self.state
