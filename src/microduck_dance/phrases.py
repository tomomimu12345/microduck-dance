"""Musical phrases and the twist gate (Task 15).

The policy steps at about 2 Hz whenever the twist command is non-zero, so the
legs looked the same in every song (similarity report, 2026-10-20). The gate
splits a track into phrases and sets the twist to zero in the quieter ones.
The robot then stands and grooves with head and body, and it walks with the
retargeted twist in the louder ones.

Phrases. librosa gives beats but no downbeats, so the first tracked beat is
taken as the first downbeat. Phrase boundaries are every PHRASE_BEATS beats
from it: beats[0], beats[16], beats[32], ... Time before the first beat
belongs to the first phrase and time after the last boundary to the last
phrase. A last phrase with fewer than MIN_TAIL_BEATS beats is merged into the
one before it.

Energy. Each phrase gets the mean beat_rms and the mean beat_onset of its beats
(compute_beats.py writes both, per beat interval). The score is
rms / mean(rms over phrases) + onset / mean(onset over phrases), so the two
terms have equal weight. Phrases scoring below the median are groove phrases
and the rest travel. When the phrase RMS has a coefficient of variation below
MIN_RMS_CV, the dynamics are too flat to rank, and the phrases alternate,
groove first.

Gate. gate(t) is 1 in travel phrases and 0 in groove phrases. At a boundary
between a travel and a groove phrase the change happens inside the groove
phrase, over fade_s, with a raised-cosine shape. Travel phrases keep their
twist unchanged, and groove phrases are exactly 0 apart from those fades.
"""

from __future__ import annotations

import numpy as np

PHRASE_BEATS = 16
MIN_TAIL_BEATS = 8
MIN_RMS_CV = 0.08
FADE_S = 0.5


def phrase_starts(n_beats: int, phrase_beats: int = PHRASE_BEATS, min_tail: int = MIN_TAIL_BEATS) -> list[int]:
    """Beat indices that start each phrase. The first is always 0."""
    if n_beats < 1:
        raise ValueError("need at least one beat")
    starts = list(range(0, n_beats, phrase_beats))
    if len(starts) > 1 and n_beats - starts[-1] < min_tail:
        starts.pop()
    return starts


def phrase_energy(beat_rms, beat_onset, starts: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-phrase mean RMS, mean onset strength and the combined score."""
    rms = np.asarray(beat_rms, float)
    onset = np.asarray(beat_onset, float)
    if rms.shape != onset.shape or rms.ndim != 1:
        raise ValueError("beat_rms and beat_onset must be 1D arrays of the same length")
    ends = list(starts[1:]) + [len(rms)]
    r = np.array([rms[a:b].mean() for a, b in zip(starts, ends)])
    o = np.array([onset[a:b].mean() for a, b in zip(starts, ends)])
    score = r / r.mean() + o / o.mean()
    return r, o, score


def classify_phrases(beat_rms, beat_onset, starts: list[int], min_cv: float = MIN_RMS_CV) -> dict:
    """Groove (False) or travel (True) for each phrase, with the rule that decided it."""
    r, o, score = phrase_energy(beat_rms, beat_onset, starts)
    cv = float(r.std() / r.mean()) if r.mean() > 0 else 0.0
    if cv < min_cv:
        travel = np.arange(len(starts)) % 2 == 1
        rule = "alternate"
    else:
        travel = score >= np.median(score)
        rule = "energy"
    return {"travel": [bool(v) for v in travel], "rule": rule, "rms_cv": cv,
            "rms": r.tolist(), "onset": o.tolist(), "score": score.tolist()}


def _smooth(x: np.ndarray) -> np.ndarray:
    """Raised cosine from 0 at x <= 0 to 1 at x >= 1."""
    x = np.clip(x, 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * x)


def twist_gate(t: np.ndarray, bounds: np.ndarray, travel, fade_s: float = FADE_S) -> np.ndarray:
    """Gate in [0, 1] for times t.

    bounds are the phrase boundary times (len(travel) - 1 of them, the starts of
    phrases 1 onward). Phrase k covers [bounds[k-1], bounds[k]), phrase 0 starts
    at -inf and the last one ends at +inf.
    """
    t = np.asarray(t, float)
    travel = np.asarray(travel, bool)
    bounds = np.asarray(bounds, float)
    if len(bounds) != len(travel) - 1:
        raise ValueError("need one boundary fewer than phrases")
    edges = np.concatenate([[-np.inf], bounds, [np.inf]])
    gate = np.zeros_like(t)
    for k, tr in enumerate(travel):
        m = (t >= edges[k]) & (t < edges[k + 1])
        if tr:
            gate[m] = 1.0
            continue
        g = np.zeros(m.sum())
        tk = t[m]
        if k > 0 and travel[k - 1] and fade_s > 0:
            g = np.maximum(g, 1.0 - _smooth((tk - edges[k]) / fade_s))
        if k + 1 < len(travel) and travel[k + 1] and fade_s > 0:
            g = np.maximum(g, _smooth((tk - (edges[k + 1] - fade_s)) / fade_s))
        gate[m] = g
    return gate


def plan_phrases(beats, beat_rms, beat_onset, phrase_beats: int = PHRASE_BEATS,
                 min_cv: float = MIN_RMS_CV) -> dict:
    """Phrase starts (beat index and time) and the groove or travel label of each."""
    beats = np.asarray(beats, float)
    if len(beat_rms) != len(beats) or len(beat_onset) != len(beats):
        raise ValueError("beat_rms and beat_onset need one value per beat")
    starts = phrase_starts(len(beats), phrase_beats)
    info = classify_phrases(beat_rms, beat_onset, starts, min_cv)
    info["start_beats"] = starts
    info["start_times_s"] = [float(beats[i]) for i in starts]
    return info
