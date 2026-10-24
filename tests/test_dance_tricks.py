"""Tests for the trick scheduler and the dance/trick hand-over machine (Task 23)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from microduck_dance.tricks import (
    DANCE, DANCE_AFTER, HOLD, TRICK, TrickHandover, climax_scores, downbeat_phase, is_upright,
    schedule_from_beats, schedule_tricks, trick_command, trick_duration,
)

REPO = Path(__file__).resolve().parents[1]
DT = 0.02


def synth_beats(duration=60.0, bpm=120.0, climax_beat=None, peak=3.0):
    """Beat features: flat, accents on every 4th beat, one loud region after a quiet one."""
    t = np.arange(0.0, duration, 60.0 / bpm)
    n = len(t)
    onset = np.ones(n)
    onset[::4] += 0.5
    rms = np.ones(n) * 0.5
    if climax_beat is not None:
        rms[climax_beat - 8 : climax_beat] = 0.2
        rms[climax_beat : climax_beat + 8] = peak
        onset[climax_beat : climax_beat + 8] += 2.0
    return t, onset, rms


def test_downbeat_phase_follows_accent():
    _, onset, _ = synth_beats()
    assert downbeat_phase(onset) == 0
    assert downbeat_phase(np.roll(onset, 2)) == 2


def test_scores_peak_at_climax_after_build():
    t, onset, rms = synth_beats(climax_beat=40)
    s = climax_scores(onset, rms)
    assert np.nanargmax(s) == 40
    assert np.isnan(s[0]) and np.isnan(s[-1])


def test_schedule_picks_climax_on_downbeat():
    t, onset, rms = synth_beats(climax_beat=40)
    ev = schedule_tricks(t, onset, rms, 60.0, n_tricks=1)
    assert len(ev) == 1
    assert ev[0].t_start == pytest.approx(t[40])
    assert ev[0].trick == "roulade" and ev[0].bridge_s == 1.0


def test_schedule_constraints():
    t, onset, rms = synth_beats(climax_beat=40)
    rms[8:12] = 5.0  # a louder burst before 8 s must still be rejected
    onset[8:12] += 5.0
    ev = schedule_tricks(t, onset, rms, 60.0, n_tricks=2)
    assert 1 <= len(ev) <= 2
    for e in ev:
        assert e.t_start >= 8.0
        assert e.t_start + trick_duration(e.trick) + e.bridge_s <= 60.0 - 6.0
    if len(ev) == 2:
        first_end = ev[0].t_start + trick_duration(ev[0].trick) + ev[0].bridge_s
        assert ev[1].t_start - first_end >= 8.0 - 1e-9
    assert [e.t_start for e in ev] == sorted(e.t_start for e in ev)


def test_no_trick_in_last_6_s():
    t, onset, rms = synth_beats(duration=60.0, climax_beat=112)  # climax at 56 s
    ev = schedule_tricks(t, onset, rms, 60.0, n_tricks=2)
    assert all(e.t_start + trick_duration(e.trick) + e.bridge_s <= 54.0 for e in ev)
    assert all(e.t_start < 54.0 for e in ev)


def test_short_song_gives_no_trick():
    t, onset, rms = synth_beats(duration=14.0, climax_beat=16)
    assert schedule_tricks(t, onset, rms, 14.0) == []


def test_real_beats_files_schedule():
    for song in ("swag", "funky_magic", "fallin_for_you"):
        beats = json.loads((REPO / f"dance/data/beats/songs/{song}.json").read_text())
        ev = schedule_from_beats(beats)
        assert 1 <= len(ev) <= 2
        assert all(e.t_start >= 8.0 for e in ev)
        assert all(e.t_start + trick_duration(e.trick) + e.bridge_s <= beats["duration_s"] - 6.0 for e in ev)
        saved = json.loads((REPO / f"dance/data/tricks/{song}.json").read_text())
        assert [round(e.t_start, 3) for e in ev] == [s["t_start"] for s in saved]


def test_missing_features_raise():
    with pytest.raises(KeyError):
        schedule_from_beats({"beat_times_s": [0, 1], "duration_s": 10})


def test_trick_commands():
    c, done = trick_command("roulade", 0.5)
    assert not c.any() and not done and c.shape == (13,)
    assert trick_command("roulade", 1.0)[1]
    assert trick_command("sitstand", 2.9)[0][0] == 1.0
    c, done = trick_command("sitstand", 3.1)
    assert c[0] == 0.0 and not done
    assert trick_command("sitstand", 6.0)[1]
    with pytest.raises(ValueError):
        trick_command("backflip", 0.0)


def test_is_upright_thresholds():
    assert is_upright(1.0, 0.14)
    assert not is_upright(math.cos(math.radians(25)), 0.14)
    assert not is_upright(1.0, 0.06)


def run_machine(m, signal, t_end):
    """signal(t) -> (up_z, height). Returns the list of (t, state)."""
    out, t = [], 0.0
    while t < t_end:
        out.append((t, m.step(t, *signal(t))))
        t += DT
    return out


def first(trace, state):
    return next((t for t, s in trace if s == state), None)


def test_roulade_upright_passes_after_bridge_and_hold():
    m = TrickHandover(10.0, "roulade", 1.0)
    trace = run_machine(m, lambda t: (1.0, 0.14), 20.0)
    assert trace[0][1] == DANCE
    assert first(trace, TRICK) == pytest.approx(10.0, abs=2 * DT)
    assert first(trace, HOLD) == pytest.approx(11.0, abs=2 * DT)
    # bridge_s is 1.0 and the upright check (0.5 s) is already satisfied inside it
    assert first(trace, DANCE_AFTER) == pytest.approx(12.0, abs=2 * DT)
    assert not m.timed_out
    assert trace[-1][1] == DANCE_AFTER


def test_upright_must_hold_for_half_a_second():
    m = TrickHandover(5.0, "sitstand", 0.0)

    def sig(t):
        # trick ends at 11.0. Upright flickers off at 11.3 s, so the 0.5 s run restarts there.
        return (0.3, 0.05) if 11.25 <= t < 11.3 else (1.0, 0.14)

    trace = run_machine(m, sig, 20.0)
    assert first(trace, HOLD) == pytest.approx(11.0, abs=2 * DT)
    assert first(trace, DANCE_AFTER) == pytest.approx(11.3 + 0.5, abs=2 * DT)


def test_timeout_path_hands_back_and_flags():
    m = TrickHandover(5.0, "roulade", 1.0, hold_timeout_s=4.0)
    trace = run_machine(m, lambda t: (-1.0, 0.03), 30.0)  # stays on its back
    assert first(trace, HOLD) == pytest.approx(6.0, abs=2 * DT)
    assert first(trace, DANCE_AFTER) == pytest.approx(10.0, abs=2 * DT)
    assert m.timed_out


def test_late_recovery_before_timeout():
    m = TrickHandover(5.0, "roulade", 1.0, hold_timeout_s=4.0)
    trace = run_machine(m, lambda t: (1.0, 0.14) if t >= 8.0 else (-1.0, 0.03), 30.0)
    assert first(trace, DANCE_AFTER) == pytest.approx(8.5, abs=2 * DT)
    assert not m.timed_out


def test_trick_command_from_machine():
    m = TrickHandover(5.0, "sitstand", 0.0)
    assert m.command(6.0)[0] == 1.0
    assert m.command(9.0)[0] == 0.0
