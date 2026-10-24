"""Tests for the beat groove layer and head_gain in the retarget (Task 11)."""

from __future__ import annotations

import dataclasses
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

from microduck_dance.command_file import load_command_file
from microduck_dance.groove import beat_phase, beat_pulse, check_beats, groove_layer, load_beats
from microduck_dance.ranges import ALL_RANGES
from microduck_dance.retarget import TASK15_OPTIONS, RetargetConfig, smpl_to_commands

sys.path.insert(0, str(Path(__file__).parent))
from test_dance_retarget import PELVIS_Y, make_motion, random_motion, still_motion  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DT = 0.02
NO_RAMP = dict(ramp_in_s=0.0, ramp_out_s=0.0)
# These tests check the layer shape at the configured amplitude (Task 11), so the
# Task 19 half amplitude is undone. test_dance_fidelity.py tests groove_scale.
FULL = dict(groove_scale=1.0)
HP, HY, BZ = 4, 5, 9
LO = np.array([r[0] for r in ALL_RANGES])
HI = np.array([r[1] for r in ALL_RANGES])


def irregular_beats():
    # 110 to 125 BPM with jitter, on and off the 20 ms grid.
    rng = np.random.default_rng(3)
    return np.cumsum(np.concatenate([[1.013], rng.uniform(0.48, 0.545, size=30)]))


def _local_maxima(x):
    return np.where((x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:]) & (x[1:-1] > 1e-6))[0] + 1


def test_pulse_peaks_land_at_beat_plus_lag():
    beats = irregular_beats()
    cfg = RetargetConfig(**NO_RAMP)
    out = smpl_to_commands(*still_motion(600, 30), fps=30, cfg=cfg, beats=beats)
    t = np.arange(len(out)) * DT
    for col, sign in ((HP, 1.0), (BZ, -1.0)):
        peaks = t[_local_maxima(sign * out[:, col])]
        expected = beats + cfg.groove_lag_s
        expected = expected[(expected > 0.3) & (expected < t[-1] - 0.3)]
        assert len(peaks) == len(expected)
        assert np.abs(peaks - expected).max() <= DT + 1e-9


def test_pulse_amplitude_and_width_match_config():
    beats = np.arange(1.0, 19.0, 0.5)  # on the frame grid, so the peak frame is the peak
    cfg = RetargetConfig(**NO_RAMP, **FULL, groove_lag_s=0.04)
    out = smpl_to_commands(*still_motion(600, 30), fps=30, cfg=cfg, beats=beats)
    assert out[:, HP].max() == pytest.approx(0.35, abs=1e-9)
    assert out[:, BZ].min() == pytest.approx(-0.015, abs=1e-9)
    assert out[:, HP].min() == pytest.approx(0.0, abs=1e-12)  # a dip only, never up
    # Width 0.6 x 0.5 s: 0.3 s of non-zero pulse per beat.
    t = np.arange(len(out)) * DT
    one = (t > 2.04 - 0.25) & (t < 2.04 + 0.25)
    nz = t[one][out[one, HP] > 1e-9]
    assert nz.max() - nz.min() == pytest.approx(0.3 - 2 * DT, abs=DT + 1e-9)
    # Raised cosine shape: 3 frames (0.06 s) from the centre of a 0.3 s wide pulse.
    i = int(round((2.04 + 0.06) / DT))
    assert out[i, HP] == pytest.approx(0.35 * 0.5 * (1 + math.cos(2 * math.pi * 0.06 / 0.3)), abs=1e-9)


def test_sway_is_half_beat_and_phase_locked():
    beats = 1.0 + 0.52 * np.arange(34)  # half a period is 13 frames
    cfg = RetargetConfig(**NO_RAMP, **FULL, groove_lag_s=0.04)
    out = smpl_to_commands(*still_motion(600, 30), fps=30, cfg=cfg, beats=beats)
    idx = np.round((beats + 0.04) / DT).astype(int)
    vals = out[idx, HY]
    np.testing.assert_allclose(vals, 0.5 * (-1.0) ** np.arange(len(beats)), atol=1e-9)
    # Between beats it crosses zero.
    mid = np.round((beats[:-1] + 0.26 + 0.04) / DT).astype(int)
    assert np.abs(out[mid, HY]).max() < 1e-9
    assert np.abs(out[:, HY]).max() == pytest.approx(0.5, abs=1e-9)


def test_beat_phase_extrapolates_with_edge_periods():
    beats = np.array([1.0, 1.5, 2.0, 2.6])
    np.testing.assert_allclose(beat_phase(np.array([0.0, 1.0, 1.25, 2.6, 3.2]), beats), [-2, 0, 0.5, 3, 4])


def test_pulse_never_exceeds_one_when_pulses_overlap():
    p = beat_pulse(np.linspace(0, 3, 3001), [1.0, 1.1, 2.0], width_frac=1.5, lag_s=0.0)
    assert p.max() == pytest.approx(1.0) and p.min() >= 0.0


def test_check_beats_rejects_bad_input():
    with pytest.raises(ValueError):
        check_beats([1.0])
    with pytest.raises(ValueError):
        check_beats([1.0, 0.9, 2.0])


def test_groove_off_and_no_beats_give_same_output():
    motion = random_motion(seed=4, F=600, fps=30)
    beats = irregular_beats()
    base = smpl_to_commands(*motion, fps=30, cfg=RetargetConfig())
    off = smpl_to_commands(*motion, fps=30, cfg=RetargetConfig(groove=False), beats=beats)
    np.testing.assert_array_equal(base, off)


def test_head_gain_scales_head_columns_only():
    a = math.radians(10)
    poses, trans = make_motion(150, 30, 1, lambda t: (0.0, PELVIS_Y, 0.0), lambda t: (0.0, 0.0, 0.0),
                               head_rotvec_fn=lambda t: (a, a, 0.0))
    one = smpl_to_commands(poses, trans, fps=30, cfg=RetargetConfig(ramp_out_s=0.0, head_gain=1.0))
    two = smpl_to_commands(poses, trans, fps=30, cfg=RetargetConfig(ramp_out_s=0.0, head_gain=2.0))
    np.testing.assert_allclose(two[100:, 3:7], 2.0 * one[100:, 3:7], atol=1e-9)
    np.testing.assert_array_equal(two[:, [0, 1, 2, 9, 10, 11]], one[:, [0, 1, 2, 9, 10, 11]])
    assert np.abs(one[100:, 3:6]).min() > 0.01  # the per-column check below is not vacuous


def test_head_gain_is_per_column_and_head_roll_default_is_one():
    assert tuple(RetargetConfig().head_gain) == (2.0, 2.0, 2.0, 1.0)
    a = math.radians(10)
    poses, trans = make_motion(150, 30, 1, lambda t: (0.0, PELVIS_Y, 0.0), lambda t: (0.0, 0.0, 0.0),
                               head_rotvec_fn=lambda t: (a, a, a / 2))
    one = smpl_to_commands(poses, trans, fps=30, cfg=RetargetConfig(ramp_out_s=0.0, head_gain=1.0))
    dflt = smpl_to_commands(poses, trans, fps=30, cfg=RetargetConfig(ramp_out_s=0.0))
    per = smpl_to_commands(poses, trans, fps=30, cfg=RetargetConfig(ramp_out_s=0.0, head_gain=(1.5, 0.5, 3.0, 1.0)))
    assert np.abs(one[100:, 3:7]).min() > 0.01
    np.testing.assert_allclose(dflt[100:, 3:7], one[100:, 3:7] * [2.0, 2.0, 2.0, 1.0], atol=1e-9)
    np.testing.assert_allclose(per[100:, 3:7], one[100:, 3:7] * [1.5, 0.5, 3.0, 1.0], atol=1e-9)
    with pytest.raises(ValueError):
        smpl_to_commands(poses, trans, fps=30, cfg=RetargetConfig(head_gain=(1.0, 2.0)))


def test_default_lag_leads_the_beat_by_measured_policy_lag():
    from microduck_dance.retarget import GROOVE_POLICY_LAG_COMP_S

    assert GROOVE_POLICY_LAG_COMP_S == -0.14 and RetargetConfig().groove_lag_s == -0.14
    beats = np.arange(1.0, 19.0, 0.5)
    out = smpl_to_commands(*still_motion(600, 30), fps=30, cfg=RetargetConfig(**NO_RAMP, **FULL), beats=beats)
    t = np.arange(len(out)) * DT
    peaks = t[_local_maxima(out[:, HP])]
    np.testing.assert_allclose(peaks, (beats - 0.14)[: len(peaks)], atol=DT / 2 + 1e-9)
    assert len(peaks) == len(beats)
    # The sway extreme is shifted by the same lag: +0.5 at beat 0 - 0.14 s.
    assert out[int(round((1.0 - 0.14) / DT)), HY] == pytest.approx(0.5, abs=1e-9)


def test_clipping_holds_with_large_groove_and_head_gain():
    cfg = RetargetConfig(head_gain=10.0, groove_nod=3.0, groove_bounce=-0.2, groove_sway=3.0)
    out = smpl_to_commands(*random_motion(seed=0, F=600, fps=30), fps=30, cfg=cfg, beats=irregular_beats())
    assert np.all(out >= LO - 1e-9) and np.all(out <= HI + 1e-9)
    assert np.isclose(out[:, HP], HI[HP]).any() and np.isclose(out[:, BZ], LO[BZ]).any()


def test_groove_ramps_in_and_out_with_the_clip():
    beats = np.arange(0.0, 13.0, 0.5)  # a beat at t = 0 and at the last frame
    cfg = RetargetConfig(**FULL)
    out = smpl_to_commands(*still_motion(360, 30), fps=30, cfg=cfg, beats=beats)
    assert np.all(out[0] == 0.0) and np.all(out[-1] == 0.0)
    t = np.arange(len(out)) * DT
    full = groove_layer(t, beats, 0.35, -0.015, 0.5, 0.6, cfg.groove_lag_s)
    ramp = np.clip(t / cfg.ramp_in_s, 0, 1) * np.clip((t[-1] - t) / cfg.ramp_out_s, 0, 1)
    np.testing.assert_allclose(out[:, [HP, HY, BZ]], full[:, [HP, HY, BZ]] * ramp[:, None], atol=1e-12)


def test_load_beats_reads_json(tmp_path):
    p = tmp_path / "b.json"
    p.write_text(json.dumps({"beat_times_s": [0.5, 1.0, 1.5], "tempo_bpm": 120.0}))
    np.testing.assert_array_equal(load_beats(p), [0.5, 1.0, 1.5])


def test_retarget_motion_cli_applies_beats(tmp_path):
    sys.path.insert(0, str(REPO / "dance/scripts/retarget"))
    import retarget_motion  # noqa: E402

    poses, trans = still_motion(300, 30)
    pkl = tmp_path / "m.pkl"
    with pkl.open("wb") as fh:
        pickle.dump({"smpl_poses": poses, "smpl_trans": trans}, fh)
    beats = tmp_path / "b.json"
    beats.write_text(json.dumps({"beat_times_s": list(np.arange(1.0, 9.0, 0.5))}))
    shared = ["--phrase-gate", "0", "--groove-pattern", "0"]  # a json without music features (Task 15)
    retarget_motion.main([str(pkl), str(tmp_path / "on.npz"), "--beats", str(beats), *shared, "--groove-scale", "1"])
    retarget_motion.main([str(pkl), str(tmp_path / "off.npz"), "--beats", str(beats), "--groove", "0", *shared])
    on, _, meta = load_command_file(tmp_path / "on.npz")
    off, _, meta_off = load_command_file(tmp_path / "off.npz")
    assert meta["groove_applied"] and not meta_off["groove_applied"]
    assert meta["retarget_config"]["head_gain"] == [2.0, 2.0, 2.0, 1.0]
    assert on[:, HP].max() > 0.3 and np.abs(off).max() < 1e-6
    assert dataclasses.asdict(RetargetConfig())["groove"] is True
