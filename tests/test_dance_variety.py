"""Tests for the Task 15 retarget options: foot velocity, phrase gate, per-song groove pattern."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

from microduck_dance.command_file import load_command_file
from microduck_dance.groove import (
    BOUNCE_DEEP, BOUNCE_SHALLOW, ONSET_MEAN_HI, ONSET_MEAN_LO, TRUNK_SWAY, accent_position, bounce_depth,
    choose_pattern,
)
from microduck_dance.phrases import classify_phrases, phrase_starts, plan_phrases, twist_gate
from microduck_dance.ranges import ALL_RANGES
from microduck_dance.retarget import TASK15_OPTIONS, RetargetConfig, smpl_to_commands

sys.path.insert(0, str(Path(__file__).parent))
from test_dance_retarget import FLOOR_Z, Y_TO_Z, flowerdance_squat, still_motion, walking_motion  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DT = 0.02
NO_RAMP = dict(ramp_in_s=0.0, ramp_out_s=0.0)
VX, VY, WZ, HP, HY, BZ, BR = 0, 1, 2, 4, 5, 9, 10
LO = np.array([r[0] for r in ALL_RANGES])
HI = np.array([r[1] for r in ALL_RANGES])


def _local_maxima(x):
    return np.where((x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:]) & (x[1:-1] > 1e-6))[0] + 1


def make_music(n_beats=64, period=0.5, t0=1.0, tempo=120.0, loud_phrases=(1, 3), flat=False,
               accent=0, onset_mean=2.0):
    """Beats json with 16-beat phrases; loud_phrases get 1.5x RMS and onset unless flat."""
    beats = t0 + period * np.arange(n_beats)
    phrase = np.arange(n_beats) // 16
    level = np.where(np.isin(phrase, loud_phrases) & (not flat), 1.5, 1.0)
    at_beat = np.where(np.arange(n_beats) % 4 == accent, 3.0, 1.0)
    return {"beat_times_s": beats.tolist(), "tempo_bpm": tempo, "onset_mean": onset_mean,
            "onset_at_beat": at_beat.tolist(), "beat_onset": (2.0 * level).tolist(),
            "beat_rms": (0.2 * level).tolist()}


# Part A: support-foot velocity

FWD = Y_TO_Z.apply([0.0, 0.0, 1.0])  # body forward in the z-up world of flowerdance_squat


def test_sliding_in_place_gives_near_zero_velocity_with_foot_velocity():
    poses, trans, joints = flowerdance_squat(depth=0.0)
    t = np.arange(len(trans)) / 30
    slide = joints + (0.2 * t)[:, None, None] * FWD  # whole body, feet included, glides forward
    off = smpl_to_commands(poses, trans, fps=30, cfg=RetargetConfig(vel_gain=1.0, foot_velocity=False), joints=slide)
    on = smpl_to_commands(poses, trans, fps=30, cfg=RetargetConfig(vel_gain=1.0, foot_velocity=True), joints=slide)
    assert off[100:-60, VX].mean() == pytest.approx(0.2, abs=0.02)
    assert np.abs(on[:, :2]).max() < 1e-3


def stepping_joints(speed=0.3, F=450, fps=30, step_s=0.5, lift=0.05):
    """Pelvis moves at speed; each foot alternates stance (planted) and swing (2x speed, lifted)."""
    poses, trans, joints = flowerdance_squat(F=F, fps=fps, depth=0.0)
    t = np.arange(F) / fps
    joints = joints + (speed * t)[:, None, None] * FWD
    for j, (foot, phase0) in enumerate(((10, 0), (11, 1))):
        k = (np.floor(t / step_s).astype(int) + phase0) % 2  # 1 while this foot swings
        v = np.where(k == 1, 2.0 * speed, 0.0)
        s = np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) / fps)])
        u = (t % step_s) / step_s
        base = joints[0, foot] - 0.0 * FWD
        joints[:, foot] = base + s[:, None] * FWD
        joints[:, foot, 2] = FLOOR_Z + np.where(k == 1, lift * np.sin(np.pi * u), 0.0)
    return poses, trans, joints


def test_stepping_forward_keeps_its_speed_with_foot_velocity():
    poses, trans, joints = stepping_joints(speed=0.3)
    on = smpl_to_commands(poses, trans, fps=30, cfg=RetargetConfig(vel_gain=1.0, foot_velocity=True), joints=joints)
    off = smpl_to_commands(poses, trans, fps=30, cfg=RetargetConfig(vel_gain=1.0, foot_velocity=False), joints=joints)
    assert off[100:-60, VX].mean() == pytest.approx(0.3, abs=0.02)
    assert on[100:-60, VX].mean() == pytest.approx(0.3, abs=0.03)


def test_foot_velocity_default_off_and_ignored_without_joints():
    assert RetargetConfig().foot_velocity is False
    m = walking_motion(speed=0.1, F=150, fps=30)
    a = smpl_to_commands(*m, fps=30, cfg=RetargetConfig(foot_velocity=True))
    b = smpl_to_commands(*m, fps=30, cfg=RetargetConfig(foot_velocity=False))
    np.testing.assert_array_equal(a, b)


# Part B: phrases and the twist gate

def test_phrase_starts_every_16_beats_and_short_tail_merged():
    assert phrase_starts(89) == [0, 16, 32, 48, 64, 80]  # last phrase has 9 beats, kept
    assert phrase_starts(117) == [0, 16, 32, 48, 64, 80, 96]  # 5-beat tail merged
    assert phrase_starts(10) == [0]


def test_classify_by_energy_median_and_alternate_when_flat():
    m = make_music(loud_phrases=(1, 3))
    starts = phrase_starts(64)
    c = classify_phrases(m["beat_rms"], m["beat_onset"], starts)
    assert c["rule"] == "energy" and c["travel"] == [False, True, False, True]
    m = make_music(loud_phrases=(0, 1))
    assert classify_phrases(m["beat_rms"], m["beat_onset"], starts)["travel"] == [True, True, False, False]
    flat = make_music(flat=True)
    c = classify_phrases(flat["beat_rms"], flat["beat_onset"], starts)
    assert c["rule"] == "alternate" and c["travel"] == [False, True, False, True]
    assert c["rms_cv"] < 0.08


def test_cv_threshold_switches_rule():
    # Phrase RMS 1.0, 1.1, 1.0, 1.1: CV 0.048, below 0.08, so alternate even though phrase 0 is quiet.
    rms = np.repeat([1.0, 1.1, 1.0, 1.1], 16)
    c = classify_phrases(rms, np.ones(64), phrase_starts(64))
    assert c["rule"] == "alternate"
    c = classify_phrases(rms, np.ones(64), phrase_starts(64), min_cv=0.04)
    assert c["rule"] == "energy" and c["travel"] == [False, True, False, True]


def test_gate_fades_inside_groove_phrase_and_is_smooth():
    t = np.arange(0, 20, DT)
    g = twist_gate(t, np.array([5.0, 10.0]), [True, False, True], fade_s=0.5)
    assert np.all(g[t < 5.0] == 1.0) and np.all(g[t >= 10.0] == 1.0)
    inner = (t >= 5.5 - 1e-9) & (t <= 9.5 + 1e-9)
    assert np.all(g[inner] == 0.0)
    assert np.all(np.diff(g[(t >= 5.0) & (t < 5.5)]) <= 0) and np.all(np.diff(g[(t >= 9.5) & (t < 10.0)]) >= 0)
    # Raised cosine: the steepest step is pi / 2 times the mean slope.
    assert np.abs(np.diff(g)).max() <= np.pi / 2 * DT / 0.5 + 1e-9


def _pipeline(music, cfg_kw=None, motion=None):
    # The Task 15 options are tested with the Task 15 defaults (gate on, full groove,
    # no posture lead, open-loop wz). Task 19 changed the defaults (test_dance_fidelity.py).
    cfg = RetargetConfig(**NO_RAMP, **{**TASK15_OPTIONS, **(cfg_kw or {})})
    motion = motion if motion is not None else walking_motion(speed=0.15, F=30 * 36, fps=30)
    return smpl_to_commands(*motion, fps=30, cfg=cfg, music=music, return_info=True)


def test_twist_zero_in_groove_phrases_and_unchanged_in_travel():
    music = make_music()
    on, info = _pipeline(music)
    off, _ = _pipeline(music, {"phrase_gate": False})
    t = np.arange(len(on)) * DT
    starts = np.asarray(info["phrases"]["start_times_s"])
    beats = np.asarray(music["beat_times_s"])
    assert np.all(np.isin(starts, beats))  # phrase boundaries on beats
    np.testing.assert_allclose(starts, beats[[0, 16, 32, 48]])
    edges = np.concatenate([[-np.inf], starts[1:], [np.inf]])
    assert np.abs(off[200:, VX]).min() > 0.1  # the twist being gated is not zero to begin with
    for k, travel in enumerate(info["phrases"]["travel"]):
        lo, hi = edges[k], edges[k + 1]
        if travel:
            m = (t >= lo) & (t < hi)
            np.testing.assert_array_equal(on[m, :3], off[m, :3])
        else:
            m = (t >= lo + 0.5 + 1e-9) & (t < hi - 0.5 - 1e-9)
            assert m.sum() > 100 and np.all(on[m, :3] == 0.0)
    # Non-twist columns differ only through the groove gain, never through the gate.
    np.testing.assert_array_equal(on[:, [3, 6, 11]], off[:, [3, 6, 11]])
    assert 0.4 < info["phrases"]["travel_time_frac"] < 0.6


def test_gated_twist_has_no_jumps():
    on, _ = _pipeline(make_music())
    # vx below 0.4 m/s; the fade moves at most pi/2 * 0.4 / 0.5 m/s per second.
    assert np.abs(np.diff(on[:, VX])).max() <= np.pi / 2 * 0.4 / 0.5 * DT + 1e-6


def test_no_music_means_no_gate_and_shared_groove():
    beats = 1.0 + 0.5 * np.arange(64)
    a, info = smpl_to_commands(*walking_motion(0.15, 30 * 36, 30), fps=30, cfg=RetargetConfig(**NO_RAMP),
                               beats=beats, return_info=True)
    assert info["phrases"] is None and info["groove"] == "uniform"
    assert np.abs(a[200:, VX]).min() > 0.03  # vel_gain 0.3 x 0.15 m/s


def test_music_missing_keys_raises():
    with pytest.raises(ValueError):
        _pipeline({"beat_times_s": [1.0, 1.5, 2.0]})


# Part C: per-song groove pattern

def test_accent_position_picks_strongest_bar_position():
    o = np.tile([1.0, 0.8, 1.4, 0.9], 8)
    assert accent_position(o) == 2
    assert choose_pattern({"onset_at_beat": o, "tempo_bpm": 120, "onset_mean": 2.0}, 0.35, 0.5)["nod_positions"] == [0, 2]
    o = np.tile([2.0, 0.8, 1.4, 0.9], 8)
    assert choose_pattern({"onset_at_beat": o, "tempo_bpm": 120, "onset_mean": 2.0}, 0.35, 0.5)["nod_positions"] == [0]


def test_nods_land_only_on_chosen_bar_positions():
    music = make_music(accent=3, flat=True)
    out, info = _pipeline(music, {"phrase_gate": False}, motion=still_motion(30 * 36, 30))
    assert info["pattern"]["nod_positions"] == [0, 3]
    t = np.arange(len(out)) * DT
    peaks = t[_local_maxima(out[:, HP])]
    beats = np.asarray(music["beat_times_s"])
    expected = beats[np.isin(np.arange(len(beats)) % 4, [0, 3])] - 0.14
    expected = expected[expected < t[-1]]
    np.testing.assert_allclose(peaks, expected, atol=DT / 2 + 1e-9)
    assert out[:, HP].max() == pytest.approx(0.35, abs=1e-6)


def test_bounce_depth_linear_in_onset_strength_and_clipped():
    assert bounce_depth(ONSET_MEAN_LO) == pytest.approx(BOUNCE_SHALLOW)
    assert bounce_depth(ONSET_MEAN_HI) == pytest.approx(BOUNCE_DEEP)
    assert bounce_depth(0.5 * (ONSET_MEAN_LO + ONSET_MEAN_HI)) == pytest.approx(-0.0175)
    assert bounce_depth(0.0) == pytest.approx(-0.010) and bounce_depth(10.0) == pytest.approx(-0.025)
    music = make_music(flat=True, onset_mean=ONSET_MEAN_HI)
    out, _ = _pipeline(music, {"phrase_gate": False}, motion=still_motion(30 * 36, 30))
    assert out[:, BZ].min() == pytest.approx(-0.025, abs=1e-6)


def test_slow_tempo_sways_trunk_fast_tempo_sways_head():
    slow, info = _pipeline(make_music(tempo=92.0, flat=True), {"phrase_gate": False}, motion=still_motion(1080, 30))
    assert info["pattern"]["sway_type"] == "trunk_roll"
    assert np.abs(slow[:, HY]).max() < 1e-9
    assert np.abs(slow[:, BR]).max() == pytest.approx(TRUNK_SWAY, abs=1e-3)
    fast, info = _pipeline(make_music(tempo=118.0, flat=True), {"phrase_gate": False}, motion=still_motion(1080, 30))
    assert info["pattern"]["sway_type"] == "head_yaw"
    assert np.abs(fast[:, BR]).max() < 1e-9
    assert np.abs(fast[:, HY]).max() == pytest.approx(0.5, abs=1e-3)
    # Trunk sway completes one cycle every two beats: it is at opposite sides on consecutive beats.
    beats = np.asarray(make_music()["beat_times_s"])
    idx = np.round((beats[2:20] - 0.14) / DT).astype(int)
    np.testing.assert_allclose(slow[idx, BR], TRUNK_SWAY * (-1.0) ** np.arange(2, 20), atol=2e-3)


def test_groove_is_1p3x_in_groove_phrases():
    music = make_music()  # phrases 0 and 2 groove
    out, info = _pipeline(music, motion=still_motion(30 * 36, 30))
    t = np.arange(len(out)) * DT
    starts = info["phrases"]["start_times_s"]
    g = (t > starts[0] + 1.0) & (t < starts[1] - 1.0)
    tr = (t > starts[1] + 1.0) & (t < starts[2] - 1.0)
    assert out[g, HP].max() == pytest.approx(1.3 * 0.35, abs=1e-6)
    assert out[tr, HP].max() == pytest.approx(0.35, abs=1e-6)
    assert np.abs(out[g, HY]).max() == pytest.approx(1.3 * 0.5, abs=1e-3)


def test_pattern_output_within_ranges():
    from test_dance_retarget import random_motion
    music = make_music(tempo=92.0, onset_mean=ONSET_MEAN_HI)
    cfg = {"head_gain": 10.0, "groove_nod": 3.0, "groove_phrase_gain": 3.0}
    out, _ = _pipeline(music, cfg, motion=random_motion(seed=1, F=1080, fps=30))
    assert np.all(out >= LO - 1e-9) and np.all(out <= HI + 1e-9)


def test_cli_stores_pattern_and_phrases_in_meta(tmp_path):
    sys.path.insert(0, str(REPO / "dance/scripts/retarget"))
    import retarget_motion  # noqa: E402

    poses, trans = walking_motion(0.15, 30 * 36, 30)
    pkl = tmp_path / "m.pkl"
    with pkl.open("wb") as fh:
        pickle.dump({"smpl_poses": poses, "smpl_trans": trans}, fh)
    bj = tmp_path / "b.json"
    bj.write_text(json.dumps(make_music(tempo=92.0, accent=1)))
    retarget_motion.main([str(pkl), str(tmp_path / "o.npz"), "--beats", str(bj), "--phrase-gate", "1"])
    _, _, meta = load_command_file(tmp_path / "o.npz")
    gi = meta["groove_info"]
    assert gi["groove"] == "pattern"
    assert gi["pattern"]["nod_positions"] == [0, 1] and gi["pattern"]["sway_type"] == "trunk_roll"
    assert gi["phrases"]["travel"] == [False, True, False, True]
    assert meta["retarget_config"]["phrase_gate"] is True
    assert meta["retarget_config"]["vel_gain"] == RetargetConfig().vel_gain


def test_real_beats_json_has_music_features():
    p = REPO / "dance/data/beats/songs/swag.json"
    if not p.exists():
        pytest.skip("beats missing")
    m = json.loads(p.read_text())
    plan = plan_phrases(m["beat_times_s"], m["beat_rms"], m["beat_onset"])
    assert plan["start_beats"][0] == 0 and len(plan["travel"]) >= 4
    assert ONSET_MEAN_LO <= m["onset_mean"] <= ONSET_MEAN_HI
