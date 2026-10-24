"""Tests for compute_stats in dance/scripts/evaluate/play_dance.py (CPU only, no env)."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_PATH = Path(__file__).resolve().parents[1] / "dance" / "scripts" / "evaluate" / "play_dance.py"
_spec = importlib.util.spec_from_file_location("play_dance", _PATH)
play_dance = importlib.util.module_from_spec(_spec)
sys.modules["play_dance"] = play_dance  # dataclass needs its module registered
_spec.loader.exec_module(play_dance)
compute_stats = play_dance.compute_stats


def test_perfect_tracking_has_zero_error():
    cmd = np.random.default_rng(0).uniform(-0.1, 0.1, (100, 13))
    s = compute_stats(cmd, cmd.copy(), np.zeros(100, bool), 0.02)
    assert s.falls == 0 and s.twist_rmse == 0 and s.head_rmse == 0 and s.body_rmse == 0 and s.duration_s == 2.0


def test_falls_count_rising_edges():
    term = np.zeros(100, bool); term[[10, 11, 50]] = True
    assert compute_stats(np.zeros((100, 13)), np.zeros((100, 13)), term, 0.02).falls == 2


def test_rmse_skips_terminated_steps():
    cmd = np.zeros((10, 13)); meas = np.zeros((10, 13))
    meas[:, 0] = 0.3            # constant vx error on every step
    meas[4, :] = 100.0          # garbage on the reset step
    term = np.zeros(10, bool); term[4] = True
    s = compute_stats(cmd, meas, term, 0.02)
    assert s.twist_rmse == pytest.approx(0.3 / np.sqrt(3))
    assert s.head_rmse == 0 and s.body_rmse == 0 and s.falls == 1


def test_fall_steps_are_rising_edges():
    term = np.zeros(20, bool); term[[3, 4, 12]] = True
    assert play_dance.fall_steps(term) == [3, 12]


def test_check_clock_flags_a_clip_that_restarted():
    T = 10
    on_clock = np.minimum(np.arange(1, 15), T - 1)[:, None].repeat(2, axis=1)
    assert play_dance._check_clock(on_clock, T) == 0
    restarted = on_clock.copy(); restarted[5:, 1] = np.arange(1, 10)  # env 1 fell at step 4 and replayed
    assert play_dance._check_clock(restarted, T) > 0


def test_report_falls_gives_time_and_resumed_index():
    term = np.zeros((10, 1), bool); term[4, 0] = True
    idx = np.arange(1, 11)[:, None]
    (f,) = play_dance._report_falls(term, idx, 0.02)
    assert f == {"step": 4, "t_s": 0.1, "clip_index_after_reset": 5, "global_step_after": 5}


def test_fix_spawn_pose_zeroes_xy_and_yaw_only():
    from types import SimpleNamespace

    pose_range = {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.12, 0.13), "yaw": (-3.14, 3.14)}
    cfg = SimpleNamespace(events={"reset_base": SimpleNamespace(params={"pose_range": pose_range})})
    play_dance.fix_spawn_pose(cfg)
    assert pose_range == {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.12, 0.13), "yaw": (0.0, 0.0)}


def test_twist_corr_smoothed_per_channel():
    t = np.arange(500) * 0.02
    cmd = np.zeros((500, 13))
    cmd[:, 0] = np.sin(t)
    cmd[:, 1] = np.cos(0.7 * t)
    meas = cmd.copy()
    meas[:, 1] *= -1.0  # anti-correlated vy
    meas[:, 0] += 0.5 * np.sign(np.sin(40 * t))  # fast wobble the 0.5 s average removes
    s = compute_stats(cmd, meas, np.zeros(500, bool), 0.02)
    assert s.twist_corr_vx > 0.95
    assert s.twist_corr_vy == pytest.approx(-1.0)
    assert np.isnan(s.twist_corr_wz)  # constant command
    assert s.twist_corr == pytest.approx((s.twist_corr_vx + s.twist_corr_vy) / 2)


def test_twist_corr_stops_at_first_fall():
    t = np.arange(400) * 0.02
    cmd = np.zeros((400, 13))
    cmd[:, 2] = np.sin(t)
    meas = cmd.copy()
    meas[200:, 2] = -cmd[200:, 2]  # after the fall the signal is garbage
    term = np.zeros(400, bool)
    term[200] = True
    assert compute_stats(cmd, meas, term, 0.02).twist_corr_wz == pytest.approx(1.0)
