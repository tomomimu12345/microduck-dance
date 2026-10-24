"""Lagged-follow camera of play_dance.py, tested on synthetic trunk paths (no GPU, no env)."""

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

_PATH = Path(__file__).resolve().parents[1] / "dance" / "scripts" / "evaluate" / "play_dance.py"
_spec = importlib.util.spec_from_file_location("play_dance", _PATH)
play_dance = importlib.util.module_from_spec(_spec)
sys.modules["play_dance"] = play_dance
_spec.loader.exec_module(play_dance)

DT = 0.02
TAU = play_dance.CAM_TAU_S


def test_time_constant_is_1_5_s():
    assert TAU == 1.5


def test_step_response_reaches_63_percent_after_one_tau():
    # The trunk jumps 0.1 m along x at t = 0 (under the cap) and stays there.
    T = int(round(3 * TAU / DT)) + 1
    pos = np.zeros((T, 3))
    pos[1:, 0] = 0.1
    tgt = play_dance.lagged_follow(pos, DT, TAU, max_offset=None)
    k = int(round(TAU / DT))
    assert tgt[k, 0] / 0.1 == pytest.approx(1 - math.exp(-1), abs=1e-9)
    assert tgt[-1, 0] / 0.1 == pytest.approx(1 - math.exp(-3), abs=1e-9)
    assert np.all(np.diff(tgt[:, 0]) >= 0)


def test_discretization_does_not_depend_on_dt():
    t_end = 2.0
    fine = play_dance.lagged_follow(np.r_[np.zeros((1, 3)), np.ones((int(t_end / 0.01), 3)) * 0.1], 0.01, TAU, None)
    coarse = play_dance.lagged_follow(np.r_[np.zeros((1, 3)), np.ones((int(t_end / 0.02), 3)) * 0.1], 0.02, TAU, None)
    assert fine[-1] == pytest.approx(coarse[-1], abs=1e-12)


def test_walk_then_stop_shows_lag_then_returns_to_centre():
    # 0.3 m/s along +y for 4 s, then standing for 10 s.
    walk, stand = int(4 / DT), int(10 / DT)
    y = np.r_[np.arange(walk) * 0.3 * DT, np.full(stand, (walk - 1) * 0.3 * DT)]
    pos = np.c_[np.zeros_like(y), y, np.full_like(y, 0.11)]
    tgt = play_dance.lagged_follow(pos, DT)
    gap = pos[:, 1] - tgt[:, 1]
    # While walking at constant speed the gap tends to v * tau (0.45 m) but the cap holds it at 0.25 m.
    assert gap[walk - 1] == pytest.approx(play_dance.CAM_MAX_OFFSET, abs=1e-9)
    assert np.all(np.hypot(*(pos[:, :2] - tgt[:, :2]).T) <= play_dance.CAM_MAX_OFFSET + 1e-12)
    # After stopping the camera catches up: under 1 cm after 7 s.
    assert abs(gap[walk + int(7 / DT)]) < 0.01
    # The height is filtered too and stays at the constant trunk height.
    assert np.allclose(tgt[:, 2], 0.11)


def test_slow_sway_is_followed_with_small_gap():
    # A 0.05 m sway at 0.1 Hz moves the target almost as much as the trunk.
    t = np.arange(int(60 / DT)) * DT
    pos = np.c_[0.05 * np.sin(2 * np.pi * 0.1 * t), np.zeros_like(t), np.zeros_like(t)]
    tgt = play_dance.lagged_follow(pos, DT)
    w = 2 * np.pi * 0.1
    gain = 1 / math.sqrt(1 + (w * TAU) ** 2)
    late = t > 20
    assert np.max(np.abs(tgt[late, 0])) == pytest.approx(0.05 * gain, rel=0.02)


def test_cap_keeps_direction():
    tgt = play_dance.lowpass_follow([0.0, 0.0, 0.0], [3.0, 4.0, 0.2], DT, TAU, max_offset=0.5)
    gap = np.array([3.0, 4.0]) - tgt[:2]
    assert np.hypot(*gap) == pytest.approx(0.5)
    assert gap[1] / gap[0] == pytest.approx(4 / 3)
    a = 1 - math.exp(-DT / TAU)
    assert tgt[2] == pytest.approx(0.2 * a)


class _Cam:
    def __init__(self):
        self.lookat = np.zeros(3)
        self.type = self.trackbodyid = self.fixedcamid = None


def test_lagged_camera_sets_free_camera_and_lookat():
    pytest.importorskip("mujoco")
    cam = _Cam()
    lc = play_dance.LaggedCamera(cam, DT)
    assert cam.trackbodyid == -1
    lc.update([1.0, 2.0, 0.1])
    assert np.allclose(cam.lookat, [1.0, 2.0, 0.1])
    lc.update([1.1, 2.0, 0.1])
    assert cam.lookat[0] == pytest.approx(1.0 + 0.1 * (1 - math.exp(-DT / TAU)))


def test_parser_cam_follow_default_is_trunk():
    a = play_dance.build_parser().parse_args(["--checkpoint", "x.pt", "--commands", "c.npz", "--out", "o.mp4"])
    assert a.cam_follow == "trunk" and a.cam_tau == TAU
    b = play_dance.build_parser().parse_args(["--checkpoint", "x.pt", "--commands", "c.npz", "--out", "o.mp4",
                                              "--cam-follow", "lagged"])
    assert b.cam_follow == "lagged"


def test_save_states_stores_feet(tmp_path):
    T = 10
    feet = np.random.default_rng(1).normal(size=(T, 2, 3))
    p = play_dance.save_states(tmp_path / "s.npz", np.zeros((T, 13)), np.zeros((T, 13)), np.zeros(T, bool),
                               np.zeros((T, 3)), np.zeros(T), DT, feet=feet)
    assert np.allclose(np.load(p)["feet"], feet, atol=1e-6)
    with pytest.raises(ValueError):
        play_dance.save_states(tmp_path / "t.npz", np.zeros((T, 13)), np.zeros((T, 13)), np.zeros(T, bool),
                               np.zeros((T, 3)), np.zeros(T), DT, feet=np.zeros((T, 3)))
