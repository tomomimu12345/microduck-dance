"""Tests for the visible-motion statistics and the play_dance state dump (CPU only, no env)."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from microduck_dance.ranges import DANCE_STAND_HEIGHT
from microduck_dance.visible_motion import (
    commanded_path,
    moving_average,
    nod_lag,
    path_length,
    visible_motion,
)

_PATH = Path(__file__).resolve().parents[1] / "dance" / "scripts" / "evaluate" / "play_dance.py"
_spec = importlib.util.spec_from_file_location("play_dance", _PATH)
play_dance = importlib.util.module_from_spec(_spec)
sys.modules["play_dance"] = play_dance
_spec.loader.exec_module(play_dance)

DT = 0.02


def _state(T=1500):
    t = np.arange(T) * DT
    cmd = np.zeros((T, 13))
    meas = np.zeros((T, 13))
    pos = np.zeros((T, 3))
    pos[:, 2] = DANCE_STAND_HEIGHT
    yaw = np.zeros(T)
    return t, cmd, meas, pos, yaw


def test_moving_average_keeps_constants_and_length():
    x = np.full(50, 3.0)
    assert np.allclose(moving_average(x, 7), 3.0)
    assert moving_average(np.arange(10.0), 4).shape == (10,)
    assert np.allclose(moving_average(np.arange(10.0), 3)[1:-1], np.arange(10.0)[1:-1])


def test_straight_walk_path_and_displacement():
    t, cmd, meas, pos, yaw = _state()
    cmd[:, 0] = 0.2
    pos[:, 0] = 0.2 * t
    s = visible_motion(cmd, meas, pos, yaw, DT)
    assert s["path_m"] == pytest.approx(0.2 * t[-1], rel=1e-9)
    assert s["max_disp_m"] == pytest.approx(0.2 * t[-1], rel=1e-9)
    assert s["cmd_path_m"] == pytest.approx(0.2 * t[-1], rel=1e-9)
    assert s["yaw_abs_change_rad"] == 0.0


def test_commanded_path_turns_with_wz():
    T = 1000
    cmd = np.zeros((T, 13))
    cmd[:, 0] = 0.1
    cmd[:, 2] = 2 * np.pi / (T * DT)  # one full circle
    xy = commanded_path(cmd, DT)
    assert np.linalg.norm(xy[-1] - xy[0]) < 0.01
    assert path_length(xy) == pytest.approx(0.1 * (T - 1) * DT, rel=1e-6)


def test_yaw_change_unwraps_across_pi():
    t, cmd, meas, pos, yaw = _state()
    total = 3 * np.pi
    yaw[:] = np.angle(np.exp(1j * (total * t / t[-1])))  # wrapped, crosses +-pi
    s = visible_motion(cmd, meas, pos, yaw, DT)
    assert s["yaw_abs_change_rad"] == pytest.approx(total, rel=1e-6)
    assert s["yaw_net_rad"] == pytest.approx(total, rel=1e-6)


def test_height_range_std_and_ratios():
    t, cmd, meas, pos, yaw = _state()
    pos[:, 2] += 0.01 * np.sin(2 * np.pi * t)
    cmd[:, 5] = 0.4 * np.sin(2 * np.pi * 0.5 * t)
    meas[:, 5] = 0.2 * np.sin(2 * np.pi * 0.5 * t)
    meas[:, 10] = 0.05 * np.sign(np.sin(2 * np.pi * t))
    s = visible_motion(cmd, meas, pos, yaw, DT)
    assert s["height_p5_p95_m"] == pytest.approx(0.02 * np.sin(np.deg2rad(72)) / 1, rel=0.05)
    assert s["head_yaw_std_rad"] == pytest.approx(0.2 / np.sqrt(2), rel=0.01)
    assert s["std_ratio"]["head_yaw"] == pytest.approx(0.5, rel=1e-6)
    assert s["corr"]["head_yaw"] == pytest.approx(1.0, abs=1e-9)
    assert s["roll_std_rad"] == pytest.approx(0.05, rel=0.02)
    assert np.isnan(s["std_ratio"]["vx"])  # zero command


def test_nod_lag_finds_delay_after_each_beat():
    t, cmd, meas, pos, yaw = _state(3000)
    beats = np.arange(1.0, 59.0, 0.6)
    lag = 0.12
    for b in beats:
        meas[:, 4] += 0.3 * np.exp(-0.5 * ((t - b - lag) / 0.05) ** 2)
        cmd[:, 4] += 0.3 * np.exp(-0.5 * ((t - b + 0.14) / 0.05) ** 2)
    s = visible_motion(cmd, meas, pos, yaw, DT, beats=beats)
    assert s["nod_lag_realized"]["median_s"] == pytest.approx(lag, abs=DT)
    assert s["nod_lag_command"]["median_s"] == pytest.approx(-0.14, abs=DT)
    assert s["nod_lag_realized"]["n"] == int(np.sum((beats >= 1.5) & (beats <= t[-1] - 1.5)))


def test_nod_lag_ignores_slow_drift():
    t = np.arange(2000) * DT
    beats = np.arange(2.0, 38.0, 0.5)
    x = 0.5 * t / t[-1]  # slow ramp would put every argmax at the window end
    for b in beats:
        x = x + 0.2 * np.exp(-0.5 * ((t - b - 0.06) / 0.04) ** 2)
    assert nod_lag(t, x, beats, DT)["median_s"] == pytest.approx(0.06, abs=DT)


def test_skip_drops_leading_steps():
    t, cmd, meas, pos, yaw = _state()
    pos[:100, 0] = np.linspace(0, 1, 100)
    pos[100:, 0] = 1.0
    s = visible_motion(cmd, meas, pos, yaw, DT, skip_s=100 * DT)
    assert s["path_m"] == 0.0
    assert s["seconds"] == pytest.approx((1500 - 100) * DT)


def test_save_states_round_trip(tmp_path):
    T = 20
    rng = np.random.default_rng(0)
    cmd, meas = rng.normal(size=(T, 13)), rng.normal(size=(T, 13))
    term = np.zeros(T, bool)
    term[5] = True
    pos, yaw = rng.normal(size=(T, 3)), rng.normal(size=T)
    p = play_dance.save_states(tmp_path / "a" / "s.npz", cmd, meas, term, pos, yaw, DT, checkpoint="ck.pt")
    d = np.load(p)
    assert np.allclose(d["cmd"], cmd, atol=1e-6) and np.allclose(d["pos"], pos, atol=1e-6)
    assert d["term"].dtype == bool and d["term"][5] and float(d["dt"]) == DT
    assert str(d["checkpoint"]) == "ck.pt"
    s = visible_motion(d["cmd"], d["meas"], d["pos"], d["yaw"], float(d["dt"]))
    assert s["seconds"] == pytest.approx(T * DT)


def test_save_states_rejects_bad_shapes(tmp_path):
    with pytest.raises(ValueError):
        play_dance.save_states(tmp_path / "s.npz", np.zeros((5, 13)), np.zeros((5, 13)), np.zeros(5, bool),
                               np.zeros((5, 2)), np.zeros(5), DT)


def test_parser_has_dump_states_and_allows_no_out():
    a = play_dance.build_parser().parse_args(["--checkpoint", "x.pt", "--commands", "c.npz",
                                              "--dump-states", "s.npz"])
    assert a.dump_states == Path("s.npz") and a.out is None


def _load(name, folder="evaluate"):
    path = Path(__file__).resolve().parents[1] / "dance" / "scripts" / folder / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_table_parses_dump_names():
    vmt = _load("visible_motion_table")
    assert vmt.parse_name("dance_v3_new_swag") == ("dance_v3", "new", "swag")
    assert vmt.parse_name("dance_v2_old_fallin_for_you") == ("dance_v2", "old", "fallin_for_you")
    with pytest.raises(ValueError):
        vmt.parse_name("dance_v3_new_unknown")

