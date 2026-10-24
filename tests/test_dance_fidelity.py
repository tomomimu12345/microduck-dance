"""Tests for the Task 19 retarget defaults: gate off, half groove, posture lead, heading servo."""

from __future__ import annotations

import dataclasses
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from microduck_dance.retarget import (
    GROOVE_POLICY_LAG_COMP_S, HEADING_SERVO_GAIN, POSTURE_LEAD_S, TASK15_OPTIONS, RetargetConfig, heading_servo,
    smpl_to_commands,
)

sys.path.insert(0, str(Path(__file__).parent))
from test_dance_retarget import PELVIS_Y, make_motion, random_motion, still_motion, walking_motion  # noqa: E402
from test_dance_variety import make_music  # noqa: E402

DT = 0.02
NO_RAMP = dict(ramp_in_s=0.0, ramp_out_s=0.0)
VX, VY, WZ, HP, HY, BZ, BR = 0, 1, 2, 4, 5, 9, 10


def turning_motion(psi_fn, F, fps=30, speed=0.0):
    """y-up clip with pelvis heading psi_fn(t) (rad, left positive), walking along the initial forward axis."""
    return make_motion(F, fps, 1, lambda t: (0.0, PELVIS_Y, speed * t), lambda t: (0.0, psi_fn(t), 0.0))


def commanded_heading(out):
    """Integral of wz before each step, as the robot sees it (psi_c of the servo)."""
    return np.concatenate([[0.0], np.cumsum(out[:, WZ])[:-1] * DT])


def commanded_path(out):
    psi = commanded_heading(out)
    c, s = np.cos(psi), np.sin(psi)
    vxw, vyw = c * out[:, VX] - s * out[:, VY], s * out[:, VX] + c * out[:, VY]
    return np.cumsum(vxw) * DT, np.cumsum(vyw) * DT


def test_new_defaults_and_old_options():
    cfg = RetargetConfig()
    assert cfg.phrase_gate is False and cfg.groove_scale == 0.5 and cfg.heading_servo is True
    assert cfg.vel_gain == 0.3
    assert cfg.posture_lead_s == POSTURE_LEAD_S == 0.12 and cfg.heading_servo_gain == HEADING_SERVO_GAIN == 1.5
    old = RetargetConfig(**TASK15_OPTIONS)
    assert old.phrase_gate is True and old.groove_scale == 1.0 and old.posture_lead_s == 0.0
    assert old.heading_servo is False and old.vel_gain == 1.3
    assert {f.name for f in dataclasses.fields(RetargetConfig)} >= set(TASK15_OPTIONS)


# 1. phrase gate

def test_phrase_gate_off_by_default_and_still_reachable():
    music = make_music()  # phrases 0 and 2 are groove phrases
    motion = walking_motion(speed=0.15, F=30 * 36, fps=30)
    off, info = smpl_to_commands(*motion, fps=30, cfg=RetargetConfig(**NO_RAMP), music=music, return_info=True)
    assert info["phrases"] is None
    assert np.abs(off[200:, VX]).min() > 0.03  # walking in every phrase (0.3 x 0.15 m/s)
    on, info_on = smpl_to_commands(*motion, fps=30, cfg=RetargetConfig(**NO_RAMP, phrase_gate=True), music=music,
                                   return_info=True)
    assert info_on["phrases"]["travel"] == [False, True, False, True]
    t = np.arange(len(on)) * DT
    s = info_on["phrases"]["start_times_s"]
    m = (t > s[0] + 1.0) & (t < s[1] - 1.0)
    assert np.all(on[m, :3] == 0.0)


# 2. groove at half amplitude

@pytest.mark.parametrize("with_music", [False, True])
def test_groove_is_half_of_the_task15_layer(with_music):
    beats = 1.0 + 0.5 * np.arange(64)
    music = make_music(flat=True) if with_music else None
    kw = dict(fps=30, beats=None if with_music else beats, music=music)
    still = still_motion(30 * 36, 30)
    half = smpl_to_commands(*still, cfg=RetargetConfig(**NO_RAMP), **kw)
    full = smpl_to_commands(*still, cfg=RetargetConfig(**NO_RAMP, groove_scale=1.0), **kw)
    assert np.abs(full[:, HP]).max() == pytest.approx(0.35, abs=1e-6)
    np.testing.assert_allclose(half, 0.5 * full, atol=1e-12)


# 3. posture lead

def test_posture_columns_lead_by_0p12_s_and_twist_does_not():
    motion = random_motion(seed=0, F=600, fps=30)
    kw = dict(**NO_RAMP, heading_servo=False)
    lead = smpl_to_commands(*motion, fps=30, cfg=RetargetConfig(**kw))
    base = smpl_to_commands(*motion, fps=30, cfg=RetargetConfig(**kw, posture_lead_s=0.0))
    n = int(round(POSTURE_LEAD_S / DT))
    assert n == 6
    np.testing.assert_array_equal(lead[:-n, 3:13], base[n:, 3:13])
    np.testing.assert_array_equal(lead[-n:, 3:13], np.repeat(base[-1:, 3:13], n, axis=0))
    np.testing.assert_array_equal(lead[:, :3], base[:, :3])


def test_groove_is_not_shifted_by_the_posture_lead():
    # The groove has its own lead (groove_lag_s); the posture lead must not add to it.
    assert GROOVE_POLICY_LAG_COMP_S == -0.14
    beats = np.arange(1.0, 19.0, 0.5)
    out = smpl_to_commands(*still_motion(600, 30), fps=30, cfg=RetargetConfig(**NO_RAMP), beats=beats)
    t = np.arange(len(out)) * DT
    hp = out[:, HP]
    peaks = t[np.where((hp[1:-1] > hp[:-2]) & (hp[1:-1] >= hp[2:]) & (hp[1:-1] > 1e-6))[0] + 1]
    np.testing.assert_allclose(peaks, beats - 0.14, atol=DT / 2 + 1e-9)


def test_lead_keeps_the_ramps():
    out = smpl_to_commands(*random_motion(seed=3, F=300, fps=30), fps=30, cfg=RetargetConfig())
    assert np.all(out[0] == 0.0) and np.all(out[-1] == 0.0)


# 5. heading servo

def test_constant_turn_510_deg_in_60_s_is_followed():
    total = math.radians(510.0)
    out = smpl_to_commands(*turning_motion(lambda t: total * t / 60.0, F=1800), fps=30, cfg=RetargetConfig())
    assert np.abs(out[:, WZ]).max() < 0.999  # not clipped
    psi_c = commanded_heading(out)[-1] + out[-1, WZ] * DT
    assert abs(psi_c - total) <= 0.15 * total
    # The open-loop wz loses the ramp-in and ramp-out turn; the servo makes up the ramp-in part.
    old = smpl_to_commands(*turning_motion(lambda t: total * t / 60.0, F=1800), fps=30,
                           cfg=RetargetConfig(heading_servo=False))
    assert abs(psi_c - total) < abs(old[:, WZ].sum() * DT - total)


def test_wz_clip_at_1_rad_s_is_respected_and_heading_catches_up():
    # A fast 3 rad turn over 1.5 s (2 rad/s), then the dancer faces the new way.
    def psi(t):
        return 3.0 * float(np.clip((t - 5.0) / 1.5, 0.0, 1.0))
    motion = turning_motion(psi, F=30 * 20)
    out, info = smpl_to_commands(*motion, fps=30, cfg=RetargetConfig(), return_info=True)
    assert np.abs(out[:, WZ]).max() <= 1.0 + 1e-12
    assert np.isclose(out[:, WZ], 1.0).sum() > 20  # the clip is reached
    psi_c = commanded_heading(out)
    assert psi_c[int(15.0 / DT)] == pytest.approx(3.0, abs=0.05)  # caught up after the clip
    assert info["heading_servo"]["final_target_rad"] == pytest.approx(3.0, abs=0.05)
    old = smpl_to_commands(*motion, fps=30, cfg=RetargetConfig(heading_servo=False))
    assert np.abs(old[:, WZ]).max() <= 1.0 + 1e-12
    assert old[:, WZ].sum() * DT < 2.6  # the open-loop command loses the clipped part


def test_heading_servo_unit_respects_clip_rate_and_scale():
    t = np.arange(1000) * DT
    psi_h = 3.0 * np.clip(t - 2.0, 0.0, 1.0)  # 3 rad/s ramp
    wz, psi_c = heading_servo(psi_h, DT, 1.5, (-1.0, 1.0), max_rate=4.0)
    assert np.abs(wz).max() <= 1.0 + 1e-12
    assert np.abs(np.diff(wz)).max() <= 4.0 * DT + 1e-12
    np.testing.assert_allclose(psi_c[1:], np.cumsum(wz)[:-1] * DT)
    assert psi_c[-1] == pytest.approx(3.0, abs=1e-3)
    zero, psi0 = heading_servo(psi_h, DT, 1.5, (-1.0, 1.0), max_rate=4.0, scale=np.zeros(len(t)))
    assert np.all(zero == 0.0) and np.all(psi0 == 0.0)


def test_heading_servo_with_a_closed_gate_winds_up_and_spins_at_the_clip_on_reopen():
    # Documented in the heading_servo docstring. psi_h turns at 0.5 rad/s. The gate is closed
    # from 2 s to 6 s: wz is 0 and psi_c stands still, so the error grows to 2 rad. The
    # rate limit acts before scale, so w already sits at the clip and wz jumps to the clip
    # in the first open frame. The servo then catches up.
    t = np.arange(int(14 / DT)) * DT
    psi_h = 0.5 * t
    gate = ((t < 2.0) | (t >= 6.0)).astype(float)
    wz, psi_c = heading_servo(psi_h, DT, 1.5, (-1.0, 1.0), max_rate=4.0, scale=gate)
    closed = (t >= 2.0) & (t < 6.0)
    assert np.all(wz[closed] == 0.0)
    k = int(np.argmax(t >= 6.0))
    assert psi_h[k] - psi_c[k] > 1.5  # about 2 rad behind
    assert wz[k] == pytest.approx(1.0) and wz[k - 1] == 0.0  # steps from 0 to the clip
    assert np.allclose(wz[k:k + 100], 1.0)  # stays at the clip while the error is large
    assert abs(psi_h[-1] - psi_c[-1]) < 0.1


def test_velocity_follows_the_human_world_direction_while_heading_lags():
    # The dancer walks straight ahead at 0.2 m/s while spinning at 2 rad/s, so the
    # commanded heading lags far behind. vx, vy are in the commanded frame, so the
    # commanded path still goes straight along the initial forward axis.
    motion = turning_motion(lambda t: 2.0 * t, F=30 * 12, speed=0.2)
    out = smpl_to_commands(*motion, fps=30, cfg=RetargetConfig(vel_gain=1.0))
    x, y = commanded_path(out)
    assert x[-1] > 1.5
    assert abs(math.degrees(math.atan2(y[-1], x[-1]))) < 5.0
    assert np.abs(out[:, VX]).max() <= 0.4 + 1e-12 and np.abs(out[:, VY]).max() <= 0.3 + 1e-12
