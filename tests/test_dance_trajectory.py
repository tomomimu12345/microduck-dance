import math

import pytest
import torch

from microduck_dance.ranges import (
    ALL_RANGES,
    BODY,
    BODY_RANGES,
    CMD_DIM,
    HEAD,
    HEAD_RANGES,
    TWIST,
    TWIST_RANGES,
)
from microduck_dance.trajectory import (
    ClipPlayer,
    SmoothTrajectory,
    SmoothTrajectorySpec,
)


def test_ranges_layout():
    assert CMD_DIM == 13 == len(ALL_RANGES)
    assert (TWIST, HEAD, BODY) == (slice(0, 3), slice(3, 7), slice(7, 13))
    assert len(TWIST_RANGES) == 3 and len(HEAD_RANGES) == 4 and len(BODY_RANGES) == 6


def test_smooth_values_stay_in_ranges():
    spec = SmoothTrajectorySpec(ranges=BODY_RANGES, freq_range_hz=(0.3, 1.5))
    tr = SmoothTrajectory(spec, num_envs=256, device="cpu", generator=torch.Generator().manual_seed(0))
    tr.resample(torch.arange(256))
    vals = torch.stack([tr.step(0.02) for _ in range(500)])  # (500, 256, 6)
    lo = torch.tensor([r[0] for r in BODY_RANGES]); hi = torch.tensor([r[1] for r in BODY_RANGES])
    assert (vals >= lo - 1e-6).all() and (vals <= hi + 1e-6).all()
    assert vals[..., 0].abs().max() == 0 and vals[..., 1].abs().max() == 0 and vals[..., 5].abs().max() == 0


def test_smooth_is_continuous():
    # 1 step の変化が、最大周波数と最大振幅から決まる上限以下であること
    spec = SmoothTrajectorySpec(ranges=HEAD_RANGES, freq_range_hz=(0.3, 2.0), hold_prob=0.0)
    tr = SmoothTrajectory(spec, 64, "cpu", torch.Generator().manual_seed(1)); tr.resample(torch.arange(64))
    vals = torch.stack([tr.step(0.02) for _ in range(200)])
    bound = 2 * math.pi * 2.0 * 1.40 * 0.02 + 1e-6
    assert (vals.diff(dim=0).abs() <= bound).all()


def test_hold_prob_one_gives_zero():
    spec = SmoothTrajectorySpec(ranges=TWIST_RANGES, freq_range_hz=(0.1, 0.6), hold_prob=1.0)
    tr = SmoothTrajectory(spec, 8, "cpu"); tr.resample(torch.arange(8))
    assert tr.step(0.02).abs().max() == 0


def test_resample_only_touches_given_envs():
    spec = SmoothTrajectorySpec(ranges=HEAD_RANGES, freq_range_hz=(0.3, 2.0), hold_prob=0.0)
    tr = SmoothTrajectory(spec, 4, "cpu", torch.Generator().manual_seed(2)); tr.resample(torch.arange(4))
    for _ in range(10): tr.step(0.02)
    before = tr.value().clone(); tr.resample(torch.tensor([1]))
    assert torch.equal(tr.value()[[0, 2, 3]], before[[0, 2, 3]])


def test_clip_player_holds_last_frame_without_random_start():
    frames = torch.arange(5, dtype=torch.float32).unsqueeze(1)  # (5,1)
    p = ClipPlayer(frames, dt=0.02, num_envs=1, device="cpu", random_start=False); p.resample(torch.tensor([0]))
    out = [p.step(0.02).item() for _ in range(8)]
    assert out == [1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0]


def test_clip_player_random_start_wraps():
    frames = torch.arange(5, dtype=torch.float32).unsqueeze(1)
    p = ClipPlayer(frames, 0.02, 3, "cpu", random_start=True, generator=torch.Generator().manual_seed(0))
    p.resample(torch.arange(3))
    for _ in range(12): v = p.step(0.02)
    assert ((v >= 0) & (v <= 4)).all()


def test_clip_player_index_does_not_drift_over_long_clip():
    n = 6000
    frames = torch.arange(n, dtype=torch.float32).unsqueeze(1)
    p = ClipPlayer(frames, dt=0.02, num_envs=1, device="cpu", random_start=False)
    p.resample(torch.tensor([0]))
    for i in range(1, n):
        assert p.step(0.02).item() == float(i), f"drift at step {i}"


def test_clip_player_seek_moves_only_given_envs():
    frames = torch.arange(20, dtype=torch.float32).unsqueeze(1)
    p = ClipPlayer(frames, 0.02, 2, "cpu", random_start=False); p.resample(torch.arange(2))
    for _ in range(3): p.step(0.02)
    p.seek(torch.tensor([1]), 7)
    assert p.index().tolist() == [3, 7] and p.value()[:, 0].tolist() == [3.0, 7.0]
    assert p.step(0.02)[:, 0].tolist() == [4.0, 8.0]


def test_clip_player_seek_ramps_in_linearly():
    frames = torch.full((20, 1), 2.0)
    p = ClipPlayer(frames, 0.02, 1, "cpu", random_start=False); p.resample(torch.tensor([0]))
    p.seek(torch.tensor([0]), 5, ramp_steps=4)
    out = [p.value().item()] + [p.step(0.02).item() for _ in range(6)]
    assert out == [0.0, 0.5, 1.0, 1.5, 2.0, 2.0, 2.0]


def test_resume_on_reset_keeps_reset_env_on_the_global_clock():
    # Mimics ManagerBasedRlEnv.step: a reset (resample) runs before the command
    # update (step), and the policy then sees value().
    frames = torch.arange(50, dtype=torch.float32).unsqueeze(1)
    p = ClipPlayer(frames, 0.02, 2, "cpu", random_start=False, resume_on_reset=True, reset_ramp_steps=0)
    p.resample(torch.arange(2))  # env construction: clock 0, plays from frame 0
    assert p.index().tolist() == [0, 0]
    for _ in range(10): p.step(0.02)
    p.resample(torch.tensor([1]))  # env 1 falls during control step 10
    p.step(0.02)
    assert p.index().tolist() == [11, 11]
    assert p.clock == 11


def test_resume_on_reset_ramps_the_reset_env_only():
    frames = torch.full((50, 1), 1.0)
    p = ClipPlayer(frames, 0.02, 2, "cpu", random_start=False, resume_on_reset=True, reset_ramp_steps=25)
    p.resample(torch.arange(2))
    assert p.value()[:, 0].tolist() == [1.0, 1.0]  # no ramp at clock 0
    for _ in range(5): p.step(0.02)
    p.resample(torch.tensor([0]))
    v = p.step(0.02)[:, 0].tolist()
    assert v[0] == pytest.approx(1 / 25) and v[1] == 1.0


def test_clip_player_zero_dt_step_does_not_advance():
    # ManagerBasedRlEnv.reset runs command_manager.compute(dt=0.0); that must not skip frame 0.
    frames = torch.arange(5, dtype=torch.float32).unsqueeze(1)
    p = ClipPlayer(frames, 0.02, 1, "cpu", random_start=False); p.resample(torch.tensor([0]))
    assert p.step(0.0).item() == 0.0 and p.clock == 0
    assert p.step(0.02).item() == 1.0 and p.clock == 1
