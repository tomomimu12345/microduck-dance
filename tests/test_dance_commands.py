import math

import numpy as np
import pytest
import torch

from microduck_dance.ranges import DANCE_STAND_HEIGHT, HEAD_RANGES, TWIST_RANGES
from microduck_dance.tasks.dance_commands import (
    DancePoseCommandCfg,
    DanceVelocityCommandCfg,
    dance_body_tracking,
)


class _Data:
    def __init__(self, n):
        self.root_link_pos_w = torch.zeros(n, 3)
        self.root_link_quat_w = torch.zeros(n, 4)
        self.root_link_quat_w[:, 0] = 1.0
        self.root_link_lin_vel_b = torch.zeros(n, 3)
        self.root_link_ang_vel_b = torch.zeros(n, 3)
        self.heading_w = torch.zeros(n)


class _Asset:
    def __init__(self, n):
        self.data = _Data(n)


class _Terrain:
    def __init__(self, n):
        self.env_origins = torch.zeros(n, 3)


class _Scene:
    def __init__(self, n):
        self._asset = _Asset(n)
        self.terrain = _Terrain(n)

    def __getitem__(self, _):
        return self._asset


class _Cmd:
    def __init__(self):
        self.cmd = None

    def get_command(self, _):
        return self.cmd


class _Env:
    def __init__(self, n):
        self.num_envs = n
        self.device = "cpu"
        self.step_dt = 0.02
        self.scene = _Scene(n)
        self.command_manager = _Cmd()


def fake_env(num_envs):
    return _Env(num_envs)


def fake_env_with_root(pos, quat_yaw_deg, cmd):
    env = _Env(len(pos))
    data = env.scene._asset.data
    data.root_link_pos_w[:] = torch.tensor(pos)
    half = torch.deg2rad(torch.tensor(quat_yaw_deg)) / 2
    data.root_link_quat_w[:] = torch.stack(
        [torch.cos(half), torch.zeros_like(half), torch.zeros_like(half), torch.sin(half)], dim=-1
    )
    env.command_manager.cmd = torch.tensor(cmd, dtype=torch.float32)
    return env


def _vel_cfg(**kw):
    return DanceVelocityCommandCfg(
        entity_name="robot",
        resampling_time_range=(4.0, 8.0),
        heading_command=False,
        rel_heading_envs=0.0,
        rel_standing_envs=0.5,
        ranges=DanceVelocityCommandCfg.Ranges(
            lin_vel_x=TWIST_RANGES[0], lin_vel_y=TWIST_RANGES[1], ang_vel_z=TWIST_RANGES[2]
        ),
        freq_range_hz=(0.1, 0.6),
        hold_prob=0.0,
        **kw,
    )


def test_pose_command_changes_every_step():
    cfg = DancePoseCommandCfg(resampling_time_range=(4.0, 8.0), ranges=HEAD_RANGES,
                              freq_range_hz=(0.3, 2.0), hold_prob=0.0, columns=(3, 7))
    term = cfg.build(fake_env(num_envs=8))
    term.reset(torch.arange(8)); a = term.command.clone()
    term.compute(0.02); b = term.command.clone()
    assert a.shape == (8, 4) and not torch.allclose(a, b)


def _write_clip(tmp_path, dt=0.02):
    cmds = np.zeros((10, 13), np.float32); cmds[:, 3] = np.arange(10) * 0.01
    np.savez(tmp_path / "c.npz", commands=cmds, dt=dt)
    return str(tmp_path / "c.npz")


def test_file_source_plays_columns_in_order(tmp_path):
    cfg = DancePoseCommandCfg(resampling_time_range=(1e6, 1e6), ranges=HEAD_RANGES, freq_range_hz=(0.3, 2.0),
                              source="file", command_file=_write_clip(tmp_path), columns=(3, 7))
    term = cfg.build(fake_env(num_envs=1)); term.reset(torch.tensor([0]))
    seen = []
    for _ in range(3): term.compute(0.02); seen.append(round(term.command[0, 0].item(), 4))
    assert seen == [0.01, 0.02, 0.03]


def test_file_dt_mismatch_raises(tmp_path):
    cfg = DancePoseCommandCfg(resampling_time_range=(1e6, 1e6), ranges=HEAD_RANGES, freq_range_hz=(0.3, 2.0),
                              source="file", command_file=_write_clip(tmp_path, dt=0.04), columns=(3, 7))
    with pytest.raises(ValueError):
        cfg.build(fake_env(num_envs=1))


def test_library_source_uses_random_start(tmp_path):
    cfg = DancePoseCommandCfg(resampling_time_range=(1e6, 1e6), ranges=HEAD_RANGES, freq_range_hz=(0.3, 2.0),
                              source="library", command_file=_write_clip(tmp_path), columns=(3, 7))
    term = cfg.build(fake_env(num_envs=64)); term.reset(torch.arange(64))
    assert term.command[:, 0].unique().numel() > 1


def test_velocity_command_writes_vel_command_b():
    term = _vel_cfg().build(fake_env(num_envs=4)); term.reset(torch.arange(4)); term.compute(0.02)
    v = term.vel_command_b
    assert v.shape == (4, 3) and (v[:, 0].abs() <= 0.4).all() and v.abs().sum() > 0
    # rel_standing_envs=0.5 must not zero the stream.
    assert not term.is_standing_env.any()


def test_body_tracking_ignores_yaw_and_xy():
    env = fake_env_with_root(pos=[[0.3, -0.2, DANCE_STAND_HEIGHT]], quat_yaw_deg=[90.0], cmd=[[0, 0, 0, 0, 0, 0]])
    assert dance_body_tracking(env).item() == pytest.approx(1.0, abs=1e-5)


def test_body_tracking_penalizes_height_error():
    env = fake_env_with_root(pos=[[0, 0, DANCE_STAND_HEIGHT]], quat_yaw_deg=[0.0], cmd=[[0, 0, -0.03, 0, 0, 0]])
    assert dance_body_tracking(env).item() < 0.7


def test_body_tracking_height_reference_is_dance_stand_height():
    # Standing at the measured stance with body_z = 0 scores full height reward.
    stand = fake_env_with_root(pos=[[0, 0, DANCE_STAND_HEIGHT]], quat_yaw_deg=[0.0], cmd=[[0, 0, 0, 0, 0, 0]])
    assert dance_body_tracking(stand).item() == pytest.approx(1.0, abs=1e-5)
    # The old 0.095 reference is now a 21 mm error, so the height Gaussian is near 0.
    low = fake_env_with_root(pos=[[0, 0, 0.095]], quat_yaw_deg=[0.0], cmd=[[0, 0, 0, 0, 0, 0]])
    assert dance_body_tracking(low).item() == pytest.approx(2.0 / 3.0, abs=0.01)
