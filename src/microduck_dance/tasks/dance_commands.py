"""Dance command terms and the heading-free body tracking reward.

The terms stream a new command every control step (smooth random trajectories
during training, a precomputed clip during playback) instead of holding a
value between resamples. They live here, apart from mdp.py, so that our own
work stays separable from upstream.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

from microduck_dance.command_file import load_command_file
from microduck_dance.ranges import DANCE_STAND_HEIGHT
from microduck_dance.trajectory import (
    ClipPlayer,
    SmoothTrajectory,
    SmoothTrajectorySpec,
)
from mjlab_microduck.tasks.mdp import (
    UniformPoseCommand,
    UniformPoseCommandCfg,
    VelocityCommandCommandOnly,
    VelocityCommandCommandOnlyCfg,
)


def _make_source(cfg, env, ranges):
    """Build the trajectory source selected by cfg.source for this term."""
    if cfg.source == "smooth":
        spec = SmoothTrajectorySpec(
            ranges=tuple(tuple(r) for r in ranges),
            freq_range_hz=cfg.freq_range_hz,
            num_harmonics=cfg.num_harmonics,
            hold_prob=cfg.hold_prob,
        )
        return SmoothTrajectory(spec, env.num_envs, env.device)
    if cfg.source in ("library", "file"):
        if cfg.command_file is None:
            raise ValueError(f"source={cfg.source!r} requires command_file")
        commands, file_dt, _ = load_command_file(cfg.command_file)
        if abs(file_dt - env.step_dt) > 1e-6:
            raise ValueError(
                f"{cfg.command_file}: dt={file_dt} differs from env.step_dt={env.step_dt}; "
                "the player advances one frame per control step"
            )
        lo, hi = cfg.columns
        frames = torch.as_tensor(commands[:, lo:hi], dtype=torch.float32)
        return ClipPlayer(
            frames, file_dt, env.num_envs, env.device, random_start=(cfg.source == "library"),
            resume_on_reset=cfg.resume_on_reset,
            reset_ramp_steps=int(round(cfg.reset_ramp_s / file_dt)),
        )
    raise ValueError(f"unknown source {cfg.source!r}")


@dataclass(kw_only=True)
class DancePoseCommandCfg(UniformPoseCommandCfg):
    freq_range_hz: tuple[float, float]
    columns: tuple[int, int]
    num_harmonics: int = 3
    hold_prob: float = 0.2
    source: Literal["smooth", "library", "file"] = "smooth"
    command_file: str | None = None
    # Playback only (source="file"): a reset env resumes at the global step
    # and fades in over reset_ramp_s instead of restarting the clip.
    resume_on_reset: bool = False
    reset_ramp_s: float = 0.0

    def build(self, env) -> "DancePoseCommand":
        return DancePoseCommand(self, env)


class DancePoseCommand(UniformPoseCommand):
    """Head or body pose command streamed from a trajectory source every step."""

    cfg: DancePoseCommandCfg

    def __init__(self, cfg: DancePoseCommandCfg, env):
        super().__init__(cfg, env)
        self._source = _make_source(cfg, env, cfg.ranges)

    @property
    def source(self):
        """The trajectory source (SmoothTrajectory or ClipPlayer)."""
        return self._source

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return
        self._source.resample(env_ids)
        self._command[env_ids] = self._source.value()[env_ids]

    def _update_command(self) -> None:
        self._command[:] = self._source.step(getattr(self, "_compute_dt", self._env.step_dt))

    def compute(self, dt: float) -> None:
        # env.reset() calls compute(dt=0.0). Forward that dt so a reset does not
        # advance the trajectory (a ClipPlayer would otherwise skip frame 0).
        self._compute_dt = dt
        super().compute(dt)


@dataclass(kw_only=True)
class DanceVelocityCommandCfg(VelocityCommandCommandOnlyCfg):
    freq_range_hz: tuple[float, float]
    num_harmonics: int = 3
    hold_prob: float = 0.2
    source: Literal["smooth", "library", "file"] = "smooth"
    command_file: str | None = None
    # Playback only (source="file"): a reset env resumes at the global step
    # and fades in over reset_ramp_s instead of restarting the clip.
    resume_on_reset: bool = False
    reset_ramp_s: float = 0.0
    columns: tuple[int, int] = (0, 3)

    def build(self, env) -> "DanceVelocityCommand":
        return DanceVelocityCommand(self, env)


class DanceVelocityCommand(VelocityCommandCommandOnly):
    """Twist command streamed from a trajectory source every step.

    Standing is expressed by the trajectory's hold segments, so mjlab's
    standing-env zeroing and the turn-in-place bucket are not used.
    """

    cfg: DanceVelocityCommandCfg

    def __init__(self, cfg: DanceVelocityCommandCfg, env):
        super().__init__(cfg, env)
        r = cfg.ranges
        self._source = _make_source(cfg, env, (r.lin_vel_x, r.lin_vel_y, r.ang_vel_z))

    @property
    def source(self):
        """The trajectory source (SmoothTrajectory or ClipPlayer)."""
        return self._source

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return
        self._source.resample(env_ids)
        self.vel_command_b[env_ids] = self._source.value()[env_ids]
        self.vel_command_w[env_ids] = self.vel_command_b[env_ids]
        self.is_standing_env[env_ids] = False

    def _update_command(self) -> None:
        self.vel_command_b[:] = self._source.step(getattr(self, "_compute_dt", self._env.step_dt))

    def compute(self, dt: float) -> None:
        # env.reset() calls compute(dt=0.0). Forward that dt so a reset does not
        # advance the trajectory (a ClipPlayer would otherwise skip frame 0).
        self._compute_dt = dt
        super().compute(dt)
        self.vel_command_w[:] = self.vel_command_b


def dance_body_tracking(
    env,
    command_name: str = "body_pose",
    nominal_height: float = DANCE_STAND_HEIGHT,
    z_std: float = 0.01,
    angle_std: float = math.radians(8.0),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Mean of three Gaussians on body height, roll and pitch.

    Unlike body_pose_tracking_6d this ignores x, y and yaw, so a dancing robot
    is free to travel and turn. cmd[:, 2] is the height delta, cmd[:, 3] roll,
    cmd[:, 4] pitch.
    """
    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)

    pos_w = torch.nan_to_num(asset.data.root_link_pos_w, nan=0.0)
    z_err = pos_w[:, 2] - env.scene.terrain.env_origins[:, 2] - (nominal_height + cmd[:, 2])

    quat = asset.data.root_link_quat_w
    qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    roll = torch.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
    pitch = torch.asin(torch.clamp(2.0 * (qw * qy - qz * qx), -1.0, 1.0))

    r_z = torch.exp(-((z_err / z_std) ** 2))
    r_r = torch.exp(-(((roll - cmd[:, 3]) / angle_std) ** 2))
    r_p = torch.exp(-(((pitch - cmd[:, 4]) / angle_std) ** 2))
    return (r_z + r_r + r_p) / 3.0
