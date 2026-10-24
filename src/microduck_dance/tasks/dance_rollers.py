"""Roller dance: twist mapping command term and the rewards that differ from the roller task.

A roller policy reads its twist slot differently from a walking policy:
cmd[0] is the push strength (0 coasts, > 0 pushes, < 0 brakes), cmd[1] is
always 0 and cmd[2] is the heading error in rad (RelativeHeadingVelocityCommand,
positive means the target is to the left). The dance library gives
(vx, vy, wz) for a walker, so this term maps every step:

    push    = clip(PUSH_GAIN * vx, PUSH_RANGE)     (backward steps become braking)
    vy      dropped (a skate cannot move sideways)
    target  = target + wz * dt, held within HEADING_MAX of the current yaw
    cmd[2]  = wrap(target - yaw)

The target is a saturating integrator: once the error reaches HEADING_MAX the
target moves with the robot, so a robot that cannot turn as fast as the dance
does not build up a debt of whole turns. The same pure function
(map_dance_twist) runs in training and in playback.

Held push pieces (roller002). The library twist pushes in short bursts: push
> 0.2 lasts 0.17 s on average (p90 0.33 s) and half of the frames brake, so a
policy trained on it alone learns to stand and turn. With push_piece_prob > 0
the training source also draws piecewise held pieces (PUSH_PIECE_KINDS, 1.5 to
4 s each, in dance units so they pass through the same map): sustained pushes
straight or turning, and brakes. Playback of a song still uses only the file.

Kept apart from mdp.py, like dance_v3.py, so our work stays separable from upstream.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

from microduck_dance.command_file import load_command_file
from microduck_dance.mix import MixedSource, PieceKind, PiecewiseHeld, ZeroSource
from microduck_dance.trajectory import ClipPlayer
from microduck_dance.tasks.dance_commands import DanceVelocityCommand, DanceVelocityCommandCfg
from mjlab_microduck.tasks.mdp import VelocityCommandCommandOnly

# Library vx has |vx| p95 0.084 m/s (songs 0.13 to 0.17). A gain of 5 puts the
# library p95 at a push of 0.42, inside the roller range, and lets the song
# peaks saturate at full push.
PUSH_GAIN = 5.0
PUSH_RANGE = (-0.5, 0.6)  # the roller task's cmd_x range
HEADING_MAX = 1.0  # rad, clip of cmd[2]

# Held push pieces in dance units (vx, vy, wz). vx 0.04 to 0.12 maps to a push
# of 0.2 to 0.6 (the top of PUSH_RANGE), vx -0.10 to -0.02 to a brake of -0.5 to -0.1.
PUSH_PIECE_KINDS = (
    PieceKind(0.5, ((0.04, 0.12), (0.0, 0.0), (-0.2, 0.2))),  # push straight
    PieceKind(0.3, ((0.04, 0.12), (0.0, 0.0), (0.0, 0.0)), abs_ranges=((2, 0.3, 0.8),)),  # push and turn
    PieceKind(0.2, ((-0.10, -0.02), (0.0, 0.0), (-0.2, 0.2))),  # brake
)
PUSH_PIECE_S = (1.5, 4.0)
PUSH_PIECE_BLEND_S = 0.25


def wrap_angle(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))


def yaw_from_quat(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def map_dance_twist(
    dance: torch.Tensor,
    yaw: torch.Tensor,
    target: torch.Tensor,
    dt: float,
    push_gain: float = PUSH_GAIN,
    push_range: tuple[float, float] = PUSH_RANGE,
    heading_max: float = HEADING_MAX,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map dance twists (N, 3) = (vx, vy, wz) to roller commands (N, 3).

    yaw is the current yaw (N,) and target the heading target (N,) from the
    previous step. Returns (command, new target).
    """
    push = torch.clamp(dance[:, 0] * push_gain, push_range[0], push_range[1])
    err = wrap_angle(target + dance[:, 2] * dt - yaw).clamp(-heading_max, heading_max)
    new_target = wrap_angle(yaw + err)
    cmd = torch.stack([push, torch.zeros_like(push), err], dim=1)
    return cmd, new_target


@dataclass(kw_only=True)
class RollerDanceTwistCommandCfg(DanceVelocityCommandCfg):
    """source "library": library clip with probability 1 - zero_prob per resample, else an exact zero.
    source "file": a song played from frame 0 (playback)."""

    source: Literal["library", "file"] = "library"
    zero_prob: float = 0.15
    # Share of resamples that draw held push pieces instead of the library.
    push_piece_prob: float = 0.0
    push_piece_kinds: tuple = PUSH_PIECE_KINDS
    push_piece_s: tuple[float, float] = PUSH_PIECE_S
    push_piece_blend_s: float = PUSH_PIECE_BLEND_S
    push_gain: float = PUSH_GAIN
    push_range: tuple[float, float] = PUSH_RANGE
    heading_max: float = HEADING_MAX

    def build(self, env) -> "RollerDanceTwistCommand":
        return RollerDanceTwistCommand(self, env)


def build_training_source(lib, cfg: RollerDanceTwistCommandCfg, num_envs: int, device,
                          generator: torch.Generator | None = None) -> MixedSource:
    """Library, held push pieces and exact zero, mixed per resample."""
    lib_prob = 1.0 - cfg.zero_prob - cfg.push_piece_prob
    if lib_prob < -1e-9 or cfg.push_piece_prob < 0 or cfg.zero_prob < 0:
        raise ValueError("zero_prob and push_piece_prob must be >= 0 and sum to <= 1")
    zero = ZeroSource(3, num_envs, device)
    if cfg.push_piece_prob <= 0.0:
        return MixedSource((lib, zero), (1.0 - cfg.zero_prob, cfg.zero_prob), num_envs, device, generator,
                           names=("library", "zero"))
    held = PiecewiseHeld(tuple(cfg.push_piece_kinds), tuple(cfg.push_piece_s), cfg.push_piece_blend_s,
                         num_envs, device, generator)
    return MixedSource((lib, held, zero), (max(lib_prob, 0.0), cfg.push_piece_prob, cfg.zero_prob), num_envs,
                       device, generator, names=("library", "push_pieces", "zero"))


class RollerDanceTwistCommand(DanceVelocityCommand):
    """Streams the dance twist from the library or a file and outputs the roller command."""

    cfg: RollerDanceTwistCommandCfg

    def __init__(self, cfg: RollerDanceTwistCommandCfg, env):
        if cfg.source == "file":
            # The file source of the dance tasks (resume_on_reset, reset ramp).
            DanceVelocityCommand.__init__(self, cfg, env)
        else:
            VelocityCommandCommandOnly.__init__(self, cfg, env)
            commands, file_dt, _ = load_command_file(cfg.command_file)
            if abs(file_dt - env.step_dt) > 1e-6:
                raise ValueError(f"{cfg.command_file}: dt={file_dt} differs from env.step_dt={env.step_dt}")
            lo, hi = cfg.columns
            frames = torch.as_tensor(commands[:, lo:hi], dtype=torch.float32)
            lib = ClipPlayer(frames, file_dt, env.num_envs, env.device, random_start=True)
            self._source = build_training_source(lib, cfg, env.num_envs, env.device)
        self.dance_twist = torch.zeros(self.num_envs, 3, device=self.device)
        self._target_yaw = torch.zeros(self.num_envs, device=self.device)
        self._anchor = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids):
        out = super().reset(env_ids)
        # The robot pose is written by the reset events, so the heading target is
        # anchored to the yaw at the next command update.
        if env_ids is None:
            self._anchor[:] = True
        else:
            self._anchor[env_ids] = True
        return out

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return
        self._source.resample(env_ids)
        self.dance_twist[env_ids] = self._source.value()[env_ids]
        self.is_standing_env[env_ids] = False

    def _update_command(self) -> None:
        dt = getattr(self, "_compute_dt", self._env.step_dt)
        self.dance_twist[:] = self._source.step(dt)
        yaw = yaw_from_quat(self.robot.data.root_link_quat_w)
        self._target_yaw = torch.where(self._anchor, yaw, self._target_yaw)
        self._anchor[:] = False
        cmd, self._target_yaw = map_dance_twist(
            self.dance_twist, yaw, self._target_yaw, dt,
            self.cfg.push_gain, self.cfg.push_range, self.cfg.heading_max,
        )
        self.vel_command_b[:] = cmd
        self.vel_command_w[:] = cmd

    def _update_metrics(self) -> None:
        pass


# --------------------------------------------------------------------------- rewards


def forward_lean_relative_reward(
    env,
    command_name: str,
    body_command_name: str = "body_pose",
    target_pitch: float = 0.262,
    std: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=("trunk_base",)),
) -> torch.Tensor:
    """forward_lean_reward with the target lean shifted by the commanded body pitch.

    push * exp(-(g_b,x - sin(target_pitch + pitch_cmd))^2 / std^2). At zero body
    command this is the roller forward_lean term up to sin(target) vs target.
    """
    asset = env.scene[asset_cfg.name]
    push = torch.clamp(env.command_manager.get_command(command_name)[:, 0], min=0.0)
    pitch_cmd = env.command_manager.get_command(body_command_name)[:, 4]
    lean = asset.data.projected_gravity_b[:, 0]
    target = torch.sin(target_pitch + pitch_cmd)
    return push * torch.exp(-((lean - target) ** 2) / (std ** 2))


def roller_body_tracking(
    env,
    command_name: str = "body_pose",
    nominal_height: float = 0.138,
    nominal_pitch: float = 0.0,
    z_std: float = 0.02,
    angle_std: float = 0.15,
    z_range: tuple[float, float] = (-0.035, 0.010),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """dance_body_tracking_clamped measured about the roller stance.

    Height target nominal_height + clamp(z_cmd), roll target roll_cmd, pitch
    target nominal_pitch + pitch_cmd. nominal_pitch is the lean the roller
    policy skates with, so a zero body command does not ask it to stand straight.
    """
    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    pos_w = torch.nan_to_num(asset.data.root_link_pos_w, nan=0.0)
    z_cmd = cmd[:, 2].clamp(z_range[0], z_range[1])
    z_err = pos_w[:, 2] - env.scene.terrain.env_origins[:, 2] - (nominal_height + z_cmd)
    quat = asset.data.root_link_quat_w
    qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    roll = torch.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
    pitch = torch.asin(torch.clamp(2.0 * (qw * qy - qz * qx), -1.0, 1.0))
    r_z = torch.exp(-((z_err / z_std) ** 2))
    r_r = torch.exp(-(((roll - cmd[:, 3]) / angle_std) ** 2))
    r_p = torch.exp(-(((pitch - nominal_pitch - cmd[:, 4]) / angle_std) ** 2))
    return (r_z + r_r + r_p) / 3.0


__all__ = [
    "HEADING_MAX",
    "PUSH_GAIN",
    "PUSH_PIECE_KINDS",
    "PUSH_RANGE",
    "RollerDanceTwistCommandCfg",
    "build_training_source",
    "forward_lean_relative_reward",
    "map_dance_twist",
    "roller_body_tracking",
    "wrap_angle",
    "yaw_from_quat",
]
