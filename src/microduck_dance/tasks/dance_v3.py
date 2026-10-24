"""dance_v3 command terms and reward functions.

Commands: each term mixes sources per env at every resample (see
microduck_dance.mix). Rewards: versions of the Velocity tracking,
upright and posture terms that price only the escapable error, so that
stepping and commanded body motion are no longer taxed. At zero body command
the upright and posture terms equal the Velocity ones exactly.

Kept apart from mdp.py, like dance_commands.py, so our work stays separable
from upstream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch

from mjlab.managers import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp.rewards import variable_posture
from mjlab.utils.lab_api.math import quat_apply_inverse

from microduck_dance.command_file import load_command_file
from microduck_dance.mix import MixedSource, PieceKind, PiecewiseHeld, ZeroSource
from microduck_dance.ranges import DANCE_STAND_HEIGHT
from microduck_dance.trajectory import ClipPlayer, SmoothTrajectory, SmoothTrajectorySpec
from microduck_dance.tasks.dance_commands import (
    DancePoseCommand,
    DancePoseCommandCfg,
    DanceVelocityCommand,
    DanceVelocityCommandCfg,
)
from mjlab_microduck.tasks.mdp import UniformPoseCommand, VelocityCommandCommandOnly


# --------------------------------------------------------------------------- commands


def load_library_columns(path: str | None, columns: tuple[int, int], step_dt: float) -> torch.Tensor:
    if path is None:
        raise ValueError("dance_v3 needs a library command file")
    if not Path(path).is_file():
        raise FileNotFoundError(
            f"dance_v3 library {path} is missing. Build it with dance/scripts/retarget/retarget_all.sh "
            "(it runs dance/scripts/retarget/build_library.py)."
        )
    commands, file_dt, _ = load_command_file(path)
    if abs(file_dt - step_dt) > 1e-6:
        raise ValueError(f"{path}: dt={file_dt} differs from env.step_dt={step_dt}")
    lo, hi = columns
    return torch.as_tensor(commands[:, lo:hi], dtype=torch.float32)


@dataclass(kw_only=True)
class DanceV3VelocityCommandCfg(DanceVelocityCommandCfg):
    source: Literal["mix"] = "mix"
    # Mode probabilities per resample: held footwork, library, idle (exact zero).
    mode_probs: tuple[float, float, float] = (0.50, 0.40, 0.10)
    piece_s: tuple[float, float] = (1.0, 3.0)
    blend_s: float = 0.2
    # Piece kinds inside the held mode: exact zero, turn in place, uniform box.
    piece_probs: tuple[float, float, float] = (0.15, 0.20, 0.65)
    turn_wz_abs: tuple[float, float] = (0.4, 1.0)
    held_ranges: tuple[tuple[float, float], ...] = ((-0.35, 0.35), (-0.25, 0.25), (-1.0, 1.0))

    def build(self, env) -> "DanceV3VelocityCommand":
        return DanceV3VelocityCommand(self, env)


def make_twist_source(cfg: DanceV3VelocityCommandCfg, frames: torch.Tensor, step_dt: float,
                      num_envs: int, device, generator: torch.Generator | None = None) -> MixedSource:
    zero3 = ((0.0, 0.0),) * 3
    kinds = (
        PieceKind(cfg.piece_probs[0], zero3),
        PieceKind(cfg.piece_probs[1], zero3, abs_ranges=((2, *cfg.turn_wz_abs),)),
        PieceKind(cfg.piece_probs[2], tuple(tuple(r) for r in cfg.held_ranges)),
    )
    held = PiecewiseHeld(kinds, cfg.piece_s, cfg.blend_s, num_envs, device, generator)
    lib = ClipPlayer(frames, step_dt, num_envs, device, random_start=True, generator=generator)
    idle = ZeroSource(3, num_envs, device)
    return MixedSource((held, lib, idle), cfg.mode_probs, num_envs, device, generator,
                       names=("held", "library", "idle"))


class DanceV3VelocityCommand(DanceVelocityCommand):
    cfg: DanceV3VelocityCommandCfg

    def __init__(self, cfg: DanceV3VelocityCommandCfg, env):
        VelocityCommandCommandOnly.__init__(self, cfg, env)
        frames = load_library_columns(cfg.command_file, cfg.columns, env.step_dt)
        self._source = make_twist_source(cfg, frames, env.step_dt, env.num_envs, env.device)


@dataclass(kw_only=True)
class DanceV3PoseCommandCfg(DancePoseCommandCfg):
    """source "mix": library with probability library_prob, otherwise the alt source.

    alt "held": piecewise-constant uniform holds over `ranges`, pieces of hold_s.
    alt "smooth": SmoothTrajectory over alt_ranges with freq_range_hz and hold_prob.
    """

    source: Literal["mix"] = "mix"
    library_prob: float = 0.85
    alt: Literal["held", "smooth"] = "held"
    hold_s: tuple[float, float] = (2.0, 5.0)
    alt_ranges: tuple[tuple[float, float], ...] = field(default_factory=tuple)

    def build(self, env) -> "DanceV3PoseCommand":
        return DanceV3PoseCommand(self, env)


def make_pose_source(cfg: DanceV3PoseCommandCfg, frames: torch.Tensor, step_dt: float,
                     num_envs: int, device, generator: torch.Generator | None = None) -> MixedSource:
    lib = ClipPlayer(frames, step_dt, num_envs, device, random_start=True, generator=generator)
    if cfg.alt == "held":
        alt = PiecewiseHeld((PieceKind(1.0, tuple(tuple(r) for r in cfg.ranges)),), cfg.hold_s, 0.0,
                            num_envs, device, generator)
    elif cfg.alt == "smooth":
        spec = SmoothTrajectorySpec(
            ranges=tuple(tuple(r) for r in cfg.alt_ranges),
            freq_range_hz=cfg.freq_range_hz,
            num_harmonics=cfg.num_harmonics,
            hold_prob=cfg.hold_prob,
        )
        alt = SmoothTrajectory(spec, num_envs, device, generator)
    else:
        raise ValueError(f"unknown alt {cfg.alt!r}")
    return MixedSource((lib, alt), (cfg.library_prob, 1.0 - cfg.library_prob), num_envs, device, generator,
                       names=("library", cfg.alt))


class DanceV3PoseCommand(DancePoseCommand):
    cfg: DanceV3PoseCommandCfg

    def __init__(self, cfg: DanceV3PoseCommandCfg, env):
        UniformPoseCommand.__init__(self, cfg, env)
        frames = load_library_columns(cfg.command_file, cfg.columns, env.step_dt)
        self._source = make_pose_source(cfg, frames, env.step_dt, env.num_envs, env.device)


# --------------------------------------------------------------------------- rewards


def track_planar_velocity(
    env,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """exp(-(vx_err^2 + vy_err^2) / std^2). Unlike mjlab's term, vz (gait bob) is not priced."""
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    actual = asset.data.root_link_lin_vel_b
    err = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
    return torch.exp(-err / std**2)


def track_yaw_rate(
    env,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """exp(-wz_err^2 / std^2). Unlike mjlab's term, roll and pitch rates are not priced."""
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    actual = asset.data.root_link_ang_vel_b
    return torch.exp(-torch.square(command[:, 2] - actual[:, 2]) / std**2)


def commanded_gravity_b(roll: torch.Tensor, pitch: torch.Tensor) -> torch.Tensor:
    """Unit gravity in the trunk frame for a trunk at (roll, pitch), ZYX Euler (yaw free)."""
    return torch.stack(
        [torch.sin(pitch), -torch.sin(roll) * torch.cos(pitch), -torch.cos(roll) * torch.cos(pitch)], dim=-1
    )


def upright_relative(
    env,
    std: float,
    command_name: str = "body_pose",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """exp(-|g_b,xy - g_cmd,xy|^2 / std^2) with g_cmd from the commanded body roll and pitch.

    At zero command g_cmd,xy = 0 and this is mjlab's flat-ground upright term.
    """
    asset = env.scene[asset_cfg.name]
    if asset_cfg.body_ids:
        quat = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :].squeeze(1)
    else:
        quat = asset.data.root_link_quat_w
    g_b = quat_apply_inverse(quat, asset.data.gravity_vec_w)
    cmd = env.command_manager.get_command(command_name)
    g_cmd = commanded_gravity_b(cmd[:, 3], cmd[:, 4])
    err = torch.sum(torch.square(g_b[:, :2] - g_cmd[:, :2]), dim=1)
    return torch.exp(-err / std**2)


def synergy_offsets(body_cmd: torch.Tensor, table: torch.Tensor, knots: torch.Tensor,
                    z_range: tuple[float, float]) -> torch.Tensor:
    """Joint offsets S(cmd) for body commands (N, 6).

    table: (J, K, 6), per body_z knot the joint offset and the coefficients of
    roll, pitch, roll^2, pitch^2 and roll*pitch. S is linear in z between knots.
    The z command is clamped to z_range and to the knot range. S(0) = 0 exactly.
    """
    z = body_cmd[:, 2].clamp(z_range[0], z_range[1]).clamp(float(knots[0]), float(knots[-1]))
    j = (torch.searchsorted(knots, z.contiguous(), right=True) - 1).clamp(0, len(knots) - 2)
    w = ((z - knots[j]) / (knots[j + 1] - knots[j])).to(table.dtype)
    t0 = table[:, j].permute(1, 0, 2)  # (N, J, 6)
    t1 = table[:, j + 1].permute(1, 0, 2)
    row = t0 + (t1 - t0) * w[:, None, None]
    r = body_cmd[:, 3].to(table.dtype)
    p = body_cmd[:, 4].to(table.dtype)
    basis = torch.stack([r, p, r * r, p * p, r * p], dim=1)  # (N, 5)
    return row[..., 0] + torch.einsum("njc,nc->nj", row[..., 1:], basis)


class posture_relative(variable_posture):
    """mjlab variable_posture with the target HOME + S(body_cmd).

    synergy maps leg joint names to one 6-tuple (offset and the roll, pitch,
    roll^2, pitch^2, roll*pitch coefficients) per entry of z_knots (fitted by dance/scripts/retarget/fit_posture_synergy.py). Joints
    not in the table keep the HOME target. The offset at z = 0 is zero, so at
    zero body command this is variable_posture.
    """

    def __init__(self, cfg: RewardTermCfg, env):
        super().__init__(cfg, env)
        asset = env.scene[cfg.params["asset_cfg"].name]
        _, names = asset.find_joints(cfg.params["asset_cfg"].joint_names)
        table = cfg.params["synergy"]
        knots = tuple(cfg.params["z_knots"])
        unknown = set(table) - set(names)
        if unknown:
            raise ValueError(f"synergy joints not in the posture selection: {sorted(unknown)}")
        zero = tuple((0.0,) * 6 for _ in knots)
        for n, rows in table.items():
            if len(rows) != len(knots) or any(len(r) != 6 for r in rows):
                raise ValueError(f"synergy {n}: {len(rows)} rows for {len(knots)} knots")
        self.knots = torch.tensor(knots, dtype=torch.float32, device=env.device)
        self.table = torch.tensor([table.get(n, zero) for n in names], dtype=torch.float32, device=env.device)

    def __call__(
        self,
        env,
        std_standing,
        std_walking,
        std_running,
        asset_cfg: SceneEntityCfg,
        command_name: str,
        body_command_name: str = "body_pose",
        synergy: dict | None = None,
        z_knots: tuple[float, ...] = (),
        z_range: tuple[float, float] = (-0.035, 0.010),
        walking_threshold: float = 0.5,
        running_threshold: float = 1.5,
    ) -> torch.Tensor:
        del synergy, z_knots
        asset = env.scene[asset_cfg.name]
        command = env.command_manager.get_command(command_name)
        total_speed = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
        standing = (total_speed < walking_threshold).float().unsqueeze(1)
        walking = ((total_speed >= walking_threshold) & (total_speed < running_threshold)).float().unsqueeze(1)
        running = (total_speed >= running_threshold).float().unsqueeze(1)
        std = self.std_standing * standing + self.std_walking * walking + self.std_running * running

        body = env.command_manager.get_command(body_command_name)
        desired = self.default_joint_pos[:, asset_cfg.joint_ids] + synergy_offsets(body, self.table, self.knots, z_range)
        err2 = torch.square(asset.data.joint_pos[:, asset_cfg.joint_ids] - desired)
        return torch.exp(-torch.mean(err2 / (std**2), dim=1))


FOOT_HEIGHT_SENSOR = "foot_height_scan"


def ground_z_under_feet_from(frame_z: torch.Tensor, heights: torch.Tensor) -> torch.Tensor:
    """(N,) ground height below the feet: the mean over feet of frame z minus the clearance.

    frame_z and heights are (N, F), the foot site heights and their clearance
    above the terrain from a TerrainHeightSensor (min reduction, so the highest
    ground point under each foot).
    """
    return torch.mean(frame_z - heights, dim=1)


def ground_z_under_feet(env, sensor_name: str = FOOT_HEIGHT_SENSOR) -> torch.Tensor:
    """(N,) ground height below the two feet, from the foot height ray sensor."""
    data = env.scene[sensor_name].data
    return ground_z_under_feet_from(data.frame_pos_w[:, :, 2], data.heights)


def body_height_reference(env, height_ref: str = "env_origin",
                          sensor_name: str = FOOT_HEIGHT_SENSOR) -> torch.Tensor:
    """(N,) the z that body height is measured from.

    "env_origin" is the env origin z (the ground at the spawn point). "feet" is
    the ground under the feet, so on stairs the reference moves with the step
    the robot stands on. On a plane both are 0.
    """
    if height_ref == "env_origin":
        return env.scene.terrain.env_origins[:, 2]
    if height_ref == "feet":
        return ground_z_under_feet(env, sensor_name)
    raise ValueError(f"unknown height_ref {height_ref!r}")


def dance_body_tracking_clamped(
    env,
    command_name: str = "body_pose",
    nominal_height: float = DANCE_STAND_HEIGHT,
    z_std: float = 0.02,
    angle_std: float = 0.15,
    z_range: tuple[float, float] = (-0.035, 0.010),
    height_ref: str = "env_origin",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """dance_body_tracking with the body_z target clamped to the reachable range.

    height_ref selects the ground the body height is measured from (see body_height_reference).
    """
    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)

    pos_w = torch.nan_to_num(asset.data.root_link_pos_w, nan=0.0)
    z_cmd = cmd[:, 2].clamp(z_range[0], z_range[1])
    ground = torch.nan_to_num(body_height_reference(env, height_ref), nan=0.0)
    z_err = pos_w[:, 2] - ground - (nominal_height + z_cmd)

    quat = asset.data.root_link_quat_w
    qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    roll = torch.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
    pitch = torch.asin(torch.clamp(2.0 * (qw * qy - qz * qx), -1.0, 1.0))

    r_z = torch.exp(-((z_err / z_std) ** 2))
    r_r = torch.exp(-(((roll - cmd[:, 3]) / angle_std) ** 2))
    r_p = torch.exp(-(((pitch - cmd[:, 4]) / angle_std) ** 2))
    return (r_z + r_r + r_p) / 3.0


__all__ = [
    "FOOT_HEIGHT_SENSOR",
    "DanceV3PoseCommandCfg",
    "DanceV3VelocityCommandCfg",
    "commanded_gravity_b",
    "body_height_reference",
    "dance_body_tracking_clamped",
    "ground_z_under_feet",
    "ground_z_under_feet_from",
    "make_pose_source",
    "make_twist_source",
    "posture_relative",
    "synergy_offsets",
    "track_planar_velocity",
    "track_yaw_rate",
    "upright_relative",
]
