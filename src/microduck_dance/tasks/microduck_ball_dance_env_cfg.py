"""Ball dance: the blind basketball task driven by the dance library commands.

Built on make_microduck_basketball_env_cfg(blind=True), so the scene, spawn on
the ball apex, DR, terminations and every basketball reward stay as they were
for the b11 policy. Only the command block changes.

* twist: library twist times BALL_TWIST_SCALE, clipped to the ball range.
  Each resample picks an exact zero interval with probability ZERO_TWIST_PROB.
* head_pose: library head columns at full scale.
* body_pose: library body columns (z, roll, pitch) times BALL_BODY_SCALE.
* Actor obs head_command and body_command slots carry these commands. The
  critic sees the head command in its head slot and keeps the ball state in
  its body slot, so its width stays 76 and the b11 critic loads strictly.
* Rewards: head_pose_tracking and body_pose_tracking added on top of the
  basketball stack. No new curriculum.

b11 was trained with zeros in the head and body slots, so its obs normalizer
has std 0 there. A continuation must reseed those slots (see
basketball_finetune.reseed_command_normalizer and library_command_stats) and
should zero their untrained input weights (zero_command_input_columns).
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import generated_commands
from mjlab.managers import ObservationTermCfg, RewardTermCfg

from microduck_dance.command_file import load_command_file
from microduck_dance.mix import MixedSource, ZeroSource
from microduck_dance.ranges import (
    BALL_TWIST_RANGES,
    BALL_TWIST_SCALE,
    BODY_RANGES,
    HEAD_RANGES,
)
from microduck_dance.trajectory import ClipPlayer
from mjlab_microduck.tasks import mdp as microduck_mdp
from microduck_dance.tasks.dance_commands import (
    DancePoseCommand,
    DancePoseCommandCfg,
    DanceVelocityCommand,
    DanceVelocityCommandCfg,
    dance_body_tracking,
)
from mjlab_microduck.tasks.mdp import UniformPoseCommand, VelocityCommandCommandOnly
from mjlab_microduck.tasks.microduck_basketball_env_cfg import (
    BALL_RADIUS,
    ROOT_HEIGHT,
    MicroduckBasketballRlCfg,
    make_microduck_basketball_env_cfg,
)
from microduck_dance.tasks.microduck_dance_env_cfg import HEAD_TRACKING_STD, RESAMPLE_S

DANCE_LIBRARY = str(Path(__file__).resolve().parents[3] / "dance" / "data" / "library.npz")

# Factor on the library body columns. Body z, roll and pitch move the trunk
# over a rolling ball, so they start at 30% of the floor dance amplitude.
BALL_BODY_SCALE = 0.3
ZERO_TWIST_PROB = 0.15
HEAD_TRACKING_WEIGHT = 2.0
BODY_TRACKING_WEIGHT = 1.0
BODY_Z_STD = 0.02
BODY_ANGLE_STD = 0.15
# Root z above the env origin that body z = 0 refers to. The same reference
# as the basketball height reward (root at ball centre + R + ROOT_HEIGHT, ball
# centre at R on the floor), so the two terms never pull apart.
BALL_STAND_HEIGHT = 2 * BALL_RADIUS + ROOT_HEIGHT
# Control step of the library (50 Hz). The env step must match it.
STEP_DT = 0.02


# --------------------------------------------------------------------------- sources


def library_frames(path: str, columns: tuple[int, int]) -> torch.Tensor:
    """Columns lo:hi of the command library as a float32 (T, D) tensor."""
    if not Path(path).is_file():
        raise FileNotFoundError(
            f"dance library {path} is missing. Build it with dance/scripts/retarget/retarget_all.sh (build_library.py)."
        )
    commands, _, _ = load_command_file(path)
    lo, hi = columns
    return torch.as_tensor(commands[:, lo:hi], dtype=torch.float32)


def make_ball_twist_source(frames: torch.Tensor, scale: float, limits, zero_prob: float,
                           num_envs: int, device, generator: torch.Generator | None = None) -> MixedSource:
    """Library twist times scale, clipped to limits, or an exact zero interval."""
    lo = torch.tensor([r[0] for r in limits], dtype=torch.float32)
    hi = torch.tensor([r[1] for r in limits], dtype=torch.float32)
    scaled = torch.clamp(frames * scale, lo, hi)
    lib = ClipPlayer(scaled, STEP_DT, num_envs, device, random_start=True, generator=generator)
    zero = ZeroSource(frames.shape[1], num_envs, device)
    return MixedSource((lib, zero), (1.0 - zero_prob, zero_prob), num_envs, device, generator,
                       names=("library", "zero"))


def make_scaled_pose_source(frames: torch.Tensor, scale: float, num_envs: int, device,
                            generator: torch.Generator | None = None) -> ClipPlayer:
    return ClipPlayer(frames * scale, STEP_DT, num_envs, device, random_start=True, generator=generator)


def library_command_stats(path: str = DANCE_LIBRARY) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Mean and std of the head and body commands this task plays, per obs term."""
    head = library_frames(path, (3, 7))
    body = library_frames(path, (7, 13)) * BALL_BODY_SCALE
    return {
        name: (x.mean(0), x.std(0, unbiased=False))
        for name, x in (("head_command", head), ("body_command", body))
    }


# --------------------------------------------------------------------------- command terms


@dataclass(kw_only=True)
class BallDanceTwistCommandCfg(DanceVelocityCommandCfg):
    source: Literal["library"] = "library"
    scale: float = BALL_TWIST_SCALE
    zero_prob: float = ZERO_TWIST_PROB

    def build(self, env) -> "BallDanceTwistCommand":
        return BallDanceTwistCommand(self, env)


class BallDanceTwistCommand(DanceVelocityCommand):
    cfg: BallDanceTwistCommandCfg

    def __init__(self, cfg: BallDanceTwistCommandCfg, env):
        VelocityCommandCommandOnly.__init__(self, cfg, env)
        _check_dt(cfg.command_file, env.step_dt)
        r = cfg.ranges
        self._source = make_ball_twist_source(
            library_frames(cfg.command_file, cfg.columns), cfg.scale,
            (r.lin_vel_x, r.lin_vel_y, r.ang_vel_z), cfg.zero_prob, env.num_envs, env.device,
        )


@dataclass(kw_only=True)
class ScaledLibraryPoseCommandCfg(DancePoseCommandCfg):
    source: Literal["library"] = "library"
    scale: float = 1.0

    def build(self, env) -> "ScaledLibraryPoseCommand":
        return ScaledLibraryPoseCommand(self, env)


class ScaledLibraryPoseCommand(DancePoseCommand):
    cfg: ScaledLibraryPoseCommandCfg

    def __init__(self, cfg: ScaledLibraryPoseCommandCfg, env):
        UniformPoseCommand.__init__(self, cfg, env)
        _check_dt(cfg.command_file, env.step_dt)
        self._source = make_scaled_pose_source(
            library_frames(cfg.command_file, cfg.columns), cfg.scale, env.num_envs, env.device,
        )


def _check_dt(path: str, step_dt: float) -> None:
    _, file_dt, _ = load_command_file(path)
    if abs(file_dt - step_dt) > 1e-6 or abs(file_dt - STEP_DT) > 1e-6:
        raise ValueError(f"{path}: dt={file_dt} differs from env.step_dt={step_dt}")


# --------------------------------------------------------------------------- cfg


def make_microduck_ball_dance_env_cfg(
    play: bool = False, command_file: str = DANCE_LIBRARY,
) -> ManagerBasedRlEnvCfg:
    cfg = make_microduck_basketball_env_cfg(play=play, blind=True, history=1)

    twist = cfg.commands["twist"]
    cfg.commands["twist"] = BallDanceTwistCommandCfg(
        **{
            **vars(twist),
            "resampling_time_range": RESAMPLE_S,
            "freq_range_hz": (0.1, 0.6),
            "command_file": command_file,
            "columns": (0, 3),
        }
    )
    r = cfg.commands["twist"].ranges
    r.lin_vel_x, r.lin_vel_y, r.ang_vel_z = BALL_TWIST_RANGES
    cfg.commands["head_pose"] = ScaledLibraryPoseCommandCfg(
        ranges=HEAD_RANGES,
        resampling_time_range=RESAMPLE_S,
        freq_range_hz=(0.3, 2.0),
        columns=(3, 7),
        command_file=command_file,
        scale=1.0,
    )
    cfg.commands["body_pose"] = ScaledLibraryPoseCommandCfg(
        ranges=tuple((lo * BALL_BODY_SCALE, hi * BALL_BODY_SCALE) for lo, hi in BODY_RANGES),
        resampling_time_range=RESAMPLE_S,
        freq_range_hz=(0.3, 1.5),
        columns=(7, 13),
        command_file=command_file,
        scale=BALL_BODY_SCALE,
    )

    # Actor head and body slots carry the commands. The critic gets the head
    # command in its zero pad slot and keeps the ball state in the body slot.
    actor = cfg.observations["actor"].terms
    critic = cfg.observations["critic"].terms
    actor["head_command"] = ObservationTermCfg(func=generated_commands, params={"command_name": "head_pose"})
    actor["body_command"] = ObservationTermCfg(func=generated_commands, params={"command_name": "body_pose"})
    critic["head_command"] = ObservationTermCfg(func=generated_commands, params={"command_name": "head_pose"})

    cfg.rewards["head_pose_tracking"] = RewardTermCfg(
        func=microduck_mdp.head_pose_tracking,
        weight=HEAD_TRACKING_WEIGHT,
        params={"command_name": "head_pose", "std": HEAD_TRACKING_STD},
    )
    cfg.rewards["body_pose_tracking"] = RewardTermCfg(
        func=dance_body_tracking,
        weight=BODY_TRACKING_WEIGHT,
        params={
            "command_name": "body_pose",
            "nominal_height": BALL_STAND_HEIGHT,
            "z_std": BODY_Z_STD,
            "angle_std": BODY_ANGLE_STD,
        },
    )

    # standing_envs zeroes the twist of some envs. The zero intervals do that here.
    cfg.curriculum.pop("standing_envs", None)
    return cfg


MicroduckBallDanceRlCfg = deepcopy(MicroduckBasketballRlCfg)
MicroduckBallDanceRlCfg.experiment_name = "ball_dance"
MicroduckBallDanceRlCfg.run_name = "ball_dance"
