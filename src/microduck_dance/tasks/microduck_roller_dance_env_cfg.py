"""Roller dance: the roller skate task driven by the dance library commands.

Experiment log: dance/docs/EXPERIMENTS.md, roller001 and roller002. Built on make_microduck_velocity_rollers_env_cfg, so the rollers
model (robot_allcollisions_rollers.xml, 4 passive wheels), its DR, sensors,
obs noise, delays and the critic wheel_vel term stay as they are. The policy
starts from Pollen's roller.onnx (scripts/onnx_to_checkpoint.py).

Commands
* twist: dance library twist mapped to the roller meaning every step
  (dance_rollers.map_dance_twist: push = clip(5 vx), vy dropped, heading
  target integrated from wz). Exact zero intervals with probability 0.15.
* head_pose, body_pose: dance_v3 mixes (library 0.85 / held, library 0.60 /
  slow sines), at full scale.
* Actor and critic head and body slots carry these commands (the roller task
  zero pads them).

Rewards against the roller task
* removed: neck_joint_pos_l2 (pulls the neck to HOME against head tracking),
  heading_hold (replaced by heading_tracking, cmd[2] is used again).
* changed: pose is posture_relative (HOME + body synergy, neck and head not
  priced), upright is upright_relative, forward_lean targets the lean plus the
  commanded pitch at half weight, glide stillness std 5 -> 10 so commanded
  knee bends do not zero it, neck_action_rate_l2 -0.5 -> -0.1, action_rate
  fixed at ROLLER_ACTION_RATE (no ramp).
* added: head_pose_tracking (std 0.25), head_pose_bias (self negating,
  positive weight), body_pose_tracking about the roller stance.
* DR curricula fixed at their final stage: wheel bearing friction 0.0015,
  CoM ranges 0.01. roller.onnx is assumed to have been through them.

roller002 (continued from roller001 model_1699). roller001 tracked head and
body but stood nearly in place (vx correlation 0.03 to 0.16, half the speed of
roller.onnx): wheel_speed logged 0.02 per s against 3.4 for head and body
tracking. Two changes push it back toward skating:
* twist source: PUSH_PIECE_PROB of the resamples draw held push pieces
  (dance_rollers.PUSH_PIECE_KINDS, 1.5 to 4 s), because the library pushes
  only in 0.2 s bursts. Song playback is unchanged.
* wheel_speed weight 10 -> WHEEL_SPEED_WEIGHT.
* braking vel_std 0.3 -> BRAKING_VEL_STD and weight 1 -> BRAKING_WEIGHT
  (second stage, from model_2999). With std 0.3 m/s a brake paid 85% of its
  maximum at 0.12 m/s, so the faster policy skated through the song brakes
  and its vx correlation fell from 0.33 to 0.24 (swag, seed 0).

ROLLER_STAGES holds the settings of the three training stages (roller001,
roller002 stage 1, roller002 stage 2). The factory's `stage` argument selects
one. The default is the final stage, and the module constants below are its
values.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import generated_commands
from mjlab.managers import ObservationTermCfg, RewardTermCfg

from microduck_dance.ranges import BODY_RANGES, HEAD_RANGES
from mjlab_microduck.tasks import mdp as microduck_mdp
from microduck_dance.tasks.dance_rollers import (
    HEADING_MAX,
    PUSH_GAIN,
    PUSH_RANGE,
    RollerDanceTwistCommandCfg,
    forward_lean_relative_reward,
    roller_body_tracking,
)
from microduck_dance.tasks.dance_v3 import (
    DanceV3PoseCommandCfg,
    posture_relative,
    upright_relative,
)
from microduck_dance.tasks.microduck_dance_env_cfg import HEAD_TRACKING_STD, RESAMPLE_S
from microduck_dance.tasks.microduck_dance_v3_env_cfg import (
    BODY_ANGLE_STD,
    BODY_LIBRARY_PROB,
    BODY_SLOW_FREQ_HZ,
    BODY_SLOW_HOLD_PROB,
    BODY_SLOW_RANGES,
    BODY_Z_STD,
    BODY_Z_TARGET_RANGE,
    HEAD_HOLD_S,
    HEAD_LIBRARY_PROB,
    POSTURE_SYNERGY,
    POSTURE_Z_KNOTS,
)
from mjlab_microduck.tasks.microduck_velocity_rollers_env_cfg import (
    MicroduckRollersRlCfg,
    make_microduck_velocity_rollers_env_cfg,
)

DANCE_LIBRARY = str(Path(__file__).resolve().parents[3] / "dance" / "data" / "library.npz")

ZERO_TWIST_PROB = 0.15
PUSH_PIECE_PROB = 0.4
WHEEL_SPEED_WEIGHT = 20.0
BRAKING_WEIGHT = 2.0
BRAKING_VEL_STD = 0.1
# Action scale of the joint position action. roller.onnx's metadata says 1.0
# and the Pollen runtime manifest says 0.8. Zero training check (task-27
# report, swag 20 s, 16 envs, DR on, mapped twist): at 1.0 roller.onnx glides
# 4.7 m/min with 0.38 falls/min, at 0.8 it glides 2.8 m/min with 0 falls and
# the same twist correlation. 0.8 is what the robot runs, so we train at 0.8.
ROLLER_ACTION_SCALE = 0.8
# Trunk (root link) height above the floor and trunk pitch of roller.onnx
# at scale 0.8 under the mapped dance twist, medians over 16 envs x 20 s
# (same check). It skates crouched and leaning forward.
ROLLER_STAND_HEIGHT = 0.116
ROLLER_STAND_PITCH = 0.104
ROLLER_ACTION_RATE = -1.5
HEAD_TRACKING_WEIGHT = 2.0
HEAD_BIAS_WEIGHT = 1.0
BODY_TRACKING_WEIGHT = 2.0
HEADING_TRACKING_WEIGHT = 1.0
HEADING_TRACKING_STD = 0.5
FORWARD_LEAN_WEIGHT = 0.75
GLIDE_STILLNESS_STD = 10.0
NECK_ACTION_RATE_WEIGHT = -0.1
WHEEL_FRICTION_FINAL = 0.0015
COM_RANGE_FINAL = 0.01
LOOSE_STD = 999.0  # posture std that prices nothing (as for the passive wheels)

# Settings that changed between the training stages (dance/docs/EXPERIMENTS.md roller001, roller002).
# Everything else is the same in all three.
ROLLER_STAGES = {
    "roller001": {"push_piece_prob": 0.0, "wheel_speed_weight": 10.0,
                  "braking_weight": 1.0, "braking_vel_std": 0.3},
    "roller002_stage1": {"push_piece_prob": PUSH_PIECE_PROB, "wheel_speed_weight": WHEEL_SPEED_WEIGHT,
                         "braking_weight": 1.0, "braking_vel_std": 0.3},
    "roller002_stage2": {"push_piece_prob": PUSH_PIECE_PROB, "wheel_speed_weight": WHEEL_SPEED_WEIGHT,
                         "braking_weight": BRAKING_WEIGHT, "braking_vel_std": BRAKING_VEL_STD},
}
FINAL_STAGE = "roller002_stage2"


def make_microduck_roller_dance_env_cfg(
    play: bool = False, command_file: str = DANCE_LIBRARY, source: str = "library", stage: str = FINAL_STAGE,
) -> ManagerBasedRlEnvCfg:
    if source not in ("library", "file"):
        raise ValueError(f"unknown source {source!r}")
    if stage not in ROLLER_STAGES:
        raise ValueError(f"unknown stage {stage!r}, choose from {list(ROLLER_STAGES)}")
    st = ROLLER_STAGES[stage]
    cfg = make_microduck_velocity_rollers_env_cfg(play=play)
    cfg.actions["joint_pos"].scale = ROLLER_ACTION_SCALE

    file_mode = source == "file"
    resample = (1e6, 1e6) if file_mode else RESAMPLE_S

    # ── commands ───────────────────────────────────────────────────────────
    twist = cfg.commands["twist"]
    fields = {k: v for k, v in vars(twist).items()}
    cfg.commands["twist"] = RollerDanceTwistCommandCfg(
        **{
            **fields,
            "resampling_time_range": resample,
            "freq_range_hz": (0.1, 0.6),
            "hold_prob": 0.0,
            "source": source,
            "command_file": command_file,
            "columns": (0, 3),
            "zero_prob": ZERO_TWIST_PROB,
            "push_piece_prob": 0.0 if file_mode else st["push_piece_prob"],
            "push_gain": PUSH_GAIN,
            "push_range": PUSH_RANGE,
            "heading_max": HEADING_MAX,
        }
    )
    r = cfg.commands["twist"].ranges
    r.lin_vel_x, r.lin_vel_y, r.ang_vel_z = PUSH_RANGE, (0.0, 0.0), (-HEADING_MAX, HEADING_MAX)

    if file_mode:
        from microduck_dance.tasks.dance_commands import DancePoseCommandCfg

        head = DancePoseCommandCfg(ranges=HEAD_RANGES, resampling_time_range=resample, freq_range_hz=(0.3, 2.0),
                                   hold_prob=0.0, columns=(3, 7), source="file", command_file=command_file)
        body = DancePoseCommandCfg(ranges=BODY_RANGES, resampling_time_range=resample, freq_range_hz=(0.3, 1.5),
                                   hold_prob=0.0, columns=(7, 13), source="file", command_file=command_file)
    else:
        head = DanceV3PoseCommandCfg(
            ranges=HEAD_RANGES, resampling_time_range=resample, freq_range_hz=(0.3, 2.0), hold_prob=0.1,
            columns=(3, 7), command_file=command_file, source="mix", library_prob=HEAD_LIBRARY_PROB,
            alt="held", hold_s=HEAD_HOLD_S,
        )
        body = DanceV3PoseCommandCfg(
            ranges=BODY_RANGES, resampling_time_range=resample, freq_range_hz=BODY_SLOW_FREQ_HZ,
            hold_prob=BODY_SLOW_HOLD_PROB, columns=(7, 13), command_file=command_file, source="mix",
            library_prob=BODY_LIBRARY_PROB, alt="smooth", alt_ranges=BODY_SLOW_RANGES,
        )
    cfg.commands["head_pose"] = head
    cfg.commands["body_pose"] = body

    for group in ("actor", "critic"):
        terms = cfg.observations[group].terms
        terms["head_command"] = ObservationTermCfg(func=generated_commands, params={"command_name": "head_pose"})
        terms["body_command"] = ObservationTermCfg(func=generated_commands, params={"command_name": "body_pose"})

    # ── rewards ────────────────────────────────────────────────────────────
    rw = cfg.rewards
    rw.pop("neck_joint_pos_l2")
    rw.pop("heading_hold")

    pose = rw["pose"]
    params = deepcopy(pose.params)
    for key in ("std_standing", "std_walking", "std_running"):
        params[key] = {**params[key], r".*neck.*": LOOSE_STD, r".*head.*": LOOSE_STD}
    rw["pose"] = RewardTermCfg(
        func=posture_relative,
        weight=pose.weight,
        params={**params, "body_command_name": "body_pose", "synergy": dict(POSTURE_SYNERGY),
                "z_knots": POSTURE_Z_KNOTS, "z_range": BODY_Z_TARGET_RANGE},
    )
    up = rw["upright"]
    rw["upright"] = RewardTermCfg(
        func=upright_relative,
        weight=up.weight,
        params={"std": up.params["std"], "asset_cfg": deepcopy(up.params["asset_cfg"]), "command_name": "body_pose"},
    )
    lean = rw["forward_lean"]
    rw["forward_lean"] = RewardTermCfg(
        func=forward_lean_relative_reward,
        weight=FORWARD_LEAN_WEIGHT,
        params={"command_name": "twist", "body_command_name": "body_pose",
                "target_pitch": lean.params["target_pitch"], "std": lean.params["std"]},
    )
    rw["glide"].params["stillness_std"] = GLIDE_STILLNESS_STD
    rw["neck_action_rate_l2"].weight = NECK_ACTION_RATE_WEIGHT
    rw["action_rate_l2"].weight = ROLLER_ACTION_RATE
    rw["wheel_speed"].weight = st["wheel_speed_weight"]
    rw["braking"].weight = st["braking_weight"]
    rw["braking"].params["vel_std"] = st["braking_vel_std"]
    rw["heading_tracking"] = RewardTermCfg(
        func=microduck_mdp.heading_tracking_reward,
        weight=HEADING_TRACKING_WEIGHT,
        params={"command_name": "twist", "std": HEADING_TRACKING_STD},
    )
    rw["head_pose_tracking"] = RewardTermCfg(
        func=microduck_mdp.head_pose_tracking,
        weight=HEAD_TRACKING_WEIGHT,
        params={"command_name": "head_pose", "std": HEAD_TRACKING_STD},
    )
    # Self negating (returns <= 0), so the weight is positive.
    rw["head_pose_bias"] = RewardTermCfg(
        func=microduck_mdp.head_pose_bias_penalty,
        weight=HEAD_BIAS_WEIGHT,
        params={"command_name": "head_pose", "tau_s": 1.0},
    )
    rw["body_pose_tracking"] = RewardTermCfg(
        func=roller_body_tracking,
        weight=BODY_TRACKING_WEIGHT,
        params={"command_name": "body_pose", "nominal_height": ROLLER_STAND_HEIGHT,
                "nominal_pitch": ROLLER_STAND_PITCH, "z_std": BODY_Z_STD, "angle_std": BODY_ANGLE_STD,
                "z_range": BODY_Z_TARGET_RANGE},
    )

    # ── curricula fixed at their final stage ──────────────────────────────
    cfg.curriculum.pop("action_rate_weight", None)
    if cfg.curriculum.pop("wheel_friction", None) is not None:
        cfg.events["randomize_wheel_friction"].params["ranges"] = (WHEEL_FRICTION_FINAL, WHEEL_FRICTION_FINAL)
    for name, event in (("com_range", "randomize_com"), ("head_com_range", "randomize_head_com")):
        if cfg.curriculum.pop(name, None) is not None:
            cfg.events[event].params["ranges"] = (-COM_RANGE_FINAL, COM_RANGE_FINAL)
    return cfg


MicroduckRollerDanceRlCfg = deepcopy(MicroduckRollersRlCfg)
MicroduckRollerDanceRlCfg.experiment_name = "roller_dance"
MicroduckRollerDanceRlCfg.run_name = "roller_dance"
MicroduckRollerDanceRlCfg.max_iterations = 1700
MicroduckRollerDanceRlCfg.actor.distribution_cfg = {
    "class_name": "GaussianDistribution",
    "init_std": 0.25,
    "std_type": "scalar",
}
MicroduckRollerDanceRlCfg.algorithm.learning_rate = 2.0e-5
MicroduckRollerDanceRlCfg.algorithm.schedule = "fixed"
MicroduckRollerDanceRlCfg.algorithm.entropy_coef = 0.005
