"""Registration of the dance tasks with the mjlab task registry.

mjlab imports this package through the mjlab.tasks entry point
(pyproject.toml), after or before mjlab_microduck.tasks. The import below makes
sure the upstream tasks and MicroduckOnPolicyRunner exist first. Task ids,
cfgs and runner settings are the same as in the fork these tasks came from.

The dance library is dance/data/library.npz at the repository root, built by
dance/scripts/retarget/retarget_all.sh. The path is absolute and the file is
read only when an env is built, so a missing library does not break import or
any other task.
"""

from pathlib import Path as _Path

from mjlab.tasks.registry import register_mjlab_task
from mjlab_microduck.tasks import MicroduckOnPolicyRunner

from microduck_dance.tasks.microduck_ball_dance_env_cfg import (
    MicroduckBallDanceRlCfg,
    make_microduck_ball_dance_env_cfg,
)
from microduck_dance.tasks.microduck_dance_env_cfg import (
    MicroduckDanceRlCfg,
    make_microduck_dance_env_cfg,
)
from microduck_dance.tasks.microduck_dance_v3_env_cfg import (
    MicroduckDanceV3RlCfg,
    MicroduckDanceV3RoughMixRlCfg,
    MicroduckDanceV3RoughRlCfg,
    make_microduck_dance_v3_env_cfg,
)
from microduck_dance.tasks.microduck_roller_dance_env_cfg import (
    MicroduckRollerDanceRlCfg,
    make_microduck_roller_dance_env_cfg,
)

_DANCE_LIBRARY = str(_Path(__file__).resolve().parents[3] / "dance" / "data" / "library.npz")

# Dance: Velocity recipe with streaming command terms (obs layout unchanged).
register_mjlab_task(
    task_id="Mjlab-Dance-Flat-MicroDuck",
    env_cfg=make_microduck_dance_env_cfg(),
    play_env_cfg=make_microduck_dance_env_cfg(play=True),
    rl_cfg=MicroduckDanceRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# DanceLib: same task, commands replayed from the retargeted dance library.
register_mjlab_task(
    task_id="Mjlab-DanceLib-Flat-MicroDuck",
    env_cfg=make_microduck_dance_env_cfg(source="library", command_file=_DANCE_LIBRARY),
    play_env_cfg=make_microduck_dance_env_cfg(play=True, source="library", command_file=_DANCE_LIBRARY),
    rl_cfg=MicroduckDanceRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# DanceV3: held footwork pieces mixed into the commands, untaxed body motion.
register_mjlab_task(
    task_id="Mjlab-DanceV3-Flat-MicroDuck",
    env_cfg=make_microduck_dance_v3_env_cfg(command_file=_DANCE_LIBRARY),
    play_env_cfg=make_microduck_dance_v3_env_cfg(play=True, command_file=_DANCE_LIBRARY),
    rl_cfg=MicroduckDanceV3RlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# DanceV3 on the Velocity rough terrain (steps up to 1.5 cm, random grid up to
# 1 cm, gentle slopes). The policy stays blind, as the rough walking policy is.
register_mjlab_task(
    task_id="Mjlab-DanceV3-Rough-MicroDuck",
    env_cfg=make_microduck_dance_v3_env_cfg(command_file=_DANCE_LIBRARY, rough=True),
    play_env_cfg=make_microduck_dance_v3_env_cfg(play=True, command_file=_DANCE_LIBRARY, rough=True),
    rl_cfg=MicroduckDanceV3RoughRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# DanceV3 on a mixed floor: half the tiles are the MuJoCo ground plane, the others the
# rough dance tiles, so one policy trains the plane contact and the box contact at once.
register_mjlab_task(
    task_id="Mjlab-DanceV3-RoughMix-MicroDuck",
    env_cfg=make_microduck_dance_v3_env_cfg(command_file=_DANCE_LIBRARY, rough=True, mixed=True),
    play_env_cfg=make_microduck_dance_v3_env_cfg(play=True, command_file=_DANCE_LIBRARY, rough=True, mixed=True),
    rl_cfg=MicroduckDanceV3RoughMixRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Ball dance: the blind basketball task driven by the dance library commands
# (twist scaled to the ball range, body columns at BALL_BODY_SCALE).
register_mjlab_task(
    task_id="Mjlab-BallDance-MicroDuck",
    env_cfg=make_microduck_ball_dance_env_cfg(),
    play_env_cfg=make_microduck_ball_dance_env_cfg(play=True),
    rl_cfg=MicroduckBallDanceRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Roller dance: the roller skate task driven by the dance library commands,
# continued from Pollen's roller.onnx (scripts/onnx_to_checkpoint.py, scripts/finetune_roller_dance.py).
register_mjlab_task(
    task_id="Mjlab-RollerDance-MicroDuck",
    env_cfg=make_microduck_roller_dance_env_cfg(),
    play_env_cfg=make_microduck_roller_dance_env_cfg(play=True),
    rl_cfg=MicroduckRollerDanceRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
