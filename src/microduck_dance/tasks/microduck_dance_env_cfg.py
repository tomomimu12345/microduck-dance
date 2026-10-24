"""Dance task: the Velocity walking recipe driven by streaming command terms.

Built from make_microduck_velocity_env_cfg so DR, observations and noise stay
identical. The 61D actor observation layout is unchanged, which lets a
Velocity checkpoint be fine-tuned here and evaluated in the same environment.
Only the command terms, the body tracking reward and the curricula differ.
"""

from __future__ import annotations

import functools
import importlib.util
from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import RewardTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg

from microduck_dance.ranges import BODY_RANGES, DANCE_STAND_HEIGHT, HEAD_RANGES
from microduck_dance.tasks.dance_commands import (
    DancePoseCommandCfg,
    DanceVelocityCommandCfg,
    dance_body_tracking,
)
from mjlab_microduck.tasks import microduck_velocity_env_cfg as _vel
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    MicroduckRlCfg,
    make_microduck_velocity_env_cfg,
)

RESAMPLE_S = (4.0, 8.0)
FILE_RESAMPLE_S = (1e6, 1e6)
BODY_TRACKING_WEIGHT = 2.0
# head_pose_tracking std in the dance tasks (Velocity keeps 0.5). Dance head
# commands are small (library p95 0.25 to 0.37 rad), and at std 0.5 a still head
# scored 1.871 of 2.0 on the library, so tracking was nearly free to ignore (exp head001).
HEAD_TRACKING_STD = 0.25


@functools.cache
def _pristine_rough_terrains_cfg():
    """MICRODUCK_ROUGH_TERRAINS_CFG as its module defines it, before any factory wrote to it.

    At upstream f010ef1 the rough factories assign the shared module object to the cfg and
    the play branch then sets curriculum False and a 5 x 5 grid on it. Registration builds
    the play cfgs at import, so by the time this package is imported the shared object
    already holds the play grid. Running the module source again in a private module gives
    the object as written, without patching upstream. The extra module is not put in
    sys.modules, and its own imports resolve to the modules already loaded.
    """
    spec = importlib.util.find_spec(_vel.__name__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MICRODUCK_ROUGH_TERRAINS_CFG


def rough_terrains_cfg(play: bool = False):
    """A private copy of the rough terrain cfg for one dance cfg.

    Training gets the full 10 x 20 grid with the module's curriculum setting, play the
    5 x 5 grid without curriculum that the Velocity play branch sets.
    """
    gen = deepcopy(_pristine_rough_terrains_cfg())
    if play:
        gen.curriculum = False
        gen.num_cols = 5
        gen.num_rows = 5
    return gen


def make_microduck_dance_env_cfg(
    play: bool = False,
    command_file: str | None = None,
    source: str = "smooth",
    rough: bool = False,
) -> ManagerBasedRlEnvCfg:
    if source not in ("smooth", "library", "file"):
        raise ValueError(f"unknown source {source!r}")
    if source in ("library", "file") and command_file is None:
        raise ValueError(f"source={source!r} requires command_file")

    # rough=True takes the Velocity rough settings: terrain generator
    # MICRODUCK_ROUGH_TERRAINS_CFG (as a private copy, see rough_terrains_cfg),
    # terrain_levels curriculum, softened terrain contacts, nconmax 200 and 30
    # solver iterations. Nothing below touches them.
    cfg = make_microduck_velocity_env_cfg(play=play, rough=rough)
    if rough:
        cfg.scene.terrain.terrain_generator = rough_terrains_cfg(play)

    resample = FILE_RESAMPLE_S if source == "file" else RESAMPLE_S

    def hold(default: float) -> float:
        return 0.0 if source == "file" else default

    twist = deepcopy(cfg.commands["twist"])
    cfg.commands["twist"] = DanceVelocityCommandCfg(
        **{
            **vars(twist),
            "resampling_time_range": resample,
            "freq_range_hz": (0.1, 0.6),
            "hold_prob": hold(0.25),
            "source": source,
            "command_file": command_file,
            "columns": (0, 3),
        }
    )
    cfg.commands["head_pose"] = DancePoseCommandCfg(
        ranges=HEAD_RANGES,
        resampling_time_range=resample,
        freq_range_hz=(0.3, 2.0),
        hold_prob=hold(0.1),
        columns=(3, 7),
        source=source,
        command_file=command_file,
    )
    cfg.commands["body_pose"] = DancePoseCommandCfg(
        ranges=BODY_RANGES,
        resampling_time_range=resample,
        freq_range_hz=(0.3, 1.5),
        hold_prob=hold(0.1),
        columns=(7, 13),
        source=source,
        command_file=command_file,
    )

    cfg.rewards["body_pose_tracking"] = RewardTermCfg(
        func=dance_body_tracking,
        weight=BODY_TRACKING_WEIGHT,
        params={"command_name": "body_pose", "nominal_height": DANCE_STAND_HEIGHT},
    )
    cfg.rewards["head_pose_tracking"].params["std"] = HEAD_TRACKING_STD

    # These curricula overwrite the command ranges every step, which would
    # undo HEAD_RANGES / BODY_RANGES set above.
    for name in ("head_pose_range", "body_pose_range", "standing_envs"):
        cfg.curriculum.pop(name, None)

    return cfg


MicroduckDanceRlCfg = deepcopy(MicroduckRlCfg)
MicroduckDanceRlCfg.experiment_name = "dance"
MicroduckDanceRlCfg.run_name = "dance"
MicroduckDanceRlCfg.max_iterations = 2000
