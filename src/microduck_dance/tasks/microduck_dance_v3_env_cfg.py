"""dance_v3: the dance task with sustained footwork commands and untaxed body motion.

Design: dance/docs/EXPERIMENTS.md v3design001 (the causes) and v3001 (the config).
Built on make_microduck_dance_env_cfg(source="library"), so DR, observations
(61D), noise and the dance head std stay identical. Changes against dance_v2:

- Commands mix sources per env at every resample (4 to 8 s). Twist: held
  footwork pieces 0.50, library 0.40, exact-zero idle 0.10. Head: library 0.85,
  uniform holds 0.15. Body: library 0.60, slow large sines 0.40.
- track_linear_velocity prices only the planar error and
  track_angular_velocity only the yaw-rate error. The base gait's vz bob and
  roll and pitch rates cost 0.5 to 0.7 per step there and made standing the
  optimum below about 0.25 m/s.
- upright is measured against the commanded roll and pitch, and pose against
  HOME + S * body command. Both equal the Velocity terms at zero body command.
- body tracking: weight 3.0, z_std 0.02, angle_std 0.15, z target clamped to
  the reachable [-0.035, +0.010] m.

No new curriculum. A resume from base5000 restores common_step_counter, so the
inherited stages (all at or below 2000 iterations) start at their final value.
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass

import mujoco
import numpy as np
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import RewardTermCfg
from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainOutput

from microduck_dance.ranges import BODY_RANGES, DANCE_STAND_HEIGHT, HEAD_RANGES
from microduck_dance.tasks.dance_v3 import (
    DanceV3PoseCommandCfg,
    DanceV3VelocityCommandCfg,
    dance_body_tracking_clamped,
    posture_relative,
    track_planar_velocity,
    track_yaw_rate,
    upright_relative,
)
from microduck_dance.tasks.microduck_dance_env_cfg import (
    MicroduckDanceRlCfg,
    make_microduck_dance_env_cfg,
)

TWIST_MODE_PROBS = (0.50, 0.40, 0.10)  # held footwork, library, idle
TWIST_PIECE_S = (1.0, 3.0)
TWIST_BLEND_S = 0.2
TWIST_PIECE_PROBS = (0.15, 0.20, 0.65)  # exact zero, turn in place, uniform
TWIST_TURN_WZ_ABS = (0.4, 1.0)
TWIST_HELD_RANGES = ((-0.35, 0.35), (-0.25, 0.25), (-1.0, 1.0))

HEAD_LIBRARY_PROB = 0.85
HEAD_HOLD_S = (2.0, 5.0)

BODY_LIBRARY_PROB = 0.60
BODY_SLOW_FREQ_HZ = (0.1, 0.4)
BODY_SLOW_RANGES = ((0.0, 0.0), (0.0, 0.0), (-0.03, 0.01), (-0.20, 0.20), (-0.20, 0.20), (0.0, 0.0))
BODY_SLOW_HOLD_PROB = 0.1

# Reachable trunk height range with flat feet (v3 design report, section 2).
BODY_Z_TARGET_RANGE = (-0.035, 0.010)
BODY_TRACKING_WEIGHT = 3.0
BODY_Z_STD = 0.02
BODY_ANGLE_STD = 0.15

TWIST_TRACKING_WEIGHT = 2.0
LIN_STD = math.sqrt(0.1)
YAW_STD = math.sqrt(0.5)

# Posture synergy S: leg joint offsets from HOME as a function of the body
# command. For each body_z knot (m) a 6-tuple: the joint offset (rad) and the
# coefficients of roll, pitch, roll^2, pitch^2 and roll*pitch (angles in rad);
# linear in z between knots. Fitted by flat-foot IK on the walk model
# (dance/scripts/retarget/fit_posture_synergy.py): squat with the CoM over the feet, roll
# by hip_roll = -roll plus a leg-length differential, pitch with the CoM kept
# over the feet. Every term is zero at zero command.
POSTURE_Z_KNOTS = (-0.035, -0.025, -0.015, -0.005, 0.0, 0.005, 0.010)
POSTURE_SYNERGY: dict[str, tuple[tuple[float, ...], ...]] = {
    'left_hip_yaw': ((+0.0000, +0.0000, +0.0002, -0.0010, -0.0003, -0.0002), (+0.0000, +0.0000, +0.0001, -0.0007, -0.0002, -0.0001), (+0.0000, +0.0000, +0.0000, -0.0003, -0.0001, -0.0001), (+0.0000, +0.0000, +0.0000, -0.0001, -0.0000, -0.0000), (+0.0000, -0.0000, +0.0001, +0.0002, -0.0002, +0.0001), (-0.0000, -0.0000, +0.0002, +0.0001, -0.0005, +0.0000), (-0.0000, -0.0004, +0.0012, -0.0011, -0.0013, -0.0004)),
    'left_hip_roll': ((+0.0000, -1.0000, +0.0000, +0.0001, -0.0442, -0.0000), (+0.0000, -1.0000, +0.0000, +0.0002, -0.0445, -0.0000), (+0.0000, -1.0000, +0.0000, +0.0002, -0.0445, +0.0000), (+0.0000, -1.0000, -0.0000, +0.0003, -0.0445, +0.0000), (+0.0000, -1.0000, +0.0000, +0.0004, -0.0446, -0.0000), (-0.0000, -0.9998, +0.0000, +0.0002, -0.0448, -0.0006), (-0.0001, -0.9814, +0.0013, -0.0394, -0.0475, -0.0525)),
    'left_hip_pitch': ((+0.6948, -0.3221, -2.2824, -0.8614, -1.4906, -0.5825), (+0.5118, -0.3370, -2.0924, -0.4992, -1.1622, -0.4413), (+0.3267, -0.3813, -1.9116, -0.3153, -0.9683, -0.5224), (+0.1215, -0.4665, -1.7468, -0.2896, -0.9200, -0.3618), (+0.0000, -0.5721, -1.6676, +0.1549, -1.0197, -0.0077), (-0.1516, -0.8007, -1.5034, -0.8335, -1.3205, -0.3522), (-0.3987, -1.5525, -1.1221, +1.9861, -1.4530, -0.1877)),
    'left_knee': ((+1.1692, -0.4940, +0.3292, -0.0854, -1.3405, +0.1167), (+0.8940, -0.5303, +0.3144, -0.2381, -1.3957, +0.0704), (+0.5867, -0.6056, +0.3119, -0.2882, -1.4600, +0.0741), (+0.2224, -0.7444, +0.3422, -0.4736, -1.6479, +0.0629), (+0.0000, -0.8587, +0.4009, -1.0028, -1.9288, -0.1518), (-0.2801, -1.2565, +0.4275, -1.7388, -2.0362, +0.2466), (-0.7258, -2.5066, +0.3291, +1.0400, -1.5634, -0.1288)),
    'left_ankle': ((+0.4744, -0.1719, +1.6081, +0.7740, +0.1497, +0.6992), (+0.3823, -0.1934, +1.4030, +0.2593, -0.2339, +0.5117), (+0.2600, -0.2242, +1.2198, +0.0265, -0.4919, +0.5965), (+0.1009, -0.2779, +1.0852, -0.1844, -0.7280, +0.4247), (+0.0000, -0.2866, +1.0647, -1.1573, -0.9094, -0.1440), (-0.1286, -0.4558, +0.9276, -0.9050, -0.7169, +0.5988), (-0.3272, -0.9538, +0.4519, -0.9710, -0.1184, +0.0581)),
    'right_hip_yaw': ((-0.0000, +0.0000, -0.0002, +0.0010, +0.0003, -0.0002), (-0.0000, +0.0000, -0.0001, +0.0007, +0.0002, -0.0001), (-0.0000, +0.0000, -0.0000, +0.0003, +0.0001, -0.0001), (-0.0000, +0.0000, -0.0000, +0.0001, +0.0000, -0.0000), (+0.0000, +0.0000, -0.0001, -0.0001, +0.0002, -0.0001), (+0.0000, -0.0000, -0.0002, -0.0001, +0.0005, -0.0003), (+0.0000, -0.0004, -0.0012, +0.0010, +0.0013, -0.0001)),
    'right_hip_roll': ((-0.0000, -1.0000, -0.0000, -0.0001, +0.0442, -0.0000), (-0.0000, -1.0000, -0.0000, -0.0002, +0.0445, +0.0000), (-0.0000, -1.0000, -0.0000, -0.0002, +0.0445, +0.0000), (-0.0000, -1.0000, +0.0000, -0.0003, +0.0445, +0.0000), (+0.0000, -1.0000, -0.0000, -0.0004, +0.0446, -0.0001), (+0.0000, -0.9998, -0.0000, -0.0001, +0.0448, -0.0007), (+0.0001, -0.9814, -0.0012, +0.0378, +0.0472, -0.0529)),
    'right_hip_pitch': ((-0.6728, -0.3203, +2.2895, +0.8265, +1.3503, -0.6759), (-0.5001, -0.3369, +2.0949, +0.6164, +1.0782, -0.4632), (-0.3224, -0.3795, +1.9145, +0.3850, +0.9527, -0.5255), (-0.1209, -0.4661, +1.7639, +0.3778, +1.0047, -0.3585), (+0.0000, -0.5652, +1.6598, +0.8816, +1.1853, -0.0743), (+0.1514, -0.8008, +1.5113, +0.7812, +1.3961, -0.4393), (+0.3987, -1.5727, +1.1285, -2.0844, +1.3968, -0.0868)),
    'right_knee': ((-1.1730, -0.4963, -0.2958, +0.0828, +1.3482, +0.0938), (-0.8961, -0.5304, -0.2923, +0.1467, +1.3976, +0.0564), (-0.5875, -0.6064, -0.3007, +0.3679, +1.4364, +0.0657), (-0.2226, -0.7461, -0.3344, +0.6038, +1.6506, +0.0494), (+0.0000, -0.8701, -0.3395, +0.7524, +1.7215, -0.0937), (+0.2802, -1.2766, -0.4248, +1.6331, +2.0938, +0.3857), (+0.7258, -2.5062, -0.3306, -1.0104, +1.5654, -0.2016)),
    'right_ankle': ((-0.5002, -0.1760, -1.5818, -0.7418, -0.0014, +0.7694), (-0.3959, -0.1935, -1.3835, -0.4686, +0.3197, +0.5195), (-0.2651, -0.2269, -1.2114, -0.0163, +0.4839, +0.5912), (-0.1017, -0.2801, -1.0945, +0.2263, +0.6460, +0.4078), (+0.0000, -0.3049, -0.9956, -0.1296, +0.5367, -0.0197), (+0.1288, -0.4757, -0.9329, +0.8517, +0.6992, +0.8241), (+0.3272, -0.9332, -0.4597, +1.1011, +0.1776, -0.1147)),
}

# Rough dance terrain (task 28). The shared rough terrain has 1.5 to 2 m flat platforms at the
# tile centres, where the robot spawns, and a dance travels too little to leave them. The rough
# dance task shrinks the platforms on its own copy of the terrain cfg, so the 0.5 m spawn noise
# already puts feet on steps, cells and slopes. Step, cell and slope heights are unchanged.
DANCE_ROUGH_PLATFORM_WIDTH = 0.5
DANCE_ROUGH_SUB_TERRAINS_WITH_PLATFORM = ("pyramid_stairs", "random_grid", "pyramid_slope")

# Mixed floor (task 30). The foot mesh against a box is convex CCD contact, against the plane it is
# the plane primitive, and dance_v4 turns differently on the two (rough-diagnosis.md). A fine-tune on
# boxes only (rough003) lost flat turning in 250 iterations. The mixed task keeps a MuJoCo plane in the
# scene and gives part of the tiles no geometry, so envs on those tiles stand on the plane while the
# others stand on boxes and height fields, all in one batch. The plane lies MIX_PLANE_DROP below z = 0:
# the lowest walkable box top is -1 cm (grid cells), so a foot on a box tile never reaches the plane,
# and a plane tile is a shallow pit whose edges are 4 m from the spawn point.
MIX_PLANE_TILE = "plane"
MIX_PLANE_SHARE = 0.5  # fraction of tiles (and so of envs) on the plane
MIX_PLANE_DROP = 0.03
MIX_PLANE_GEOM = "mix_ground_plane"

STEP_CEILING_INHERITED = 120_000  # 5000 iterations x 24 steps: base5000's counter on resume


def make_microduck_dance_v3_env_cfg(
    play: bool = False, command_file: str | None = None, rough: bool = False, mixed: bool = False
) -> ManagerBasedRlEnvCfg:
    if command_file is None:
        raise ValueError("dance_v3 requires command_file (the dance library)")
    if mixed and not rough:
        raise ValueError("mixed is a variant of the rough terrain")
    cfg = make_microduck_dance_env_cfg(play=play, command_file=command_file, source="library", rough=rough)
    if rough:
        _dance_rough_terrain(cfg)
    if mixed:
        _dance_mixed_floor(cfg)

    twist = cfg.commands["twist"]
    cfg.commands["twist"] = DanceV3VelocityCommandCfg(
        **{
            **vars(twist),
            "source": "mix",
            "hold_prob": 0.0,
            "mode_probs": TWIST_MODE_PROBS,
            "piece_s": TWIST_PIECE_S,
            "blend_s": TWIST_BLEND_S,
            "piece_probs": TWIST_PIECE_PROBS,
            "turn_wz_abs": TWIST_TURN_WZ_ABS,
            "held_ranges": TWIST_HELD_RANGES,
        }
    )
    head = cfg.commands["head_pose"]
    cfg.commands["head_pose"] = DanceV3PoseCommandCfg(
        **{
            **vars(head),
            "ranges": HEAD_RANGES,
            "source": "mix",
            "library_prob": HEAD_LIBRARY_PROB,
            "alt": "held",
            "hold_s": HEAD_HOLD_S,
        }
    )
    body = cfg.commands["body_pose"]
    cfg.commands["body_pose"] = DanceV3PoseCommandCfg(
        **{
            **vars(body),
            "ranges": BODY_RANGES,
            "source": "mix",
            "library_prob": BODY_LIBRARY_PROB,
            "alt": "smooth",
            "freq_range_hz": BODY_SLOW_FREQ_HZ,
            "hold_prob": BODY_SLOW_HOLD_PROB,
            "alt_ranges": BODY_SLOW_RANGES,
        }
    )

    r = cfg.rewards
    r["track_linear_velocity"] = RewardTermCfg(
        func=track_planar_velocity,
        weight=TWIST_TRACKING_WEIGHT,
        params={"command_name": "twist", "std": LIN_STD},
    )
    r["track_angular_velocity"] = RewardTermCfg(
        func=track_yaw_rate,
        weight=TWIST_TRACKING_WEIGHT,
        params={"command_name": "twist", "std": YAW_STD},
    )
    up = r["upright"]
    r["upright"] = RewardTermCfg(
        func=upright_relative,
        weight=up.weight,
        params={"std": up.params["std"], "asset_cfg": deepcopy(up.params["asset_cfg"]), "command_name": "body_pose"},
    )
    pose = r["pose"]
    r["pose"] = RewardTermCfg(
        func=posture_relative,
        weight=pose.weight,
        params={
            **deepcopy(pose.params),
            "body_command_name": "body_pose",
            "synergy": dict(POSTURE_SYNERGY),
            "z_knots": POSTURE_Z_KNOTS,
            "z_range": BODY_Z_TARGET_RANGE,
        },
    )
    r["body_pose_tracking"] = RewardTermCfg(
        func=dance_body_tracking_clamped,
        weight=BODY_TRACKING_WEIGHT,
        params={
            "command_name": "body_pose",
            "nominal_height": DANCE_STAND_HEIGHT,
            "z_std": BODY_Z_STD,
            "angle_std": BODY_ANGLE_STD,
            "z_range": BODY_Z_TARGET_RANGE,
            # On rough ground the env origin is the ground at the spawn point only. Off the
            # platform the robot stands on other steps, so measure from the ground under the feet.
            "height_ref": "feet" if rough else "env_origin",
        },
    )
    return cfg


def _dance_rough_terrain(cfg) -> None:
    """Small platforms and a fixed uniform spread over the terrain tiles for the rough dance task.

    Works on a copy of the terrain generator cfg. terrain_levels moves envs by walked distance, which
    a dance does not produce, so the levels sank to the easiest row (mean 0.51 of 9 after 340
    iterations). It is removed, and envs start on any row (max_init_terrain_level None). The
    generator samples type and difficulty per tile at random, so this spreads them uniformly.
    """
    gen = deepcopy(cfg.scene.terrain.terrain_generator)
    for name in DANCE_ROUGH_SUB_TERRAINS_WITH_PLATFORM:
        gen.sub_terrains[name].platform_width = DANCE_ROUGH_PLATFORM_WIDTH
    cfg.scene.terrain.terrain_generator = gen
    cfg.scene.terrain.max_init_terrain_level = None
    cfg.curriculum.pop("terrain_levels", None)


@dataclass(kw_only=True)
class PlaneTileCfg(SubTerrainCfg):
    """A tile without geometry. The robot stands on the scene's ground plane (add_mix_ground_plane)."""

    plane_z: float = -MIX_PLANE_DROP

    def function(self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator) -> TerrainOutput:
        del difficulty, spec, rng
        return TerrainOutput(origin=np.array([self.size[0] / 2, self.size[1] / 2, self.plane_z]), geometries=[])


def add_mix_ground_plane(spec: mujoco.MjSpec) -> None:
    """The ground plane of the mixed floor, in the terrain body so the foot contact and ray sensors see it.

    Built like mjlab's plane terrain (TerrainEntity._import_ground_plane), with the default contact
    parameters, so it is not softened like the boxes. Static geoms of one body never collide with each
    other, so the plane only meets the robot.
    """
    spec.body("terrain").add_geom(
        name=MIX_PLANE_GEOM, type=mujoco.mjtGeom.mjGEOM_PLANE, size=(0, 0, 0.01), pos=(0.0, 0.0, -MIX_PLANE_DROP),
        rgba=(0.45, 0.5, 0.55, 1.0),
    )


def _dance_mixed_floor(cfg) -> None:
    """Plane tiles next to the rough tiles: MIX_PLANE_SHARE of the tile proportion goes to PlaneTileCfg.

    The rough tiles keep their relative proportions and the dance platforms of _dance_rough_terrain.
    The generator samples every tile's type at random (curriculum False), envs go to random tiles and
    stay there (no terrain_levels), so the batch splits in about the same shares.
    """
    gen = cfg.scene.terrain.terrain_generator
    assert not gen.curriculum, "the mixed floor needs random tiles"
    rough_total = sum(sub.proportion for sub in gen.sub_terrains.values())
    plane = PlaneTileCfg(proportion=rough_total * MIX_PLANE_SHARE / (1.0 - MIX_PLANE_SHARE), size=gen.size)
    gen.sub_terrains = {MIX_PLANE_TILE: plane, **gen.sub_terrains}
    prev = cfg.scene.spec_fn

    def spec_fn(spec: mujoco.MjSpec) -> None:
        if prev is not None:
            prev(spec)  # soften the terrain boxes first, the plane keeps the default contact
        add_mix_ground_plane(spec)

    spec_fn.mixed_floor = True
    spec_fn.previous = prev
    cfg.scene.spec_fn = spec_fn


MicroduckDanceV3RlCfg = deepcopy(MicroduckDanceRlCfg)
MicroduckDanceV3RlCfg.experiment_name = "dance_v3"
MicroduckDanceV3RlCfg.run_name = "dance_v3"
MicroduckDanceV3RlCfg.max_iterations = 3000
MicroduckDanceV3RlCfg.save_interval = 250

# Rough terrain twin (Mjlab-DanceV3-Rough-MicroDuck). Same runner settings, own log folder.
MicroduckDanceV3RoughRlCfg = deepcopy(MicroduckDanceV3RlCfg)
MicroduckDanceV3RoughRlCfg.experiment_name = "dance_v3_rough"
MicroduckDanceV3RoughRlCfg.run_name = "dance_v3_rough"

# Mixed floor twin (Mjlab-DanceV3-RoughMix-MicroDuck, task 30). Same runner settings, own log folder.
MicroduckDanceV3RoughMixRlCfg = deepcopy(MicroduckDanceV3RlCfg)
MicroduckDanceV3RoughMixRlCfg.experiment_name = "dance_v3_rough_mix"
MicroduckDanceV3RoughMixRlCfg.run_name = "dance_v3_rough_mix"
