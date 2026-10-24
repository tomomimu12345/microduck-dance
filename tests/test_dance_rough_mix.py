"""Mjlab-DanceV3-RoughMix-MicroDuck: plane tiles and rough tiles in one terrain (task 30, CPU)."""

from __future__ import annotations


import mjlab.tasks  # noqa: F401
import microduck_dance.tasks  # noqa: F401
import mujoco
import numpy as np
import pytest
from mjlab.scene import Scene
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

from microduck_dance.tasks.microduck_dance_v3_env_cfg import (
    DANCE_ROUGH_PLATFORM_WIDTH,
    DANCE_ROUGH_SUB_TERRAINS_WITH_PLATFORM,
    MIX_PLANE_DROP,
    MIX_PLANE_GEOM,
    MIX_PLANE_SHARE,
    MIX_PLANE_TILE,
    PlaneTileCfg,
    make_microduck_dance_v3_env_cfg,
)
from mjlab_microduck.tasks.microduck_velocity_env_cfg import MICRODUCK_ROUGH_TERRAINS_CFG, _soften_terrain_contacts

MIX = "Mjlab-DanceV3-RoughMix-MicroDuck"
ROUGH = "Mjlab-DanceV3-Rough-MicroDuck"
FLAT = "Mjlab-DanceV3-Flat-MicroDuck"


def _small_scene(num_envs: int = 64):
    cfg = load_env_cfg(MIX)
    g = cfg.scene.terrain.terrain_generator
    g.num_rows, g.num_cols, g.seed = 4, 4, 0
    cfg.scene.num_envs = num_envs
    import torch

    torch.manual_seed(0)  # env rows are drawn with torch
    scene = Scene(cfg.scene, device="cpu")
    return cfg, scene, scene.compile()


@pytest.mark.parametrize("play", [False, True])
def test_mix_adds_plane_tiles_to_the_rough_dance_terrain(play):
    mix, rough = load_env_cfg(MIX, play=play), load_env_cfg(ROUGH, play=play)
    gm, gr = mix.scene.terrain.terrain_generator, rough.scene.terrain.terrain_generator
    assert list(gm.sub_terrains) == [MIX_PLANE_TILE, *gr.sub_terrains]
    assert isinstance(gm.sub_terrains[MIX_PLANE_TILE], PlaneTileCfg)
    total = sum(s.proportion for s in gm.sub_terrains.values())
    assert gm.sub_terrains[MIX_PLANE_TILE].proportion / total == pytest.approx(MIX_PLANE_SHARE)
    # Rough tiles unchanged: proportions, heights and the 0.5 m dance platforms.
    for name, sub in gr.sub_terrains.items():
        assert vars(gm.sub_terrains[name]) == vars(sub), name
    for name in DANCE_ROUGH_SUB_TERRAINS_WITH_PLATFORM:
        assert gm.sub_terrains[name].platform_width == DANCE_ROUGH_PLATFORM_WIDTH
    assert gm.sub_terrains["pyramid_stairs"].step_height_range == (0.0, 0.015)
    assert gm.sub_terrains["pyramid_stairs"].step_width == 0.15
    assert gm.sub_terrains["random_grid"].grid_width == 0.45
    assert gm.sub_terrains["random_grid"].grid_height_range == (0.0, 0.010)
    assert not gm.curriculum and mix.scene.terrain.max_init_terrain_level is None
    assert "terrain_levels" not in mix.curriculum
    # The shared and the rough task cfgs keep their own sub-terrains.
    assert MIX_PLANE_TILE not in MICRODUCK_ROUGH_TERRAINS_CFG.sub_terrains
    assert MIX_PLANE_TILE not in gr.sub_terrains
    # Physics, rewards and commands as on the rough task.
    assert (mix.sim.nconmax, mix.sim.mujoco.iterations, mix.sim.mujoco.ls_iterations) == (200, 30, 50)
    assert list(mix.rewards) == list(rough.rewards)
    for name, term in rough.rewards.items():
        assert mix.rewards[name].weight == term.weight, name
    assert mix.rewards["body_pose_tracking"].params["height_ref"] == "feet"
    assert list(mix.commands) == list(rough.commands)
    assert mix.scene.spec_fn.previous is _soften_terrain_contacts


def test_mix_runner_has_its_own_log_folder():
    assert load_rl_cfg(MIX).experiment_name == "dance_v3_rough_mix"
    assert load_rl_cfg(ROUGH).experiment_name == "dance_v3_rough"


def test_mixed_needs_rough():
    with pytest.raises(ValueError):
        make_microduck_dance_v3_env_cfg(command_file="x.npz", mixed=True)


def test_plane_tile_has_no_geometry_and_spawns_on_the_plane():
    out = PlaneTileCfg(size=(8.0, 8.0)).function(0.5, mujoco.MjSpec(), np.random.default_rng(0))
    assert out.geometries == []
    np.testing.assert_allclose(out.origin, [4.0, 4.0, -MIX_PLANE_DROP])


def test_plane_matches_the_flat_task_plane_and_is_not_softened():
    _, _, m = _small_scene(4)
    planes = [i for i in range(m.ngeom) if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE]
    assert [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) for i in planes] == [MIX_PLANE_GEOM]
    p = planes[0]
    assert m.body(m.geom_bodyid[p]).name == "terrain"  # seen by feet_ground_contact and foot_height_scan
    assert m.geom_pos[p][2] == pytest.approx(-MIX_PLANE_DROP)
    assert m.geom_group[p] == 0 and m.geom_contype[p] == 1 and m.geom_conaffinity[p] == 1
    flat_cfg = load_env_cfg(FLAT)
    flat_cfg.scene.num_envs = 1
    fm = Scene(flat_cfg.scene, device="cpu").compile()
    fp = [i for i in range(fm.ngeom) if fm.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE]
    assert len(fp) == 1
    for attr in ("geom_friction", "geom_solref", "geom_solimp", "geom_priority", "geom_condim", "geom_margin"):
        np.testing.assert_allclose(getattr(m, attr)[p], getattr(fm, attr)[fp[0]], err_msg=attr)
    boxes = [i for i in range(m.ngeom) if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX
             and m.body(m.geom_bodyid[i]).name == "terrain"]
    assert boxes and all(m.geom_solref[i][0] == pytest.approx(0.04) for i in boxes)


def test_lowest_walkable_box_top_is_above_the_plane():
    # Grid cells reach -grid_height, so the plane must lie below that with room for foot penetration.
    g = load_env_cfg(MIX).scene.terrain.terrain_generator
    assert MIX_PLANE_DROP >= g.sub_terrains["random_grid"].grid_height_range[1] + 0.015


def test_one_batch_has_plane_contact_and_box_contact():
    """64 envs on a 4 x 4 mixed terrain: some envs on plane tiles, some on box tiles, and a robot dropped
    at those origins touches only the plane or only the boxes (MuJoCo C on the CPU, 0.5 s)."""
    cfg, scene, m = _small_scene(64)
    origins = scene.terrain.env_origins.numpy().astype(float)
    on_plane = np.abs(origins[:, 2] + MIX_PLANE_DROP) < 1e-6
    assert 0 < on_plane.sum() < len(origins)
    feet = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"robot/{n}")
            for n in ("left_foot_collision", "right_foot_collision")]
    assert min(feet) >= 0
    terrain = m.body("terrain").id

    def foot_contact_types(origin):
        d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, 0)
        d.qpos[:3] = 0.0
        mujoco.mj_kinematics(m, d)
        low = min(float(d.geom_xpos[f, 2] - m.geom_rbound[f]) for f in feet)
        d.qpos[:3] = origin + np.array([0.0, 0.0, 0.005 - low])
        types = set()
        for _ in range(100):
            mujoco.mj_step(m, d)
            for c in d.contact[: d.ncon]:
                for a, b in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                    if a in feet and m.geom_bodyid[b] == terrain:
                        types.add(int(m.geom_type[b]))
        return types

    plane_env = int(np.flatnonzero(on_plane)[0])
    flat_box = [i for i in np.flatnonzero(~on_plane) if abs(origins[i, 2]) < 0.02]
    assert flat_box, "no low box tile in this terrain"
    assert foot_contact_types(origins[plane_env]) == {int(mujoco.mjtGeom.mjGEOM_PLANE)}
    assert foot_contact_types(origins[flat_box[0]]) == {int(mujoco.mjtGeom.mjGEOM_BOX)}
