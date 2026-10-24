"""Mjlab-DanceV3-Rough-MicroDuck: the dance_v3 task on the Velocity rough terrain (CPU, cfg only)."""

from __future__ import annotations

import mjlab.tasks  # noqa: F401
import microduck_dance.tasks  # noqa: F401
import pytest
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    MICRODUCK_ROUGH_TERRAINS_CFG,
    _soften_terrain_contacts,
    make_microduck_velocity_env_cfg,
)

from microduck_dance.tasks.microduck_dance_env_cfg import rough_terrains_cfg

ROUGH = "Mjlab-DanceV3-Rough-MicroDuck"
FLAT = "Mjlab-DanceV3-Flat-MicroDuck"


@pytest.mark.parametrize("play", [False, True])
def test_rough_task_uses_the_velocity_rough_settings(play):
    cfg = load_env_cfg(ROUGH, play=play)
    vel = make_microduck_velocity_env_cfg(play=play, rough=True)
    t = cfg.scene.terrain
    assert t.terrain_type == "generator"
    g = t.terrain_generator
    assert list(g.sub_terrains) == list(MICRODUCK_ROUGH_TERRAINS_CFG.sub_terrains)
    assert g is not MICRODUCK_ROUGH_TERRAINS_CFG
    # Same sub-terrains apart from the platform width (task 28, see test_dance_rough_terrain_spread).
    for name, sub in MICRODUCK_ROUGH_TERRAINS_CFG.sub_terrains.items():
        a, b = dict(vars(g.sub_terrains[name])), dict(vars(sub))
        a.pop("platform_width", None), b.pop("platform_width", None)
        assert a == b, name
    # Grid and curriculum of the private copy (rough_terrains_cfg), not of the shared
    # upstream object, which the upstream play cfgs shrink at import.
    ref = rough_terrains_cfg(play)
    assert (g.size, g.num_rows, g.num_cols, g.curriculum) == (ref.size, ref.num_rows, ref.num_cols, ref.curriculum)
    assert (g.num_rows, g.num_cols) == ((5, 5) if play else (10, 20))
    # Steps up to 1.5 cm, grid up to 1 cm.
    assert g.sub_terrains["pyramid_stairs"].step_height_range == (0.0, 0.015)
    assert g.sub_terrains["random_grid"].grid_height_range == (0.0, 0.010)
    assert t.max_init_terrain_level is None
    assert cfg.scene.spec_fn is _soften_terrain_contacts
    assert cfg.sim.nconmax == vel.sim.nconmax == 200
    assert cfg.sim.mujoco.iterations == vel.sim.mujoco.iterations == 30
    assert cfg.sim.mujoco.ls_iterations == vel.sim.mujoco.ls_iterations == 50
    assert "terrain_levels" not in cfg.curriculum and "terrain_levels" in vel.curriculum
    assert cfg.events["reset_base"].params["pose_range"] == vel.events["reset_base"].params["pose_range"]


def test_rough_obs_is_61d_and_blind():
    rough, flat = load_env_cfg(ROUGH), load_env_cfg(FLAT)
    for group in flat.observations:
        assert list(rough.observations[group].terms) == list(flat.observations[group].terms)
    names = [n for g in rough.observations.values() for n in g.terms]
    assert not any("height" in n and "scan" in n for n in names)
    assert not any(type(s).__name__ == "RayCastSensorCfg" and "scan" in s.name for s in rough.scene.sensors
                   if s.name not in [x.name for x in flat.scene.sensors])


def test_rough_obs_terms_match_velocity():
    # Velocity flat is the 61D reference layout (48 proprioception + 13D command block).
    # The env-built dimension is checked in the smoke test and in play_dance.py.
    rough, vel = load_env_cfg(ROUGH), make_microduck_velocity_env_cfg()
    for group in vel.observations:
        assert list(rough.observations[group].terms) == list(vel.observations[group].terms)


def test_rough_rewards_and_commands_match_flat():
    rough, flat = load_env_cfg(ROUGH), load_env_cfg(FLAT)
    assert list(rough.rewards) == list(flat.rewards)
    for name, term in flat.rewards.items():
        assert rough.rewards[name].func is term.func
        assert rough.rewards[name].weight == term.weight
    assert list(rough.commands) == list(flat.commands)
    for name in flat.commands:
        assert type(rough.commands[name]) is type(flat.commands[name])
        assert vars(rough.commands[name]) == vars(flat.commands[name])
    # The flat task keeps its own curricula (no terrain levels) and a plane.
    assert "terrain_levels" not in flat.curriculum
    assert set(rough.curriculum) == set(flat.curriculum)
    assert flat.scene.terrain.terrain_type == "plane"
    assert flat.sim.nconmax != 200


def test_rough_runner_has_its_own_experiment():
    rough, flat = load_rl_cfg(ROUGH), load_rl_cfg(FLAT)
    assert rough.experiment_name == "dance_v3_rough"
    assert flat.experiment_name == "dance_v3"
    a, b = vars(rough).copy(), vars(flat).copy()
    for k in ("experiment_name", "run_name"):
        a.pop(k), b.pop(k)
    assert a == b
