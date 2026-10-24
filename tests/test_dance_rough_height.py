"""Body height on rough ground is measured from the ground under the feet (CPU, no env is built)."""

from __future__ import annotations

from types import SimpleNamespace

import mjlab.tasks  # noqa: F401
import microduck_dance.tasks  # noqa: F401
import pytest
import torch
from mjlab.tasks.registry import load_env_cfg

from microduck_dance.ranges import DANCE_STAND_HEIGHT
from microduck_dance.tasks.dance_v3 import (
    body_height_reference,
    dance_body_tracking_clamped,
    ground_z_under_feet_from,
)

ROUGH = "Mjlab-DanceV3-Rough-MicroDuck"
FLAT = "Mjlab-DanceV3-Flat-MicroDuck"


class _Scene(dict):
    pass


def _env(root_z, origin_z, feet_z, clearance, body_cmd=None):
    """Fake env with N envs. feet_z and clearance are (N, 2)."""
    n = len(root_z)
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * n)
    pos = torch.zeros(n, 3)
    pos[:, 2] = torch.as_tensor(root_z)
    robot = SimpleNamespace(data=SimpleNamespace(root_link_pos_w=pos, root_link_quat_w=quat))
    frame = torch.zeros(n, 2, 3)
    frame[:, :, 2] = torch.as_tensor(feet_z)
    sensor = SimpleNamespace(data=SimpleNamespace(frame_pos_w=frame, heights=torch.as_tensor(clearance)))
    origins = torch.zeros(n, 3)
    origins[:, 2] = torch.as_tensor(origin_z)
    scene = _Scene(robot=robot, foot_height_scan=sensor)
    scene.terrain = SimpleNamespace(env_origins=origins)
    cmd = torch.zeros(n, 6) if body_cmd is None else torch.as_tensor(body_cmd)
    return SimpleNamespace(scene=scene, command_manager=SimpleNamespace(get_command=lambda name: cmd))


def test_ground_under_feet_is_the_mean_hit_height():
    frame_z = torch.tensor([[0.030, 0.045], [0.010, 0.010]])
    heights = torch.tensor([[0.010, 0.010], [0.010, 0.000]])
    assert torch.allclose(ground_z_under_feet_from(frame_z, heights), torch.tensor([0.0275, 0.005]))


def test_height_reference_modes():
    env = _env([0.2, 0.2], [0.0, 0.03], [[0.05, 0.05], [0.05, 0.05]], [[0.02, 0.02], [0.02, 0.04]])
    assert torch.allclose(body_height_reference(env, "env_origin"), torch.tensor([0.0, 0.03]))
    assert torch.allclose(body_height_reference(env, "feet"), torch.tensor([0.03, 0.02]))
    with pytest.raises(ValueError):
        body_height_reference(env, "trunk")


def test_feet_reference_follows_the_step():
    # Spawned at ground z 0 (env origin), now standing one 1.5 cm step higher at nominal height.
    step = 0.015
    root = step + DANCE_STAND_HEIGHT
    env = _env([root], [0.0], [[step + 0.01, step + 0.01]], [[0.01, 0.01]])
    on_step = dance_body_tracking_clamped(env, height_ref="feet")
    from_origin = dance_body_tracking_clamped(env, height_ref="env_origin")
    assert on_step.item() == pytest.approx(1.0, abs=1e-6)
    # From the origin the 1.5 cm step reads as a height error: r_z = exp(-(0.015 / 0.02)^2).
    expected = (torch.exp(torch.tensor(-(step / 0.02) ** 2)).item() + 2.0) / 3.0
    assert from_origin.item() == pytest.approx(expected, abs=1e-6)


def test_feet_and_origin_agree_on_flat_ground():
    env = _env([0.10, 0.09], [0.0, 0.0], [[0.01, 0.02], [0.01, 0.01]], [[0.01, 0.02], [0.01, 0.01]],
               body_cmd=[[0, 0, -0.01, 0.1, 0.0, 0], [0, 0, 0.005, 0.0, -0.1, 0]])
    a = dance_body_tracking_clamped(env, height_ref="feet")
    b = dance_body_tracking_clamped(env, height_ref="env_origin")
    assert torch.allclose(a, b, atol=1e-7)


@pytest.mark.parametrize("play", [False, True])
def test_rough_task_measures_height_from_the_feet(play):
    rough, flat = load_env_cfg(ROUGH, play=play), load_env_cfg(FLAT, play=play)
    assert rough.rewards["body_pose_tracking"].params["height_ref"] == "feet"
    assert flat.rewards["body_pose_tracking"].params["height_ref"] == "env_origin"
    assert any(s.name == "foot_height_scan" for s in rough.scene.sensors)
    # Nothing else in the term differs.
    a = dict(rough.rewards["body_pose_tracking"].params)
    b = dict(flat.rewards["body_pose_tracking"].params)
    a.pop("height_ref"), b.pop("height_ref")
    assert a == b


@pytest.mark.parametrize("play", [False, True])
def test_dance_rough_terrain_spread(play):
    from microduck_dance.tasks.microduck_dance_v3_env_cfg import DANCE_ROUGH_PLATFORM_WIDTH
    from mjlab_microduck.tasks.microduck_velocity_env_cfg import MICRODUCK_ROUGH_TERRAINS_CFG as shared

    cfg = load_env_cfg(ROUGH, play=play)
    g = cfg.scene.terrain.terrain_generator
    assert g is not shared
    for name in ("pyramid_stairs", "random_grid", "pyramid_slope"):
        assert g.sub_terrains[name].platform_width == DANCE_ROUGH_PLATFORM_WIDTH == 0.5
    # The shared cfg keeps its platforms. Heights stay within the caps.
    assert shared.sub_terrains["pyramid_stairs"].platform_width == 2.0
    assert shared.sub_terrains["random_grid"].platform_width == 1.5
    assert max(g.sub_terrains["pyramid_stairs"].step_height_range) <= 0.015
    assert max(g.sub_terrains["random_grid"].grid_height_range) <= 0.010
    assert max(g.sub_terrains["pyramid_slope"].slope_range) <= 0.10
    # Uniform over tiles: no distance driven levels, envs start on any row of random tiles.
    assert "terrain_levels" not in cfg.curriculum
    assert cfg.scene.terrain.max_init_terrain_level is None
    assert g.curriculum is False
    # The 0.5 m spawn noise reaches past the platform edge.
    pr = cfg.events["reset_base"].params["pose_range"]
    assert pr["x"][1] > DANCE_ROUGH_PLATFORM_WIDTH / 2 and pr["y"][1] > DANCE_ROUGH_PLATFORM_WIDTH / 2
