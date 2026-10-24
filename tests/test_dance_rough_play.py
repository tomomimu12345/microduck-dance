"""Rough terrain helpers of dance/scripts/evaluate/play_dance.py (CPU only, no env is stepped)."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_PATH = Path(__file__).resolve().parents[1] / "dance" / "scripts" / "evaluate" / "play_dance.py"
_spec = importlib.util.spec_from_file_location("play_dance_rough", _PATH)
play_dance = importlib.util.module_from_spec(_spec)
sys.modules["play_dance_rough"] = play_dance
_spec.loader.exec_module(play_dance)

SONG = Path(__file__).resolve().parents[1] / "dance" / "data" / "commands" / "songs" / "swag.npz"
needs_song = pytest.mark.skipif(
    not SONG.exists(),
    reason="song command files are not in git (made from the songs by dance/scripts/retarget/retarget_all.sh)",
)


@pytest.mark.parametrize("u", np.linspace(0.0, 0.999, 37))
def test_ring_point_is_at_chebyshev_distance(u):
    p = play_dance.ring_point((3.0, -2.0), 1.6, u)
    rel = np.abs(p - np.array([3.0, -2.0]))
    assert rel.max() == pytest.approx(1.6)
    assert rel.min() <= 1.6 + 1e-12


def test_off_platform_d_clears_every_platform():
    from mjlab_microduck.tasks.microduck_velocity_env_cfg import MICRODUCK_ROUGH_TERRAINS_CFG

    halves = [play_dance.platform_half_width(c) for c in MICRODUCK_ROUGH_TERRAINS_CFG.sub_terrains.values()]
    assert max(halves) == 1.0
    assert play_dance.OFF_PLATFORM_D > max(halves) + 0.3  # a foot about 0.05 m off the trunk stays outside
    assert play_dance.OFF_PLATFORM_D < MICRODUCK_ROUGH_TERRAINS_CFG.size[0] / 2 - 1.5


def test_platform_half_width():
    assert play_dance.platform_half_width(SimpleNamespace(platform_width=1.5)) == 0.75
    assert play_dance.platform_half_width(SimpleNamespace()) == 0.0


def test_off_platform_fraction():
    feet = np.zeros((4, 2, 2))
    feet[1, 0] = (1.2, 0.0)   # one ankle out
    feet[2, :] = (0.0, -1.1)  # both out
    assert play_dance.off_platform_fraction(feet, (0.0, 0.0), 1.0) == 0.5
    assert np.isnan(play_dance.off_platform_fraction(feet, (0.0, 0.0), 0.0))


@needs_song
@pytest.mark.parametrize("play", [False, True])
def test_rough_env_cfg(tmp_path, play):
    args = play_dance.build_parser().parse_args(["--checkpoint", "x.pt", "--commands", str(SONG),
                                                 "--terrain", "rough"])
    cfg = play_dance.build_env_cfg(args, play=play, num_envs=2, seed=7, steps=100, dt=0.02, render=False,
                                   command_file=tmp_path / "c.npz")
    g = cfg.scene.terrain.terrain_generator
    assert cfg.scene.terrain.terrain_type == "generator"
    assert (g.curriculum, g.num_rows, g.num_cols, g.seed) == (False, 5, 5, 7)
    assert "terrain_levels" not in cfg.curriculum
    pr = cfg.events["reset_base"].params["pose_range"]
    assert pr["x"] == pr["y"] == (0.0, 0.0)
    assert cfg.sim.nconmax == 200


@needs_song
def test_flat_env_cfg_unchanged(tmp_path):
    args = play_dance.build_parser().parse_args(["--checkpoint", "x.pt", "--commands", str(SONG)])
    assert args.terrain == "flat"
    cfg = play_dance.build_env_cfg(args, play=False, num_envs=2, seed=0, steps=100, dt=0.02, render=False,
                                   command_file=tmp_path / "c.npz")
    assert cfg.scene.terrain.terrain_type == "plane"
    assert cfg.events["reset_base"].params["pose_range"]["x"] == (-0.5, 0.5)


def test_terrain_band_color_separates_steps():
    # Two surfaces 2.5 to 15 mm apart never share a colour, heights within one band do.
    c = play_dance.terrain_band_color
    assert c(0.0) == c(0.001)
    for z in np.linspace(-0.05, 0.05, 41):
        for h in np.linspace(0.0025, 0.015, 26):
            assert c(z) != c(z - h), (z, h)
