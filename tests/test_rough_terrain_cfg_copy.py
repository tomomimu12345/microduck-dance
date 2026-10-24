"""The dance cfgs take a private copy of the rough terrain cfg with the grid as upstream defines it (CPU).

At upstream f010ef1 the rough factories write the play grid (5 x 5, no curriculum)
into the shared MICRODUCK_ROUGH_TERRAINS_CFG, and registration builds the play cfgs
at import. The dance cfgs do not patch upstream. rough_terrains_cfg rebuilds the
module object from its source and copies it per cfg.
"""

from __future__ import annotations

import mjlab.tasks  # noqa: F401
import microduck_dance.tasks  # noqa: F401
import pytest
from mjlab.tasks.registry import load_env_cfg

from microduck_dance.tasks.microduck_dance_env_cfg import _pristine_rough_terrains_cfg, rough_terrains_cfg
from mjlab_microduck.tasks import microduck_velocity_env_cfg as vel_mod

ROUGH_TASKS = ("Mjlab-DanceV3-Rough-MicroDuck", "Mjlab-DanceV3-RoughMix-MicroDuck")


def test_pristine_cfg_is_the_module_definition():
    p = _pristine_rough_terrains_cfg()
    shared = vel_mod.MICRODUCK_ROUGH_TERRAINS_CFG
    assert p is not shared
    assert (p.num_rows, p.num_cols, p.size, p.border_width) == (10, 20, (8.0, 8.0), 20.0)
    assert list(p.sub_terrains) == list(shared.sub_terrains)
    for name, sub in shared.sub_terrains.items():
        assert vars(p.sub_terrains[name]) == vars(sub), name
    # Fields the play branches write are the only ones that may differ from the shared object.
    a, b = dict(vars(p)), dict(vars(shared))
    for k in ("curriculum", "num_rows", "num_cols", "sub_terrains"):
        a.pop(k), b.pop(k)
    assert a == b


def test_copies_are_private():
    p = _pristine_rough_terrains_cfg()
    train, play = rough_terrains_cfg(), rough_terrains_cfg(play=True)
    assert train is not p and play is not p and train is not play
    assert (train.num_rows, train.num_cols, train.curriculum) == (10, 20, p.curriculum)
    assert (play.num_rows, play.num_cols, play.curriculum) == (5, 5, False)
    assert (p.num_rows, p.num_cols) == (10, 20)


@pytest.mark.parametrize("task", ROUGH_TASKS)
def test_rough_training_cfgs_get_the_full_grid(task):
    g = load_env_cfg(task).scene.terrain.terrain_generator
    assert g is not vel_mod.MICRODUCK_ROUGH_TERRAINS_CFG
    assert (g.num_rows, g.num_cols) == (10, 20)
    gp = load_env_cfg(task, play=True).scene.terrain.terrain_generator
    assert (gp.num_rows, gp.num_cols, gp.curriculum) == (5, 5, False)
