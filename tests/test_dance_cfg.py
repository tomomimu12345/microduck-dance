import pytest

from microduck_dance.ranges import BODY_RANGES, HEAD_RANGES
from microduck_dance.tasks.dance_commands import (
    DancePoseCommandCfg,
    DanceVelocityCommandCfg,
    dance_body_tracking,
)
from microduck_dance.tasks.microduck_dance_env_cfg import (
    MicroduckDanceRlCfg,
    make_microduck_dance_env_cfg,
)
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
)


def test_dance_cfg_uses_dance_commands():
    cfg = make_microduck_dance_env_cfg()
    assert isinstance(cfg.commands["head_pose"], DancePoseCommandCfg)
    assert isinstance(cfg.commands["body_pose"], DancePoseCommandCfg)
    assert isinstance(cfg.commands["twist"], DanceVelocityCommandCfg)
    assert cfg.commands["body_pose"].ranges == BODY_RANGES
    assert cfg.commands["head_pose"].ranges == HEAD_RANGES
    assert cfg.commands["head_pose"].columns == (3, 7)
    assert cfg.commands["body_pose"].columns == (7, 13)
    assert cfg.commands["twist"].freq_range_hz == (0.1, 0.6)
    assert cfg.commands["twist"].hold_prob == 0.25


def test_twist_values_inherited_from_velocity():
    d = make_microduck_dance_env_cfg().commands["twist"]
    v = make_microduck_velocity_env_cfg().commands["twist"]
    assert d.ranges.lin_vel_x == v.ranges.lin_vel_x
    assert d.ranges.ang_vel_z == v.ranges.ang_vel_z
    assert d.rel_turn_in_place_envs == v.rel_turn_in_place_envs


def test_dance_cfg_removes_range_curricula():
    cfg = make_microduck_dance_env_cfg()
    for k in ("head_pose_range", "body_pose_range", "standing_envs"):
        assert k not in cfg.curriculum


def test_dance_cfg_enables_body_tracking():
    r = make_microduck_dance_env_cfg().rewards["body_pose_tracking"]
    assert r.func is dance_body_tracking and r.weight == 2.0


def test_observation_layout_matches_velocity():
    d = make_microduck_dance_env_cfg()
    v = make_microduck_velocity_env_cfg()
    for group in v.observations:
        assert list(d.observations[group].terms) == list(v.observations[group].terms)


def test_file_mode_sets_all_terms(tmp_path):
    cfg = make_microduck_dance_env_cfg(
        play=True, source="file", command_file=str(tmp_path / "x.npz")
    )
    for k in ("twist", "head_pose", "body_pose"):
        c = cfg.commands[k]
        assert c.source == "file" and c.hold_prob == 0.0
        assert c.resampling_time_range == (1e6, 1e6)
        assert c.command_file == str(tmp_path / "x.npz")


def test_file_mode_requires_command_file():
    with pytest.raises(ValueError):
        make_microduck_dance_env_cfg(source="file")


def test_library_mode_requires_command_file():
    with pytest.raises(ValueError):
        make_microduck_dance_env_cfg(source="library")


def test_rl_cfg():
    assert MicroduckDanceRlCfg.experiment_name == "dance"
    assert MicroduckDanceRlCfg.max_iterations == 2000


def test_library_mode_sets_sources(tmp_path):
    cfg = make_microduck_dance_env_cfg(source="library", command_file=str(tmp_path / "lib.npz"))
    for k in ("twist", "head_pose", "body_pose"):
        assert cfg.commands[k].source == "library"
    assert cfg.commands["head_pose"].resampling_time_range == (4.0, 8.0)


def test_dance_lib_task_registered_without_library_file():
    from mjlab.tasks.registry import list_tasks

    import microduck_dance.tasks  # noqa: F401

    assert "Mjlab-DanceLib-Flat-MicroDuck" in list_tasks()


def test_dance_lib_task_path_is_repo_root_relative():
    from pathlib import Path

    from mjlab.tasks.registry import load_env_cfg

    import microduck_dance.tasks  # noqa: F401

    cfg = load_env_cfg("Mjlab-DanceLib-Flat-MicroDuck")
    path = Path(cfg.commands["twist"].command_file)
    assert path.is_absolute()
    assert path.parts[-3:] == ("dance", "data", "library.npz")


def _load_build_library():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "dance/scripts/retarget/build_library.py"
    spec = importlib.util.spec_from_file_location("build_library", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_library


def test_build_library_concatenates(tmp_path):
    import numpy as np

    from microduck_dance.command_file import load_command_file, save_command_file
    build_library = _load_build_library()

    a = np.full((5, 13), 0.1, dtype=np.float32)
    b = np.full((7, 13), -0.2, dtype=np.float32)
    save_command_file(tmp_path / "a.npz", a, 0.02, {})
    save_command_file(tmp_path / "b.npz", b, 0.02, {})
    n = build_library([tmp_path / "a.npz", tmp_path / "b.npz"], tmp_path / "lib.npz")
    assert n == 12
    cmds, dt, meta = load_command_file(tmp_path / "lib.npz")
    assert cmds.shape == (12, 13) and dt == 0.02
    assert np.allclose(cmds[:5], 0.1) and np.allclose(cmds[5:], -0.2)
    assert meta["clip_lengths"] == [5, 7]


def test_build_library_rejects_mixed_dt(tmp_path):
    import numpy as np

    from microduck_dance.command_file import save_command_file
    build_library = _load_build_library()

    z = np.zeros((4, 13), dtype=np.float32)
    save_command_file(tmp_path / "a.npz", z, 0.02, {})
    save_command_file(tmp_path / "b.npz", z, 0.04, {})
    with pytest.raises(ValueError):
        build_library([tmp_path / "a.npz", tmp_path / "b.npz"], tmp_path / "lib.npz")


def test_dance_head_tracking_std_is_tighter_than_velocity(tmp_path):
    from mjlab.tasks.registry import load_env_cfg

    import microduck_dance.tasks  # noqa: F401

    for cfg in (
        make_microduck_dance_env_cfg(),
        make_microduck_dance_env_cfg(source="library", command_file=str(tmp_path / "lib.npz")),
        make_microduck_dance_env_cfg(play=True, source="file", command_file=str(tmp_path / "x.npz")),
        load_env_cfg("Mjlab-Dance-Flat-MicroDuck"),
        load_env_cfg("Mjlab-DanceLib-Flat-MicroDuck"),
    ):
        assert cfg.rewards["head_pose_tracking"].params["std"] == 0.25
    assert make_microduck_velocity_env_cfg().rewards["head_pose_tracking"].params["std"] == 0.5
    assert load_env_cfg("Mjlab-Velocity-Flat-MicroDuck").rewards["head_pose_tracking"].params["std"] == 0.5
    # Building a dance cfg must not leak the tighter std into a later Velocity cfg.
    make_microduck_dance_env_cfg()
    assert make_microduck_velocity_env_cfg().rewards["head_pose_tracking"].params["std"] == 0.5


def test_body_tracking_uses_dance_stand_height():
    import inspect

    from microduck_dance.ranges import DANCE_STAND_HEIGHT

    assert DANCE_STAND_HEIGHT == 0.116
    r = make_microduck_dance_env_cfg().rewards["body_pose_tracking"]
    assert r.params["nominal_height"] == DANCE_STAND_HEIGHT
    default = inspect.signature(dance_body_tracking).parameters["nominal_height"].default
    assert default == DANCE_STAND_HEIGHT


def test_play_dance_measures_against_dance_stand_height():
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "dance/scripts/evaluate/play_dance.py").read_text()
    assert "from microduck_dance.ranges import DANCE_STAND_HEIGHT" in src
    # Flat: above the env origin. Rough: above the ground under the feet (task 28).
    assert "ground = self.env.scene.terrain.env_origins[:, 2]" in src
    assert "ground = ground_z_under_feet(self.env)" in src
    assert "z = d.root_link_pos_w[:, 2] - ground - DANCE_STAND_HEIGHT" in src
    assert "0.095" not in src


# SHA-256 of library.npz["commands"]. Task 18 kept the array bit for bit through the
# rename to synthetic_NN (9690021b...1de0a8, the library dance_v4 trained on). Task 19
# regenerated it with the new retarget defaults (gate off, half groove, posture lead,
# heading servo, vel_gain 0.3), so the value changed to the one below.
LIBRARY_COMMANDS_SHA256 = "57e74da0e8d3b487f636929699b98f0a3931128dae106d5081ef2962c0ad0197"


def _library_path():
    from pathlib import Path

    return Path(__file__).resolve().parents[1] / "dance" / "data" / "library.npz"


def test_library_names_only_synthetic_sources():
    import re

    from microduck_dance.command_file import load_command_file

    path = _library_path()
    if not path.is_file():
        pytest.skip("dance/data/library.npz missing")
    _, dt, meta = load_command_file(path)
    assert dt == 0.02
    assert meta["sources"] == [f"synthetic_{i:02d}.npz" for i in range(33)]
    assert all(re.fullmatch(r"synthetic_\d\d\.npz", s) for s in meta["sources"])


def test_library_commands_unchanged_by_the_rename():
    import hashlib

    import numpy as np

    from microduck_dance.command_file import load_command_file

    path = _library_path()
    if not path.is_file():
        pytest.skip("dance/data/library.npz missing")
    cmds, _, _ = load_command_file(path)
    assert cmds.dtype == np.float32 and cmds.shape == (99000, 13)
    assert hashlib.sha256(np.ascontiguousarray(cmds).tobytes()).hexdigest() == LIBRARY_COMMANDS_SHA256


def test_library_equals_the_training_command_files():
    import numpy as np

    from microduck_dance.command_file import load_command_file

    path = _library_path()
    files = sorted((path.parent / "commands" / "training").glob("synthetic_*.npz"))
    if not path.is_file() or len(files) != 33:
        pytest.skip("library or the 33 training command files missing (commands are not in git)")
    cmds, _, _ = load_command_file(path)
    parts = [load_command_file(f)[0] for f in files]
    np.testing.assert_array_equal(np.concatenate(parts, axis=0), cmds)
