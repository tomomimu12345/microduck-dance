"""Task 28 additions to play_dance.py (video terrain, A/B switches, eval dumps) and rough_metrics.py (CPU)."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


_EVAL = Path(__file__).resolve().parents[1] / "dance" / "scripts" / "evaluate"


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, _EVAL / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


play_dance = _load("play_dance_t28", "play_dance.py")
rough_metrics = _load("rough_metrics_t28", "rough_metrics.py")

SONG = Path(__file__).resolve().parents[1] / "dance" / "data" / "commands" / "songs" / "funky_magic.npz"
needs_song = pytest.mark.skipif(
    not SONG.exists(),
    reason="song command files are not in git (made from the songs by dance/scripts/retarget/retarget_all.sh)",
)


def _cfg(tmp_path, extra, play=False, render=False, num_envs=2):
    args = play_dance.build_parser().parse_args(["--checkpoint", "x.pt", "--commands", str(SONG), *extra])
    return args, play_dance.build_env_cfg(args, play=play, num_envs=num_envs, seed=3, steps=100, dt=0.02,
                                          render=render, command_file=tmp_path / "c.npz")


def test_video_terrain_cfg_values():
    g = play_dance.video_terrain_cfg()
    assert list(g.sub_terrains) == ["pyramid_stairs", "random_grid"]
    st, gr = g.sub_terrains["pyramid_stairs"], g.sub_terrains["random_grid"]
    assert st.step_height_range == (0.015, 0.015)
    # The rough dance training geometry (task 30).
    assert st.step_width == 0.15 and st.platform_width == 0.5
    assert gr.grid_width == 0.45 and gr.grid_height_range == (0.0, 0.010)
    assert gr.merge_similar_heights is False
    assert gr.platform_width == 0.5
    # 0.45 m cells must not divide the 8 m tile (the generator needs a border).
    assert (g.size[0] / gr.grid_width) % 1 > 1e-6
    # The off-platform spawn stays on the stairs: outside the platform, inside the 1 m border.
    assert st.platform_width / 2 + 0.3 < play_dance.OFF_PLATFORM_D < g.size[0] / 2 - st.border_width


@needs_song
def test_video_terrain_leaves_the_training_terrain_alone(tmp_path):
    from mjlab_microduck.tasks.microduck_velocity_env_cfg import MICRODUCK_ROUGH_TERRAINS_CFG as shared

    before = (shared.num_rows, shared.num_cols, shared.sub_terrains["pyramid_stairs"].step_width)
    _, cfg = _cfg(tmp_path, ["--terrain", "video", "--tile-type", "pyramid_stairs", "--spawn-u", "0.125"],
                  play=True, render=True, num_envs=1)
    g = cfg.scene.terrain.terrain_generator
    assert g is not shared and list(g.sub_terrains) == ["pyramid_stairs", "random_grid"]
    assert (g.num_rows, g.num_cols, g.curriculum, g.seed) == (5, 5, False, 3)
    assert (shared.num_rows, shared.num_cols, shared.sub_terrains["pyramid_stairs"].step_width) == before
    # Rough physics settings stay.
    assert cfg.sim.nconmax == 200 and cfg.sim.mujoco.iterations == 30
    assert "terrain_levels" not in cfg.curriculum


def test_video_box_colors():
    c = play_dance.video_box_color
    h = play_dance.VIDEO_STEP_HEIGHT
    tones = {play_dance.VIDEO_CREAM, play_dance.VIDEO_SLATE}
    for k in range(12):
        a, b = c("step", k * h), c("step", (k + 1) * h)
        assert {a, b} == tones
    assert c("step", 3 * h + 0.002) == c("step", 3 * h)
    assert c("platform", 0.2) == play_dance.VIDEO_PLATFORM_GREY
    assert c("cell", 0.0, (0, 0)) == play_dance.VIDEO_CREAM
    assert c("cell", 0.0, (0, 1)) == play_dance.VIDEO_SLATE
    assert c("cell", 0.0, (-1, 0)) == play_dance.VIDEO_SLATE
    top = c("cell", 0.0, (1, 1), frac=1.0)
    assert top == tuple(min(1.0, v * 1.15) for v in play_dance.VIDEO_CREAM)
    with pytest.raises(ValueError):
        c("ramp", 0.0)


@needs_song
def test_ground_box_replaces_the_plane(tmp_path):
    import mujoco
    from mjlab.scene import Scene

    _, cfg = _cfg(tmp_path, ["--ground", "box", "--eval"])
    assert cfg.scene.terrain.terrain_type == "plane"
    m = Scene(cfg.scene, device="cpu").compile()
    terrain = m.body("terrain").id
    geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == terrain]
    plane = [g for g in geoms if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE]
    box = [g for g in geoms if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX]
    assert len(plane) == 1 and len(box) == 1
    p, b = plane[0], box[0]
    assert (m.geom_contype[p], m.geom_conaffinity[p]) == (0, 0)
    assert (m.geom_contype[b], m.geom_conaffinity[b]) == (1, 1)
    for field in ("geom_friction", "geom_solref", "geom_solimp", "geom_priority", "geom_condim", "geom_group"):
        assert np.array_equal(getattr(m, field)[b], getattr(m, field)[p]), field
    data = mujoco.MjData(m)
    mujoco.mj_kinematics(m, data)
    assert data.geom_xpos[b, 2] + m.geom_size[b, 2] == pytest.approx(0.0)


@needs_song
def test_ground_box_is_flat_only(tmp_path):
    with pytest.raises(SystemExit):
        _cfg(tmp_path, ["--ground", "box", "--terrain", "rough"])


@needs_song
def test_solver_iters_and_no_dr(tmp_path):
    _, cfg = _cfg(tmp_path, ["--terrain", "rough", "--solver-iters", "10", "20", "--eval", "--no-dr"])
    assert (cfg.sim.mujoco.iterations, cfg.sim.mujoco.ls_iterations) == (10, 20)
    assert "push_robot" not in cfg.events and "foot_friction" not in cfg.events
    _, dr = _cfg(tmp_path, ["--terrain", "rough", "--eval"])
    assert (dr.sim.mujoco.iterations, dr.sim.mujoco.ls_iterations) == (30, 50)
    assert "push_robot" in dr.events


@needs_song
def test_defaults_unchanged(tmp_path):
    args, cfg = _cfg(tmp_path, ["--eval"])
    assert args.ground == "plane" and args.solver_iters is None and not args.no_dr and args.dump_eval is None
    assert args.wide_intro_s == 0.0 and args.spawn_u is None
    assert cfg.scene.spec_fn is None
    assert "push_robot" in cfg.events
    assert (cfg.sim.mujoco.iterations, cfg.sim.mujoco.ls_iterations) == (10, 20)


def test_eval_dump_path():
    assert play_dance.eval_dump_path("a/b.npz", 0, 1) == Path("a/b.npz")
    assert play_dance.eval_dump_path("a/b.npz", 2, 3) == Path("a/b_seed2.npz")


def test_realized_channels_start_frame():
    # Start facing +y (yaw pi/2) and walk 1 m forward: in the start frame that is +x.
    T = 50
    yaw = np.full(T, np.pi / 2)
    pos = np.zeros((T, 3))
    pos[:, 1] = np.linspace(0.0, 1.0, T)
    pos[:, 0] = 5.0
    r = rough_metrics.realized_channels(pos, yaw)
    assert np.allclose(r["x"], np.linspace(0.0, 1.0, T), atol=1e-12)
    assert np.allclose(r["y"], 0.0, atol=1e-12)
    assert np.allclose(r["heading"], 0.0)


def test_env_correlations_window():
    dt, T = 0.02, 3000
    t = np.arange(T) * dt
    human = {"heading": np.sin(0.3 * t), "x": 0.01 * t}
    yaw = 0.4 + np.sin(0.3 * t)
    pos = np.zeros((T, 3))
    c, s = np.cos(0.4), np.sin(0.4)
    pos[:, 0], pos[:, 1] = c * 0.02 * t, s * 0.02 * t
    term = np.zeros(T, bool)
    r = rough_metrics.env_correlations(human, pos, yaw, term, dt)
    assert r["heading"] == pytest.approx(1.0) and r["x"] == pytest.approx(1.0)
    assert r["window_s"] == pytest.approx(60.0)
    term[500] = True  # first fall at 10 s: too short to score
    r = rough_metrics.env_correlations(human, pos, yaw, term, dt)
    assert np.isnan(r["heading"]) and r["window_s"] == pytest.approx(10.0)
    assert rough_metrics.first_fall(np.zeros(4, bool)) == 4


def test_by_terrain_type():
    out = rough_metrics.by_terrain_type(["flat", "stairs", "flat"], [1, 0, 0], [0.5, 0.2, np.nan], [0.1, 0.3, 0.2], 1.0)
    assert out["flat"]["n_envs"] == 2 and out["flat"]["falls_per_min"] == 0.5
    assert out["flat"]["heading_r"] == 0.5 and out["flat"]["path_x_r"] == pytest.approx(0.15)
    assert out["stairs"]["heading_r"] == 0.2


def test_video_terrain_is_the_rough_dance_training_geometry():
    import mjlab.tasks  # noqa: F401
    import microduck_dance.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg

    train = load_env_cfg("Mjlab-DanceV3-Rough-MicroDuck").scene.terrain.terrain_generator.sub_terrains
    v = play_dance.video_terrain_cfg().sub_terrains
    ts, vs = train["pyramid_stairs"], v["pyramid_stairs"]
    assert (vs.step_width, vs.platform_width, vs.border_width) == (ts.step_width, ts.platform_width, ts.border_width)
    assert vs.step_height_range == (ts.step_height_range[1],) * 2
    tg, vg = train["random_grid"], v["random_grid"]
    assert (vg.grid_width, vg.platform_width) == (tg.grid_width, tg.platform_width)
    assert vg.grid_height_range == (0.0, tg.grid_height_range[1])
