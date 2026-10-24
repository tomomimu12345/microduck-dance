"""Figure of the rough dance terrain types: renders with the robot for scale, side profiles, proportions.

Two steps, so the renders can run where EGL works and the figure where the Japanese font is:
    MUJOCO_GL=egl uv run python dance/scripts/evaluate/terrain_figure.py render --out-dir out/terrain_fig
    uv run python dance/scripts/evaluate/terrain_figure.py compose --out-dir out/terrain_fig \
        --figure dance/docs/figures/terrain_types.png

render builds one 8 m tile per (type, difficulty) from the Mjlab-DanceV3-Rough-MicroDuck training cfg
(platforms 0.5 m, task 28), puts the robot in its init pose on the ground next to the platform and renders it. The boxes are recoloured for rendering
only: steps alternate two tones, grid cells form a checkerboard, platforms are grey.
The video terrain of play_dance.py has the same geometry as the training stairs and grid (task 30), so
the figure has no separate video column.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
TYPES = ("flat", "pyramid_stairs", "random_grid", "pyramid_slope")
DIFFICULTIES = (1.0,)
# (name, distance m, elevation deg, azimuth deg). The close view is low and looks across the step
# edges toward the tile centre (azimuth 180, the robot stands on the +x side), so the risers face it.
VIEWS = (("wide", 1.5, -35.0, 150.0), ("close", 0.75, -14.0, 180.0))
CREAM, SLATE, GREY, SLOPE = (0.92, 0.88, 0.78), (0.25, 0.32, 0.42), (0.6, 0.6, 0.6), (0.80, 0.70, 0.55)
FLAT_BLUE = (0.55, 0.62, 0.72)
SIZE = (720, 540)  # width, height of one render


def training_sub_terrains():
    import mjlab.tasks  # noqa: F401
    import microduck_dance.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg

    cfg = load_env_cfg("Mjlab-DanceV3-Rough-MicroDuck")
    return cfg.scene.terrain.terrain_generator


def _scene_model(name: str, sub_cfg, difficulty: float):
    """Compiled MjModel of one tile of sub_cfg at difficulty with the robot, and the tile origin."""
    from copy import deepcopy

    from mjlab.scene import Scene
    from mjlab.tasks.registry import load_env_cfg
    from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

    cfg = load_env_cfg("Mjlab-DanceV3-Rough-MicroDuck", play=True)
    sub = deepcopy(sub_cfg)
    sub.proportion = 1.0
    cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
        size=(8.0, 8.0), border_width=2.0, num_rows=1, num_cols=1, curriculum=True,
        difficulty_range=(difficulty, difficulty), sub_terrains={name: sub}, add_lights=False)
    cfg.scene.num_envs = 1
    sc = Scene(cfg.scene, device="cpu")
    model = sc.compile()
    origin = np.asarray(sc.terrain.terrain_origins[0, 0].cpu().numpy(), float)
    return model, origin


def _colour(model, data, name: str, origin, cell: float | None):
    import mujoco

    body = model.body("terrain").id
    gs = [g for g in range(model.ngeom) if model.geom_bodyid[g] == body]
    boxes = [g for g in gs if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX]
    tops = {g: float(data.geom_xpos[g, 2] + model.geom_size[g, 2]) for g in boxes}
    levels = sorted({round(t, 5) for t in tops.values()})
    hi_cells = [tops[g] for g in boxes if cell and abs(model.geom_size[g, 0] - cell / 2) < 1e-4]
    lo, hi = (min(hi_cells), max(hi_cells)) if hi_cells else (0.0, 0.0)
    for g in gs:
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX:
            model.geom_rgba[g] = (*SLOPE, 1.0)
            model.geom_matid[g] = -1
            continue
        xy = data.geom_xpos[g, :2]
        if np.max(np.abs(xy - origin[:2])) < 1e-3 and name != "flat":
            rgb = GREY
        elif cell and abs(model.geom_size[g, 0] - cell / 2) < 1e-4:
            i, j = (int(v) for v in np.floor((xy - origin[:2]) / cell))
            base = CREAM if (i + j) % 2 == 0 else SLATE
            s = 1.0 + 0.15 * ((tops[g] - lo) / (hi - lo) if hi > lo else 0.0)
            rgb = tuple(min(1.0, c * s) for c in base)
        elif name == "flat":
            rgb = FLAT_BLUE
        else:
            rgb = CREAM if levels.index(round(tops[g], 5)) % 2 == 0 else SLATE
        model.geom_rgba[g] = (*rgb, 1.0)


def _place_robot(model, data, origin, offset):
    """Root of the robot onto the ground at origin + offset (xy), init keyframe joints."""
    import mujoco

    from microduck_dance.ranges import DANCE_STAND_HEIGHT

    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_kinematics(model, data)
    x, y = origin[0] + offset[0], origin[1] + offset[1]
    geomid = np.array([-1], dtype=np.int32)
    terrain = model.body("terrain").id
    # Cast against the terrain only (geom group 0 includes the robot, so start above everything and
    # drop through robot hits).
    z0 = origin[2] + 1.0
    while True:
        d = mujoco.mj_ray(model, data, np.array([x, y, z0]), np.array([0.0, 0.0, -1.0]), None, 1, -1, geomid)
        if d < 0:
            raise RuntimeError("no ground under the robot")
        if model.geom_bodyid[geomid[0]] == terrain:
            ground = z0 - d
            break
        z0 = z0 - d - 1e-4
    jid = next(j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE)
    adr = model.jnt_qposadr[jid]
    data.qpos[adr:adr + 3] = (x, y, ground + DANCE_STAND_HEIGHT + 0.003)
    data.qpos[adr + 3:adr + 7] = (1.0, 0.0, 0.0, 0.0)
    mujoco.mj_forward(model, data)
    return ground


def render(out_dir: Path) -> None:
    import mujoco

    out_dir.mkdir(parents=True, exist_ok=True)
    gen = training_sub_terrains()
    jobs = [(t, gen.sub_terrains[t], d, "train") for t in TYPES for d in DIFFICULTIES]
    meta = {"proportions": {t: float(gen.sub_terrains[t].proportion) for t in TYPES}, "renders": []}
    for name, sub, d, kind in jobs:
        model, origin = _scene_model(name, sub, d)
        data = mujoco.MjData(model)
        mujoco.mj_kinematics(model, data)
        _colour(model, data, name, origin, getattr(sub, "grid_width", None))
        pw = getattr(sub, "platform_width", 0.0) or 0.0
        offset = (pw / 2 + 0.35, 0.0)
        ground = _place_robot(model, data, origin, offset)
        # Soft headlight only, no shadows (the shadow map smears bars over the thin boxes).
        model.light_castshadow[:] = 0
        model.vis.headlight.ambient[:] = (0.35, 0.35, 0.35)
        model.vis.headlight.diffuse[:] = (0.55, 0.55, 0.55)
        model.vis.headlight.specular[:] = (0.0, 0.0, 0.0)
        model.vis.global_.offwidth, model.vis.global_.offheight = SIZE
        import imageio

        for view, dist, elev, azim in VIEWS:
            cam = mujoco.MjvCamera()
            if view == "wide":
                cam.lookat[:] = (origin[0] + offset[0] - 0.1, origin[1] + offset[1], ground + 0.08)
            else:
                cam.lookat[:] = (origin[0] + offset[0] - 0.12, origin[1] + offset[1], ground + 0.05)
            cam.distance, cam.elevation, cam.azimuth = dist, elev, azim
            with mujoco.Renderer(model, SIZE[1], SIZE[0]) as r:
                r.update_scene(data, cam)
                img = r.render()
            f = out_dir / f"{kind}_{name}_d{int(d * 100):03d}_{view}.png"
            imageio.imwrite(f, img)
            meta["renders"].append({"file": f.name, "type": name, "difficulty": d, "kind": kind, "view": view,
                                    "params": {k: (list(v) if isinstance(v, tuple) else v)
                                               for k, v in vars(sub).items()
                                               if isinstance(v, (int, float, tuple, bool))}})
            print("wrote", f)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))


def _robot(ax, x0: float, y0: float):
    """25 cm robot silhouette (cm): two legs, trunk, head."""
    from matplotlib.patches import Rectangle

    for dx in (0.0, 5.0):
        ax.add_patch(Rectangle((x0 + dx, y0), 3.0, 9.0, color="0.35"))
    ax.add_patch(Rectangle((x0 - 1.0, y0 + 9.0), 10.0, 8.0, color="0.35"))
    ax.add_patch(Rectangle((x0 + 0.5, y0 + 17.5), 7.0, 7.5, color="0.35"))


PROFILE_W, PROFILE_H = 90.0, 30.0  # cm, the same for every sketch


def _profile(ax, name: str, p: dict, note: str | None = None):
    """Side profile to a common scale (cm) with a 25 cm robot, dimensions written above."""
    ax.set_xlim(0, PROFILE_W)
    ax.set_ylim(-4, PROFILE_H)
    ax.set_aspect("equal")
    ax.axis("off")
    lines = []
    x0 = 30.0  # where the profile starts
    if name == "flat":
        xs, ys = [0, PROFILE_W], [0, 0]
        lines = ["平らな箱 (段差なし)"]
    elif name == "pyramid_stairs":
        h, w = 100 * max(p["step_height_range"]), 100 * p["step_width"]
        xs, ys, x, y = [0, x0], [0, 0], x0, 0.0
        while x + w <= PROFILE_W + w:
            y += h
            xs += [x, x + w]
            ys += [y, y]
            x += w
        lines = [f"段差 {h:.1f} cm、踏面 {w:.0f} cm", f"中央の台 {100 * p['platform_width']:.0f} cm 角"]
        if min(p["step_height_range"]) < max(p["step_height_range"]):
            lines.append(f"(段差は難易度で 0 から {h:.1f} cm)")
    elif name == "random_grid":
        g, w = 100 * max(p["grid_height_range"]), 100 * p["grid_width"]
        hs = np.random.default_rng(3).uniform(-g, g, 8)
        xs, ys = [0, 0], [0, hs[0]]
        for i, hh in enumerate(hs):
            xs += [i * w, (i + 1) * w]
            ys += [hh, hh]
        lines = [f"マス {w:.0f} cm", f"高さ -{g:.0f} から +{g:.0f} cm", "(マスごとに一様乱数)"]
    elif name == "pyramid_slope":
        sl = max(p["slope_range"])
        xs, ys = [0, x0, PROFILE_W], [0, 0, sl * (PROFILE_W - x0)]
        lines = [f"勾配 {100 * min(p['slope_range']):.0f} から {100 * sl:.0f} %", f"(図は {100 * sl:.0f} %)", "頂上に平らな台"]
    ax.fill_between(xs, -4, ys, color="0.85", step=None)
    ax.plot(xs, ys, color="k", lw=0.8)
    _robot(ax, 8.0, float(np.interp(10.0, xs, ys)))
    if note:
        lines.append(note)
    ax.text(0, PROFILE_H + 1, "\n".join(lines), fontsize=11, va="bottom", linespacing=1.4)


def compose(out_dir: Path, figure: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    for f in font_manager.findSystemFonts():
        if "ipaexg" in f.lower():
            font_manager.fontManager.addfont(f)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=f).get_name()
    meta = json.loads((out_dir / "meta.json").read_text())
    by = {(r["kind"], r["type"], r["view"]): r for r in meta["renders"]}
    label = {"flat": "平らな箱", "pyramid_stairs": "ピラミッド階段", "random_grid": "ランダムな格子",
             "pyramid_slope": "斜面 (ハイトフィールド)"}
    cols = len(TYPES)
    fig = plt.figure(figsize=(3.2 * cols, 8.4))
    gs = fig.add_gridspec(3, cols, height_ratios=[1, 0.75, 0.75], hspace=0.12, wspace=0.04)

    def show(ax, r):
        img = plt.imread(out_dir / r["file"])
        if r["view"] == "close":
            img = img[60:]  # drop most of the empty sky above the horizon
        ax.imshow(img)
        ax.set_xticks([]), ax.set_yticks([])

    for c, t in enumerate(TYPES):
        ax = fig.add_subplot(gs[0, c])
        show(ax, by[("train", t, "wide")])
        ax.set_title(f"{label[t]}\n学習での割合 {100 * meta['proportions'][t]:.0f} %", fontsize=12)
        ax2 = fig.add_subplot(gs[1, c])
        show(ax2, by[("train", t, "close")])
        if c == 0:
            ax.set_ylabel("全体 (難易度 最大)", fontsize=12)
            ax2.set_ylabel("足元の拡大", fontsize=12)
        _profile(fig.add_subplot(gs[2, c]), t, by[("train", t, "close")]["params"])
    fig.text(0.5, 0.06, "下段の断面図はすべて同じ縮尺 (横 90 cm)。灰色の人形は身長 25 cm のロボット。"
             "動画の凹凸地形は、この学習用の階段と格子を同じ寸法のまま2色に塗ったもの。", ha="center", fontsize=11)
    fig.suptitle("ダンス学習用の凹凸地形 (Mjlab-DanceV3-Rough-MicroDuck、1 区画 8 m 角)", fontsize=14, y=0.99)
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure, dpi=110, bbox_inches="tight")
    print("wrote", figure)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=("render", "compose"))
    ap.add_argument("--out-dir", type=Path, default=REPO / "out/terrain_fig")
    ap.add_argument("--figure", type=Path, default=REPO / "dance/docs/figures/terrain_types.png")
    a = ap.parse_args(argv)
    if a.step == "render":
        render(a.out_dir)
    else:
        compose(a.out_dir, a.figure)


if __name__ == "__main__":
    main()
