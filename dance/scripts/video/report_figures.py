"""Explanatory figures for dance/docs/report.md.

One function per figure. Every figure is a 300 dpi PNG on white, sized for a 16 cm wide
column, with Arial for Latin text and BIZ UDGothic for Japanese (the report fonts).
Figures carry short names and data labels only; the caption in the report explains.

Run from the repo root with the microduck venv:
    .venv/bin/python dance/scripts/video/report_figures.py            # all figures
    .venv/bin/python dance/scripts/video/report_figures.py workflow   # one figure

Data used by the data figures (all under out/, not tracked):
- fidelity: human pkl dance/data/motions/songs/<song>.pkl, robot dumps before the Task 19
  retarget change out/rec_v4/eval003_old/<song>_v4_states.npz and after it
  out/fid001/sim/<song>_vg0.3_states.npz (fid001 in EXPERIMENTS.md).
- rough_ab: out/t28/ab/metrics.json (rough002 A/B) and out/t31/eval/table3.json (video003).
- results: out/eval_v4/*.json (eval004), out/t31/eval/table3.json (video003),
  out/roller/final/roller_onnx_*.json and out/roller/final2/roller002_*.json (roller002).
- metrics: the results sources plus the per-run video003 files out/t31/eval/<pol>_<song>_s<k>.json
  and .npz (Travel on the rough floor is the trunk path length from the npz), and the fidelity
  dumps for Human r. Flat Travel is missing: eval004 saved no positions.
The roller path figure is not built: no eval or recording saved the robot positions.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle, Arc, Circle  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
OUT_DIR = REPO / "dance/docs/figures"
FONT_DIR = Path("/mnt/c/Windows/Fonts")
FONT_FILES = ["arial.ttf", "arialbd.ttf", "BIZ-UDGothicR.ttc", "BIZ-UDGothicB.ttc"]
CM = 1 / 2.54
WIDTH_CM = 16.0
DPI = 300
FS = 9  # base font size in pt at print size
FS_SMALL = 8

SONGS = ["swag", "funky_magic", "fallin_for_you"]

# Colors (dataviz reference palette, light mode).
INK = "#0b0b0b"
INK2 = "#52514e"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
GRAY = "#8a8985"
KIND = {  # fill, edge
    "own": ("#fde3d6", ORANGE),
    "pretrained": ("#dbe8f8", BLUE),
    "tool": ("#ebebea", GRAY),
    "plain": ("#ffffff", INK2),
}
KIND_NAME = {"own": "自作", "pretrained": "学習済みを使用", "tool": "既存モジュール"}


def setup_fonts() -> None:
    """Arial first, BIZ UDGothic as the fallback for Japanese glyphs."""
    for name in FONT_FILES:
        p = FONT_DIR / name
        if p.exists():
            font_manager.fontManager.addfont(str(p))
    have = {f.name for f in font_manager.fontManager.ttflist}
    fam = [f for f in ("Arial", "BIZ UDGothic") if f in have]
    if len(fam) < 2:
        print(f"warning: report fonts missing, using {fam or 'default'}", file=sys.stderr)
    plt.rcParams.update({
        "font.family": fam + ["DejaVu Sans"],
        "font.size": FS,
        "axes.titlesize": FS,
        "axes.labelsize": FS,
        "xtick.labelsize": FS_SMALL + 0.5,
        "ytick.labelsize": FS_SMALL + 0.5,
        "legend.fontsize": FS_SMALL + 0.5,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })


def save(fig, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")
    return out


# diagram helpers

def canvas(h_cm: float):
    """Figure WIDTH_CM x h_cm whose axes use cm as data units."""
    fig = plt.figure(figsize=(WIDTH_CM * CM, h_cm * CM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, WIDTH_CM)
    ax.set_ylim(0, h_cm)
    ax.axis("off")
    return fig, ax


def box(ax, x0, y0, w, h, text, kind="plain", fs=FS, dashed=False, bold_first=True):
    """Rounded box; the first line is the name (bold), further lines are small data labels."""
    fill, edge = KIND[kind]
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=0.15",
                                fc=fill, ec=edge, lw=1.0, ls="--" if dashed else "-"))
    lines = text.split("\n")
    cx, cy = x0 + w / 2, y0 + h / 2
    if bold_first and len(lines) > 1:
        heights = [0.44] + [0.36] * (len(lines) - 1)
        y = cy + sum(heights) / 2
        for i, (t, lh) in enumerate(zip(lines, heights)):
            y -= lh / 2
            ax.text(cx, y, t, ha="center", va="center", fontsize=fs if i == 0 else FS_SMALL,
                    color=INK if i == 0 else INK2, weight="bold" if i == 0 else "normal")
            y -= lh / 2
    else:
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color=INK,
                weight="bold" if bold_first else "normal", linespacing=1.15)
    return (x0, y0, w, h)


def arrow(ax, p0, p1, color=INK2, lw=1.0, style="-|>", conn="arc3", ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=9, color=color, lw=lw,
                                 connectionstyle=conn, shrinkA=0, shrinkB=0, ls=ls))


def poly_arrow(ax, pts, color=INK2, lw=1.0):
    """Polyline through pts with an arrow head on the last segment."""
    xs, ys = zip(*pts)
    ax.plot(xs[:-1], ys[:-1], color=color, lw=lw, solid_capstyle="butt")
    arrow(ax, pts[-2], pts[-1], color=color, lw=lw)


def label(ax, x, y, text, fs=FS_SMALL, ha="center", va="center", color=INK2, **kw):
    ax.text(x, y, text, fontsize=fs, ha=ha, va=va, color=color, linespacing=1.15, **kw)


def legend_kinds(ax, x, y, kinds=("own", "pretrained", "tool")):
    for k in kinds:
        fill, edge = KIND[k]
        ax.add_patch(Rectangle((x, y - 0.17), 0.5, 0.34, fc=fill, ec=edge, lw=1.0))
        ax.text(x + 0.65, y, KIND_NAME[k], fontsize=FS_SMALL, va="center", color=INK)
        x += 0.65 + 0.3 * len(KIND_NAME[k]) + 0.6


# 1. workflow

def fig_workflow(out_dir: Path = OUT_DIR) -> Path:
    fig, ax = canvas(8.45)
    # lane frames
    for y0, h, name in ((4.15, 4.05, "推論時"), (0.95, 2.95, "学習時")):
        ax.add_patch(FancyBboxPatch((0.08, y0), 15.84, h, boxstyle="round,pad=0,rounding_size=0.2",
                                    fc="none", ec="#d4d3cf", lw=0.8))
        label(ax, 0.3, y0 + h - 0.32, name, fs=FS, ha="left", color=INK, weight="bold")

    yc = 6.1  # inference row centre
    yb, yf = 6.9, 5.3  # beat and feature boxes
    box(ax, 0.3, yc - 0.5, 1.2, 1.0, "曲\nwav", "plain")
    box(ax, 2.2, yb - 0.45, 2.0, 0.9, "拍の検出\nlibrosa", "tool")
    box(ax, 2.2, yf - 0.45, 2.0, 0.9, "音楽特徴量\nJukebox", "pretrained")
    poly_arrow(ax, [(1.5, yc), (1.85, yc), (1.85, yb), (2.2, yb)])
    poly_arrow(ax, [(1.85, yc), (1.85, yf), (2.2, yf)])
    box(ax, 4.75, yc - 0.6, 1.6, 1.2, "EDGE\n踊りの生成", "pretrained")
    poly_arrow(ax, [(4.2, yf), (4.48, yf), (4.48, yc - 0.2), (4.75, yc - 0.2)])
    box(ax, 7.7, yc - 0.6, 2.1, 1.2, "リターゲット", "own")
    arrow(ax, (6.35, yc), (7.7, yc))
    label(ax, 7.02, yc - 0.2, "人の動き\nSMPL\n60 秒\n30 fps", va="top")
    poly_arrow(ax, [(4.2, yb), (8.75, yb), (8.75, yc + 0.6)])
    label(ax, 6.4, yb + 0.12, "拍の時刻", va="bottom")
    box(ax, 11.0, yc - 0.6, 1.4, 1.2, "policy", "own")
    arrow(ax, (9.8, yc), (11.0, yc))
    label(ax, 10.4, yc + 0.12, "指示\n13 個\n50 Hz", va="bottom")
    box(ax, 13.75, yc - 0.6, 2.0, 1.2, "MuJoCo", "tool")
    arrow(ax, (12.4, yc), (13.75, yc))
    label(ax, 13.08, yc + 0.12, "目標角\n14 個", va="bottom")
    poly_arrow(ax, [(14.75, yc - 0.6), (14.75, 4.95), (12.1, 4.95), (12.1, yc - 0.6)])
    label(ax, 13.45, 4.8, "体の状態 48 個", va="top")

    # training lane
    yt = 2.2
    box(ax, 0.3, yt - 0.55, 2.0, 1.1, "合成した曲\n33 曲", "own")
    box(ax, 2.95, yt - 0.55, 2.4, 1.1, "上と同じ処理", "plain", dashed=True)
    arrow(ax, (2.3, yt), (2.95, yt))
    box(ax, 6.25, 2.85 - 0.48, 3.0, 0.96, "学習用ライブラリ\n33 分", "own")
    box(ax, 6.25, 1.5 - 0.42, 3.0, 0.84, "乱数の指示", "own")
    poly_arrow(ax, [(5.35, yt), (5.8, yt), (5.8, 2.8), (6.25, 2.8)])
    box(ax, 10.5, yt - 0.6, 2.4, 1.2, "PPO, mjlab\n2048 環境", "tool")
    poly_arrow(ax, [(9.25, 2.8), (9.9, 2.8), (9.9, yt + 0.15), (10.5, yt + 0.15)])
    poly_arrow(ax, [(9.25, 1.5), (9.9, 1.5), (9.9, yt - 0.15), (10.5, yt - 0.15)])
    box(ax, 13.5, yt - 0.55, 2.0, 1.1, "", "own")
    ax.text(14.5, yt, "報酬と\nランダム化", ha="center", va="center", fontsize=FS, color=INK,
            weight="bold", linespacing=1.15)
    arrow(ax, (13.5, yt), (12.9, yt))
    arrow(ax, (11.4, yt + 0.6), (11.4, yc - 0.6), color=ORANGE, lw=1.3)
    label(ax, 11.55, 3.45, "学習した重み", ha="left", color=INK2)

    legend_kinds(ax, 0.3, 0.45)
    return save(fig, out_dir / "workflow.png")


# 2. architecture

ACTOR_ROWS = [("角速度", 3), ("重力の向き", 3), ("関節角", 14), ("関節角速度", 14), ("前回の出力", 14),
              ("足運び", 3), ("頭", 4), ("体", 6)]



def mlp(ax, x0, yc, name, sizes, kind="own"):
    """Draw an MLP between x0 and x0 + 2.6 cm as layer bars, centred on yc."""
    fill, edge = KIND[kind]
    xs = [x0 + 0.35 + i * 0.62 for i in range(len(sizes))]
    hs = [{512: 1.7, 256: 1.25, 128: 0.9}.get(n, 0.3) for n in sizes]
    for i in range(len(xs) - 1):
        for s in (-1, 1):
            ax.plot([xs[i] + 0.09, xs[i + 1] - 0.09], [yc + s * hs[i] / 2, yc + s * hs[i + 1] / 2],
                    color=edge, lw=0.5, alpha=0.6)
    for x, h, n in zip(xs, hs, sizes):
        ax.add_patch(Rectangle((x - 0.09, yc - h / 2), 0.18, h, fc=fill, ec=edge, lw=0.9))
        label(ax, x, yc - 1.05, str(n), fs=FS_SMALL - 1, color=INK2)
    label(ax, x0 + 1.3, yc + 1.15, name, fs=FS, color=INK, weight="bold")

def fig_architecture(out_dir: Path = OUT_DIR) -> Path:
    fig, ax = canvas(9.7)
    # input column
    x0, w, rh = 0.3, 3.1, 0.5
    y = 8.55
    label(ax, x0, y + 0.3, "体の状態 48", fs=FS, ha="left", color=INK, weight="bold")
    rows_y = []
    for i, (name, n) in enumerate(ACTOR_ROWS):
        if i == 5:
            y -= 0.55
            label(ax, x0, y + 0.3, "指示 13", fs=FS, ha="left", color=INK, weight="bold")
        y -= rh
        kind = "plain" if i < 5 else "own"
        fill, edge = KIND[kind]
        ax.add_patch(Rectangle((x0, y), w, rh - 0.06, fc=fill, ec=edge, lw=0.8))
        label(ax, x0 + 0.15, y + (rh - 0.06) / 2, name, fs=FS_SMALL, ha="left", color=INK)
        label(ax, x0 + w - 0.15, y + (rh - 0.06) / 2, str(n), fs=FS_SMALL, ha="right", color=INK)
        rows_y.append(y)
    top, bot = 8.55 - 0.06, rows_y[-1]
    bx = x0 + w + 0.25  # bracket that gathers the 61 inputs
    ax.plot([x0 + w + 0.1, bx, bx, x0 + w + 0.1], [top, top, bot, bot], color=INK2, lw=1.0)

    ya = 6.9  # actor row
    mlp(ax, 5.0, ya, "actor", [512, 256, 128, 14])
    arrow(ax, (bx, ya), (5.0, ya))
    label(ax, 4.3, ya + 0.12, "61", va="bottom")
    box(ax, 9.2, ya - 0.65, 2.1, 1.3, "モーターの\nモデル BAM", "tool", bold_first=False)
    arrow(ax, (7.6, ya), (9.2, ya))
    label(ax, 8.4, ya + 0.12, "目標角 14", va="bottom")
    box(ax, 12.4, ya - 0.65, 3.3, 1.3, "MuJoCo\n2048 環境", "tool")
    arrow(ax, (11.3, ya), (12.4, ya))
    poly_arrow(ax, [(14.05, ya + 0.65), (14.05, 9.4), (2.4, 9.4), (2.4, 9.05)])
    label(ax, 8.7, 9.27, "体の状態, 50 Hz", va="top")

    # critic, PPO, rewards, randomization
    yc = 2.9
    mlp(ax, 5.0, yc, "critic", [512, 256, 128, 1])
    poly_arrow(ax, [(bx, bot), (bx, yc + 0.25), (5.0, yc + 0.25)])
    label(ax, 4.3, yc + 0.37, "61", va="bottom")
    box(ax, 0.3, 1.0, 3.1, 0.8, "速度と足の情報 15", "plain", bold_first=False, fs=FS_SMALL)
    poly_arrow(ax, [(3.4, 1.4), (4.5, 1.4), (4.5, yc - 0.25), (5.0, yc - 0.25)])
    box(ax, 9.2, yc - 0.65, 2.1, 1.3, "PPO で更新", "tool", bold_first=False)
    arrow(ax, (7.6, yc), (9.2, yc))
    label(ax, 8.4, yc + 0.12, "価値", va="bottom")
    arrow(ax, (10.25, yc + 0.65), (6.6, ya - 1.3), color=ORANGE, lw=1.3)
    label(ax, 8.95, 4.95, "更新", color=ORANGE, ha="left")
    rx, rw = 12.4, 3.3
    box(ax, rx, 0.9, rw, 2.75, "", "own")
    label(ax, rx + 0.2, 3.3, "報酬", fs=FS, ha="left", color=INK, weight="bold")
    for i, t in enumerate(["追従: 足運び, 頭, 体", "姿勢と安定", "足の上げ方", "滑らかさ"]):
        label(ax, rx + 0.3, 2.75 - i * 0.5, t, ha="left", color=INK)
    arrow(ax, (rx, yc), (11.3, yc))
    box(ax, 12.4, 4.2, 3.3, 1.45, "ランダム化\n押す, 摩擦, 質量\nセンサ誤差, 遅れ", "own")
    arrow(ax, (14.05, 5.65), (14.05, ya - 0.65))
    return save(fig, out_dir / "architecture.png")


# 3. retarget

def _stick_figure(ax, px, py, s=1.0):
    """Dancer seen from the side, leaning forward. Returns pelvis, neck, head centre, floor y."""
    pel = np.array([px, py])
    neck = pel + s * np.array([0.45, 2.0])
    head_r = 0.36 * s
    head = neck + s * np.array([0.12, 0.2]) + np.array([0.0, head_r])
    knee_l = pel + s * np.array([0.35, -1.0]); foot_l = knee_l + s * np.array([-0.15, -1.05])
    knee_r = pel + s * np.array([-0.3, -1.05]); foot_r = knee_r + s * np.array([-0.05, -1.0])
    elb_l = neck + s * np.array([0.6, -0.6]); hand_l = elb_l + s * np.array([0.55, 0.25])
    elb_r = neck + s * np.array([-0.55, -0.55]); hand_r = elb_r + s * np.array([-0.2, -0.65])
    neck_top = neck + s * np.array([0.12, 0.2])
    kw = dict(color="#3c3b38", lw=2.4, solid_capstyle="round")
    for a, b in ((pel, neck), (pel, knee_l), (knee_l, foot_l), (pel, knee_r), (knee_r, foot_r),
                 (neck, elb_l), (elb_l, hand_l), (neck, elb_r), (elb_r, hand_r), (neck, neck_top)):
        ax.plot([a[0], b[0]], [a[1], b[1]], **kw)
    ax.add_patch(Circle(head, head_r, fc="white", ec="#3c3b38", lw=2.0, zorder=4))
    ax.plot([head[0] + head_r, head[0] + head_r + 0.16], [head[1] - 0.04, head[1] - 0.08],
            color="#3c3b38", lw=2.0)  # nose: the face direction
    ax.add_patch(Circle(pel, 0.13, fc=ORANGE, ec="none", zorder=5))
    return pel, neck, head, min(foot_l[1], foot_r[1])


GREEN = "#1baf7a"


def fig_retarget(out_dir: Path = OUT_DIR) -> Path:
    fig, ax = canvas(8.6)
    pel, neck, head, floor_y = _stick_figure(ax, 3.0, 3.75, s=1.2)
    ax.plot([0.6, 5.8], [floor_y - 0.04] * 2, color="#bdbcb8", lw=1.2)
    label(ax, 0.6, 8.25, "人の動き, 30 fps", fs=FS, ha="left", color=INK, weight="bold")
    bbox = dict(fc="white", ec="none", pad=1.0)

    # boxes, top to bottom in body order
    bx, bw, bh = 11.4, 3.2, 1.05
    yh, yb, yt = head[1], 5.0, 3.2
    rows = [(yh, "頭 4", "0.12 秒早める", GREEN), (yb, "体 6", "0.12 秒早める", BLUE),
            (yt, "足運び 3", "前後, 左右, 旋回", ORANGE)]
    for y, t, sub, c in rows:
        ax.add_patch(FancyBboxPatch((bx, y - bh / 2), bw, bh, boxstyle="round,pad=0,rounding_size=0.15",
                                    fc="white", ec=c, lw=1.4))
        ax.text(bx + bw / 2, y + 0.19, t, ha="center", va="center", fontsize=FS, color=INK, weight="bold")
        ax.text(bx + bw / 2, y - 0.22, sub, ha="center", va="center", fontsize=FS_SMALL, color=INK2)
    label(ax, bx + bw / 2, 8.25, "指示 13 個, 50 Hz", fs=FS, color=INK, weight="bold")

    # head orientation
    ax.add_patch(Arc(head, 1.25, 1.25, theta1=-35, theta2=35, color=GREEN, lw=1.5))
    arrow(ax, (head[0] + 0.53, head[1] + 0.3), (head[0] + 0.46, head[1] + 0.40), color=GREEN, lw=1.5)
    arrow(ax, (head[0] + 0.85, head[1]), (bx, yh), color=GREEN, lw=1.2)
    label(ax, 8.0, yh + 0.08, "頭の向き", va="bottom", bbox=bbox)
    # trunk tilt and pelvis height
    ax.plot([pel[0], pel[0]], [pel[1], pel[1] + 1.9], color=BLUE, lw=0.8, ls=":")
    ang = np.degrees(np.arctan2(neck[1] - pel[1], neck[0] - pel[0]))
    ax.add_patch(Arc(pel, 2.6, 2.6, theta1=ang, theta2=90, color=BLUE, lw=1.4))
    arrow(ax, (1.3, floor_y), (1.3, pel[1]), style="<|-|>", color=BLUE, lw=1.2)
    ax.plot([1.2, pel[0] - 0.15], [pel[1], pel[1]], color=BLUE, lw=0.8, ls=":")
    arrow(ax, (pel[0] + 1.9, yb), (bx, yb), color=BLUE, lw=1.2)
    label(ax, 8.0, yb + 0.08, "腰の高さ, 体の傾き", va="bottom", bbox=bbox)
    # pelvis heading and velocity
    arrow(ax, pel + [0.2, 0.0], pel + [1.45, 0.0], color=ORANGE, lw=1.6)
    ax.add_patch(Arc(pel, 1.2, 0.45, theta1=200, theta2=340, color=ORANGE, lw=1.4))
    arrow(ax, (pel[0] + 0.5, pel[1] - 0.19), (pel[0] + 0.57, pel[1] - 0.12), color=ORANGE, lw=1.4)
    arrow(ax, (pel[0] + 1.7, pel[1] - 0.1), (bx, yt), color=ORANGE, lw=1.2)
    label(ax, 8.0, 3.55, "腰の向きと速さ", va="bottom", bbox=bbox)

    # beat layer, added to head and trunk from the right
    ty = 0.55
    ax.plot([0.6, 5.8], [ty, ty], color="#bdbcb8", lw=1.0)
    for i in range(9):
        x = 0.8 + i * 0.6
        ax.plot([x, x], [ty - 0.15, ty + 0.15], color=INK2, lw=1.4)
    label(ax, 0.6, ty + 0.42, "拍", fs=FS, ha="left", color=INK, weight="bold")
    rx = bx + bw + 0.6
    ax.plot([5.8, rx, rx], [ty, ty, yh], color=INK2, lw=1.0)
    arrow(ax, (rx, yh), (bx + bw, yh))
    arrow(ax, (rx, yb), (bx + bw, yb))
    label(ax, 9.0, ty + 0.12, "うなずき, 沈み込み", va="bottom")
    return save(fig, out_dir / "retarget.png")


# 4. fidelity

def _fidelity_headings(song: str, before: Path, after: Path):
    sys.path.insert(0, str(REPO / "dance/scripts/evaluate"))
    import fidelity_eval as F  # needs the microduck venv (microduck_dance.retarget)

    h, _ = F.human_channels(REPO / f"dance/data/motions/songs/{song}.pkl")
    res = {}
    for tag, p in (("before", before), ("after", after)):
        st = np.load(p)
        r = F.cmd_channels(st["meas"].astype(float), st["pos"].astype(float), st["yaw"])
        res[tag] = (np.degrees(r["heading"]), F.ncc(h["heading"], r["heading"]))
    t = np.arange(F.N_STEPS) * F.DT
    return t, np.degrees(h["heading"]), res


def fig_fidelity(out_dir: Path = OUT_DIR, song: str = "funky_magic",
                 before_dir: Path = REPO / "out/rec_v4/eval003_old",
                 after_dir: Path = REPO / "out/fid001/sim") -> Path:
    before = before_dir / f"{song}_v4_states.npz"
    after = after_dir / f"{song}_vg0.3_states.npz"
    for p in (before, after):
        if not p.exists():
            raise FileNotFoundError(p)
    t, hum, res = _fidelity_headings(song, before, after)
    fig, axs = plt.subplots(1, 2, figsize=(WIDTH_CM * CM, 6.0 * CM), sharey=True)
    lo = min(hum.min(), *(r[0].min() for r in res.values()))
    hi = max(hum.max(), *(r[0].max() for r in res.values()))
    for ax, tag, name in zip(axs, ("before", "after"), ("変更前", "変更後")):
        rob, r = res[tag]
        ax.plot(t, hum, color=GRAY, lw=1.6, label="人")
        ax.plot(t, rob, color=BLUE, lw=1.3, label="ロボット")
        ax.set_xlim(0, 60)
        ax.set_ylim(lo - 30, hi + 60)
        ax.set_xticks([0, 20, 40, 60])
        ax.set_xlabel("時間 (秒)")
        ax.grid(axis="y", color="#e6e5e1", lw=0.6)
        ax.text(0.03, 0.95, name, transform=ax.transAxes, fontsize=FS, weight="bold", va="top", color=INK)
        ax.text(0.97, 0.05, f"r = {r:.2f}", transform=ax.transAxes, fontsize=FS_SMALL + 0.5, ha="right",
                va="bottom", color=INK)
    axs[0].set_ylabel("体の向き (度)")
    axs[1].legend(loc="upper center", frameon=False, ncol=1, bbox_to_anchor=(0.45, 1.0))
    fig.tight_layout(pad=0.4, w_pad=1.2)
    print(f"fidelity {song}: r before {res['before'][1]:.3f}, after {res['after'][1]:.3f}")
    return save(fig, out_dir / "fidelity.png")


# shared bar helpers

def _seed_pooled(per_song_seed):
    """per_song_seed: {song: [falls per seed]} -> (mean, SE over the seed means of the 3 songs)."""
    arr = np.array([per_song_seed[s] for s in SONGS])  # songs x seeds
    seed_means = arr.mean(axis=0)
    return float(seed_means.mean()), float(seed_means.std(ddof=1) / np.sqrt(len(seed_means)))


def _bars(ax, names, vals, errs, colors, fmt="{:.2f}", ymax=None):
    x = np.arange(len(names))
    ax.bar(x, vals, width=0.6, color=colors, yerr=errs, capsize=3,
           error_kw=dict(ecolor=INK, lw=0.9, capthick=0.9))
    ax.set_xticks(x, names)
    ax.tick_params(axis="x", length=0)
    top = ymax if ymax is not None else max(v + e for v, e in zip(vals, errs)) * 1.25
    bottom = min(0, min(v - e for v, e in zip(vals, errs)) * 1.6)
    ax.set_ylim(bottom, top)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
    for xi, v, e in zip(x, vals, errs):
        txt = fmt.format(v).replace("-", "\u2212")
        if v >= 0:
            ax.text(xi, v + e + top * 0.03, txt, ha="center", va="bottom", fontsize=FS_SMALL + 0.5,
                    color=INK)
        else:
            ax.text(xi, v - e - top * 0.03, txt, ha="center", va="top", fontsize=FS_SMALL + 0.5,
                    color=INK)
    ax.grid(axis="y", color="#e6e5e1", lw=0.6)
    ax.set_axisbelow(True)


# 5. rough A/B

def fig_rough_ab(out_dir: Path = OUT_DIR) -> Path:
    ab = {Path(r["npz"]).stem: r for r in json.load(open(REPO / "out/t28/ab/metrics.json"))}
    t3 = json.load(open(REPO / "out/t31/eval/table3.json"))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(WIDTH_CM * CM, 6.0 * CM),
                                 gridspec_kw=dict(width_ratios=[3, 2]))
    keys = [("flat", "平らな床"), ("A_box", "箱 1 枚"), ("rough", "凸凹の床")]
    vals = [ab[k]["heading_r"] for k, _ in keys]
    errs = [ab[k]["heading_r_se"] for k, _ in keys]
    _bars(a1, [n for _, n in keys], vals, errs, [GRAY] * 3, ymax=1.05)
    a1.axhline(0, color=INK2, lw=0.8)
    a1.set_ylabel("人との向きの相関")
    a2_names = ["平らな床用", "凸凹用"]
    v2 = [t3["v4_video"]["falls"], t3["ft750_video"]["falls"]]
    e2 = [t3["v4_video"]["falls_se"], t3["ft750_video"]["falls_se"]]
    _bars(a2, a2_names, v2, e2, [GRAY, BLUE], fmt="{:.1f}")
    a2.set_ylabel("凸凹の床での転倒 (回/分)")
    fig.tight_layout(pad=0.4, w_pad=2.0)
    return save(fig, out_dir / "rough_ab.png")


# 6. roller paths (not built)

def fig_roller_paths(out_dir: Path = OUT_DIR) -> Path | None:
    """Not built: the roller evals (out/roller/final*, video*) saved summary numbers only, no
    positions or speeds over time, so there is nothing to draw without re-running the policies."""
    print("roller_paths: skipped, no saved positions or speed series for roller001 and roller002")
    return None


# 7. results

def _results_data():
    flat = {}
    for pol in ("dance_v4", "base5000"):
        flat[pol] = _seed_pooled({s: [ps["mean"]["falls"] for ps in
                                      json.load(open(REPO / f"out/eval_v4/{pol}_{s}.json"))["per_seed"]]
                                  for s in SONGS})
    t3 = json.load(open(REPO / "out/t31/eval/table3.json"))
    rough = {"ft750": (t3["ft750_video"]["falls"], t3["ft750_video"]["falls_se"]),
             "dance_v4": (t3["v4_video"]["falls"], t3["v4_video"]["falls_se"])}
    roller = {}
    for pol, d in (("roller002", "final2"), ("roller_onnx", "final")):
        roller[pol] = _seed_pooled({s: [m["falls_per_min"] for m in
                                        json.load(open(REPO / f"out/roller/{d}/{pol}_{s}.json"))["per_seed_means"]]
                                    for s in SONGS})
    return flat, rough, roller


def fig_results(out_dir: Path = OUT_DIR) -> Path:
    flat, rough, roller = _results_data()
    fig, axs = plt.subplots(1, 3, figsize=(WIDTH_CM * CM, 6.2 * CM))
    panels = [("平らな床", ["平らな床用", "歩行だけ"], [flat["dance_v4"], flat["base5000"]]),
              ("凸凹の床", ["凸凹用", "平らな床用"], [rough["ft750"], rough["dance_v4"]]),
              ("ローラー", ["ローラー用", "元の policy"], [roller["roller002"], roller["roller_onnx"]])]
    for ax, (title, names, ve) in zip(axs, panels):
        vals = [v for v, _ in ve]
        errs = [e for _, e in ve]
        _bars(ax, names, vals, errs, [BLUE, GRAY], fmt="{:.2f}" if max(vals) < 1 else "{:.1f}")
        ax.set_title(title, fontsize=FS, weight="bold", color=INK, pad=4)
        print(f"results {title}: " + ", ".join(f"{n} {v:.3f} ± {e:.3f}" for n, (v, e) in zip(names, ve)))
    axs[0].set_ylabel("転倒 (回/分)")
    fig.tight_layout(pad=0.4, w_pad=1.5)
    return save(fig, out_dir / "results.png")



# policies: how the three policies used in the video relate

def fig_policies(out_dir: Path = OUT_DIR) -> Path:
    fig, ax = canvas(4.6)
    w, h = 3.3, 1.3
    box(ax, 0.3, 2.9, w, h, "歩行 policy\n5000 回", "own")
    box(ax, 4.6, 2.9, w, h, "平らな床用\n10000 回", "own")
    box(ax, 8.9, 2.9, w, h, "凸凹用\n10750 回", "own")
    box(ax, 4.6, 0.4, w, h, "Pollen の\n滑るだけの policy", "pretrained")
    label(ax, 2.45, 1.05, "車輪付きの機体", color=INK)
    box(ax, 8.9, 0.4, w, h, "ローラー用\n滑りながら踊る", "own")
    arrow(ax, (3.6, 3.55), (4.6, 3.55))
    arrow(ax, (7.9, 3.55), (8.9, 3.55))
    arrow(ax, (7.9, 1.05), (8.9, 1.05))
    label(ax, 4.25, 4.4, "踊りの学習")
    label(ax, 8.4, 4.4, "凸凹で追加学習")
    label(ax, 8.4, 1.9, "踊りの学習")
    for x, s in ((6.25, "動画: 平らな床の3曲"), (10.55, "動画: 凸凹の床")):
        label(ax, x, 2.55, s, color=INK)
    label(ax, 12.45, 1.05, "動画: ローラースケート", ha="left", color=INK)
    return save(fig, out_dir / "policies.png")

# metrics: every 3.5 metric per scene, the video policy against its comparison

def _t31_travel(npz: Path) -> float:
    """Mean trunk path length (m/min) over the envs of one video003 dump, as roller_dance_eval does."""
    d = np.load(npz)
    xy, term = d["pos"][:, :, :2].astype(float), d["term"].astype(bool)
    minutes = len(term) * float(d["dt"]) / 60.0
    lens = []
    for i in range(xy.shape[1]):
        step = np.linalg.norm(np.diff(xy[:, i], axis=0), axis=1)
        bad = term[1:, i] | term[:-1, i]
        lens.append(step[~bad].sum() / minutes)
    return float(np.mean(lens))


def _metrics_data():
    """{metric: {scene: {"ours"|"base": (mean, se)}}}; a missing value is left out of the dict.

    Command r is the mean of the forward and turning correlations (the roller eval has no
    sideways one). All values are seed means of the 3-song average, SE over the 3 seeds.
    """
    def pooled(get):
        return _seed_pooled({s: get(s) for s in SONGS})

    seeds = (0, 1, 2)
    keys = {"fall": "falls", "head": "head_rmse", "body": "body_rmse"}
    out = {m: {} for m in ("fall", "head", "body", "cmd", "travel", "human")}

    def cmd_r(m):
        return 0.5 * (m["twist_corr_vx"] + m["twist_corr_wz"])

    # flat: eval004
    for role, pol in (("ours", "dance_v4"), ("base", "base5000")):
        ms = {s: [ps["mean"] for ps in json.load(open(REPO / f"out/eval_v4/{pol}_{s}.json"))["per_seed"]]
              for s in SONGS}
        for m, k in keys.items():
            out[m].setdefault("flat", {})[role] = pooled(lambda s: [x[k] for x in ms[s]])
        out["cmd"].setdefault("flat", {})[role] = pooled(lambda s: [cmd_r(x) for x in ms[s]])
    # rough: video003, rough policy (ft750) and dance_v4 on the video terrain
    for role, pol in (("ours", "ft750_video"), ("base", "v4_video")):
        ms = {s: [json.load(open(REPO / f"out/t31/eval/{pol}_{s}_s{k}.json"))["summary"] for k in seeds]
              for s in SONGS}
        for m, k in keys.items():
            out[m].setdefault("rough", {})[role] = pooled(lambda s: [x[k] for x in ms[s]])
        out["cmd"].setdefault("rough", {})[role] = pooled(lambda s: [cmd_r(x) for x in ms[s]])
        out["travel"].setdefault("rough", {})[role] = pooled(
            lambda s: [_t31_travel(REPO / f"out/t31/eval/{pol}_{s}_s{k}.npz") for k in seeds])
    # roller: roller002 (final2) and Pollen's roller.onnx (final)
    for role, pol, d in (("ours", "roller002", "final2"), ("base", "roller_onnx", "final")):
        ms = {s: json.load(open(REPO / f"out/roller/{d}/{pol}_{s}.json"))["per_seed_means"] for s in SONGS}
        for m, k in (("fall", "falls_per_min"), ("head", "head_rmse"), ("body", "body_rmse"),
                     ("travel", "distance_m_per_min")):
            out[m].setdefault("roller", {})[role] = pooled(lambda s: [x[k] for x in ms[s]])
        out["cmd"].setdefault("roller", {})[role] = pooled(lambda s: [cmd_r(x) for x in ms[s]])
    # human heading r before (eval003_old) and after (fid001) the retarget change, 1 run per song
    per_song = {"before": [], "after": []}
    for s in SONGS:
        _, _, res = _fidelity_headings(s, REPO / f"out/rec_v4/eval003_old/{s}_v4_states.npz",
                                       REPO / f"out/fid001/sim/{s}_vg0.3_states.npz")
        for tag in per_song:
            per_song[tag].append(res[tag][1])
    out["human"] = per_song
    return out


SCENES = [("flat", "平らな床"), ("rough", "凸凹"), ("roller", "ローラー")]


def fig_metrics(out_dir: Path = OUT_DIR) -> Path:
    data = _metrics_data()
    fig, axs = plt.subplots(2, 3, figsize=(WIDTH_CM * CM, 10.0 * CM))
    panels = [("fall", "Fall rate", "回/分"), ("head", "Head RMSE", "RMSE"),
              ("body", "Body RMSE", "RMSE"), ("cmd", "Command r", "r"), ("travel", "Travel", "m/分")]
    w = 0.36
    for ax, (m, title, unit) in zip(axs.flat, panels):
        tops = []
        for xi, (sc, _) in enumerate(SCENES):
            for role, dx, col in (("ours", -w / 2, BLUE), ("base", w / 2, GRAY)):
                ve = data[m].get(sc, {}).get(role)
                if ve is None:  # no saved data for this scene: leave the bar out, mark the gap
                    if role == "ours":
                        ax.text(xi, 0, "なし", transform=ax.get_xaxis_transform(), ha="center",
                                va="bottom", fontsize=FS_SMALL, color=INK2)
                    continue
                ax.bar(xi + dx, ve[0], width=w, color=col, yerr=ve[1], capsize=2.5,
                       error_kw=dict(ecolor=INK, lw=0.8, capthick=0.8))
                tops.append(ve[0] + ve[1])
                print(f"metrics {m} {sc} {role}: {ve[0]:.3f} ± {ve[1]:.3f}")
        ax.set_xticks(range(len(SCENES)), [n for _, n in SCENES])
        ax.tick_params(axis="x", length=0)
        ax.set_xlim(-0.6, len(SCENES) - 0.4)
        ax.set_ylim(0, max(tops) * 1.12 if m != "cmd" else 1.0)
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        ax.set_title(title, fontsize=FS, weight="bold", color=INK, pad=4)
        ax.set_ylabel(unit)
        ax.grid(axis="y", color="#e6e5e1", lw=0.6)
        ax.set_axisbelow(True)
    # human r: before and after the retarget change, bar = mean of the 3 songs, dots = songs
    ax = axs.flat[5]
    for xi, (tag, col) in enumerate((("before", GRAY), ("after", BLUE))):
        v = data["human"][tag]
        ax.bar(xi, np.mean(v), width=0.55, color=col)
        ax.scatter(np.full(len(v), xi), v, s=9, color=INK, zorder=3, lw=0)
        print(f"metrics human {tag}: mean {np.mean(v):.3f}, songs " + ", ".join(f"{x:.3f}" for x in v))
    ax.set_xticks([0, 1], ["変更前", "変更後"])
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
    ax.set_title("Human r", fontsize=FS, weight="bold", color=INK, pad=4)
    ax.set_ylabel("r")
    ax.grid(axis="y", color="#e6e5e1", lw=0.6)
    ax.set_axisbelow(True)
    handles = [Rectangle((0, 0), 1, 1, color=BLUE), Rectangle((0, 0), 1, 1, color=GRAY)]
    fig.legend(handles, ["動画で使った policy", "学習の出発点の policy"], loc="upper center", ncol=2,
               frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(pad=0.4, w_pad=1.2, h_pad=1.0, rect=(0, 0, 1, 0.94))
    return save(fig, out_dir / "metrics.png")



# EDGE: redrawn from Tseng et al. (CVPR 2023) with the values of the public code

def fig_edge(out_dir: Path = OUT_DIR) -> Path:
    fig, ax = canvas(7.6)
    bw, bh = 3.0, 1.1
    # inputs (left) and their projections (middle)
    box(ax, 0.2, 5.4, bw, bh, "音楽特徴量\nJukebox, 4800 次元", "pretrained")
    box(ax, 3.8, 5.4, 2.8, bh, "Transformer\nエンコーダ ×2", "pretrained")
    arrow(ax, (3.2, 5.95), (3.8, 5.95))
    box(ax, 0.2, 3.0, bw, bh, "拡散の時刻 t", "plain", bold_first=False)
    box(ax, 3.8, 3.0, 2.8, bh, "MLP", "pretrained", bold_first=False)
    arrow(ax, (3.2, 3.55), (3.8, 3.55))
    box(ax, 0.2, 1.25, bw, bh, "雑音入りの動き\n151 × 150", "plain")
    # decoder, layers stacked bottom to top
    dx, dw = 7.3, 4.3
    ax.add_patch(Rectangle((dx, 0.95), dw, 6.35, fc="white", ec=INK2, lw=1.0, ls="--"))
    label(ax, dx + dw / 2, 7.0, "デコーダ ×8", fs=FS, color=INK, weight="bold")
    label(ax, dx + dw / 2, 6.55, "各層で t による FiLM", fs=FS_SMALL)
    ys = [1.3, 3.0, 4.7]
    for y, s in zip(ys, ["自己注意", "交差注意", "全結合"]):
        box(ax, dx + 0.5, y, dw - 1.0, bh, s, "pretrained", bold_first=False)
    for y0, y1 in zip(ys[:-1], ys[1:]):
        arrow(ax, (dx + dw / 2, y0 + bh), (dx + dw / 2, y1))
    arrow(ax, (3.2, 1.85), (dx + 0.5, 1.85))
    arrow(ax, (6.6, 3.55), (dx + 0.5, 3.4))
    poly_arrow(ax, [(6.6, 5.95), (7.0, 5.95), (7.0, 3.8), (dx + 0.5, 3.8)])
    label(ax, 6.75, 3.15, "t", ha="left")
    label(ax, 7.1, 5.3, "音楽", ha="left")
    # output and the sampling loop
    box(ax, 12.5, 4.7, 3.2, bh, "推定した動き\n151 × 150", "plain")
    arrow(ax, (dx + dw - 0.5, 5.25), (12.5, 5.25))
    poly_arrow(ax, [(14.1, 4.7), (14.1, 0.5), (1.7, 0.5), (1.7, 1.25)], color=ORANGE)
    label(ax, 7.9, 0.12, "拡散 1000 段で学習、50 段の DDIM で生成", color=ORANGE, va="bottom")
    return save(fig, out_dir / "edge.png")

FIGURES = {
    "workflow": fig_workflow,
    "architecture": fig_architecture,
    "retarget": fig_retarget,
    "fidelity": fig_fidelity,
    "rough_ab": fig_rough_ab,
    "roller_paths": fig_roller_paths,
    "results": fig_results,
    "policies": fig_policies,
    "edge": fig_edge,
    "metrics": fig_metrics,
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("figures", nargs="*", help=f"figures to build, any of {', '.join(FIGURES)} (default: all)")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args(argv)
    unknown = [f for f in args.figures if f not in FIGURES]
    if unknown:
        ap.error(f"unknown figure(s): {', '.join(unknown)}")
    setup_fonts()
    for name in args.figures or list(FIGURES):
        FIGURES[name](args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
