"""Build the submission video (1920x1080, 29.97 fps, AAC encoded once, at most 180 s).

The video says one thing: give microduck a song and it dances to it, also on a rough floor and on
roller skates. On-screen text says what is shown and how it was made. Every source and license is on the
credits card only.

    course opening -> highlight with the title over its bottom (10 s, music) -> workflow (inference 8 s,
    training 6 s) -> three songs (human left, robot right) -> rough floor -> roller skates -> credits ->
    course ending. Each dance section has one subtitle line (two for the songs) saying what is shown.

The video is a list of sections (SECTIONS below). A section is enabled or not and builds one or more
segments. Switch sections with --enable / --disable, list them with --list-sections.

Inputs are full-song recordings that start at song time 0, all with the lagged camera
(play_dance.py --cam-follow lagged --dump-states):
    out/rec_v4/<song>_v4.mp4                                   (dance_v4)
    out/rec_v4/<song>_human3d.mp4                              (render_human_3d.py, EDGE motions as a
                                                                3D mannequin)
    out/rough/t31/funky_magic_video_15s_wideintro_rough.mp4    (dance_rough on the video terrain,
                                                                Funky Magic 0 to 15 s)
    out/roller/video2/roller002_funky_magic_15s.mp4            (roller_dance, Funky Magic 28 to 43 s)
    out/course_assets/op.mp4, ed.mp4                           (course opening and ending, no audio)
Nothing under out/ is committed. The course clips are course material and are not in the repository.
A section whose input files are missing is skipped with a warning, so a fresh clone builds a shorter
video instead of stopping.

    uv run python dance/scripts/video/make_final_video.py --out out/final/microduck_dance.mp4

The song segments are the liveliest 25 s of each song by retargeted command energy (pick_segment.py).
The highlight moments were picked by eye from contact sheets (clear robot, big motion, no fall).
Slides are drawn with Pillow in IPAexGothic.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compose_video import (  # noqa: E402
    BAND_H, JA_FONT, ClipSegment, Segment, SlideSegment, compose_video, probe_duration)
from pick_segment import command_energy, groove_to_travel_times, liveliest_window  # noqa: E402

from microduck_dance.command_file import load_command_file  # noqa: E402

W, H = 1920, 1080
MAX_TOTAL_S = 180.0
BG = (20, 24, 44)
FG = (240, 242, 248)
DIM = (170, 178, 200)
ACCENT = (242, 183, 5)
BOX = (38, 46, 80)

ALL_SONGS = {
    # stem: (on-screen title, composer)
    "swag": ("Swag", "Phalene"),
    "funky_magic": ("Funky Magic", "FLASH☆BEAT"),
    "fallin_for_you": ("Fallin For You", "Flehmann"),
    "redwing_beats": ("Redwing Beats", "Anonyment"),
    "pastel_house": ("パステルハウス", "かずち"),
}
SONGS = [(s, *ALL_SONGS[s]) for s in ("swag", "funky_magic", "fallin_for_you")]  # replaced by --songs
SONG_LEN_S = 25.0
HUMAN_TAG = "human3d"  # <song>_human3d.mp4 (3D mannequin)
MUSIC_DIR = Path("dance/data/music/opentracks")
COURSE_DIR = Path("out/course_assets")

# On-screen text. This is all the text the video has outside the credits card.
TITLE = "曲に合わせて踊る microduck"
SUBTITLE = "Physical AI 応用1 最終課題  tom11111111"
# Workflow slides. Each step is (label, small lines, kind). kind "own" is what I wrote or trained,
# "pre" is a trained model used as is, "other" is neither.
KIND_COLOR = {"own": ACCENT, "pre": (110, 170, 240), "other": DIM}
KIND_LABEL = {"own": "自作", "pre": "学習済みを使用"}
INFER_TITLE = "推論時 (踊らせるとき)"
INFER_STEPS = [
    ("曲", ["wav ファイル"], "other"),
    ("拍と音楽特徴量", ["拍の検出 (librosa)", "Jukebox"], "pre"),
    ("EDGE", ["人の踊り 60 秒"], "pre"),
    ("リターゲット", ["自作の規則", "13 次元の指示、50 Hz"], "own"),
    ("policy", ["自分で強化学習", "14 関節の目標角"], "own"),
    ("MuJoCo", ["microduck が踊る"], "other"),
]
TRAIN_TITLE = "学習時"
TRAIN_STEPS = [
    ("合成した 33 曲", ["numpy で作った曲"], "own"),
    ("指示を作る", ["推論時と同じ手順", "学習用ライブラリ 33 分"], "own"),
    ("mjlab + PPO", ["ランダムな乱れを加え", "2048 環境で学習"], "own"),
    ("policy", ["推論時に使う"], "own"),
]
TRAIN_NOTE = "本物の曲は AI の学習に使っていない"
HUMAN_LABEL = "人の踊り (AI が生成)"
ROBOT_LABEL = "microduck"
ROUGH_TEXT = "凸凹の床"
ROLLER_TEXT = "ローラースケート"
# One subtitle per dance section saying what the viewer sees.
SONG_SUB = ("左: EDGE が曲から作った人の踊り",
            "右: その踊りを指示に変換し、自分で学習した policy で動く microduck")
ROUGH_SUB = ("段差 1.5 cm の凸凹の床。凸凹用に追加学習した policy",)
ROLLER_SUB = ("足に車輪の付いた microduck。Pollen Robotics のローラー用 policy から追加学習",)

# Song layout: song title at the top left, labels above the panels, panels 640 + 1280 wide, 720 tall.
SONG_TITLE_SIZE = 56
LABEL_SIZE = 44
BODY_Y = 170
LABEL_Y = 104
CLIP_TEXT_SIZE = 72

ROUGH_CLIP = Path("out/rough/t31/funky_magic_video_15s_wideintro_rough.mp4")  # song time 0 to 15 s
ROUGH_SONG = "funky_magic"
ROLLER_CLIP = Path("out/roller/video2/roller002_funky_magic_15s.mp4")  # song time 28 to 43 s
ROLLER_SONG = "funky_magic"
ROLLER_START_S = 28.0
CLIP_LEN_S = 15.0

# Highlight, all Funky Magic: (robot clip ("rec" is out/rec_v4/<song>_v4.mp4), song, song time at
# which the clip starts, cut start in song time, length).
# Picked by eye from 2 fps contact sheets: big head swings and a side step in Funky Magic, head moves
# on the stairs, and the roller robot close to the camera while it slides. Every cut starts on a beat
# of dance/data/beats/songs/funky_magic.json (117.5 BPM) and ends within 20 ms of one.
HIGHLIGHT = [
    ("rec", "funky_magic", 0.0, 33.065, 4.0),
    (ROUGH_CLIP, ROUGH_SONG, 0.0, 6.571, 3.0),
    (ROLLER_CLIP, ROLLER_SONG, ROLLER_START_S, 31.556, 3.0),
]

# Credits card. Names and licenses are from dance/data/CREDITS.md and the rights audit.
MESH_LICENSE_URL = "https://creativecommons.org/licenses/by-nc-sa/4.0/"
MESH_SOURCE_URL = "https://github.com/pollen-robotics/microduck_rl"


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(JA_FONT), size)


def centered(d: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, size: int, fill=FG) -> None:
    d.text(xy, text, font=font(size), fill=fill, anchor="mm")


# ---------------------------------------------------------------- slides


def title_band(path: Path) -> None:
    """Transparent 1920x280 PNG laid over the bottom of the highlight: a soft dark gradient, the title
    and the small course line on one line at the bottom edge, below the robot."""
    im = Image.new("RGBA", (W, BAND_H), (0, 0, 0, 0))
    grad = Image.new("L", (1, BAND_H))
    for y in range(BAND_H):
        grad.putpixel((0, y), round(210 * min(1.0, max(0.0, y - 60) / (BAND_H - 100)) ** 1.5))
    im.putalpha(grad.resize((W, BAND_H)))
    d = ImageDraw.Draw(im)
    # One line near the bottom edge: the robot's feet reach down to about y = 910 of the frame
    # (y = 140 of the band) in the highlight cuts, and the text starts below that.
    tf, sf = font(76), font(38)
    x = (W - (d.textlength(TITLE, font=tf) + 50 + d.textlength(SUBTITLE, font=sf))) / 2
    d.text((x, 245), TITLE, font=tf, fill=FG + (255,), anchor="ls")
    d.text((x + d.textlength(TITLE, font=tf) + 50, 245), SUBTITLE, font=sf, fill=DIM + (255,), anchor="ls")
    assert x >= 40, "title line does not fit"
    im.save(path)


def _box(d: ImageDraw.ImageDraw, x: float, y: float, bw: float, bh: float, step) -> None:
    """A workflow box: the label (44 px) and one or two small lines (32 px), outlined by its kind."""
    label, lines, kind = step
    color = KIND_COLOR[kind]
    d.rounded_rectangle([x, y, x + bw, y + bh], radius=20, fill=BOX, outline=color, width=6 if kind != "other" else 2)
    n = len(lines)
    ly = y + bh / 2 - (n * 44) / 2 - 8
    centered(d, (x + bw / 2, ly), label, 44, color if kind != "other" else FG)
    for k, line in enumerate(lines):
        centered(d, (x + bw / 2, ly + 66 + k * 44), line, 32, FG)


def _arrow(d: ImageDraw.ImageDraw, x0: float, x1: float, y: float) -> None:
    d.polygon([(x0, y - 24), (x1, y), (x0, y + 24)], fill=DIM)


def _legend(d: ImageDraw.ImageDraw, y: float, kinds=("own", "pre")) -> None:
    """Swatch and label per kind, centred as a group, 90 px between items."""
    f = font(36)
    widths = [80 + d.textlength(KIND_LABEL[k], font=f) for k in kinds]
    x = (W - (sum(widths) + 90 * (len(kinds) - 1))) / 2
    for kind, w in zip(kinds, widths):
        d.rounded_rectangle([x, y - 20, x + 60, y + 20], radius=8, fill=BOX, outline=KIND_COLOR[kind], width=6)
        d.text((x + 80, y), KIND_LABEL[kind], font=f, fill=FG, anchor="lm")
        x += w + 90


def slide_inference(path: Path) -> None:
    """Six boxes in two rows of three. Row 1 runs left to right, an elbow arrow leads to row 2."""
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    centered(d, (W / 2, 80), INFER_TITLE, 60)
    bw, bh, gap = 520, 220, 90
    x0 = (W - (3 * bw + 2 * gap)) / 2
    tops = [190, 560]
    for i, step in enumerate(INFER_STEPS):
        r, c = divmod(i, 3)
        x, y = x0 + c * (bw + gap), tops[r]
        _box(d, x, y, bw, bh, step)
        if c < 2:
            _arrow(d, x + bw + 16, x + bw + gap - 16, y + bh / 2)
    # elbow from the bottom of box 3 to the top of box 4
    xa, xb = x0 + 2 * (bw + gap) + bw / 2, x0 + bw / 2
    ym = (tops[0] + bh + tops[1]) / 2
    d.line([(xa, tops[0] + bh + 8), (xa, ym), (xb, ym), (xb, tops[1] - 34)], fill=DIM, width=6)
    d.polygon([(xb - 22, tops[1] - 36), (xb + 22, tops[1] - 36), (xb, tops[1] - 8)], fill=DIM)
    _legend(d, 950)
    im.save(path)


def slide_training(path: Path) -> None:
    """Four boxes in one row, the legend and one footnote."""
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    centered(d, (W / 2, 120), TRAIN_TITLE, 60)
    bw, bh, gap = 400, 260, 70
    x0 = (W - (4 * bw + 3 * gap)) / 2
    y = 330
    for i, step in enumerate(TRAIN_STEPS):
        x = x0 + i * (bw + gap)
        _box(d, x, y, bw, bh, step)
        if i < len(TRAIN_STEPS) - 1:
            _arrow(d, x + bw + 12, x + bw + gap - 12, y + bh / 2)
    _legend(d, 760, ("own",))
    centered(d, (W / 2, 880), TRAIN_NOTE, 36, DIM)
    im.save(path)


def credit_lines(songs) -> list[tuple[str, int, tuple]]:
    """(text, size, color) of the credits card. The heading of each group is dim, the source bright."""
    return [
        ("音楽  OpenTracks (https://opentracks.com/)", 34, DIM),
        ("   ".join(f'"{t}" by {c}' for _, t, c in songs), 38, FG),
        ("ダンス生成  EDGE (Tseng et al., CVPR 2023, MIT License)", 36, FG),
        ("EDGE の学習データ  AIST++ (Li et al., ICCV 2021, CC BY 4.0)、AIST Dance Video Database", 34, FG),
        ("音楽特徴量  OpenAI Jukebox (Dhariwal et al., 2020, OpenAI Noncommercial Use License)", 34, FG),
        ("ロボットのメッシュ  Microduck, Pollen Robotics (CC BY-NC-SA 4.0、改変なし)", 36, FG),
        (f"{MESH_LICENSE_URL}", 30, DIM),
        ("ローラーの policy  roller.onnx, pollen-robotics/microduck-policies (Apache-2.0)、追加学習で改変", 34, FG),
        (f"学習環境  microduck-playground (Apache-2.0)、元は {MESH_SOURCE_URL}", 32, FG),
        ("シミュレーションと学習  MuJoCo, mjlab, rsl_rl", 34, FG),
    ]


def slide_credits(path: Path, songs) -> None:
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    centered(d, (W / 2, 90), "クレジット", 60)
    y = 190
    for text, size, color in credit_lines(songs):
        centered(d, (W / 2, y), text, size, color)
        y += size + 44
    im.save(path)


# ---------------------------------------------------------------- sections


@dataclass
class Context:
    rec: Path
    slides: Path
    songs: list
    sections: list = field(default_factory=list)


@dataclass
class Section:
    """One part of the video. `build(ctx)` returns the segments. `requires` are files the section needs:
    a list of paths, or a function of the context that returns one. A section with a missing file is
    skipped (skip_missing)."""
    key: str
    title: str
    build: callable
    enabled: bool = True
    requires: object = field(default_factory=list)

    def needs(self, ctx) -> list:
        return list(self.requires(ctx) if callable(self.requires) else self.requires)


def skip_missing(sections: list, ctx) -> list[str]:
    """Disable every enabled section whose input files are missing and print a warning for it."""
    warnings = []
    for s in sections:
        if not s.enabled:
            continue
        missing = [str(p) for p in s.needs(ctx) if not Path(p).exists()]
        if missing:
            s.enabled = False
            msg = f"WARNING: skipping section {s.key} ({s.title}), missing {missing}"
            print(msg, file=sys.stderr, flush=True)
            warnings.append(msg)
    return warnings


def song_window(song: str, length_s: float = SONG_LEN_S, phrase_change: bool = True) -> float:
    """Start of the liveliest window. With phrase_change it must contain a groove to travel change.
    The current command files have no phrases, so then the window is only the liveliest one."""
    commands, dt, meta = load_command_file(Path("dance/data/commands/songs") / f"{song}.npz")
    has_phrases = bool((meta.get("groove_info") or {}).get("phrases"))
    need = groove_to_travel_times(meta) if phrase_change and has_phrases else None
    start, _ = liveliest_window(command_energy(commands), dt, length_s, min_start_s=1.0, max_end_s=58.5,
                                must_contain=need)
    return start


def song_segment(ctx: Context, song: str, start: float) -> Segment:
    title, composer = ALL_SONGS[song]
    return Segment(ctx.rec / f"{song}_v4.mp4", ctx.rec / f"{song}_{HUMAN_TAG}.mp4", MUSIC_DIR / f"{song}.wav",
                   f"{title} / {composer}", start, SONG_LEN_S, left_label=HUMAN_LABEL, right_label=ROBOT_LABEL,
                   title_align="left", title_size=SONG_TITLE_SIZE, label_size=LABEL_SIZE, body_y=BODY_Y,
                   label_y=LABEL_Y, subtitle=SONG_SUB)


def clip_segment(video: Path, song: str, offset: float, start: float, length: float, text: str,
                 short_fades: bool = False, subtitle=()) -> Segment:
    """A robot clip over the whole frame with one line of text at the top left (or none) and
    optional subtitle lines at the bottom."""
    fades = dict(fade_in_s=0.05, fade_out_s=0.05) if short_fades else {}
    return Segment(video, None, MUSIC_DIR / f"{song}.wav", text, start, length, robot_offset_s=offset,
                   title_align="left", title_size=CLIP_TEXT_SIZE, full_frame=True, subtitle=tuple(subtitle),
                   **fades)


def highlight_clip(ctx: Context, video, song: str) -> Path:
    return ctx.rec / f"{song}_v4.mp4" if video == "rec" else Path(video)


def b_op(ctx):
    return [ClipSegment(COURSE_DIR / "op.mp4", probe_duration(COURSE_DIR / "op.mp4"), "course opening")]


def b_ed(ctx):
    return [ClipSegment(COURSE_DIR / "ed.mp4", probe_duration(COURSE_DIR / "ed.mp4"), "course ending")]


def b_highlight(ctx):
    """The highlight with the title laid over its bottom for its whole length."""
    band = ctx.slides / "title_band.png"
    title_band(band)
    segs = [clip_segment(highlight_clip(ctx, v, song), song, off, st, ln, "", short_fades=True)
            for v, song, off, st, ln in HIGHLIGHT]
    for seg in segs:
        seg.band_png = band
    return segs


def highlight_needs(ctx) -> list:
    out = []
    for v, song, _, _, _ in HIGHLIGHT:
        out += [highlight_clip(ctx, v, song), MUSIC_DIR / f"{song}.wav"]
    return out


def b_workflow(ctx):
    slide_inference(ctx.slides / "inference.png")
    slide_training(ctx.slides / "training.png")
    return [SlideSegment(ctx.slides / "inference.png", 8.0, "workflow (inference)"),
            SlideSegment(ctx.slides / "training.png", 6.0, "workflow (training)")]


def b_songs(ctx):
    return [song_segment(ctx, song, song_window(song)) for song, _, _ in ctx.songs]


def songs_needs(ctx) -> list:
    out = []
    for song, _, _ in ctx.songs:
        out += [ctx.rec / f"{song}_v4.mp4", ctx.rec / f"{song}_{HUMAN_TAG}.mp4", MUSIC_DIR / f"{song}.wav"]
    return out


def b_rough(ctx):
    return [clip_segment(ROUGH_CLIP, ROUGH_SONG, 0.0, 0.0, CLIP_LEN_S, ROUGH_TEXT,
                         subtitle=ROUGH_SUB)]


def b_roller(ctx):
    return [clip_segment(ROLLER_CLIP, ROLLER_SONG, ROLLER_START_S, ROLLER_START_S, CLIP_LEN_S, ROLLER_TEXT,
                         subtitle=ROLLER_SUB)]


def b_credits(ctx):
    slide_credits(ctx.slides / "credits.png", ctx.songs)
    return [SlideSegment(ctx.slides / "credits.png", 12.0, "credits")]


SECTIONS = [
    Section("op", "コース オープニング", b_op, requires=[COURSE_DIR / "op.mp4"]),
    Section("highlight", "ハイライトとタイトル", b_highlight, requires=highlight_needs),
    Section("workflow", "しくみ (推論時と学習時)", b_workflow),
    Section("songs", "3 曲", b_songs, requires=songs_needs),
    Section("rough", "凸凹の床", b_rough, requires=[ROUGH_CLIP, MUSIC_DIR / f"{ROUGH_SONG}.wav"]),
    Section("roller", "ローラースケート", b_roller, requires=[ROLLER_CLIP, MUSIC_DIR / f"{ROLLER_SONG}.wav"]),
    Section("credits", "クレジット", b_credits),
    Section("ed", "コース エンディング", b_ed, requires=[COURSE_DIR / "ed.mp4"]),
]


def print_table(rows: list[tuple[str, str, float]]) -> float:
    print(f"{'section':10s} {'segment':40s} {'start':>7s} {'dur':>7s}")
    t = 0.0
    for key, title, dur in rows:
        print(f"{key:10s} {(title or '-')[:40]:40s} {t:7.2f} {dur:7.2f}")
        t += dur
    print(f"{'total':10s} {'':40s} {'':7s} {t:7.2f}")
    return t


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rec-dir", type=Path, default=Path("out/rec_v4"))
    ap.add_argument("--songs", nargs=3, default=["swag", "funky_magic", "fallin_for_you"], choices=list(ALL_SONGS))
    ap.add_argument("--slide-dir", type=Path, default=Path("out/final/slides"))
    ap.add_argument("--enable", nargs="*", default=[], help="section keys to switch on")
    ap.add_argument("--disable", nargs="*", default=[], help="section keys to switch off")
    ap.add_argument("--list-sections", action="store_true")
    ap.add_argument("--table-only", action="store_true", help="build the segments and print the table, no encoding")
    ap.add_argument("--out", type=Path, default=Path("out/final/microduck_dance.mp4"))
    a = ap.parse_args()
    global SONGS
    SONGS = [(s, *ALL_SONGS[s]) for s in a.songs]

    keys = [s.key for s in SECTIONS]
    for k in a.enable + a.disable:
        if k not in keys:
            ap.error(f"unknown section {k!r}, choose from {keys}")
    for s in SECTIONS:
        s.enabled = (s.enabled or s.key in a.enable) and s.key not in a.disable
    if a.list_sections:
        for s in SECTIONS:
            print(f"{s.key:10s} {'on ' if s.enabled else 'off'} {s.title}")
        return

    a.slide_dir.mkdir(parents=True, exist_ok=True)
    ctx = Context(a.rec_dir, a.slide_dir, SONGS, SECTIONS)
    skip_missing(SECTIONS, ctx)
    segs, rows = [], []
    for s in SECTIONS:
        if not s.enabled:
            continue
        for seg in s.build(ctx):
            segs.append(seg)
            rows.append((s.key, seg.title, float(seg.duration_s)))
    total = print_table(rows)
    if total > MAX_TOTAL_S:
        raise SystemExit(f"total {total:.1f} s exceeds the {MAX_TOTAL_S:.0f} s limit")
    if a.table_only:
        return
    compose_video(segs, a.out)
    print(json.dumps({"out": str(a.out), "duration_s": round(probe_duration(a.out), 3)}, indent=2))


if __name__ == "__main__":
    main()
