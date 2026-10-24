"""Build the submission video (1920x1080, 29.97 fps, AAC encoded once, at most 180 s).

The video is a list of sections (SECTIONS below). A section is enabled or not, builds one or more
segments, and may add credit lines. Disabled sections are placeholders for material that is not ready
(ball, tricks). Switch sections with --enable / --disable, list them with --list-sections.

    opening clip (course asset, 3 s) -> title -> microduck intro -> pipeline -> three songs
    (human left, robot right) -> comparison with the walking-only policy -> rough floor -> roller
    skates -> credits -> ending clip (course asset, 3 s)

Inputs are full-song recordings that start at song time 0, all with the lagged camera
(play_dance.py --cam-follow lagged --dump-states):
    out/rec_v4/<song>_v4.mp4, out/rec_v4/<song>_base5000.mp4  (dance_v4 and base5000)
    out/rec_v4/<song>_human3d.mp4                              (render_human_3d.py, EDGE motions as a
                                                                3D mannequin with a head visor)
    out/rough/t31/funky_magic_video_15s_wideintro_rough.mp4    (dance_rough on the video terrain, 1.5 cm
                                                                steps, first 15 s, 2 s wide intro)
    out/roller/video2/roller002_funky_magic_15s.mp4            (roller_dance, Funky Magic 28 to 43 s)
    out/course_assets/op.mp4, ed.mp4                           (course opening and ending, no audio)
and the eval json files out/eval_v4/<policy>_<song>.json (play_dance.py --eval).
Nothing under out/ is committed. The course opening and ending clips are course material and are not in
the repository. A section whose input files are missing is skipped with a warning, so a fresh clone
builds a shorter video instead of stopping.

    uv run python dance/scripts/video/make_final_video.py --out out/final/microduck_dance.mp4

The song segments are the liveliest 25 s of each song by retargeted command energy (pick_segment.py)
among the windows that contain a change from a groove phrase to a travel phrase. The current command
files have no phrases, so the window is the liveliest one.
Slides and bands are drawn with Pillow in IPAexGothic.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compose_video import (  # noqa: E402
    JA_FONT, BAND_H, CompareSegment, ClipSegment, Segment, SlideSegment, compose_video, probe_duration)
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
POLICY = "dance_v4"  # the policy shown in the song segments
POLICY_LABEL = "dance_v4: 計10000回 (歩行5000 + ダンス3000 + 2000)"
BASE_LABEL = "base5000: 歩行タスクのみ、計5000回"
HUMAN_TAG = "human3d"  # <song>_human3d.mp4 (3D mannequin); "human" is the older 2D stick figure
HUMAN_LABEL = "生成した人の動き (EDGE, 3D)"
MUSIC_DIR = Path("dance/data/music/opentracks")
COURSE_DIR = Path("out/course_assets")
INTRO_FIGURE = Path("dance/docs/figures/microduck.png")
TITLE_LINE = "MuJoCo シミュレーション  /  Physical AI 応用1 最終課題"

# Credit lines of the bands under the video and of the end card. Names and licenses are from
# dance/data/CREDITS.md and the rights audit.
DANCE_CREDIT = "ダンス生成 EDGE (Tseng et al., CVPR 2023)。EDGE の学習データは AIST++ (Li et al., ICCV 2021)。音楽特徴量は OpenAI Jukebox で生成。"
ROBOT_CREDIT = "ロボットの 3D モデル Microduck, Pollen Robotics (CC BY-NC-SA 4.0)。非営利の授業課題。"
ROLLER_CREDIT = "roller.onnx: Pollen Robotics microduck-policies (Apache-2.0), 改変して使用"
MESH_LICENSE_URL = "https://creativecommons.org/licenses/by-nc-sa/4.0/"
MESH_SOURCE_URL = "https://github.com/pollen-robotics/microduck_rl"

# Rough floor section. The policy and the fall rate are from the Task 31 eval on the video terrain
# (DR on, 16 envs, 60 s, seeds 0 1 2, mean of swag, funky_magic and fallin_for_you):
# out/t31/eval/table3.json. The clip is the fixed Task 30 setup (seed 0, hardest stairs tile, DR off).
ROUGH_POLICY = "dance_rough"  # rough004 +750, logs/rsl_rl/dance_v3_rough_mix/..._rough004/model_10750.pt
ROUGH_CLIP = Path("out/rough/t31/funky_magic_video_15s_wideintro_rough.mp4")
ROUGH_FALLS_PER_MIN = 1.264  # 3 seeds x 3 songs mean (dance_v4: 2.812)
# Roller section: roller002 (logs/rsl_rl/roller_dance/2026-10-23_18-50-41_roller002b/model_3399.pt).
# The clip starts at song time 28 s. Falls per minute on funky_magic, DR on, seeds 0 1 2 (Task 29).
ROLLER_POLICY = "roller_dance"
ROLLER_SONG = "funky_magic"
ROLLER_CLIP = Path("out/roller/video2/roller002_funky_magic_15s.mp4")
ROLLER_START_S = 28.0
ROLLER_FALLS_PER_MIN = 0.083


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(JA_FONT), size)


def centered(d: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, size: int, fill=FG) -> None:
    d.text(xy, text, font=font(size), fill=fill, anchor="mm")


# ---------------------------------------------------------------- slides and images


def slide_title(path: Path) -> None:
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    centered(d, (W / 2, 430), "microduck が曲に合わせて踊る", 96)
    centered(d, (W / 2, 560), TITLE_LINE, 46, DIM)
    d.line([(W / 2 - 300, 640), (W / 2 + 300, 640)], fill=ACCENT, width=4)
    centered(d, (W / 2, 720), "映像はすべて等速 (x1) です", 36, DIM)
    im.save(path)


def slide_intro(path: Path) -> None:
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    centered(d, (W / 2, 90), "microduck とは", 64)
    fig = Image.open(INTRO_FIGURE).convert("RGB")
    scale = 1.4
    fig = fig.resize((round(fig.width * scale), round(fig.height * scale)), Image.LANCZOS)
    im.paste(fig, ((W - fig.width) // 2, 170))
    y = 170 + fig.height + 70
    centered(d, (W / 2, y), "身長約 25 cm の小型の二足歩行ロボット。サーボは 14 個で、腕はない。", 40)
    centered(d, (W / 2, y + 70), "物をつかめないので、ペットのように踊って楽しませる。", 40)
    im.save(path)


def slide_pipeline(path: Path) -> None:
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    centered(d, (W / 2, 110), "しくみ", 64)
    steps = [
        ("曲", ["音楽ファイル (wav)"]),
        ("EDGE", ["曲から人のダンスを生成", "AIST++ で学習した", "生成モデル", "(推論のみ)"]),
        ("リターゲット", ["13次元コマンドに変換", "歩行 3 / 頭 4 / 胴体 6", "人の向きと動きに",
                          "合わせ、拍に合わせた", "動きを少し足す", "(自作のルール)"]),
        ("RL policy (追従)", ["コマンドを追従する policy", "自分で学習 (mjlab, PPO)"]),
        ("MuJoCo", ["microduck が踊る", "物理シミュレーション"]),
    ]
    bw, bh, gap, top = 320, 440, 60, 220
    x0 = (W - (5 * bw + 4 * gap)) / 2
    for i, (head, lines) in enumerate(steps):
        x = x0 + i * (bw + gap)
        own = i == 3
        d.rounded_rectangle([x, top, x + bw, top + bh], radius=18, fill=BOX,
                            outline=ACCENT if own else DIM, width=5 if own else 2)
        centered(d, (x + bw / 2, top + 64), head, 36 if own else 40, ACCENT if own else FG)
        for j, line in enumerate(lines):
            centered(d, (x + bw / 2, top + 140 + j * 44), line, 24, FG)
        if i < 4:
            ax = x + bw + 8
            d.polygon([(ax, top + bh / 2 - 18), (ax + gap - 16, top + bh / 2), (ax, top + bh / 2 + 18)], fill=ACCENT)
    centered(d, (W / 2, 770), "黄色の枠が自分で学習した部分です", 36, ACCENT)
    centered(d, (W / 2, 850), "この後の画面  左は EDGE が生成した人の動き、右はそれを追従する microduck", 32, DIM)
    centered(d, (W / 2, 910), "カメラはロボットを少し遅れて追うので、移動すると画面の中で位置が変わります", 28, DIM)
    im.save(path)


def eval_table(eval_dir: Path, songs=None) -> dict:
    """{(policy, song): (falls mean, SE over seeds)}. Falls per env per 60 s episode, DR on."""
    out = {}
    for pol in ("base5000", POLICY):
        for song, _, _ in songs or SONGS:
            d = json.loads((eval_dir / f"{pol}_{song}.json").read_text())
            f = np.array([s["mean"]["falls"] for s in d["per_seed"]])
            assert d["steps"] * d["dt"] == 60.0 and d["num_envs"] == 16 and len(f) == 3
            out[(pol, song)] = (float(f.mean()), float(f.std(ddof=1) / np.sqrt(len(f))))
    return out


def footer_png(path: Path, table: dict) -> None:
    im = Image.new("RGB", (W, 180), BG)
    d = ImageDraw.Draw(im)
    centered(d, (W / 2, 24), "ランダム化 (DR) ありの評価: 1分あたりの転倒回数 (3 seeds × 16 envs、平均 ± 標準誤差)", 28, DIM)
    cols = [360, 860, 1260, 1660]
    centered(d, (cols[0], 66), "policy", 30, DIM)
    for c, (_, title, _) in zip(cols[1:], SONGS):
        centered(d, (c, 66), title, 30, DIM)
    for r, (pol, name) in enumerate([("base5000", "base5000 (歩行のみ)"), (POLICY, f"{POLICY} (ダンス)")]):
        y = 106 + r * 42
        color = ACCENT if pol == POLICY else FG
        centered(d, (cols[0], y), name, 30, color)
        for c, (song, _, _) in zip(cols[1:], SONGS):
            m, se = table[(pol, song)]
            centered(d, (c, y), f"{m:.2f} ± {se:.2f}", 30, color)
    im.save(path)


def band_png(path: Path, lines: list[str]) -> None:
    """1920x280 band under a song or comparison segment. Lines are drawn top down, the first brightest."""
    im = Image.new("RGB", (W, BAND_H), BG)
    d = ImageDraw.Draw(im)
    d.line([(0, 1), (W, 1)], fill=BOX, width=2)
    y = 44
    for i, text in enumerate(lines):
        centered(d, (W / 2, y), text, 34 if i == 0 else 28, FG if i == 0 else DIM)
        y += 56 if i == 0 else 46
    im.save(path)


def slide_credits(path: Path, songs, extra: list[str]) -> None:
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    centered(d, (W / 2, 80), "クレジット", 56)
    lines = [("Music (OpenTracks, formerly DOVA-SYNDROME, https://opentracks.com/)", 28, DIM),
             (" / ".join(f'"{t}" by {c}' for _, t, c in songs), 32, FG),
             ("", 4, FG),
             ("ダンス生成  EDGE (Tseng et al., CVPR 2023, MIT License)。推論のみ", 30, FG),
             ("音楽特徴量  OpenAI Jukebox を使って生成 (Dhariwal et al., 2020, 非営利ライセンス)", 30, FG),
             ("EDGE の学習データ  AIST++ (Li et al., ICCV 2021, CC BY 4.0)", 30, FG),
             ("AIST Dance Video Database (Tsuchida et al., ISMIR 2019)", 30, FG),
             ("ロボットのモデルとメッシュ  Microduck, Pollen Robotics (CC BY-NC-SA 4.0、メッシュは改変なし)", 30, FG),
             (f"{MESH_LICENSE_URL}   {MESH_SOURCE_URL}", 26, DIM),
             ("学習環境  microduck-playground (Apache-2.0)", 30, FG)]
    lines += [(t, 30, FG) for t in extra]
    lines += [("シミュレーション  MuJoCo, mjlab, rsl_rl", 30, FG),
              ("", 4, FG),
              ("非営利の授業課題です。", 28, DIM)]
    y = 170
    for text, size, color in lines:
        if text:
            centered(d, (W / 2, y), text, size, color)
        y += size + 26
    im.save(path)


# ---------------------------------------------------------------- sections


@dataclass
class Context:
    rec: Path
    eval_dir: Path
    slides: Path
    songs: list
    compare_song: str
    compare_start: float
    compare_length: float
    compare_note: str
    sections: list = field(default_factory=list)  # the end card reads their credit lines


@dataclass
class Section:
    """One part of the video. `build(ctx)` returns the segments. `requires` are files the section needs:
    a list of paths, or a function of the context that returns one. A section with a missing file is
    skipped (skip_missing). `credits` are extra lines for the end card (only when enabled)."""
    key: str
    title: str
    build: callable
    enabled: bool = True
    requires: object = field(default_factory=list)
    credits: list = field(default_factory=list)

    def needs(self, ctx) -> list:
        return list(self.requires(ctx) if callable(self.requires) else self.requires)


def skip_missing(sections: list, ctx) -> list[str]:
    """Disable every enabled section whose input files are missing and print a warning for it.
    Returns the warnings. Run it before the credits section is built, so a skipped section adds no
    credit lines."""
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
    The current command files use the beat pattern groove and have no phrases (groove_info.phrases is
    None), so then the window is only the liveliest one."""
    commands, dt, meta = load_command_file(Path("dance/data/commands/songs") / f"{song}.npz")
    has_phrases = bool((meta.get("groove_info") or {}).get("phrases"))
    need = groove_to_travel_times(meta) if phrase_change and has_phrases else None
    start, _ = liveliest_window(command_energy(commands), dt, length_s, min_start_s=1.0, max_end_s=58.5,
                                must_contain=need)
    return start


def song_band(ctx: Context, name: str, out: Path, note: str | None = None, extra=()) -> Path:
    title, composer = ALL_SONGS[name]
    lines = [f'音楽 "{title}" by {composer} (OpenTracks)', DANCE_CREDIT, ROBOT_CREDIT, *extra]
    if note:
        lines.insert(1, note)
    band_png(out, lines)
    return out


def clip_section(key: str, title: str, video: Path, song: str, start: float, length: float, label: str,
                 note: str, requires=(), credits=(), enabled=False, robot_offset_s: float = 0.0,
                 band_extra=()) -> Section:
    """A recorded robot clip of one song window with the mannequin on the left and a band note.
    robot_offset_s is the song time at which the clip starts (0 for a clip recorded from the song start).
    band_extra lines go at the bottom of the band."""
    def needs(ctx: Context) -> list:
        return [video, ctx.rec / f"{song}_{HUMAN_TAG}.mp4", MUSIC_DIR / f"{song}.wav", *requires]

    def build(ctx: Context) -> list:
        band = song_band(ctx, song, ctx.slides / f"band_{key}.png", note, list(band_extra))
        st, sc = ALL_SONGS[song]
        return [Segment(video, ctx.rec / f"{song}_{HUMAN_TAG}.mp4", MUSIC_DIR / f"{song}.wav",
                        f"{title}   {st}  ({sc})   {start:.1f} 〜 {start + length:.1f} 秒",
                        start, length, left_label=HUMAN_LABEL, right_label=label, band_png=band,
                        robot_offset_s=robot_offset_s)]
    return Section(key, title, build, enabled, needs, list(credits))


def b_op(ctx):
    return [ClipSegment(COURSE_DIR / "op.mp4", probe_duration(COURSE_DIR / "op.mp4"), "course opening")]


def b_ed(ctx):
    return [ClipSegment(COURSE_DIR / "ed.mp4", probe_duration(COURSE_DIR / "ed.mp4"), "course ending")]


def b_title(ctx):
    slide_title(ctx.slides / "title.png")
    return [SlideSegment(ctx.slides / "title.png", 4.0, "title")]


def b_intro(ctx):
    slide_intro(ctx.slides / "intro.png")
    return [SlideSegment(ctx.slides / "intro.png", 7.0, "microduck intro")]


def b_pipeline(ctx):
    slide_pipeline(ctx.slides / "pipeline.png")
    return [SlideSegment(ctx.slides / "pipeline.png", 10.0, "pipeline")]


def b_songs(ctx):
    segs = []
    for i, (song, title, composer) in enumerate(ctx.songs):
        start = song_window(song)
        band = song_band(ctx, song, ctx.slides / f"band_{song}.png")
        segs.append(Segment(ctx.rec / f"{song}_v4.mp4", ctx.rec / f"{song}_{HUMAN_TAG}.mp4", MUSIC_DIR / f"{song}.wav",
                            f"{i + 1}/{len(ctx.songs)}   {title}  ({composer})   {start:.1f} 〜 {start + SONG_LEN_S:.1f} 秒",
                            start, SONG_LEN_S, left_label=HUMAN_LABEL,
                            right_label=f"microduck (policy: {POLICY})", band_png=band))
    return segs


def songs_needs(ctx) -> list:
    out = []
    for song, _, _ in ctx.songs:
        out += [ctx.rec / f"{song}_v4.mp4", ctx.rec / f"{song}_{HUMAN_TAG}.mp4", MUSIC_DIR / f"{song}.wav"]
    return out


def compare_needs(ctx) -> list:
    song = ctx.compare_song
    out = [ctx.rec / f"{song}_base5000.mp4", ctx.rec / f"{song}_v4.mp4", MUSIC_DIR / f"{song}.wav"]
    for pol in ("base5000", POLICY):
        out += [ctx.eval_dir / f"{pol}_{s}.json" for s, _, _ in ctx.songs]
    return out


def b_compare(ctx):
    footer_png(ctx.slides / "eval_table.png", eval_table(ctx.eval_dir, ctx.songs))
    song = ctx.compare_song
    band = song_band(ctx, song, ctx.slides / "band_compare.png")
    return [CompareSegment(ctx.rec / f"{song}_base5000.mp4", ctx.rec / f"{song}_v4.mp4", MUSIC_DIR / f"{song}.wav",
                           f"比較   {ALL_SONGS[song][0]}   {ctx.compare_start:.1f} 〜 "
                           f"{ctx.compare_start + ctx.compare_length:.1f} 秒   {ctx.compare_note}",
                           BASE_LABEL, POLICY_LABEL, ctx.compare_start, ctx.compare_length,
                           ctx.slides / "eval_table.png", band)]


def b_credits(ctx):
    extra = []
    for s in ctx.sections:
        if s.enabled:
            extra += s.credits
    slide_credits(ctx.slides / "credits.png", ctx.songs, extra)
    return [SlideSegment(ctx.slides / "credits.png", 12.0, "credits")]


SECTIONS = [
    Section("op", "コース オープニング", b_op, requires=[COURSE_DIR / "op.mp4"]),
    Section("title", "タイトル", b_title),
    Section("intro", "microduck とは", b_intro, requires=[INTRO_FIGURE]),
    Section("pipeline", "しくみ", b_pipeline),
    Section("songs", "3 曲", b_songs, requires=songs_needs),
    Section("compare", "比較 (歩行のみとの違い)", b_compare, requires=compare_needs),
    clip_section("rough", "凸凹の床で踊る", ROUGH_CLIP, "funky_magic", 0.0, 15.0,
                 f"凸凹 (段差 1.5 cm) / policy: {ROUGH_POLICY} / 転倒 {ROUGH_FALLS_PER_MIN:.1f} 回/分",
                 f"床は段差 1.5 cm の階段と、高さ ±1 cm の格子。{ROUGH_POLICY} は {POLICY} を凸凹の床と"
                 "平らな床で 750 回追加学習した policy",
                 enabled=True),
    clip_section("roller", "ローラースケートで踊る", ROLLER_CLIP, ROLLER_SONG, ROLLER_START_S, 15.0,
                 f"ローラースケート / policy: {ROLLER_POLICY} / 転倒 {ROLLER_FALLS_PER_MIN:.2f} 回/分",
                 f"足に車輪を付けたモデル。{ROLLER_POLICY} は roller.onnx を元に、滑って移動しながら踊るよう追加学習した policy",
                 credits=[ROLLER_CREDIT], enabled=True, robot_offset_s=ROLLER_START_S,
                 band_extra=[ROLLER_CREDIT]),
    # Placeholders. Enable with --enable ball / --enable tricks once the clip is final. Set the song and
    # the window to the ones the clip was recorded with.
    clip_section("ball", "ボールの上で踊る", Path("out/ball/b11_swag_10s.mp4"), "swag", 0.0, 10.0,
                 "microduck (policy: b11 を元にしたボール用)  ボールの上",
                 "ボールの上で踊る。policy は HannesVonEssen/microduck-basketball の b11 を元にしている。",
                 credits=["ボール上の policy  basketball b11, HannesVonEssen/microduck-basketball (Apache-2.0)"]),
    clip_section("tricks", "芸を挟む", Path("logs/tricks/clips/dance_roulade_dance_15s.mp4"), "swag", 0.0, 15.0,
                 "microduck (policy: Pollen 公式 roulade と dance_v4)",
                 "ダンスの途中で前転 (roulade) を挟み、そのまま踊りに戻る。",
                 credits=["芸の policy  pollen-robotics/microduck-policies (Apache-2.0)"]),
    Section("credits", "クレジット", b_credits),
    Section("ed", "コース エンディング", b_ed, requires=[COURSE_DIR / "ed.mp4"]),
]


def print_table(rows: list[tuple[str, str, float]]) -> float:
    print(f"{'section':10s} {'segment':64s} {'start':>7s} {'dur':>7s}")
    t = 0.0
    for key, title, dur in rows:
        print(f"{key:10s} {title[:64]:64s} {t:7.2f} {dur:7.2f}")
        t += dur
    print(f"{'total':10s} {'':64s} {'':7s} {t:7.2f}")
    return t


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rec-dir", type=Path, default=Path("out/rec_v4"))
    ap.add_argument("--eval-dir", type=Path, default=Path("out/eval_v4"))
    ap.add_argument("--songs", nargs=3, default=["swag", "funky_magic", "fallin_for_you"], choices=list(ALL_SONGS))
    ap.add_argument("--slide-dir", type=Path, default=Path("out/final/slides"))
    ap.add_argument("--compare-song", default="fallin_for_you")
    ap.add_argument("--compare-start", type=float, default=20.0)
    ap.add_argument("--compare-length", type=float, default=25.0)
    ap.add_argument("--compare-note", default="(この録画は DR なし)",
                    help="shown after the comparison title, e.g. whether the recorded robots fell")
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
    ctx = Context(a.rec_dir, a.eval_dir, a.slide_dir, SONGS, a.compare_song, a.compare_start, a.compare_length,
                  a.compare_note)
    ctx.sections = SECTIONS
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
