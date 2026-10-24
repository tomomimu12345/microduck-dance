"""Compose dance videos: human stick figure on the left, microduck on the right, title on top, song underneath.

Sync convention: every input of a segment (robot mp4, human mp4, wav) is a
full-length recording that starts at song time 0. A segment with start_s > 0
cuts all of them with the same `-ss start_s -t duration_s`, so the three stay
aligned. Record the robot from frame 0, or set robot_offset_s to the song time at which a
robot clip starts (a clip of one song window).

    uv run python dance/scripts/video/compose_video.py --robot out/swag.mp4 --human out/swag_human.mp4 \
        --audio dance/data/music/opentracks/swag.wav --title swag --start 0 --duration 10 --out out/compose.mp4

Segments are encoded identically and joined with the concat demuxer.

Besides the song Segment there are two more segment kinds, used by make_final_video.py:
- SlideSegment: a still 1920x1080 image with silent audio (title, explanation, credits).
- CompareSegment: two robot recordings side by side (960x540 each) on the same song
  segment, a label under each, and an optional 1920x180 footer image (the eval table).
- ClipSegment: a ready made clip (the course opening and ending) scaled to 1920x1080 with silent audio.
All kinds give 1920x1080, 29.97 fps (the course clips' rate), AAC 48 kHz stereo, so the parts concatenate.
Song and compare segments use the top 800 px (title, panels, footer) and may carry a 1920x280 band image
(`band_png`) below them. The song Segment has layout fields (title position and size, label size, panel
top), subtitle lines at the bottom, and a full_frame mode that scales a robot clip to the whole frame.
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

FPS = "30000/1001"  # 29.97 fps, as the course opening and ending clips
OUT_W, OUT_H, BAND_H = 1920, 1080, 280
HUMAN_W, ROBOT_W, BODY_H, TITLE_H = 640, 1280, 720, 80
CMP_W, CMP_H, FOOTER_H = 960, 540, 180
# IPAexGothic has the Japanese glyphs (and ☆). DejaVu is the fallback for ASCII text.
JA_FONT = Path("/usr/share/fonts/opentype/ipaexfont-gothic/ipaexg.ttf")
FONT = JA_FONT if JA_FONT.exists() else Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
VIDEO_ENC = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-r", str(FPS)]
AAC = ["-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2"]
PCM = ["-c:a", "pcm_s16le", "-ar", "48000", "-ac", "2"]


def _enc(out: Path) -> list[str]:
    """Encoder args. Parts written as .mkv keep PCM audio: AAC parts joined by the concat
    demuxer each bring their own encoder priming, which put the audio 70 to 100 ms
    behind the video in a 7-part join. The join encodes AAC once."""
    return VIDEO_ENC + (PCM if Path(out).suffix == ".mkv" else AAC)


@dataclass
class Segment:
    robot_mp4: Path
    human_mp4: Path | None
    audio_wav: Path
    title: str
    start_s: float
    duration_s: float
    left_label: str | None = None
    right_label: str | None = None
    band_png: Path | None = None
    # Song time at which robot_mp4 starts. A clip recorded from a song window (the roller clip starts
    # at song time 28 s) is cut from start_s - robot_offset_s, the human mp4 and the wav from start_s.
    robot_offset_s: float = 0.0
    # Layout. The defaults are the original layout: title centred in the top 80 px, panels below it,
    # labels inside the top of the panels. title_align "left" puts the title at the top left.
    # body_y is the top of the panels, label_y the top of the labels (None: 14 px into the panels).
    # full_frame scales the robot clip to the whole 1920x1080 frame (no human panel) and draws the
    # title on a dark box over the video.
    title_align: str = "center"
    title_size: int = 44
    label_size: int = 32
    body_y: int = TITLE_H
    label_y: int | None = None
    full_frame: bool = False
    fade_in_s: float = 0.3
    fade_out_s: float = 0.5
    # Subtitle lines drawn centred at the bottom of the frame (on a dark box), last line lowest.
    subtitle: tuple = ()
    subtitle_size: int = 40


@dataclass
class SlideSegment:
    image: Path
    duration_s: float
    title: str = "slide"


@dataclass
class CompareSegment:
    left_mp4: Path
    right_mp4: Path
    audio_wav: Path
    title: str
    left_label: str
    right_label: str
    start_s: float
    duration_s: float
    footer_png: Path | None = None
    band_png: Path | None = None


@dataclass
class ClipSegment:
    """A ready made clip, scaled to 1920x1080 and 29.97 fps, with silent audio (the clips have none)."""
    path: Path
    duration_s: float
    title: str = "clip"


def _escape_drawtext(text: str) -> str:
    for ch in ("\\", ":", "'", "%"):
        text = text.replace(ch, "\\" + ch)
    return text


def _font_opt() -> str:
    return f"fontfile={FONT}" if FONT.exists() else "font=Sans"


def _label(text: str, x_center: str, y: str, size: int = 32) -> str:
    """drawtext filter for a label on a dark box, centered at x_center."""
    return (f"drawtext={_font_opt()}:text='{_escape_drawtext(text)}':fontcolor=white:fontsize={size}:"
            f"box=1:boxcolor=black@0.55:boxborderw=10:x={x_center}-text_w/2:y={y}")


def _subtitles(lines, size: int) -> str:
    """drawtext filters for subtitle lines at the bottom, joined with commas ('' for none)."""
    out = []
    n = len(lines)
    for i, text in enumerate(lines):
        y = OUT_H - 40 - (n - i) * size - (n - i - 1) * 22
        out.append(_label(text, "w/2", str(y), size))
    return ",".join(out)


def _afade(dur: float, fade_in: float = 0.3, fade_out: float = 0.5) -> str:
    """Resample to 48 kHz and fade the cut song in and out, so segment joins do not click."""
    return (f"aresample=48000,afade=t=in:d={fade_in}"
            f",afade=t=out:st={max(dur - fade_out, 0.0):.3f}:d={fade_out}")


def _title(text: str, align: str = "center", size: int = 44) -> str:
    if not text:
        return "null"
    if align == "left":
        return (f"drawtext={_font_opt()}:text='{_escape_drawtext(text)}':fontcolor=white:fontsize={size}:"
                f"box=1:boxcolor=black@0.55:boxborderw=14:x=40:y=36")
    return (f"drawtext={_font_opt()}:text='{_escape_drawtext(text)}':fontcolor=white:fontsize={size}:"
            f"x=(w-text_w)/2:y=({TITLE_H}-text_h)/2")


def build_ffmpeg_cmd(segment, out: Path) -> list[str]:
    """ffmpeg command for one segment of any kind."""
    if isinstance(segment, SlideSegment):
        return _slide_cmd(segment, out)
    if isinstance(segment, ClipSegment):
        return _clip_cmd(segment, out)
    if isinstance(segment, CompareSegment):
        return _compare_cmd(segment, out)
    return _song_cmd(segment, out)


def _slide_cmd(seg: SlideSegment, out: Path) -> list[str]:
    dur = float(seg.duration_s)
    return ["ffmpeg", "-v", "error", "-y",
            "-loop", "1", "-framerate", str(FPS), "-t", str(dur), "-i", str(seg.image),
            "-f", "lavfi", "-t", str(dur), "-i", "anullsrc=r=48000:cl=stereo",
            "-filter_complex", f"[0:v]scale={OUT_W}:{OUT_H},setsar=1,format=yuv420p[v]",
            "-map", "[v]", "-map", "1:a", *_enc(out), "-t", str(dur), str(out)]


def _clip_cmd(seg: ClipSegment, out: Path) -> list[str]:
    dur = float(seg.duration_s)
    vf = f"fps={FPS},scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=decrease,pad={OUT_W}:{OUT_H}:-1:-1,setsar=1,format=yuv420p"
    return ["ffmpeg", "-v", "error", "-y", "-i", str(seg.path),
            "-f", "lavfi", "-t", str(dur), "-i", "anullsrc=r=48000:cl=stereo",
            "-filter_complex", f"[0:v]{vf}[v]",
            "-map", "[v]", "-map", "1:a", *_enc(out), "-t", str(dur), str(out)]


def _compare_cmd(seg: CompareSegment, out: Path) -> list[str]:
    """Two robot clips side by side. Both clips and the wav are cut with the same -ss and -t."""
    start, dur = float(seg.start_s), float(seg.duration_s)
    cmd = ["ffmpeg", "-v", "error", "-y"]
    for p in (seg.left_mp4, seg.right_mp4, seg.audio_wav):
        cmd += ["-ss", str(start), "-t", str(dur), "-i", str(p)]
    width = 2 * CMP_W
    n_in = 3
    panel = f"fps={FPS},scale={CMP_W}:{CMP_H}:force_original_aspect_ratio=decrease,pad={CMP_W}:{CMP_H}:-1:-1"
    parts = [f"[0:v]{panel}[a]", f"[1:v]{panel}[b]"]
    label_y = f"{TITLE_H + 14}"
    chain = (f"[a][b]hstack=inputs=2,pad={OUT_W}:{OUT_H}:0:{TITLE_H}:color=black,{_title(seg.title)},"
             f"{_label(seg.left_label, str(CMP_W // 2), label_y)},{_label(seg.right_label, str(CMP_W + CMP_W // 2), label_y)}")
    if seg.footer_png is not None:
        cmd += ["-loop", "1", "-framerate", str(FPS), "-t", str(dur), "-i", str(seg.footer_png)]
        parts.append(f"{chain}[base]")
        parts.append(f"[{n_in}:v]scale={width}:{FOOTER_H}[foot]")
        parts.append(f"[base][foot]overlay=0:{TITLE_H + CMP_H}:shortest=1[v0]")
        n_in += 1
    else:
        parts.append(f"{chain}[v0]")
    parts.append(_band_stage(cmd, seg.band_png, dur, n_in))
    cmd += ["-filter_complex", ";".join(parts), "-map", "[v]", "-map", "2:a", "-af", _afade(dur),
            *_enc(out), "-t", str(dur), str(out)]
    return cmd


def _band_stage(cmd: list, band_png, dur: float, idx: int) -> str:
    """Filter stage that turns [v0] into [v]: the 1920x280 band image overlaid under the content, or
    nothing. Adds the band input to cmd (it must be the last input added so far)."""
    if band_png is None:
        return "[v0]format=yuv420p[v]"
    cmd += ["-loop", "1", "-framerate", str(FPS), "-t", str(dur), "-i", str(band_png)]
    return f"[v0][{idx}:v]overlay=0:{OUT_H - BAND_H}:shortest=1,format=yuv420p[v]"


def _song_cmd(segment: Segment, out: Path) -> list[str]:
    """Human left, robot right. All inputs are cut with the same -ss and -t."""
    start, dur = float(segment.start_s), float(segment.duration_s)
    cmd = ["ffmpeg", "-v", "error", "-y"]
    inputs = [segment.robot_mp4] + ([segment.human_mp4] if segment.human_mp4 is not None else []) + [segment.audio_wav]
    for i, p in enumerate(inputs):
        ss = start - float(segment.robot_offset_s) if i == 0 else start
        if ss < 0:
            raise ValueError(f"segment {segment.title!r} starts before its robot clip (robot_offset_s)")
        cmd += ["-ss", str(ss), "-t", str(dur), "-i", str(p)]
    if segment.full_frame:
        if segment.human_mp4 is not None:
            raise ValueError("a full_frame segment has no human panel")
        chain = (f"[0:v]fps={FPS},scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=decrease,"
                 f"pad={OUT_W}:{OUT_H}:-1:-1,setsar=1,"
                 f"{_title(segment.title, segment.title_align, segment.title_size)}"
                 f"{',' + _subtitles(segment.subtitle, segment.subtitle_size) if segment.subtitle else ''}[v0]")
        band = _band_stage(cmd, segment.band_png, dur, 2)
        cmd += ["-filter_complex", f"{chain};{band}", "-map", "[v]", "-map", "1:a",
                "-af", _afade(dur, segment.fade_in_s, segment.fade_out_s), *_enc(out), "-t", str(dur), str(out)]
        return cmd
    robot = f"[0:v]fps={FPS},scale={ROBOT_W}:{BODY_H}:force_original_aspect_ratio=decrease,pad={ROBOT_W}:{BODY_H}:-1:-1[r]"
    if segment.human_mp4 is not None:
        human = f"[1:v]fps={FPS},scale={HUMAN_W}:{BODY_H}:force_original_aspect_ratio=decrease,pad={HUMAN_W}:{BODY_H}:-1:-1[h]"
        stack = "[h][r]hstack=inputs=2"
        audio_idx, width = 2, HUMAN_W + ROBOT_W
    else:
        human, stack = None, "[r]null"
        audio_idx, width = 1, ROBOT_W
    x0 = (OUT_W - width) // 2  # a robot-only panel is centred
    labels = ""
    label_y = str(segment.label_y if segment.label_y is not None else segment.body_y + 14)
    size = segment.label_size
    if segment.left_label and segment.human_mp4 is not None:
        labels += "," + _label(segment.left_label, str(HUMAN_W // 2), label_y, size)
    if segment.right_label:
        x = HUMAN_W + ROBOT_W // 2 if segment.human_mp4 is not None else x0 + ROBOT_W // 2
        labels += "," + _label(segment.right_label, str(x), label_y, size)
    chain = (f"{stack},pad={OUT_W}:{OUT_H}:{x0}:{segment.body_y}:color=black,"
             f"{_title(segment.title, segment.title_align, segment.title_size)}{labels}"
             f"{',' + _subtitles(segment.subtitle, segment.subtitle_size) if segment.subtitle else ''}[v0]")
    band = _band_stage(cmd, segment.band_png, dur, audio_idx + 1)
    graph = ";".join(f for f in (robot, human, chain, band) if f)
    cmd += ["-filter_complex", graph, "-map", "[v]", "-map", f"{audio_idx}:a",
            "-af", _afade(dur, segment.fade_in_s, segment.fade_out_s), *_enc(out), "-t", str(dur), str(out)]
    return cmd


def probe_duration(path: Path) -> float:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                       check=True, capture_output=True, text=True)
    return float(r.stdout.strip())


def compose_video(segments: list, out: Path) -> None:
    if not segments:
        raise ValueError("no segments")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        parts = []
        for i, seg in enumerate(segments):
            part = Path(tmp) / f"seg{i:02d}.mkv"
            subprocess.run(build_ffmpeg_cmd(seg, part), check=True)
            got = probe_duration(part)
            if abs(got - seg.duration_s) > 0.1:
                raise RuntimeError(f"segment {i} ({seg.title}) is {got:.2f} s, expected {seg.duration_s:.2f} s "
                                   "(an input is shorter than start_s + duration_s?)")
            parts.append(part)
        lst = Path(tmp) / "list.txt"
        lst.write_text("".join(f"file '{p}'\n" for p in parts))
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
                        "-c:v", "copy", *AAC, "-movflags", "+faststart", str(out)], check=True)
    want = sum(float(s.duration_s) for s in segments)
    got = probe_duration(out)
    if abs(got - want) > 0.1 * len(segments):
        raise RuntimeError(f"joined video is {got:.2f} s, the segments sum to {want:.2f} s")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", type=Path, required=True)
    ap.add_argument("--human", type=Path, default=None)
    ap.add_argument("--audio", type=Path, required=True)
    ap.add_argument("--title", required=True)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--duration", type=float, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    compose_video([Segment(a.robot, a.human, a.audio, a.title, a.start, a.duration)], a.out)
    print(f"{a.out}: {probe_duration(a.out):.3f} s")


if __name__ == "__main__":
    main()
