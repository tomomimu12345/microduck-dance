"""Tests for dance/scripts/video/compose_video.py and render_human.py (CPU only; ffmpeg ones skip without ffmpeg)."""

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_DIR = Path(__file__).resolve().parents[1] / "dance" / "scripts" / "video"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


compose_video_mod = _load("compose_video")
render_human = _load("render_human")
Segment = compose_video_mod.Segment
build_ffmpeg_cmd = compose_video_mod.build_ffmpeg_cmd
compose_video = compose_video_mod.compose_video
probe_duration = compose_video_mod.probe_duration

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="ffmpeg not available"
)


def test_ffmpeg_cmd_trims_audio_to_segment():
    seg = Segment(robot_mp4=Path("r.mp4"), human_mp4=Path("h.mp4"), audio_wav=Path("a.wav"),
                  title="Song A", start_s=5.0, duration_s=30.0)
    cmd = build_ffmpeg_cmd(seg, Path("o.mp4"))
    assert "-t" in cmd and cmd[cmd.index("-t") + 1] == "30.0"
    assert "-ss" in cmd and "5.0" in cmd


def test_all_inputs_are_trimmed_from_the_same_song_time():
    seg = Segment(robot_mp4=Path("r.mp4"), human_mp4=Path("h.mp4"), audio_wav=Path("a.wav"),
                  title="x", start_s=5.0, duration_s=30.0)
    cmd = build_ffmpeg_cmd(seg, Path("o.mp4"))
    for name in ("r.mp4", "h.mp4", "a.wav"):
        i = cmd.index(name)
        assert cmd[i - 1] == "-i"
        assert cmd[i - 5:i - 3] == ["-ss", "5.0"], name
        assert cmd[i - 3:i - 1] == ["-t", "30.0"], name


def test_robot_clip_with_offset_is_cut_from_its_own_start():
    seg = Segment(robot_mp4=Path("r.mp4"), human_mp4=Path("h.mp4"), audio_wav=Path("a.wav"),
                  title="x", start_s=28.0, duration_s=15.0, robot_offset_s=28.0)
    cmd = build_ffmpeg_cmd(seg, Path("o.mp4"))
    for name, ss in (("r.mp4", "0.0"), ("h.mp4", "28.0"), ("a.wav", "28.0")):
        i = cmd.index(name)
        assert cmd[i - 5:i - 3] == ["-ss", ss], name
    bad = Segment(robot_mp4=Path("r.mp4"), human_mp4=None, audio_wav=Path("a.wav"),
                  title="x", start_s=10.0, duration_s=5.0, robot_offset_s=28.0)
    with pytest.raises(ValueError):
        build_ffmpeg_cmd(bad, Path("o.mp4"))


def test_robot_only_segment_has_no_human_input():
    seg = Segment(robot_mp4=Path("r.mp4"), human_mp4=None, audio_wav=Path("a.wav"),
                  title="x", start_s=0.0, duration_s=3.0)
    cmd = build_ffmpeg_cmd(seg, Path("o.mp4"))
    assert cmd.count("-i") == 2


def test_human_pose_projection_faces_camera_and_floor_is_lowest_point():
    pose = np.zeros((2, 24, 3))
    pose[:, :, 2] = np.linspace(-0.9, 0.5, 24)
    xy, floor = render_human.project(pose)
    assert xy.shape == (2, 24, 2)
    assert floor == pytest.approx(-0.9)


@needs_ffmpeg
def test_composed_length_matches_audio(tmp_path):
    def run(*a):
        subprocess.run(["ffmpeg", "-v", "error", "-y", *a], check=True)

    robot = tmp_path / "robot.mp4"
    human = tmp_path / "human.mp4"
    wav = tmp_path / "a.wav"
    run("-f", "lavfi", "-i", "color=c=blue:s=1280x720:r=30:d=4", "-pix_fmt", "yuv420p", str(robot))
    run("-f", "lavfi", "-i", "color=c=white:s=640x720:r=30:d=4", "-pix_fmt", "yuv420p", str(human))
    run("-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100:duration=4", str(wav))
    out = tmp_path / "out.mp4"
    compose_video([Segment(robot, human, wav, "Song", start_s=0.0, duration_s=3.0)], out)
    assert abs(probe_duration(out) - 3.0) <= 0.1
    # nonzero start, and two segments concatenated
    out2 = tmp_path / "out2.mp4"
    segs = [Segment(robot, human, wav, "A", start_s=0.5, duration_s=2.0),
            Segment(robot, None, wav, "B", start_s=1.0, duration_s=2.0)]
    compose_video(segs, out2)
    assert abs(probe_duration(out2) - 4.0) <= 0.1


SlideSegment = compose_video_mod.SlideSegment
CompareSegment = compose_video_mod.CompareSegment
ClipSegment = compose_video_mod.ClipSegment
pick_segment = _load("pick_segment")


def test_compare_cuts_both_robots_and_audio_from_the_same_song_time():
    seg = CompareSegment(left_mp4=Path("base.mp4"), right_mp4=Path("v2.mp4"), audio_wav=Path("a.wav"),
                         title="t", left_label="base", right_label="v2", start_s=12.5, duration_s=25.0,
                         footer_png=Path("table.png"))
    cmd = build_ffmpeg_cmd(seg, Path("o.mp4"))
    for name in ("base.mp4", "v2.mp4", "a.wav"):
        i = cmd.index(name)
        assert cmd[i - 5:i - 1] == ["-ss", "12.5", "-t", "25.0"], name
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert "text='base'" in graph and "text='v2'" in graph and "overlay=" in graph
    assert cmd[cmd.index("-map", cmd.index("-map") + 1) + 1] == "2:a"


def test_slide_is_silent_still_of_given_length():
    cmd = build_ffmpeg_cmd(SlideSegment(Path("s.png"), 7.0), Path("o.mp4"))
    assert "-loop" in cmd and "anullsrc=r=48000:cl=stereo" in cmd
    assert cmd[-3:-1] == ["-t", "7.0"]


def test_song_labels_are_drawn_on_both_panels():
    seg = Segment(robot_mp4=Path("r.mp4"), human_mp4=Path("h.mp4"), audio_wav=Path("a.wav"),
                  title="x", start_s=0.0, duration_s=3.0, left_label="human", right_label="robot")
    graph = build_ffmpeg_cmd(seg, Path("o.mp4"))
    graph = graph[graph.index("-filter_complex") + 1]
    assert "text='human'" in graph and "text='robot'" in graph


def test_command_energy_skips_zero_range_columns_and_scales_by_half_range():
    cmd = np.zeros((2, 13))
    cmd[0, 0] = 0.4   # vx at its limit: energy 1
    cmd[1, 7] = 5.0   # body_x has range 0: ignored
    e = pick_segment.command_energy(cmd)
    assert e[0] == pytest.approx(1.0)
    assert e[1] == pytest.approx(0.0)


def test_liveliest_window_finds_the_burst_within_bounds():
    dt = 0.02
    energy = np.zeros(3000)                  # 60 s
    energy[int(40 / dt):int(50 / dt)] = 1.0  # burst at 40 to 50 s
    start, mean = pick_segment.liveliest_window(energy, dt, 10.0)
    assert start == pytest.approx(40.0)
    assert mean == pytest.approx(1.0)
    start, _ = pick_segment.liveliest_window(energy, dt, 10.0, max_end_s=45.0)
    assert start + 10.0 <= 45.0 + 1e-9
    with pytest.raises(ValueError):
        pick_segment.liveliest_window(energy, dt, 30.0, min_start_s=40.0, max_end_s=60.0)


@needs_ffmpeg
def test_slide_compare_and_song_segments_concatenate(tmp_path):
    from PIL import Image

    def run(*a):
        subprocess.run(["ffmpeg", "-v", "error", "-y", *a], check=True)

    robot = tmp_path / "robot.mp4"
    human = tmp_path / "human.mp4"
    wav = tmp_path / "a.wav"
    run("-f", "lavfi", "-i", "color=c=blue:s=1280x720:r=30:d=4", "-pix_fmt", "yuv420p", str(robot))
    run("-f", "lavfi", "-i", "color=c=white:s=640x720:r=30:d=4", "-pix_fmt", "yuv420p", str(human))
    run("-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100:duration=4", str(wav))
    slide = tmp_path / "slide.png"
    Image.new("RGB", (1920, 1080), (20, 20, 40)).save(slide)
    footer = tmp_path / "footer.png"
    Image.new("RGB", (1920, 180), (0, 0, 0)).save(footer)
    band = tmp_path / "band.png"
    Image.new("RGB", (1920, 280), (10, 10, 10)).save(band)
    clip = tmp_path / "clip.mp4"
    run("-f", "lavfi", "-i", "color=c=red:s=1920x1080:r=30000/1001:d=1.5", "-pix_fmt", "yuv420p", str(clip))
    segs = [SlideSegment(slide, 1.5),
            Segment(robot, human, wav, "曲 ☆", 0.5, 2.0, left_label="人", right_label="ロボット", band_png=band),
            CompareSegment(robot, robot, wav, "比較", "base", "v2", 1.0, 2.0, footer, band),
            Segment(robot, None, wav, "ロボットのみ", 0.5, 1.0, right_label="ロボット"),
            ClipSegment(clip, 1.5)]
    out = tmp_path / "all.mp4"
    compose_video(segs, out)
    assert abs(probe_duration(out) - 8.0) <= 0.2
    streams = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width,height,r_frame_rate",
                              "-of", "csv=p=0", str(out)], check=True, capture_output=True, text=True).stdout
    assert "video,1920,1080,30000/1001" in streams and "audio" in streams


@needs_ffmpeg
def test_audio_stays_aligned_after_joining_several_parts(tmp_path):
    """A click at song time 3.0 s must land at the expected output time after a multi-part join."""
    from PIL import Image

    def run(*a):
        subprocess.run(["ffmpeg", "-v", "error", "-y", *a], check=True)

    sr = 48000
    song = np.zeros(5 * sr, dtype=np.int16)
    song[3 * sr:3 * sr + 480] = 20000  # 10 ms click at 3.0 s
    wav = tmp_path / "click.wav"
    raw = tmp_path / "click.raw"
    song.tofile(raw)
    run("-f", "s16le", "-ar", str(sr), "-ac", "1", "-i", str(raw), str(wav))
    robot = tmp_path / "robot.mp4"
    run("-f", "lavfi", "-i", "color=c=blue:s=1280x720:r=30:d=5", "-pix_fmt", "yuv420p", str(robot))
    slide = tmp_path / "slide.png"
    Image.new("RGB", (1920, 800)).save(slide)
    segs = [SlideSegment(slide, 1.0), SlideSegment(slide, 1.0),
            Segment(robot, robot, wav, "a", 0.5, 1.0), Segment(robot, robot, wav, "b", 2.0, 2.0)]
    out = tmp_path / "out.mp4"
    compose_video(segs, out)
    pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", str(out), "-ac", "1", "-ar", str(sr), "-f", "s16le", "-"],
                         check=True, capture_output=True).stdout
    a = np.abs(np.frombuffer(pcm, dtype=np.int16).astype(float))
    onset = np.flatnonzero(a > 5000)[0] / sr
    # The click is 1.0 s into the last segment, which starts at 1 + 1 + 1 = 3.0 s.
    assert onset == pytest.approx(4.0, abs=0.025)


def test_liveliest_window_can_require_a_phrase_change():
    dt = 0.02
    energy = np.zeros(3000)
    energy[0:1500] = 1.0  # the first 30 s are the liveliest
    start, _ = pick_segment.liveliest_window(energy, dt, 20.0)
    assert start == 0.0
    # The change at 40 s must lie 3 s or more inside the window, so the start is in [23, 37] s.
    start, _ = pick_segment.liveliest_window(energy, dt, 20.0, must_contain=[40.0], margin_s=3.0)
    assert start == 23.0
    with pytest.raises(ValueError):
        pick_segment.liveliest_window(energy, dt, 20.0, max_end_s=30.0, must_contain=[40.0])


def test_groove_to_travel_times_from_meta():
    meta = {"groove_info": {"phrases": {"travel": [False, True, False, False, True, True],
                                        "start_times_s": [0.2, 10.0, 20.0, 30.0, 40.0, 50.0]}}}
    assert pick_segment.groove_to_travel_times(meta) == [10.0, 40.0]
    with pytest.raises(ValueError):
        pick_segment.groove_to_travel_times({})


def _final_video_module():
    pytest.importorskip("PIL")
    return _load("make_final_video")


def _sections_ctx(mfv, tmp_path):
    import copy

    sections = copy.deepcopy(mfv.SECTIONS)
    ctx = mfv.Context(tmp_path / "rec", tmp_path / "eval", tmp_path / "slides", mfv.SONGS, "fallin_for_you",
                      20.0, 25.0, "")
    ctx.sections = sections
    return sections, ctx


def test_final_video_skips_sections_with_missing_inputs(tmp_path, monkeypatch, capsys):
    """A fresh clone has no recordings and no course clips. Those sections are skipped with a warning
    instead of stopping the build, and a skipped section adds no credit line."""
    mfv = _final_video_module()
    monkeypatch.setattr(mfv, "COURSE_DIR", tmp_path / "course")
    sections, ctx = _sections_ctx(mfv, tmp_path)
    for s in sections:
        if s.key in ("op", "ed"):
            s.requires = [tmp_path / "course" / f"{s.key}.mp4"]
    warnings = mfv.skip_missing(sections, ctx)
    on = {s.key for s in sections if s.enabled}
    assert on == {"title", "intro", "pipeline", "credits"}
    assert {w.split()[3] for w in warnings} == {"op", "songs", "compare", "rough", "roller", "ed"}
    assert "WARNING: skipping section songs" in capsys.readouterr().err
    extra = [line for s in sections if s.enabled for line in s.credits]
    assert mfv.ROLLER_CREDIT not in extra


def test_final_video_keeps_a_section_whose_inputs_exist(tmp_path, monkeypatch):
    mfv = _final_video_module()
    sections, ctx = _sections_ctx(mfv, tmp_path)
    music = tmp_path / "music"
    monkeypatch.setattr(mfv, "MUSIC_DIR", music)
    ctx.rec.mkdir()
    music.mkdir()
    for song, _, _ in ctx.songs:
        for p in (ctx.rec / f"{song}_v4.mp4", ctx.rec / f"{song}_{mfv.HUMAN_TAG}.mp4", music / f"{song}.wav"):
            p.write_bytes(b"")
    mfv.skip_missing(sections, ctx)
    on = {s.key for s in sections if s.enabled}
    assert "songs" in on and "compare" not in on
