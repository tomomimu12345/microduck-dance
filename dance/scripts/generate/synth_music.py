"""Synthesize simple beat tracks for the dance library (numpy only).

The tracks are generated from scratch, so they carry no third-party rights and
may be used as training data. Each track has a kick, snare, hi-hat, a bass line
and optional chord stabs. BPM is drawn from 90 to 130 and the groove from four
patterns, both from a seeded generator.

Usage:
    python3 dance/scripts/generate/synth_music.py --out dance/data/music/synthetic --n 33
"""

from __future__ import annotations

import argparse
import wave
from pathlib import Path

import numpy as np

SR = 44100
DURATION_S = 60.0

# Steps are 16th notes inside one 4/4 bar (16 steps).
GROOVES = {
    "four_on_floor": {
        "kick": [0, 4, 8, 12],
        "snare": [4, 12],
        "hat": [2, 6, 10, 14],
        "bass": [0, 3, 6, 8, 11, 14],
    },
    "boom_bap": {
        "kick": [0, 7, 10],
        "snare": [4, 12],
        "hat": list(range(0, 16, 2)),
        "bass": [0, 7, 10, 13],
    },
    "half_time": {
        "kick": [0, 10],
        "snare": [8],
        "hat": list(range(0, 16, 2)),
        "bass": [0, 6, 8, 14],
    },
    "syncopated": {
        "kick": [0, 3, 6, 10, 13],
        "snare": [4, 12, 15],
        "hat": [1, 3, 5, 7, 9, 11, 13, 15],
        "bass": [0, 2, 5, 8, 10, 13],
    },
}

# Minor pentatonic degrees in semitones from the root.
SCALE = [0, 3, 5, 7, 10]


def _env(n: int, decay: float) -> np.ndarray:
    return np.exp(-np.arange(n) / (decay * SR))


def kick(rng: np.random.Generator) -> np.ndarray:
    n = int(0.28 * SR)
    t = np.arange(n) / SR
    freq = 45 + 110 * np.exp(-t * 28)
    phase = 2 * np.pi * np.cumsum(freq) / SR
    return np.sin(phase) * _env(n, 0.09) * 0.95


def snare(rng: np.random.Generator) -> np.ndarray:
    n = int(0.22 * SR)
    t = np.arange(n) / SR
    noise = rng.standard_normal(n) * _env(n, 0.05)
    tone = np.sin(2 * np.pi * 190 * t) * _env(n, 0.04)
    return (0.6 * noise + 0.5 * tone) * 0.7


def hat(rng: np.random.Generator, open_hat: bool = False) -> np.ndarray:
    n = int((0.18 if open_hat else 0.05) * SR)
    noise = rng.standard_normal(n)
    noise = np.diff(noise, prepend=0.0)  # crude high-pass
    return noise * _env(n, 0.09 if open_hat else 0.015) * 0.25


def bass_note(freq: float, dur: float) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n) / SR
    wave_ = np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(4 * np.pi * freq * t)
    attack = np.minimum(1.0, t / 0.005)
    return wave_ * attack * np.exp(-t / (dur * 0.9)) * 0.45


def chord_stab(root: float, dur: float) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n) / SR
    out = np.zeros(n)
    for semi in (0, 3, 7, 10):
        f = root * 2 ** (semi / 12)
        out += np.sign(np.sin(2 * np.pi * f * t)) * 0.5 + np.sin(2 * np.pi * f * t)
    attack = np.minimum(1.0, t / 0.004)
    return out * attack * _env(n, dur * 0.35) * 0.06


def _add(buf: np.ndarray, pos: int, sample: np.ndarray) -> None:
    end = min(len(buf), pos + len(sample))
    if pos < end:
        buf[pos:end] += sample[: end - pos]


def synth_track(seed: int, duration_s: float = DURATION_S) -> tuple[np.ndarray, dict]:
    """Return (mono float array in [-1, 1], info dict)."""
    rng = np.random.default_rng(seed)
    bpm = float(rng.integers(90, 131))
    groove_name = list(GROOVES)[seed % len(GROOVES)]
    g = GROOVES[groove_name]
    root_midi = int(rng.integers(33, 41))  # A1 to E2 area
    root = 440.0 * 2 ** ((root_midi - 69) / 12)
    use_chords = bool(rng.random() < 0.6)
    swing = float(rng.uniform(0.0, 0.12)) if groove_name in ("boom_bap", "half_time") else 0.0

    step_s = 60.0 / bpm / 4
    n_total = int(duration_s * SR)
    buf = np.zeros(n_total + SR)
    n_bars = int(duration_s / (16 * step_s)) + 1

    kick_s, snare_s, open_s = kick(rng), snare(rng), hat(rng, True)
    # Each 8-bar phrase draws a new bass note per bass step from the scale.
    phrase = None
    for bar in range(n_bars):
        if bar % 8 == 0:
            phrase = rng.choice(len(SCALE), size=len(g["bass"]))
        chord_deg = SCALE[int(phrase[0])]
        for step in range(16):
            off = step_s * (1 + swing) if step % 2 == 1 else 0.0
            pos = int(((bar * 16 + step) * step_s + off) * SR)
            if step in g["kick"]:
                _add(buf, pos, kick_s * rng.uniform(0.9, 1.0))
            if step in g["snare"]:
                _add(buf, pos, snare_s * rng.uniform(0.85, 1.0))
            if step in g["hat"]:
                _add(buf, pos, open_s if (step == 14 and bar % 4 == 3) else hat(rng))
            if step in g["bass"]:
                deg = SCALE[int(phrase[g["bass"].index(step)])]
                _add(buf, pos, bass_note(root * 2 ** (deg / 12), step_s * 3))
            if use_chords and step in (0, 6, 10) and bar % 2 == 0:
                _add(buf, pos, chord_stab(root * 4 * 2 ** (chord_deg / 12), step_s * 2))

    buf = buf[:n_total]
    fade = int(1.5 * SR)
    buf[-fade:] *= np.linspace(1.0, 0.0, fade)
    peak = np.abs(buf).max()
    buf = buf / peak * 0.9
    return buf, {"seed": seed, "bpm": bpm, "groove": groove_name, "chords": use_chords}


def write_wav(path: Path, mono: np.ndarray) -> None:
    pcm = (np.clip(mono, -1, 1) * 32767).astype("<i2")
    stereo = np.repeat(pcm[:, None], 2, axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(stereo.tobytes())


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("dance/data/music/synthetic"))
    ap.add_argument("--n", type=int, default=33)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--duration", type=float, default=DURATION_S)
    args = ap.parse_args(argv)
    for i in range(args.n):
        seed = args.seed0 + i
        audio, info = synth_track(seed, args.duration)
        path = args.out / f"synthetic_{i:02d}.wav"
        write_wav(path, audio)
        print(f"{path.name} bpm={info['bpm']:.0f} groove={info['groove']} chords={info['chords']}")


if __name__ == "__main__":
    main()
