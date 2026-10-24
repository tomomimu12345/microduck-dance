"""Beat tracking for the groove layer. Run with the EDGE venv (it has librosa).

    ~/.venvs/edge/bin/python dance/scripts/generate/compute_beats.py <song.wav> [...] --out-dir dance/data/beats/songs

Writes <out-dir>/<wav stem>.json with
    beat_times_s  librosa.beat.beat_track beat times in seconds from the start of the wav
    tempo_bpm     the tempo librosa estimated
plus the wav name, its duration, the sample rate used and the librosa version.
Default librosa settings (sr 22050, hop 512), mono mix.

Music features for the phrase gate and the groove pattern (Task 15), from the
librosa onset strength envelope (onset.onset_strength) and the frame RMS
(feature.rms), both at hop 512:
    onset_mean     mean onset strength over the whole wav
    rms_mean       mean RMS over the whole wav
    onset_at_beat  onset strength at each beat time (linear interpolation)
    beat_onset     mean onset strength over each beat interval [b_i, b_i+1)
    beat_rms       mean RMS over each beat interval
The last beat interval has the length of the previous one, cut at the end of
the wav. The beat times are the same as before these keys were added.

--synth-seed0 N (for dance/data/music/synthetic/synthetic_XX.wav only). The tempo
is fixed to the BPM that synth_music.py drew for seed N + XX (beat_track
bpm=...), and librosa only places the beats. Without it librosa picked 2x,
1/2 or 2/3 of the true tempo on 4 of the 33 synthetic tracks (synthetic_00, 12,
18, 20), and a start_bpm prior did not fix synthetic_12. The real songs need no prior (librosa gives
92 and 118 BPM, matching the track pages within 2 %).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import librosa
import numpy as np


def _synth_bpm(stem: str, seed0: int) -> float:
    """The BPM synth_music.synth_track drew for synthetic_XX.wav (a 2 s render reads it)."""
    spec = importlib.util.spec_from_file_location("synth_music", Path(__file__).with_name("synth_music.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return float(mod.synth_track(seed0 + int(stem.split("_")[-1]), 2.0)[1]["bpm"])


def beats_of(wav: Path, start_bpm: float | None = None) -> dict:
    y, sr = librosa.load(str(wav), sr=22050, mono=True)
    kw = {"bpm": start_bpm} if start_bpm else {}
    tempo, frames = librosa.beat.beat_track(y=y, sr=sr, **kw)
    times = librosa.frames_to_time(frames, sr=sr)
    feats = music_features(y, sr, times)
    return {
        "wav": wav.name,
        "duration_s": round(len(y) / sr, 4),
        "sr": sr,
        "librosa": librosa.__version__,
        "fixed_bpm": start_bpm,
        "tempo_bpm": round(float(np.atleast_1d(tempo)[0]), 3),
        "num_beats": int(len(times)),
        "beat_times_s": [round(float(t), 4) for t in times],
        **feats,
    }


def interval_means(t: np.ndarray, x: np.ndarray, beats: np.ndarray, end: float) -> np.ndarray:
    """Mean of x(t) over each beat interval [b_i, b_i+1); the last one has the previous length."""
    if len(beats) < 2:
        raise ValueError("need at least 2 beats")
    edges = np.append(beats, min(end, beats[-1] + (beats[-1] - beats[-2])))
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (t >= lo) & (t < hi)
        out.append(float(x[m].mean()) if m.any() else float(np.interp(lo, t, x)))
    return np.asarray(out)


def music_features(y: np.ndarray, sr: int, beats: np.ndarray) -> dict:
    oenv = librosa.onset.onset_strength(y=y, sr=sr)
    t_on = librosa.times_like(oenv, sr=sr)
    rms = librosa.feature.rms(y=y)[0]
    t_rms = librosa.times_like(rms, sr=sr)
    end = len(y) / sr
    r4 = lambda a: [round(float(v), 5) for v in a]  # noqa: E731
    return {
        "onset_mean": round(float(oenv.mean()), 5),
        "rms_mean": round(float(rms.mean()), 5),
        "onset_at_beat": r4(np.interp(beats, t_on, oenv)),
        "beat_onset": r4(interval_means(t_on, oenv, beats, end)),
        "beat_rms": r4(interval_means(t_rms, rms, beats, end)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wav", type=Path, nargs="+")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--synth-seed0", type=int, default=None, help="tempo prior for synthetic_XX.wav (synth_music --seed0)")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for wav in args.wav:
        bpm = _synth_bpm(wav.stem, args.synth_seed0) if args.synth_seed0 is not None else None
        d = beats_of(wav, bpm)
        if bpm and abs(d["tempo_bpm"] / bpm - 1) > 0.03:
            raise SystemExit(f"{wav}: tempo {d['tempo_bpm']} differs from the synth BPM {bpm} by more than 3 %")
        out = args.out_dir / f"{wav.stem}.json"
        out.write_text(json.dumps(d, indent=1) + "\n")
        print(f"{out}: {d['tempo_bpm']:.1f} BPM, {d['num_beats']} beats, first {d['beat_times_s'][:1]}")


if __name__ == "__main__":
    main()
