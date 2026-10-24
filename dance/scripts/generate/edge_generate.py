"""Generate dances with EDGE (Tseng et al., CVPR 2023) from wav files.

Usage (repo root, EDGE venv from dance/scripts/setup/setup_edge.sh):
    python -I dance/scripts/generate/edge_generate.py --music SONG.wav --out OUT.pkl --seed N [--start S --duration D]
    python -I dance/scripts/generate/edge_generate.py --jobs JOBS.tsv

A jobs file has one job per line, "music<TAB>out<TAB>seed", and runs all jobs
in one process: phase 1 extracts the Jukebox features of every song that is not
cached yet, phase 2 frees Jukebox, loads EDGE once and samples every job. A job
whose output already exists is skipped.

Steps per song and seed:
1. Jukebox features (as EDGE data/audio_extraction/jukebox_features.py through
   jukemirlib: top prior layer 66, 4800-D, resampled to 30 fps) of 5 s windows
   at 2.5 s stride (as EDGE data/slice.py). The 5B prior stays in CPU RAM and
   each of its layers is moved to the GPU only while it runs (peak about 3.8 GB
   instead of the 16 GB the EDGE README asks for). Cached under --feat-cache.
2. EDGE long-form sampling over all windows (50 DDIM steps, windows stitched
   with slerp), seeded with --seed after the model is loaded.
3. motion_fix.fix_floor (remove the lowest-joint height below 0.5 Hz, floor at
   z = -0.94) and motion_fix.align_heading (start facing -y, -90 deg).

Output pkl: smpl_poses (F, 72), smpl_trans (F, 3), full_pose (F, 24, 3),
float32, z-up metres, 30 fps, F = 30 * duration. EDGE produces 75 frames per
window plus 75, so the duration must be a multiple of 2.5 s (else exit 1).
--raw-out also writes the untouched EDGE output.

EDGE, jukemirlib and the Jukebox fork stay in ~/external (setup_edge.sh). EDGE
code is MIT, Jukebox weights are under OpenAI's non-commercial license.
"""

from __future__ import annotations

import argparse
import gc
import glob
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
from microduck_dance.motion_fix import align_heading, fix_floor, floor_drift, foot_slide  # noqa: E402

EXT = Path(os.path.expanduser("~/external"))
FPS = 30
WINDOW_S = 5.0
STRIDE_S = 2.5
LAYER = 66


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--music")
    ap.add_argument("--out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=None, help="seconds (default: whole song)")
    ap.add_argument("--jobs", help="tsv of music, out, seed (same --start/--duration for all)")
    ap.add_argument("--raw-out", help="also write the untouched EDGE output (single job only)")
    ap.add_argument("--feat-cache", default=str(EXT / "edge_feats"))
    ap.add_argument("--edge-dir", default=str(EXT / "EDGE"))
    ap.add_argument("--ckpt", default=str(EXT / "dl_edge" / "checkpoint.pt"))
    ap.add_argument("--jukebox-dir", default=str(EXT / "dl_jukebox"))
    ap.add_argument("--heading", type=float, default=-90.0)
    a = ap.parse_args()
    if a.jobs:
        jobs = []
        for line in Path(a.jobs).read_text().splitlines():
            if line.strip() and not line.startswith("#"):
                m, o, s = line.split("\t")
                jobs.append((m, o, int(s)))
        if a.raw_out:
            ap.error("--raw-out needs a single job")
    else:
        if not (a.music and a.out):
            ap.error("give --music and --out, or --jobs")
        jobs = [(a.music, a.out, a.seed)]
    return a, jobs


def song_duration(music: str, start: float, duration: float | None) -> float:
    import soundfile as sf

    info = sf.info(music)
    total = info.frames / info.samplerate - start
    d = total if duration is None else duration
    if d > total + 1e-6:
        raise SystemExit(f"{music}: asked for {d} s from {start} s, only {total:.3f} s left")
    return d


def check_duration(music: str, d: float) -> int:
    """Number of EDGE windows for a d-second clip; exits when d is not a multiple of 2.5 s."""
    n = int(round(d / STRIDE_S)) - 1
    if n < 1 or abs((n + 1) * STRIDE_S - d) > 0.5 / FPS:
        raise SystemExit(f"{music}: duration {d:.3f} s is not a multiple of {STRIDE_S} s, "
                         f"EDGE output would not match the song length")
    return n


def feat_dir(cache: str, music: str, start: float, d: float) -> Path:
    return Path(cache) / f"{Path(music).stem}_s{start:g}_d{d:g}"


def slice_song(music: str, start: float, d: float, out: Path, n: int) -> list[Path]:
    """5 s windows at 2.5 s stride, as EDGE data/slice.py slice_audio."""
    import librosa
    import soundfile as sf

    y, sr = librosa.load(music, sr=None, offset=start, duration=d)
    stride, win = int(STRIDE_S * sr), int(WINDOW_S * sr)
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for k in range(n):
        seg = y[k * stride: k * stride + win]
        if len(seg) != win:
            raise SystemExit(f"{music}: window {k} is short ({len(seg)} of {win} samples)")
        f = out / f"{Path(music).stem}_slice{k}.wav"
        sf.write(str(f), seg, sr)
        files.append(f)
    return files


class Jukebox:
    """Jukebox VQ-VAE and top prior, the prior streamed to the GPU layer by layer."""

    def __init__(self, cache_dir: str):
        import torch
        import jukemirlib  # noqa: F401
        import jukemirlib.lib as jl

        jsm = sys.modules["jukemirlib.setup_models"]
        t0 = time.time()
        # torch.load uses weights_only=True by default; the checkpoint pickles only
        # reference OrderedDict and tensor rebuild functions.
        vqvae, prior = jsm.setup_models(cache_dir=cache_dir, device="cpu")
        self.dev = "cuda"
        self.vqvae = vqvae.to(self.dev).eval()
        layers = prior.prior.transformer._attn_mods
        prior.prior.transformer._attn_mods = torch.nn.ModuleList()
        prior.to(self.dev)
        prior.prior.transformer._attn_mods = layers  # these stay on the CPU
        jl.TOP_PRIOR = prior
        jukemirlib.DEVICE = self.dev
        self.prior, self.layers, self.jl, self.jukemirlib, self.torch = prior, layers, jl, jukemirlib, torch
        print(f"jukebox loaded in {time.time() - t0:.0f} s", flush=True)

    def features(self, files: list[Path]) -> None:
        """Writes <slice>.npy (150, 4800) next to each slice wav."""
        torch, jl = self.torch, self.jl
        t0 = time.time()
        with torch.no_grad():
            zs = [jl.get_z(self.vqvae, self.jukemirlib.load_audio(str(f))) for f in files]
            x_cond, y_cond = jl.get_cond(self.prior)
            pr = self.prior.prior
            xs = []
            for z in zs:
                x = z[:, : jl.T]
                n = x.shape[1]
                x = pr.x_emb(pr.preprocess(x))
                x = jl.roll(x, 1)
                x[:, 0] = y_cond.view(1, pr.width)
                x = pr.x_emb_dropout(x) + pr.pos_emb_dropout(pr.pos_emb())[:n] + x_cond[:, :n]
                xs.append(x.cpu())
            for i, layer in enumerate(self.layers):
                if i >= LAYER:
                    break
                layer.to(self.dev)
                for j in range(len(xs)):
                    layer.attn.del_cache()
                    xs[j] = layer(xs[j].to(self.dev), encoder_kv=None, sample=True).cpu()
                    layer.attn.del_cache()
                layer.to("cpu")
                torch.cuda.empty_cache()
            for f, x in zip(files, xs):
                rep = jl.downsample(np.array(x.squeeze().float()), target_rate=FPS).astype(np.float32)
                np.save(str(f.with_suffix(".npy")), rep)
        print(f"  {len(files)} windows in {time.time() - t0:.0f} s, "
              f"max GPU {torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)


def ensure_features(songs, a) -> None:
    todo = []
    for music, d, n in songs:
        fd = feat_dir(a.feat_cache, music, a.start, d)
        if not (fd / "done").exists():
            todo.append((music, d, n, fd))
    if not todo:
        return
    juke = Jukebox(a.jukebox_dir)
    for music, d, n, fd in todo:
        print(f"features {music} -> {fd}", flush=True)
        files = slice_song(music, a.start, d, fd, n)
        juke.features(files)
        (fd / "done").write_text(f"{n}\n")
    del juke
    gc.collect()
    import torch

    torch.cuda.empty_cache()


class Edge:
    def __init__(self, edge_dir: str, ckpt: str):
        import torch

        sys.path.insert(0, edge_dir)
        from dataset.preprocess import Normalizer
        from dataset.scaler import MinMaxScaler

        torch.serialization.add_safe_globals([Normalizer, MinMaxScaler])
        from EDGE import EDGE

        self.torch = torch
        self.model = EDGE("jukebox", ckpt)
        self.model.eval()

    def sample(self, fd: Path, n: int, seed: int, tmp: Path) -> dict:
        torch = self.torch
        key = lambda p: int(Path(p).stem.split("slice")[-1])  # noqa: E731
        npys = sorted(glob.glob(str(fd / "*.npy")), key=key)
        if len(npys) != n:
            raise SystemExit(f"{fd}: {len(npys)} feature files, expected {n}")
        cond = torch.from_numpy(np.stack([np.load(p) for p in npys]))
        torch.manual_seed(seed)
        np.random.seed(seed)
        tmp.mkdir(parents=True, exist_ok=True)
        for old in tmp.glob("*.pkl"):
            old.unlink()
        names = [str(Path(p).with_suffix(".wav")) for p in npys]
        with torch.no_grad():
            self.model.render_sample((None, cond, names), "test", str(tmp), render_count=-1,
                                     fk_out=str(tmp), render=False)
        pk = list(tmp.glob("*.pkl"))
        assert len(pk) == 1, pk
        with open(pk[0], "rb") as fh:
            d = pickle.load(fh)
        pk[0].unlink()
        return {k: np.asarray(v, dtype=np.float32) for k, v in d.items()}


def finish(raw: dict, heading_deg: float):
    trans, joints, info = fix_floor(raw["smpl_trans"], raw["full_pose"])
    poses, trans, joints, yaw = align_heading(raw["smpl_poses"], trans, joints, heading_deg)
    out = {"smpl_poses": poses.astype(np.float32), "smpl_trans": trans.astype(np.float32),
           "full_pose": joints.astype(np.float32)}
    return out, info, yaw


def main():
    a, jobs = parse_args()
    songs = {}
    for music, _, _ in jobs:
        if music not in songs:
            d = song_duration(music, a.start, a.duration)
            songs[music] = (d, check_duration(music, d))
    pending = [j for j in jobs if not Path(j[1]).exists()]
    for music, out, seed in jobs:
        if (music, out, seed) not in pending:
            print(f"skip {out} (exists)", flush=True)
    if not pending:
        return
    ensure_features([(m, *songs[m]) for m in dict.fromkeys(j[0] for j in pending)], a)
    edge = Edge(a.edge_dir, a.ckpt)
    tmp = Path(a.feat_cache) / "_tmp_sample"
    for music, out, seed in pending:
        d, n = songs[music]
        t0 = time.time()
        raw = edge.sample(feat_dir(a.feat_cache, music, a.start, d), n, seed, tmp)
        want = int(round(d * FPS))
        if raw["smpl_poses"].shape != (want, 72) or raw["full_pose"].shape != (want, 24, 3):
            raise SystemExit(f"{music}: got {raw['smpl_poses'].shape}, want {want} frames")
        if a.raw_out:
            Path(a.raw_out).parent.mkdir(parents=True, exist_ok=True)
            with open(a.raw_out, "wb") as fh:
                pickle.dump(raw, fh)
        fixed, info, yaw = finish(raw, a.heading)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as fh:
            pickle.dump(fixed, fh)
        print(f"wrote {out} seed {seed} frames {want}: drift raw {floor_drift(raw['full_pose']):.3f} -> "
              f"{floor_drift(fixed['full_pose']):.3f} m, float std {info['float_std']:.3f} m, "
              f"foot slide {foot_slide(fixed['full_pose']):.3f} m/s, yaw {np.degrees(yaw):+.1f} deg, "
              f"{time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
