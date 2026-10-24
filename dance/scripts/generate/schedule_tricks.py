"""Pick trick moments for songs from their beats json (Task 23).

    UV_PROJECT_ENVIRONMENT=$HOME/.venvs/microduck uv run python -u dance/scripts/generate/schedule_tricks.py \
        swag funky_magic fallin_for_you

Reads dance/data/beats/songs/<song>.json (beat_times_s, beat_onset, beat_rms from
compute_beats.py, so librosa is not needed here) and writes
dance/data/tricks/<song>.json, a list of {t_start, trick, bridge_s, score}.
t_start is in seconds from the start of the song and sits on a guessed downbeat.
The first (best) moment gets a roulade, the second a sitstand. See
src/microduck_dance/tricks.py for the rules.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from microduck_dance.tricks import END_MARGIN_S, MIN_GAP_S, MIN_START_S, TRICKS, downbeat_phase, schedule_from_beats

REPO = Path(__file__).resolve().parents[3]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("songs", nargs="+", help="song stems, e.g. swag")
    ap.add_argument("--beats-dir", type=Path, default=REPO / "dance/data/beats/songs")
    ap.add_argument("--out-dir", type=Path, default=REPO / "dance/data/tricks")
    ap.add_argument("-n", "--n-tricks", type=int, default=2, help="1 or 2 tricks per song")
    ap.add_argument("--order", nargs="+", choices=TRICKS, default=list(TRICKS), help="trick for the best, next best moment")
    ap.add_argument("--min-start-s", type=float, default=MIN_START_S)
    ap.add_argument("--min-gap-s", type=float, default=MIN_GAP_S)
    ap.add_argument("--end-margin-s", type=float, default=END_MARGIN_S, help="no trick may run into the last N seconds")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for song in args.songs:
        beats = json.loads((args.beats_dir / f"{song}.json").read_text())
        events = schedule_from_beats(
            beats, n_tricks=args.n_tricks, order=args.order, min_start_s=args.min_start_s,
            min_gap_s=args.min_gap_s, end_margin_s=args.end_margin_s,
        )
        out = args.out_dir / f"{song}.json"
        out.write_text(json.dumps([e.as_dict() for e in events], indent=1) + "\n")
        print(f"{song}: bar phase {downbeat_phase(beats['beat_onset'])}, duration {beats['duration_s']:.1f} s")
        for e in events:
            print(f"  t={e.t_start:6.2f}  {e.trick:9s} bridge {e.bridge_s:.1f} s  score {e.score:.2f}")
        print(f"  -> {out}")


if __name__ == "__main__":
    main()
