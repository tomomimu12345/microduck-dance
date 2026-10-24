#!/usr/bin/env bash
# Generate the library dances with EDGE: synthetic wav -> EDGE pkl (no commands).
# Usage: bash dance/scripts/generate/generate_library_edge.sh [music_dir] [pkl_dir] [reject_dir]
#
# Clip i (synthetic_XX.wav, XX = i) uses seed i. All clips run in one process
# (generate_motion_edge.sh --jobs), so Jukebox and EDGE load once. A clip whose
# floor drift (p98 - p2 of the lowest joint, after the floor correction) is
# 0.10 m or more is moved to reject_dir and generated again with the next seed,
# up to 3 more seeds. The features are cached, so a retry costs seconds.
# Commands are made later with retarget_motion.py (Task 15 changes the retarget).
# Prints OK / REGEN / FAIL lines; the summary is the last lines of the output.
set -u
music=${1:-dance/data/music/synthetic}
pkls=${2:-dance/data/motions/training}
reject=${3:-out/gen_edge/lib_rejected}
py=${EDGE_PY:-$HOME/.venvs/edge/bin/python}
here=$(dirname "$(realpath "$0")")
mkdir -p "$pkls" "$reject"
jobs=$(mktemp)
trap 'rm -f "$jobs"' EXIT

declare -A seed
for wav in "$music"/synthetic_*.wav; do
    stem=$(basename "$wav" .wav)
    seed[$stem]=$((10#${stem#synthetic_}))
done

for round in 0 1 2 3; do
    : > "$jobs"
    for wav in "$music"/synthetic_*.wav; do
        stem=$(basename "$wav" .wav)
        printf '%s\t%s\t%d\n' "$wav" "$pkls/$stem.pkl" "${seed[$stem]}" >> "$jobs"
    done
    start=$(date +%s)
    bash "$here/generate_motion_edge.sh" --jobs "$jobs" 60 || echo "generator exited with $?"
    echo "round $round generation $(( $(date +%s) - start )) s"
    bad=0
    for wav in "$music"/synthetic_*.wav; do
        stem=$(basename "$wav" .wav)
        out="$pkls/$stem.pkl"
        if [[ ! -f $out ]]; then
            echo "FAIL $stem seed=${seed[$stem]} no output"
            bad=1
            continue
        fi
        drift=$("$py" -I -c '
import pickle, sys
sys.path.insert(0, sys.argv[2])
from microduck_dance.motion_fix import floor_drift
with open(sys.argv[1], "rb") as fh:
    print("%.4f" % floor_drift(pickle.load(fh)["full_pose"]))
' "$out" "$here/../../../src")
        if "$py" -I -c 'import sys; sys.exit(float(sys.argv[1]) >= 0.10)' "$drift"; then
            echo "OK $stem seed=${seed[$stem]} drift=$drift"
        else
            echo "REGEN $stem seed=${seed[$stem]} drift=$drift -> seed $(( ${seed[$stem]} + 1 ))"
            mv "$out" "$reject/${stem}_seed${seed[$stem]}.pkl"
            seed[$stem]=$(( ${seed[$stem]} + 1 ))
            bad=1
        fi
    done
    [[ $bad -eq 0 ]] && break
done
echo "DONE"
