#!/usr/bin/env bash
# Retarget every dance pkl to commands and rebuild the library (CPU, about 1 min).
# Usage (repo root): bash dance/scripts/retarget/retarget_all.sh
#
# Real songs: dance/data/motions/songs/<song>.pkl -> dance/data/commands/songs/<song>.npz
# Training: dance/data/motions/training/synthetic_XX.pkl
#   -> dance/data/commands/training/synthetic_XX.npz,
#   then dance/data/library.npz (build_library.py). Only the training files go
#   into the library. The real songs never do.
# Every clip uses the default RetargetConfig and its beats json from
# dance/data/beats/{songs,training} (dance/scripts/generate/compute_beats.py, which also
# writes the music features for the phrase gate and the groove pattern).
# One rule set for all.
set -euo pipefail
py=${MICRODUCK_PY:-.venv/bin/python}
d=dance/data
for kind in songs training; do
    for pkl in "$d/motions/$kind"/*.pkl; do
        stem=$(basename "$pkl" .pkl)
        "$py" dance/scripts/retarget/retarget_motion.py "$pkl" "$d/commands/$kind/$stem.npz" \
            --beats "$d/beats/$kind/$stem.json"
    done
done
"$py" dance/scripts/retarget/build_library.py "$d"/commands/training/synthetic_*.npz --out "$d/library.npz"
