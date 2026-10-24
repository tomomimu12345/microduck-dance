#!/usr/bin/env bash
# Generate a dance motion (SMPL pkl, z-up, 30 fps) from a song with EDGE.
#
# Usage:
#   bash dance/scripts/generate/generate_motion_edge.sh <song.wav> <out.pkl> [seed] [start_s] [duration_s]
#   bash dance/scripts/generate/generate_motion_edge.sh --jobs <jobs.tsv> [duration_s]
#
# Runs dance/scripts/generate/edge_generate.py (Jukebox features, EDGE sampling, floor
# correction, heading alignment) with the EDGE venv. Install with
# dance/scripts/setup/setup_edge.sh. Environment:
#   EDGE_PY          default ~/.venvs/edge/bin/python
#   EDGE_FEAT_CACHE  default ~/external/edge_feats (Jukebox features, reused across seeds)
# seed defaults to 0, start to 0, duration to the whole song. The duration must
# be a multiple of 2.5 s. The output has 30 * duration frames. GPU job: about
# 1.5 min to load Jukebox, 1.5 to 4.5 min of features per 60 s song (the GPU is
# shared), 5 s of sampling per seed.
set -euo pipefail

py=${EDGE_PY:-$HOME/.venvs/edge/bin/python}
cache=${EDGE_FEAT_CACHE:-$HOME/external/edge_feats}
here=$(dirname "$(realpath "$0")")

if [[ ${1:-} == --jobs ]]; then
    [[ $# -ge 2 && $# -le 3 ]] || { echo "usage: $0 --jobs <jobs.tsv> [duration_s]" >&2; exit 2; }
    extra=()
    [[ $# -eq 3 ]] && extra=(--duration "$3")
    exec "$py" -I -u "$here/edge_generate.py" --jobs "$2" --feat-cache "$cache" "${extra[@]}"
fi

if [[ $# -lt 2 || $# -gt 5 ]]; then
    echo "usage: $0 <song.wav> <out.pkl> [seed] [start_s] [duration_s]" >&2
    exit 2
fi
extra=(--seed "${3:-0}" --start "${4:-0}")
[[ $# -eq 5 ]] && extra+=(--duration "$5")
exec "$py" -I -u "$here/edge_generate.py" --music "$1" --out "$2" --feat-cache "$cache" "${extra[@]}"
