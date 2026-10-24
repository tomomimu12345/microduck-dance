#!/usr/bin/env bash
# Fall rate on flat and rough terrain for the three songs (seed 0, 16 envs, 60 s, DR on).
# Usage (repo root): CKPT=<model.pt> OUT=out/rough/eval/<name> bash dance/scripts/evaluate/rough_eval.sh [flat] [rough] [origin]
#   flat: play_dance --eval on the plane. rough: --terrain rough, spawn off the tile platforms.
#   origin: rough with the spawn at the tile origin (on the platform), swag only, as a check.
#   Writes $OUT/<terrain>_<song>.json and .log. Waits while the GPU uses GPU_MAX_MB or more (default 7000).
set -u
case "${1:-}" in -h|--help) sed -n '2,7p' "$0"; exit 0;; esac
cd "$(dirname "$(realpath "$0")")/../../.."
export MUJOCO_GL=egl
py=${MICRODUCK_PY:-.venv/bin/python}
ck=${CKPT:-logs/rsl_rl/dance_v3/2026-10-20_15-42-27_dance_v3/model_9996.pt}
out=${OUT:-out/rough/eval/dance_v4}
gpu_max=${GPU_MAX_MB:-7000}
modes=${*:-flat rough}
mkdir -p "$out"
run() {  # name song extra-args...
    local name=$1 song=$2; shift 2
    while true; do
        m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
        [ "$m" -lt "$gpu_max" ] && break
        echo "gpu $m MB, wait"; sleep 300
    done
    echo "$(date +%T) start $name $song"
    "$py" -u dance/scripts/evaluate/play_dance.py --checkpoint "$ck" --commands dance/data/commands/songs/$song.npz \
        --eval --seeds 0 --num-envs 16 --json "$out/${name}_$song.json" "$@" > "$out/${name}_$song.log" 2>&1
    echo "$(date +%T) done $name $song exit $?"
}
for mode in $modes; do
    case $mode in
        flat) for s in swag funky_magic fallin_for_you; do run flat $s; done;;
        rough) for s in swag funky_magic fallin_for_you; do run rough $s --terrain rough; done;;
        origin) run origin swag --terrain rough --rough-spawn origin;;
    esac
done
echo ALL_DONE
