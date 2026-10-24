#!/usr/bin/env bash
# Fall rate of dance_v4 on the three songs with the current commands (dance/data/commands/songs).
# Usage (repo root): bash dance/scripts/evaluate/fall_eval.sh
#   Eval mode of play_dance.py, seed 0, 16 envs.
#   Writes out/fid001/eval/<song>.json and .log. Waits while the GPU uses GPU_MAX_MB or more (default 7000).
# Environment: CKPT, OUT (default out/fid001), MICRODUCK_PY, GPU_MAX_MB, as in sweep_vel_gain.sh.
set -u
case "${1:-}" in -h|--help) sed -n '2,6p' "$0"; exit 0;; esac
cd "$(dirname "$(realpath "$0")")/../../.."
export MUJOCO_GL=egl
py=${MICRODUCK_PY:-.venv/bin/python}
ck=${CKPT:-logs/rsl_rl/dance_v3/2026-10-20_15-42-27_dance_v3/model_9996.pt}
out=${OUT:-out/fid001}
gpu_max=${GPU_MAX_MB:-7000}
mkdir -p "$out/eval"
for song in swag funky_magic fallin_for_you; do
    while true; do
        m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
        [ "$m" -lt "$gpu_max" ] && break
        echo "gpu $m MB, wait"; sleep 60
    done
    echo "$(date +%T) start $song"
    "$py" -u dance/scripts/evaluate/play_dance.py --checkpoint "$ck" --commands dance/data/commands/songs/$song.npz \
        --eval --seeds 0 --num-envs 16 --json "$out/eval/$song.json" > "$out/eval/$song.log" 2>&1
    echo "$(date +%T) done $song exit $?"
done
echo ALL_DONE
