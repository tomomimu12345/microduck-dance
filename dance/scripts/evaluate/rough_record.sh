#!/usr/bin/env bash
# Record-mode state dumps (no DR, 1 env, 60 s) for fidelity_eval.py on flat and rough, and the rough footage.
# Usage (repo root): CKPT=<model.pt> bash dance/scripts/evaluate/rough_record.sh [dumps] [video]
#   dumps: out/rough/sim/<song>_{flat,stairs,grid}_states.npz, then
#          python dance/scripts/evaluate/fidelity_eval.py flat stairs grid --sim-dir out/rough/sim --out-dir out/rough
#   video: out/rough/funky_magic_stairs_15s.mp4 (lagged camera, as the video), with its state dump.
#   An existing output is skipped. Waits while the GPU uses GPU_MAX_MB or more (default 7000).
set -u
case "${1:-}" in -h|--help) sed -n '2,8p' "$0"; exit 0;; esac
cd "$(dirname "$(realpath "$0")")/../../.."
export MUJOCO_GL=egl
py=${MICRODUCK_PY:-.venv/bin/python}
ck=${CKPT:-logs/rsl_rl/dance_v3/2026-10-20_15-42-27_dance_v3/model_9996.pt}
gpu_max=${GPU_MAX_MB:-7000}
modes=${*:-dumps video}
mkdir -p out/rough/sim
wait_gpu() {
    while true; do
        m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
        [ "$m" -lt "$gpu_max" ] && break
        echo "gpu $m MB, wait"; sleep 300
    done
}
for mode in $modes; do
    case $mode in
    dumps)
        for song in swag funky_magic fallin_for_you; do
            for tag in flat stairs grid; do
                dump=out/rough/sim/${song}_${tag}_states.npz
                [ -f "$dump" ] && { echo "skip $dump"; continue; }
                case $tag in
                    flat) extra=();;
                    stairs) extra=(--terrain rough --tile-type pyramid_stairs);;
                    grid) extra=(--terrain rough --tile-type random_grid);;
                esac
                wait_gpu; echo "$(date +%T) start $song $tag"
                "$py" -u dance/scripts/evaluate/play_dance.py --checkpoint "$ck" --commands dance/data/commands/songs/$song.npz \
                    --cam-follow lagged --dump-states "$dump" "${extra[@]}" > out/rough/sim/${song}_${tag}.log 2>&1
                echo "$(date +%T) done $song $tag exit $?"
            done
        done;;
    video)
        mp4=out/rough/funky_magic_stairs_15s.mp4
        [ -f "$mp4" ] && { echo "skip $mp4"; continue; }
        wait_gpu; echo "$(date +%T) start video"
        "$py" -u dance/scripts/evaluate/play_dance.py --checkpoint "$ck" --commands dance/data/commands/songs/funky_magic.npz \
            --cam-follow lagged --terrain rough --tile-type pyramid_stairs --max-seconds 15 --out "$mp4" \
            --dump-states out/rough/funky_magic_stairs_15s_states.npz > out/rough/funky_magic_stairs_15s.log 2>&1
        echo "$(date +%T) done video exit $?";;
    esac
done
echo ALL_DONE
