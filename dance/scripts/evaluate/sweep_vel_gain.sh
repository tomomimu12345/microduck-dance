#!/usr/bin/env bash
# Record-mode rollouts (no DR, 1 env, 60 s) of dance_v4 on the three songs, one retarget per vel_gain.
# Usage (repo root): bash dance/scripts/evaluate/sweep_vel_gain.sh vg0.3 vg0.6 ...
#   A tag is vg<value>. If out/fid001/cmds/<song>_<tag>.npz is missing, it is made with
#   retarget_motion.py --vel-gain <value> and the default options of the repo.
#   The state dump goes to out/fid001/sim/<song>_<tag>_states.npz (log next to it), then
#   python dance/scripts/evaluate/fidelity_eval.py <tags> --sim-dir out/fid001/sim.
#   An existing dump is skipped. Before each rollout it waits until the GPU uses less than
#   GPU_MAX_MB (default 7000), because the GPU is shared.
# Environment: CKPT (default the dance_v4 checkpoint logs/rsl_rl/dance_v3/2026-10-20_15-42-27_dance_v3/model_9996.pt),
#   OUT (default out/fid001), MICRODUCK_PY (default .venv/bin/python, made by uv sync), GPU_MAX_MB.
# One run takes about 3 minutes per song and tag.
set -u
case "${1:-}" in -h|--help|"") sed -n '2,12p' "$0"; exit 0;; esac
cd "$(dirname "$(realpath "$0")")/../../.."
export MUJOCO_GL=egl
py=${MICRODUCK_PY:-.venv/bin/python}
ck=${CKPT:-logs/rsl_rl/dance_v3/2026-10-20_15-42-27_dance_v3/model_9996.pt}
out=${OUT:-out/fid001}
gpu_max=${GPU_MAX_MB:-7000}
mkdir -p "$out/cmds" "$out/sim"
for tag in "$@"; do
    case "$tag" in vg*) ;; *) echo "bad tag $tag (want vg<value>)"; exit 2;; esac
    gain=${tag#vg}
    for song in swag funky_magic fallin_for_you; do
        dump=$out/sim/${song}_${tag}_states.npz
        [ -f "$dump" ] && { echo "skip $song $tag"; continue; }
        cmd=$out/cmds/${song}_${tag}.npz
        [ -f "$cmd" ] || "$py" dance/scripts/retarget/retarget_motion.py dance/data/motions/songs/$song.pkl "$cmd" \
            --beats dance/data/beats/songs/$song.json --vel-gain "$gain" || exit 1
        while true; do
            m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
            [ "$m" -lt "$gpu_max" ] && break
            echo "gpu $m MB, wait"; sleep 60
        done
        echo "$(date +%T) start $song $tag"
        "$py" -u dance/scripts/evaluate/play_dance.py --checkpoint "$ck" --commands "$cmd" --cam-follow lagged \
            --dump-states "$dump" > "$out/sim/${song}_${tag}.log" 2>&1
        echo "$(date +%T) done $song $tag exit $?"
    done
done
echo ALL_DONE
