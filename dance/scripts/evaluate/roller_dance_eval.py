"""Play a dance command file on roller skates (Mjlab-RollerDance-MicroDuck): measure or record.

Evaluate (training cfg with its DR and pushes, num_envs envs per seed):
    uv run dance/scripts/evaluate/roller_dance_eval.py --checkpoint <model.pt or actor.pt> \
        --commands dance/data/commands/songs/<song>.npz --eval --seeds 0 1 2 --num-envs 16 --json out/roller/<x>.json

Record (one env, DR removed, lagged camera, 1280x720 at 30 fps):
    uv run dance/scripts/evaluate/roller_dance_eval.py --checkpoint <...> --commands <song>.npz \
        --out out/roller/<x>.mp4 --start-seconds 20 --max-seconds 15

The twist column of the file goes through dance_rollers.map_dance_twist inside
the command term (push from vx, vy dropped, heading target from wz). The
checkpoint can be a run checkpoint or the actor-only file of
scripts/onnx_to_checkpoint.py (roller.onnx). --zero-pose-obs zeroes the actor
head and body slots (51:61) before every policy call, which is how roller.onnx
must be run (it never saw those slots non-zero); head and body are still
scored against the file. --action-scale overrides the joint action scale.

Metrics per env (compute_stats of play_dance on a 13D measurement):
falls per minute, head RMSE (4 head joints against the command), body RMSE
(trunk z above ROLLER_STAND_HEIGHT, roll, pitch above ROLLER_STAND_PITCH,
against the body command), twist correlation (mean of the vx and wz
correlations between the dance twist of the file and the measured trunk
velocity, 0.5 s moving average, up to the first fall; vy is not followed on
skates and is left out), distance travelled (trunk path length in the floor
plane, m per minute, reset steps excluded). Also stance medians (trunk z,
pitch) for the zero training check.

Rendering needs MUJOCO_GL=egl.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import play_dance as pd  # noqa: E402

TASK = "Mjlab-RollerDance-MicroDuck"
COMMAND_TERMS = ("twist", "head_pose", "body_pose")


def twist_corr(dance: np.ndarray, measured_vx: np.ndarray, measured_wz: np.ndarray, dt: float,
               first: int) -> tuple[float, float, float]:
    """(corr vx, corr wz, mean of the finite ones) up to step first."""
    w = max(1, int(round(pd.TWIST_CORR_SMOOTH_S / dt)))
    cx = pd.smoothed_corr(dance[:first, 0], measured_vx[:first], w)
    cz = pd.smoothed_corr(dance[:first, 2], measured_wz[:first], w)
    fin = [c for c in (cx, cz) if np.isfinite(c)]
    return cx, cz, float(np.mean(fin)) if fin else float("nan")


def path_length(xy: np.ndarray, term: np.ndarray) -> float:
    """Trunk path length (m) for one env. xy (T, 2), term (T,) bool. Steps into or out of a reset are skipped."""
    d = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    bad = term[1:] | term[:-1]
    return float(np.sum(d[~bad]))


def liveliest_window(commands: np.ndarray, dt: float, window_s: float = 15.0, stride_s: float = 1.0) -> float:
    """Start time (s) of the window with the most head motion plus turning in the file."""
    head_speed = np.abs(np.diff(commands[:, 3:7], axis=0)).sum(1) / dt
    score = head_speed + np.abs(commands[1:, 2])
    n = int(round(window_s / dt))
    best, best_t = -1.0, 0.0
    for s in range(0, max(1, len(score) - n + 1), int(round(stride_s / dt))):
        v = float(score[s:s + n].mean())
        if v > best:
            best, best_t = v, s * dt
    return best_t


# --------------------------------------------------------------------------- env side


def build_cfg(args, *, command_file: str, play: bool, num_envs: int, seed: int, steps: int, dt: float,
              render: bool):
    import mjlab.tasks  # noqa: F401
    import microduck_dance.tasks  # noqa: F401
    from microduck_dance.tasks.microduck_roller_dance_env_cfg import make_microduck_roller_dance_env_cfg

    cfg = make_microduck_roller_dance_env_cfg(play=play, command_file=command_file, source="file")
    if args.action_scale is not None:
        cfg.actions["joint_pos"].scale = args.action_scale
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    cfg.episode_length_s = steps * dt + 2.0
    for name in COMMAND_TERMS:
        cfg.commands[name].resume_on_reset = True
        cfg.commands[name].reset_ramp_s = args.reset_ramp_s
    if play:
        pd.strip_randomization(cfg)
        pd.fix_spawn_pose(cfg)
    if render:
        cfg.viewer.body_name = "trunk_base"
        cfg.viewer.width, cfg.viewer.height = pd.VIDEO_SIZE
        cfg.viewer.distance = args.cam_distance
        cfg.viewer.elevation = args.cam_elevation
        cfg.viewer.azimuth = args.cam_azimuth
        cfg.viewer.max_extra_envs = 0
        cfg.viewer.enable_shadows = False
        prev = cfg.scene.spec_fn

        def spec_fn(spec):
            if prev is not None:
                prev(spec)
            pd.add_sky(spec)

        cfg.scene.spec_fn = spec_fn
    return cfg


def build(args, *, commands: np.ndarray, dt: float, play: bool, num_envs: int, seed: int, render: bool):
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_rl_cfg

    from microduck_dance.command_file import save_command_file
    from mjlab_microduck.tasks import MicroduckOnPolicyRunner

    tmp = Path(tempfile.mkdtemp(prefix="roller_dance_"))
    try:
        path = tmp / "commands.npz"
        save_command_file(path, commands, dt, {"source": str(args.commands)})
        cfg = build_cfg(args, command_file=str(path), play=play, num_envs=num_envs, seed=seed,
                        steps=len(commands), dt=dt, render=render)
        device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        env = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode="rgb_array" if render else None)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    agent = asdict(load_rl_cfg(TASK))
    venv = RslRlVecEnvWrapper(env, clip_actions=agent["clip_actions"])
    runner = MicroduckOnPolicyRunner(venv, agent, device=device)
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    runner.alg.actor.load_state_dict(ck["actor_state_dict"], strict=True)
    policy = runner.get_inference_policy(device=device)
    if venv.get_observations()["actor"].shape[-1] != pd.ACTOR_OBS_DIM:
        raise RuntimeError("actor obs is not 61D")
    return env, venv, policy


def measure(env, nominal_height: float, nominal_pitch: float):
    """13D measurement in the command layout plus (vx, wz, xy, z, pitch)."""
    import torch

    robot = env.scene["robot"]
    d = robot.data
    if not hasattr(env, "_roller_head_ids"):
        ids, _ = robot.find_joints(pd.HEAD_JOINT_PATTERNS, preserve_order=True)
        env._roller_head_ids = ids
    ids = env._roller_head_ids
    q = d.root_link_quat_w
    qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    yaw = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    roll = torch.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
    pitch = torch.asin(torch.clamp(2.0 * (qw * qy - qz * qx), -1.0, 1.0))
    v = d.root_link_lin_vel_w
    c, s = torch.cos(yaw), torch.sin(yaw)
    vx = c * v[:, 0] + s * v[:, 1]
    vy = -s * v[:, 0] + c * v[:, 1]
    wz = d.root_link_ang_vel_w[:, 2]
    head = d.joint_pos[:, ids] - d.default_joint_pos[:, ids]
    pos = d.root_link_pos_w - env.scene.terrain.env_origins
    z = pos[:, 2]
    zeros = torch.zeros_like(z)
    meas = torch.stack([vx, vy, wz, head[:, 0], head[:, 1], head[:, 2], head[:, 3], zeros, zeros,
                        z - nominal_height, roll, pitch - nominal_pitch, zeros], dim=1)
    return meas, pos[:, :2], z, pitch


def rollout(env, venv, policy, steps: int, zero_pose: bool, nominal_height: float, nominal_pitch: float,
            on_state=None):
    import torch

    cm = env.command_manager
    twist_term = cm.get_term("twist")
    obs = venv.get_observations()
    rec = {k: [] for k in ("cmd", "dance", "roller", "meas", "term", "xy", "z", "pitch")}
    for k in range(steps):
        if on_state is not None:
            on_state(k)
        rec["cmd"].append(torch.cat([twist_term.dance_twist, cm.get_command("head_pose"),
                                     cm.get_command("body_pose")], dim=1).cpu().numpy())
        rec["roller"].append(cm.get_command("twist")[:, :3].cpu().numpy())
        if zero_pose:
            obs = obs.clone()
            obs["actor"][:, 51:61] = 0.0
        with torch.inference_mode():
            actions = policy(obs)
        obs, _, dones, extras = venv.step(actions)
        tout = extras.get("time_outs", torch.zeros_like(dones, dtype=torch.bool))
        rec["term"].append((dones.bool() & ~tout.bool()).cpu().numpy())
        meas, xy, z, pitch = measure(env, nominal_height, nominal_pitch)
        rec["meas"].append(meas.cpu().numpy())
        rec["xy"].append(xy.cpu().numpy())
        rec["z"].append(z.cpu().numpy())
        rec["pitch"].append(pitch.cpu().numpy())
    out = {k: np.stack(v) for k, v in rec.items() if v}
    return out


def env_stats(r: dict, i: int, dt: float) -> dict:
    cmd, meas, term = r["cmd"][:, i], r["meas"][:, i], r["term"][:, i]
    st = asdict(pd.compute_stats(cmd, meas, term, dt))
    T = len(cmd)
    first = int(np.argmax(term)) if term.any() else T
    cx, cz, cm = twist_corr(cmd, meas[:, 0], meas[:, 2], dt, first)
    minutes = T * dt / 60.0
    ok = ~term
    st.update(
        falls_per_min=st["falls"] / minutes,
        twist_corr_vx=cx, twist_corr_wz=cz, twist_corr=cm,
        push_speed_corr=pd.smoothed_corr(r["roller"][:first, i, 0], meas[:first, 0],
                                         max(1, int(round(pd.TWIST_CORR_SMOOTH_S / dt)))),
        distance_m_per_min=path_length(r["xy"][:, i], term) / minutes,
        mean_abs_vx=float(np.mean(np.abs(meas[ok, 0]))) if ok.any() else float("nan"),
        stance_z_median=float(np.median(r["z"][ok, i])) if ok.any() else float("nan"),
        stance_pitch_median=float(np.median(r["pitch"][ok, i])) if ok.any() else float("nan"),
    )
    st.pop("twist_corr_vy", None)
    return st


KEYS = ("falls", "falls_per_min", "head_rmse", "body_rmse", "twist_corr", "twist_corr_vx", "twist_corr_wz",
        "push_speed_corr", "distance_m_per_min", "mean_abs_vx", "stance_z_median", "stance_pitch_median")


def mean_stats(envs: list[dict]) -> dict:
    out = {}
    for k in KEYS:
        v = [e[k] for e in envs if np.isfinite(e[k])]
        out[k] = float(np.mean(v)) if v else float("nan")
    return out


def load_commands(args) -> tuple[np.ndarray, float]:
    from microduck_dance.command_file import load_command_file

    commands, dt, _ = load_command_file(args.commands)
    start = 0
    if args.start_seconds is not None:
        start = int(round((liveliest_window(commands, dt, args.max_seconds or 15.0)
                           if args.start_seconds < 0 else args.start_seconds) / dt))
    end = len(commands) if args.max_seconds is None else min(len(commands), start + int(round(args.max_seconds / dt)))
    args.start_index = start
    return np.asarray(commands[start:end], dtype=np.float32), dt


def nominal(args) -> tuple[float, float]:
    from microduck_dance.tasks import microduck_roller_dance_env_cfg as rd

    return rd.ROLLER_STAND_HEIGHT, rd.ROLLER_STAND_PITCH


def evaluate(args) -> dict:
    commands, dt = load_commands(args)
    nh, npitch = nominal(args)
    per_seed = []
    for seed in args.seeds:
        env, venv, policy = build(args, commands=commands, dt=dt, play=False, num_envs=args.num_envs, seed=seed,
                                  render=False)
        r = rollout(env, venv, policy, len(commands), args.zero_pose_obs, nh, npitch)
        env.close()
        envs = [env_stats(r, i, dt) for i in range(r["cmd"].shape[1])]
        per_seed.append({"seed": seed, "mean": mean_stats(envs), "envs": envs})
        print(f"[roller_dance_eval] seed {seed}: " + json.dumps({k: round(v, 4) for k, v in per_seed[-1]["mean"].items()}),
              flush=True)
    all_envs = [e for s in per_seed for e in s["envs"]]
    result = {"mode": "eval", "task": TASK, "checkpoint": str(args.checkpoint), "commands": str(args.commands),
              "start_index": args.start_index, "steps": len(commands), "dt": dt, "num_envs": args.num_envs,
              "seeds": list(args.seeds), "zero_pose_obs": args.zero_pose_obs, "action_scale": args.action_scale,
              "nominal_height": nh, "nominal_pitch": npitch,
              "summary": {**mean_stats(all_envs), "falls_total": int(sum(e["falls"] for e in all_envs)),
                          "n_envs": len(all_envs)},
              "per_seed_means": [s["mean"] for s in per_seed], "per_seed": per_seed}
    print(json.dumps(result["summary"], indent=2))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2) + "\n")
        print(f"wrote {args.json}")
    return result


def record(args) -> dict:
    import mediapy as media

    commands, dt = load_commands(args)
    nh, npitch = nominal(args)
    env, venv, policy = build(args, commands=commands, dt=dt, play=True, num_envs=1, seed=args.seeds[0], render=True)
    steps = len(commands)
    n_frames = int(math.ceil(steps * dt * pd.VIDEO_FPS - 1e-9))
    cam = pd.LaggedCamera(env._offline_renderer._cam, dt, tau=args.cam_tau)
    robot = env.scene["robot"]
    written = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with media.VideoWriter(args.out, shape=(pd.VIDEO_SIZE[1], pd.VIDEO_SIZE[0]), fps=pd.VIDEO_FPS) as writer:

        def on_state(k: int) -> None:
            nonlocal written
            if k < steps:
                cam.update(robot.data.root_link_pos_w[0].cpu().numpy())
            while written < n_frames and written / pd.VIDEO_FPS <= k * dt + 1e-9:
                writer.add_image(env.render())
                written += 1

        r = rollout(env, venv, policy, steps, args.zero_pose_obs, nh, npitch, on_state=on_state)
        on_state(steps)
    env.close()
    st = env_stats(r, 0, dt)
    result = {"mode": "record", "out": str(args.out), "frames": written, "start_index": args.start_index,
              "start_s": args.start_index * dt, "video_s": written / pd.VIDEO_FPS, "stats": st}
    print(json.dumps(result, indent=2))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2) + "\n")
    return result


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--commands", type=Path, required=True)
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--out", type=Path, default=None, help="mp4 path (record mode)")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--num-envs", type=int, default=16)
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--start-seconds", type=float, default=None,
                    help="start of the played part of the file (s). Negative: the liveliest window")
    ap.add_argument("--zero-pose-obs", action="store_true", help="zero actor obs 51:61 (roller.onnx)")
    ap.add_argument("--action-scale", type=float, default=None)
    ap.add_argument("--reset-ramp-s", type=float, default=0.5)
    ap.add_argument("--device", default=None)
    ap.add_argument("--cam-distance", type=float, default=0.6)
    ap.add_argument("--cam-elevation", type=float, default=-15.0)
    ap.add_argument("--cam-azimuth", type=float, default=150.0)
    ap.add_argument("--cam-tau", type=float, default=pd.CAM_TAU_S)
    return ap


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    if args.eval:
        evaluate(args)
    elif args.out is not None:
        record(args)
    else:
        raise SystemExit("pass --eval or --out")


if __name__ == "__main__":
    main()
