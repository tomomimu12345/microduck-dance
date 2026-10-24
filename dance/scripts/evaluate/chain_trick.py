"""Play a song's dance with trick hand-overs in one continuous simulation (Task 23).

The dance policy (rsl_rl checkpoint) runs the whole time except during a trick
event. An event is {t_start, trick, bridge_s}, read from --schedule (the json that
dance/scripts/generate/schedule_tricks.py writes, dance/data/tricks/<song>.json)
or built from --trick and --dance-s for a single event. At t_start the trick
policy (Pollen ONNX, MLP with the normalizer baked in, converted to torch so it
runs batched) takes over with its trigger command in the actor command slots.
When the trick ends a standing policy holds (velstand at zero command when
bridge_s > 0, the trick policy at its final command otherwise) until the upright
check has held for 0.5 s or the hold times out, then the dance policy takes over
again. The state machine is TrickHandover in microduck_dance.tricks. All
terminations except time_out are removed, so nothing resets mid-run. Falls are
measured from the trunk state.

Robot model: robot_allcollisions (MICRODUCK_STANDUP_ROBOT_CFG), the model the
trick policies were trained on (the roll needs the head collision geoms).
The dance policy was trained on robot_walk. --robot walk keeps the dance model
(control run).

Needs a GPU for realistic env counts (--device cpu works for a few envs). Trick
policies: pollen-robotics/microduck-policies on Hugging Face (Apache-2.0), see
dance/data/CREDITS.md. Example:

    $UV dance/scripts/evaluate/chain_trick.py --checkpoint <model_XXXX.pt> \\
        --commands dance/data/commands/songs/funky_magic.npz \\
        --policy-dir logs/tricks/pollen-microduck-policies \\
        --schedule dance/data/tricks/funky_magic.json --json logs/tricks/results/funky_magic.json
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np

from microduck_dance.tricks import (
    DANCE, DANCE_AFTER, DEFAULT_BRIDGE_S, HOLD, TRICK, TRICKS, TrickHandover, trick_command, trick_duration,
)

DT = 0.02
CMD = slice(48, 61)
STATE_CODE = {DANCE: 0, TRICK: 1, HOLD: 2, DANCE_AFTER: 3}


def onnx_to_torch(path):
    """Torch callable equal to the ONNX MLP graph (Sub, Div, Gemm, Elu), batched."""
    import onnx
    import torch
    from onnx import numpy_helper

    m = onnx.load(path)
    inits = {t.name: torch.tensor(numpy_helper.to_array(t).copy()) for t in m.graph.initializer}
    consts = {}
    for n in m.graph.node:
        if n.op_type == "Constant":
            consts[n.output[0]] = torch.tensor(numpy_helper.to_array(n.attribute[0].t).copy())
    nodes = [n for n in m.graph.node if n.op_type != "Constant"]
    in_name = m.graph.input[0].name
    out_name = m.graph.output[0].name

    def run(x):
        env = {in_name: x, **{k: v.to(x.device) for k, v in inits.items()}, **{k: v.to(x.device) for k, v in consts.items()}}
        for n in nodes:
            a = [env[i] for i in n.input if i]
            if n.op_type == "Sub":
                y = a[0] - a[1]
            elif n.op_type == "Div":
                y = a[0] / a[1]
            elif n.op_type == "Gemm":
                attrs = {at.name: at for at in n.attribute}
                tb = attrs["transB"].i if "transB" in attrs else 0
                ta = attrs["transA"].i if "transA" in attrs else 0
                alpha = attrs["alpha"].f if "alpha" in attrs else 1.0
                beta = attrs["beta"].f if "beta" in attrs else 1.0
                A = a[0].T if ta else a[0]
                B = a[1].T if tb else a[1]
                y = alpha * (A @ B) + (beta * a[2] if len(a) > 2 else 0.0)
            elif n.op_type == "Elu":
                y = torch.nn.functional.elu(a[0])
            elif n.op_type in ("Clip",):
                y = torch.clamp(a[0], a[1], a[2])
            elif n.op_type == "Max":
                y = torch.maximum(a[0], a[1])
            elif n.op_type == "Min":
                y = torch.minimum(a[0], a[1])
            else:
                raise NotImplementedError(n.op_type)
            env[n.output[0]] = y
        return env[out_name]

    return run


def check_onnx_equal(path, fn):
    import onnxruntime as ort
    import torch

    sess = ort.InferenceSession(str(path))
    rng = np.random.default_rng(0)
    err = 0.0
    for _ in range(8):
        x = rng.normal(size=(1, 61)).astype(np.float32)
        ref = sess.run(None, {sess.get_inputs()[0].name: x})[0]
        got = fn(torch.tensor(x)).numpy()
        err = max(err, float(np.abs(ref - got).max()))
    return err


TRICK_FILES = {"roulade": "roulade.onnx", "sitstand": "alpha_sitstand.onnx"}
BRIDGE_FILE = "velstand.onnx"


def build(args, dr: bool, num_envs: int, render: bool):
    import torch
    import mjlab.tasks  # noqa: F401
    import microduck_dance.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_rl_cfg, load_runner_cls
    from rsl_rl.runners import OnPolicyRunner
    from mjlab_microduck.robot.microduck_constants import MICRODUCK_STANDUP_ROBOT_CFG
    from microduck_dance.tasks.microduck_dance_env_cfg import make_microduck_dance_env_cfg

    task = "Mjlab-Dance-Flat-MicroDuck"
    cfg = make_microduck_dance_env_cfg(play=not dr, command_file=str(args.commands), source="file")
    if args.robot == "allcol":
        cfg.scene.entities = {"robot": MICRODUCK_STANDUP_ROBOT_CFG}
    cfg.scene.num_envs = num_envs
    cfg.seed = args.seed
    cfg.episode_length_s = args.total_s + 5.0
    for name in ("twist", "head_pose", "body_pose"):
        cfg.commands[name].resume_on_reset = True
        cfg.commands[name].reset_ramp_s = 0.5
    removed = [k for k in list(cfg.terminations) if k not in ("time_out",)]
    for k in removed:
        cfg.terminations.pop(k)
    print("[chain] removed terminations", removed)
    if not dr:
        # same as play_dance.strip_randomization
        DR_EVENTS = ("push_robot", "foot_friction", "encoder_bias", "base_com", "randomize_com", "randomize_head_com",
                     "randomize_motor_gains", "randomize_mass_inertia", "randomize_joint_friction",
                     "randomize_joint_damping", "randomize_armature", "randomize_base_orientation")
        rem = [n for n in DR_EVENTS if cfg.events.pop(n, None) is not None]
        for n in [n for n, c in cfg.curriculum.items() if (c.params or {}).get("event_name") in rem]:
            cfg.curriculum.pop(n)
        for g in cfg.observations.values():
            g.enable_corruption = False
            for term in g.terms.values():
                if term is not None and "max_angle_deg" in (term.params or {}):
                    term.params["max_angle_deg"] = 0.0
        print("[chain] removed DR events", rem)
    if render:
        pr = cfg.events["reset_base"].params["pose_range"]
        for k in ("x", "y", "yaw"):
            pr[k] = (0.0, 0.0)
        cfg.viewer.body_name = "trunk_base"
        cfg.viewer.width, cfg.viewer.height = 1280, 720
        cfg.viewer.distance = 0.6
        cfg.viewer.elevation = -15.0
        cfg.viewer.azimuth = 150.0
        cfg.viewer.max_extra_envs = 0
        cfg.viewer.enable_shadows = False
        prev = cfg.scene.spec_fn

        def spec_fn(spec):
            import mujoco
            if prev is not None:
                prev(spec)
            spec.add_texture(name="dance_sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                             builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT, rgb1=[0.20, 0.42, 0.70],
                             rgb2=[0.78, 0.88, 0.96], width=512, height=3072)

        cfg.scene.spec_fn = spec_fn
    device = args.device
    env = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode="rgb_array" if render else None)
    agent = asdict(load_rl_cfg(task))
    venv = RslRlVecEnvWrapper(env, clip_actions=agent["clip_actions"])
    print("[chain] clip_actions", agent["clip_actions"])
    runner = (load_runner_cls(task) or OnPolicyRunner)(venv, agent, device=device)
    runner.load(str(args.checkpoint), map_location=device)
    policy = runner.get_inference_policy(device=device)
    return env, venv, policy


def tilt_z(env):
    import torch
    d = env.scene["robot"].data
    q = d.root_link_quat_w
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    # body z axis world-z component
    bz = 1.0 - 2.0 * (x * x + y * y)
    tilt = torch.acos(torch.clamp(bz, -1.0, 1.0))
    h = d.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2]
    # body-frame angular velocity y (pitch rate)
    # body-frame angular velocity: R^T w_world, R from (w, x, y, z)
    ww = d.root_link_ang_vel_w
    qv = torch.stack([x, y, z], dim=1)
    # rotate by the conjugate quaternion
    t2 = 2.0 * torch.cross(-qv, ww, dim=1)
    wb = ww + w[:, None] * t2 + torch.cross(-qv, t2, dim=1)
    return tilt, h, wb


def load_schedule(args):
    """list of {t_start, trick, bridge_s} dicts."""
    if args.schedule:
        ev = json.loads(Path(args.schedule).read_text())
    elif args.trick != "none":
        b = DEFAULT_BRIDGE_S[args.trick] if args.bridge_s is None else args.bridge_s
        ev = [{"t_start": args.dance_s, "trick": args.trick, "bridge_s": b}]
    else:
        ev = []
    for e in ev:
        if e["trick"] not in TRICKS:
            raise SystemExit(f"unknown trick {e['trick']!r}")
    return sorted(ev, key=lambda e: e["t_start"])


def run(args, events, dr: bool, num_envs: int, render_out: Path | None = None):
    import torch
    env, venv, policy = build(args, dr, num_envs, render_out is not None)
    robot = env.scene["robot"]
    dj = robot.data.default_joint_pos[0].cpu().numpy()
    print("[chain] default_joint_pos", np.round(dj, 3).tolist(), "joints", robot.joint_names)
    pdir = Path(args.policy_dir)
    fns = {}
    for name in sorted({e["trick"] for e in events}):
        p = pdir / TRICK_FILES[name]
        fns[name] = onnx_to_torch(p)
        print("[chain]", name, "onnx vs torch max err", check_onnx_equal(p, fns[name]))
    bridge_fn = None
    if any(e["bridge_s"] > 0 for e in events):
        bp = pdir / BRIDGE_FILE
        bridge_fn = onnx_to_torch(bp)
        print("[chain] bridge", bp, "err", check_onnx_equal(bp, bridge_fn))
    kw = dict(roulade_s=args.roulade_s, sit_s=args.sit_s, stand_s=args.stand_s)
    N = num_envs
    machines = [[TrickHandover(e["t_start"], e["trick"], e["bridge_s"], hold_timeout_s=args.hold_timeout_s, **kw)
                 for e in events] for _ in range(N)]
    active = np.zeros(N, int)  # index of the event an env is working on
    total_s = args.total_s
    if total_s is None:
        total_s = max([e["t_start"] + trick_duration(e["trick"], **kw) + e["bridge_s"] + args.hold_timeout_s + 4.0
                       for e in events] or [10.0])
    steps = int(round(total_s / DT))
    log = {k: [] for k in ("tilt", "z", "pitch_rate", "state", "event")}
    obs = venv.get_observations()
    writer = None
    if render_out is not None:
        import mediapy as media
        render_out.parent.mkdir(parents=True, exist_ok=True)
        writer = media.VideoWriter(render_out, shape=(720, 1280), fps=30)
        writer.__enter__()
        written = 0
    for k in range(steps):
        t = k * DT
        if writer is not None:
            while written / 30.0 <= t + 1e-9 and written < int(total_s * 30):
                writer.add_image(env.render())
                written += 1
        tilt, h, wb = tilt_z(env)
        up_z = np.cos(tilt.cpu().numpy())
        hh = h.cpu().numpy()
        state = np.zeros(N, int)
        for i in range(N):
            if active[i] >= len(events):
                state[i] = STATE_CODE[DANCE]
                continue
            m = machines[i][active[i]]
            s = m.step(t, up_z[i], hh[i])
            if s == DANCE_AFTER and active[i] + 1 < len(events):
                active[i] += 1
                s = DANCE
            elif s == DANCE_AFTER:
                active[i] = len(events)
                s = DANCE_AFTER
            state[i] = STATE_CODE[s]
        # who acts: the trick policy of the event (TRICK, or HOLD with bridge_s == 0), or the bridge (HOLD, bridge_s > 0)
        who = [None] * N
        cmds = np.zeros((N, 13), np.float32)
        for i in range(N):
            if active[i] >= len(events) or state[i] not in (STATE_CODE[TRICK], STATE_CODE[HOLD]):
                continue
            ev = events[active[i]]
            if state[i] == STATE_CODE[HOLD] and ev["bridge_s"] > 0:
                who[i] = "bridge"
            else:
                who[i] = ev["trick"]
                cmds[i] = machines[i][active[i]].command(t)
        with torch.inference_mode():
            actions = policy(obs)
            for name in set(w for w in who if w):
                sel = torch.tensor([w == name for w in who], device=actions.device)
                ob = obs["actor"].clone()
                ob[:, CMD] = torch.tensor(cmds, device=ob.device)
                a = bridge_fn(ob) if name == "bridge" else fns[name](ob)
                actions = torch.where(sel[:, None], a, actions)
        log["state"].append(state.copy())
        log["event"].append(active.copy())
        obs, _, dones, extras = venv.step(actions)
        tilt, h, wb = tilt_z(env)
        log["tilt"].append(tilt.cpu().numpy())
        log["z"].append(h.cpu().numpy())
        log["pitch_rate"].append(wb[:, 1].cpu().numpy())
    if writer is not None:
        writer.__exit__(None, None, None)
    env.close()
    L = {k: np.stack(v) for k, v in log.items()}
    return L, machines, total_s


def score(args, events, L, machines):
    """Per env and event. Thresholds and criteria as in the tricks study (tricks-report.md section 4)."""
    tilt, z, pr = L["tilt"], L["z"], L["pitch_rate"]
    T, N = tilt.shape
    fall_tilt, fall_z = math.radians(60), 0.07
    out = []
    for i in range(N):
        rows = []
        prev_end = 0
        for k, ev in enumerate(events):
            m = machines[i][k]
            s0 = int(round(ev["t_start"] / DT))
            r = {"event": k, "trick": ev["trick"], "t_start": ev["t_start"]}
            before = slice(prev_end, s0)
            r["fell_before_trick"] = bool((tilt[before, i] > fall_tilt).any() or (z[before, i] < fall_z).any())
            if m.t_hold is None or m.t_back is None:
                r.update(trick_done=False, fell_within_3s_after=True, timed_out=False, success=False, reached_handback=False)
                rows.append(r)
                prev_end = T
                continue
            te, e = int(round(m.t_hold / DT)), int(round(m.t_back / DT))
            tr = slice(s0, te)
            cum_pitch = float(np.degrees(np.sum(pr[tr, i]) * DT))
            r["cum_pitch_deg"] = round(cum_pitch, 1)
            r["handover_tilt_deg"] = round(float(np.degrees(tilt[te - 1, i])), 1)
            r["handover_z"] = round(float(z[te - 1, i]), 4)
            r["dance_resume_t"] = round(m.t_back, 2)
            r["hold_s"] = round(m.t_back - m.t_hold, 2)
            r["timed_out"] = bool(m.timed_out)
            if ev["trick"] == "roulade":
                done = abs(cum_pitch) >= 300 and tilt[te - 1, i] < math.radians(60)
            else:
                sit_end = s0 + int(round(args.sit_s / DT))
                zz, tl = z[s0:sit_end, i], tilt[s0:sit_end, i]
                seated = bool(((zz < 0.085) & (tl < math.radians(35))).any())
                r["seated"] = seated
                r["sit_min_z"] = round(float(zz.min()), 4)
                done = seated and tilt[te - 1, i] < math.radians(25) and z[te - 1, i] > 0.10
            r["trick_done"] = bool(done)
            nxt = int(round(events[k + 1]["t_start"] / DT)) if k + 1 < len(events) else T
            post = slice(e, min(T, nxt, e + int(3.0 / DT)))
            late = slice(min(post.stop, e + int(1.0 / DT)), post.stop)
            fell_post = bool((tilt[post, i] > fall_tilt).any() or (z[late, i] < fall_z).any())
            r["fell_within_3s_after"] = fell_post
            r["reached_handback"] = True
            r["success"] = bool((not r["fell_before_trick"]) and done and not fell_post and not m.timed_out)
            rows.append(r)
            prev_end = post.stop
        out.append({"env": i, "events": rows, "success": all(r["success"] for r in rows) if rows else None,
                    "fell_any": bool((tilt[:, i] > fall_tilt).any() or (z[:, i] < fall_z).any())})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="dance policy rsl_rl checkpoint (.pt)")
    ap.add_argument("--commands", required=True, help="dance command file (.npz) of the song")
    ap.add_argument("--policy-dir", required=True, help="folder with roulade.onnx, alpha_sitstand.onnx, velstand.onnx")
    ap.add_argument("--schedule", default=None, help="trick schedule json (dance/data/tricks/<song>.json)")
    ap.add_argument("--trick", default="none", choices=["none", *TRICKS], help="single event instead of --schedule")
    ap.add_argument("--dance-s", type=float, default=10.0, help="start time of the single --trick event")
    ap.add_argument("--bridge-s", type=float, default=None, help="single event bridge (default 1.0 roulade, 0 sitstand)")
    ap.add_argument("--robot", default="allcol", choices=["allcol", "walk"])
    ap.add_argument("--total-s", type=float, default=None, help="default: last event end + hold timeout + 4 s")
    ap.add_argument("--roulade-s", type=float, default=1.0)
    ap.add_argument("--sit-s", type=float, default=3.0)
    ap.add_argument("--stand-s", type=float, default=3.0)
    ap.add_argument("--hold-timeout-s", type=float, default=4.0, help="give up the upright check after this long")
    ap.add_argument("--num-envs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dr", action="store_true", help="keep the training domain randomization")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--render", default=None, help="mp4 path (1 env)")
    ap.add_argument("--json", default=None, help="result json path")
    ap.add_argument("--trace", default=None, help="npz with tilt/z/pitch_rate/state traces")
    args = ap.parse_args()
    events = load_schedule(args)
    print("[chain] events", events)
    render = Path(args.render) if args.render else None
    L, machines, total_s = run(args, events, args.dr, 1 if render else args.num_envs, render)
    args.total_s = total_s
    res = score(args, events, L, machines)
    n_ev = len(events)
    summary = {"n": len(res), "success": int(sum(bool(r["success"]) for r in res)), "fell_any": int(sum(r["fell_any"] for r in res)),
               "events": n_ev}
    for k in range(n_ev):
        rk = [r["events"][k] for r in res]
        summary[f"event{k}"] = {"trick": events[k]["trick"], "t_start": events[k]["t_start"],
                               "success": int(sum(x["success"] for x in rk)), "trick_done": int(sum(x["trick_done"] for x in rk)),
                               "timed_out": int(sum(x.get("timed_out", False) for x in rk))}
    summary.update({k: getattr(args, k) for k in ("robot", "dr", "seed", "total_s", "roulade_s", "sit_s", "stand_s", "hold_timeout_s")})
    summary["schedule"] = args.schedule
    dump = json.dumps({"summary": summary, "envs": res}, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(dump)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(dump)
    if args.trace:
        np.savez(args.trace, **L)


if __name__ == "__main__":
    main()
