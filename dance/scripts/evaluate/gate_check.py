"""dance_v3 go/no-go gate on a checkpoint (v3 design report, section 3).

Measures, in 16 envs without DR, the realized forward speed under a constant
vx 0.2 command and the realized yaw rate under a constant wz 0.5 command,
each started from standstill (2 s of zero command, then a step to the value
held for 10 s, measured over the last 8 s). Envs 0 to 7 get vx, 8 to 15 get wz.
Optionally reads the last logged Episode_Reward/air_time from a training log.

Gate: air_time >= 0.5, vx 0.2 gives >= 0.10 m/s, wz 0.5 gives >= 0.3 rad/s.
Abort the run if air_time < 0.3.

Run: uv run python dance/scripts/evaluate/gate_check.py --checkpoint logs/rsl_rl/dance_v3/<run>/model_XXXX.pt \
        --log logs/dance_v3.log [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import play_dance as pd  # noqa: E402

from microduck_dance.command_file import save_command_file  # noqa: E402

DT = 0.02
N = 16
REST_S, TEST_S, SKIP_S = 2.0, 10.0, 2.0
VX, WZ = 0.2, 0.5
GATE = {"air_time": 0.5, "vx": 0.10, "wz": 0.3}
ABORT_AIR_TIME = 0.3


def last_air_time(log: Path) -> tuple[float | None, int | None]:
    val, it = None, None
    pat_it = re.compile(r"Learning iteration (\d+)/")
    pat_air = re.compile(r"Episode_Reward/air_time:\s*([-0-9.eE]+)")
    with log.open(errors="replace") as f:
        for line in f:
            m = pat_it.search(line)
            if m:
                it = int(m.group(1))
            m = pat_air.search(line)
            if m:
                val = float(m.group(1))
    return val, it


def frames() -> np.ndarray:
    steps = int(round((REST_S + TEST_S) / DT))
    t = np.arange(steps) * DT
    fr = np.zeros((steps, N, 13), dtype=np.float32)
    on = t >= REST_S
    fr[on, : N // 2, 0] = VX
    fr[on, N // 2 :, 2] = WZ
    return fr


def install_frames(env, fr):
    import torch

    ar = torch.arange(N, device=env.device)
    for name, (lo, hi) in (("twist", (0, 3)), ("head_pose", (3, 7)), ("body_pose", (7, 13))):
        src = env.command_manager.get_term(name).source
        src.frames = torch.as_tensor(fr[:, :, lo:hi], dtype=torch.float32, device=env.device)

        def value(self, ar=ar):
            return self.frames[self.index(), ar]

        src.value = types.MethodType(value, src)
        src.clock = 0
        src.start[:] = 0
        src.n[:] = 0
        src.ramp[:] = 0


def measure(checkpoint: Path, task: str, device: str | None) -> dict:
    import torch

    fr = frames()
    steps = fr.shape[0]
    with tempfile.TemporaryDirectory() as tmp:
        dummy = Path(tmp) / "zero.npz"
        save_command_file(dummy, np.zeros((steps, 13), np.float32), DT, {"song": "gate"})
        args = pd.build_parser().parse_args(
            ["--checkpoint", str(checkpoint), "--commands", str(dummy), "--task", task]
            + (["--device", device] if device else [])
        )
        env, venv, policy = pd._build(args, play=True, num_envs=N, seed=0, steps=steps, dt=DT, render=False)
    probe = pd._Probe(env)
    venv.reset()
    install_frames(env, fr)
    obs = venv.get_observations()
    meas, fell = [], np.zeros(N, dtype=bool)
    for _ in range(steps):
        with torch.inference_mode():
            act = policy(obs)
        obs, _, dones, extras = venv.step(act)
        to = extras.get("time_outs", torch.zeros_like(dones, dtype=torch.bool))
        fell |= (dones.bool() & ~to.bool()).cpu().numpy()
        meas.append(probe.measured().cpu().numpy().copy())
    m = np.stack(meas)
    t = np.arange(steps) * DT
    win = t >= REST_S + SKIP_S
    vx = m[win, : N // 2, 0].mean(0)
    wz = m[win, N // 2 :, 2].mean(0)
    rest = (t > 0.5) & (t < REST_S)
    return {
        "vx_cmd": VX,
        "vx_realized_mean": float(vx.mean()),
        "vx_realized_per_env": vx.round(3).tolist(),
        "wz_cmd": WZ,
        "wz_realized_mean": float(wz.mean()),
        "wz_realized_per_env": wz.round(3).tolist(),
        "rest_speed_mean": float(np.abs(m[rest][:, :, 0]).mean()),
        "falls": int(fell.sum()),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--log", type=Path, default=None, help="training log to read the last air_time from")
    ap.add_argument("--task", default="Mjlab-DanceV3-Flat-MicroDuck", help="registry id for the RL cfg and runner")
    ap.add_argument("--device", default=None)
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args(argv)

    res = measure(a.checkpoint, a.task, a.device)
    res["checkpoint"] = str(a.checkpoint)
    if a.log is not None:
        res["air_time"], res["log_iteration"] = last_air_time(a.log)
    checks = {
        "vx": res["vx_realized_mean"] >= GATE["vx"],
        "wz": res["wz_realized_mean"] >= GATE["wz"],
    }
    if res.get("air_time") is not None:
        checks["air_time"] = res["air_time"] >= GATE["air_time"]
        res["abort"] = res["air_time"] < ABORT_AIR_TIME
    res["checks"] = checks
    res["pass"] = all(checks.values())
    print(f"checkpoint {a.checkpoint}")
    if "air_time" in res:
        print(f"air_time (log, iter {res.get('log_iteration')}): {res['air_time']}  gate >= {GATE['air_time']}, abort < {ABORT_AIR_TIME}")
    print(f"vx {VX} from rest: realized {res['vx_realized_mean']:.3f} m/s  gate >= {GATE['vx']}")
    print(f"wz {WZ} from rest: realized {res['wz_realized_mean']:.3f} rad/s  gate >= {GATE['wz']}")
    print(f"falls {res['falls']} / {N}")
    print("GATE", "PASS" if res["pass"] else "FAIL", checks, "ABORT" if res.get("abort") else "")
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(res, indent=2))
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
