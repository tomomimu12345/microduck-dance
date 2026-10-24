"""Measure the b11 standing root height on the ball (zero commands, free ball, no pushes)."""
import json
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from microduck_dance.basketball_finetune import BasketballContinuationRunner, continuation_configs

CK = "logs/rsl_rl/basketball/b11_6999/checkpoint.pt"
ck = torch.load(CK, map_location="cpu", weights_only=False)
out = {}
for dr in (True, False):
    cfg, agent = continuation_configs(ck)
    cfg.scene.num_envs = 64
    cfg.episode_length_s = 20.0
    cfg.events.pop("push_robot", None)
    if not dr:
        for k in [k for k, e in cfg.events.items() if e.mode in ("startup", "interval")]:
            if k != "expand_bam_friction_fields":
                cfg.events.pop(k)
    r = cfg.commands["twist"].ranges
    r.lin_vel_x = r.lin_vel_y = r.ang_vel_z = (0.0, 0.0)
    cfg.commands["twist"].rel_standing_envs = 1.0
    raw = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    env = RslRlVecEnvWrapper(raw, clip_actions=agent["clip_actions"])
    runner = BasketballContinuationRunner(env, agent, None, "cuda:0")
    runner.load(CK, map_location="cuda:0")
    policy = runner.get_inference_policy(device="cuda:0")
    obs = env.get_observations()
    alive = torch.ones(64, dtype=torch.bool, device="cuda:0")
    zs, zb = [], []
    with torch.inference_mode():
        for t in range(350):
            a = policy(obs)
            obs, _, dones, _ = env.step(a)
            policy.reset(dones)
            alive &= ~dones.bool()
            if t >= 100:
                root = raw.scene["robot"].data.root_link_pos_w
                ball = raw.scene["ball"].data.root_link_pos_w
                zs.append((root[:, 2] - raw.scene.env_origins[:, 2])[alive].cpu())
                zb.append((root[:, 2] - ball[:, 2])[alive].cpu())
    zs, zb = torch.cat(zs), torch.cat(zb)
    out["dr" if dr else "nodr"] = {
        "alive_envs": int(alive.sum()),
        "root_above_origin_median": float(zs.median()),
        "root_above_origin_p10_p90": [float(zs.quantile(0.1)), float(zs.quantile(0.9))],
        "root_above_ball_centre_median": float(zb.median()),
    }
    env.close()
print("STANCE", json.dumps(out, indent=1))
