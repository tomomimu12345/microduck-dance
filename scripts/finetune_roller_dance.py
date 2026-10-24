"""Continue Pollen's roller.onnx actor on the roller dance task (Mjlab-RollerDance-MicroDuck).

Start (actor from scripts/onnx_to_checkpoint.py, new critic):
    uv run scripts/finetune_roller_dance.py --init logs/rsl_rl/roller_dance/roller_onnx_init/actor.pt \
        --stage roller001 --run-name roller001 --num-envs 4096 --warmup-iterations 200 --iterations 1500

--stage selects the push pieces, wheel_speed weight and braking settings of one training stage
(ROLLER_STAGES in microduck_roller_dance_env_cfg.py): roller001, roller002_stage1 or roller002_stage2.
The default is roller002_stage2, the final one. dance/README.md has the three commands that made
roller_dance.

Resume a run of this script (actor, critic, Adam state, iteration and env step counter):
    uv run scripts/finetune_roller_dance.py --resume <run>/model_XXXX.pt --run-name roller_dance \
        --rewarm-iterations 50 --iterations 1250
On resume, --iterations counts from the resumed iteration, after --rewarm-iterations
updates with the actor rate at 0 (a critic rewarm when the rewards changed).

The actor head and body command slots, which roller.onnx only saw as zeros,
get their normalizer columns reseeded with the library statistics and their
input weight columns zeroed, so step 0 acts exactly like roller.onnx whatever
the head and body commands are. The check is printed as STEP0_PARITY.

Adam has two parameter groups, actor and critic, with fixed learning rates.
The first --warmup-iterations updates run with the actor learning rate at 0
(critic warmup: the critic is new, roller.onnx ships without one), the rest
with --actor-lr. Iterations are counted from 0 at the start.
"""

from __future__ import annotations

import argparse
import json
import math
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch

TASK = "Mjlab-RollerDance-MicroDuck"
HEAD_BODY_TERMS = ("head_command", "body_command")


def library_command_stats(path: str) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Mean and std of the library head (3:7) and body (7:13) columns."""
    from microduck_dance.command_file import load_command_file

    commands, _, _ = load_command_file(path)
    c = torch.as_tensor(commands, dtype=torch.float32)
    return {name: (c[:, lo:hi].mean(0), c[:, lo:hi].std(0, unbiased=False))
            for name, (lo, hi) in (("head_command", (3, 7)), ("body_command", (7, 13)))}


def install_optimizer(alg, actor_lr: float, critic_lr: float) -> None:
    """Adam with an actor group and a critic group. schedule 'fixed' keeps both rates."""
    for v in (actor_lr, critic_lr):
        if not math.isfinite(v) or v < 0:
            raise ValueError("learning rates must be finite and non-negative")
    alg.optimizer = torch.optim.Adam([
        {"params": list(alg.actor.parameters()), "lr": actor_lr, "name": "actor"},
        {"params": list(alg.critic.parameters()), "lr": critic_lr, "name": "critic"},
    ])
    alg.schedule = "fixed"
    alg.learning_rate = actor_lr


def set_rates(alg, actor_lr: float, critic_lr: float) -> None:
    g = alg.optimizer.param_groups
    g[0]["lr"], g[1]["lr"] = actor_lr, critic_lr
    alg.learning_rate = actor_lr


def prepare_actor(runner, obs_manager, init_sd: dict, library: str) -> dict:
    """Load the roller.onnx actor, reseed and zero its head and body columns. Returns a manifest dict."""
    from microduck_dance.basketball_finetune import reseed_normalizer, term_slice

    actor = runner.alg.actor
    actor.load_state_dict(init_sd, strict=True)
    stats = library_command_stats(library)
    done = {}
    with torch.no_grad():
        w = dict(actor.named_parameters())["mlp.0.weight"]
        for term in HEAD_BODY_TERMS:
            cols = term_slice(obs_manager, "actor", term)
            mean, std = stats[term]
            reseed_normalizer(actor.obs_normalizer, cols, mean, std)
            w[:, cols] = 0.0
            done[term] = [cols.start, cols.stop]
    return {"reseeded_and_zeroed_actor_columns": done,
            "library_stats": {k: [v[0].tolist(), v[1].tolist()] for k, v in stats.items()}}


def step0_parity(runner, obs) -> float:
    """Max |actor(obs) - actor(obs with head and body slots zeroed)|. 0 means step 0 is roller.onnx."""
    actor = runner.alg.actor
    zeroed = obs.clone()
    zeroed["actor"][:, 51:61] = 0.0
    with torch.no_grad():
        a = actor(obs)
        b = actor(zeroed)
    return float((a - b).abs().max())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--init", type=Path, help="actor-only checkpoint from onnx_to_checkpoint.py")
    src.add_argument("--resume", type=Path, help="full checkpoint of a run of this script")
    p.add_argument("--run-name", required=True)
    p.add_argument("--stage", default=None,
                   help="training stage settings, roller001 / roller002_stage1 / roller002_stage2 (default)")
    p.add_argument("--num-envs", type=int, default=4096)
    p.add_argument("--warmup-iterations", type=int, default=200)
    p.add_argument("--iterations", type=int, default=1500, help="iterations after the warmup")
    p.add_argument("--rewarm-iterations", type=int, default=0,
                   help="resume only: updates with the actor rate at 0 right after the resumed iteration")
    p.add_argument("--actor-lr", type=float, default=2e-5)
    p.add_argument("--critic-lr", type=float, default=3e-4)
    p.add_argument("--save-interval", type=int, default=250)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-root", type=Path, default=Path("logs/rsl_rl"))
    a = p.parse_args()
    if min(a.num_envs, a.save_interval) < 1 or min(a.warmup_iterations, a.iterations, a.rewarm_iterations) < 0:
        p.error("bad sizes")
    if a.rewarm_iterations and a.resume is None:
        p.error("--rewarm-iterations needs --resume")

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_rl_cfg
    from mjlab.utils.os import dump_yaml
    from mjlab.utils.torch import configure_torch_backends

    import microduck_dance.tasks  # noqa: F401  (registers the task)
    from mjlab_microduck.tasks import MicroduckOnPolicyRunner
    from microduck_dance.tasks import microduck_roller_dance_env_cfg as rd

    configure_torch_backends()
    stage = a.stage or rd.FINAL_STAGE
    if stage not in rd.ROLLER_STAGES:
        p.error(f"--stage must be one of {list(rd.ROLLER_STAGES)}")
    cfg = rd.make_microduck_roller_dance_env_cfg(stage=stage)
    cfg.scene.num_envs = a.num_envs
    cfg.seed = a.seed
    agent = asdict(load_rl_cfg(TASK))
    agent.update(run_name=a.run_name, seed=a.seed, save_interval=a.save_interval,
                 max_iterations=a.warmup_iterations + a.iterations)
    agent["algorithm"]["learning_rate"] = a.actor_lr
    agent["algorithm"]["schedule"] = "fixed"
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = a.log_root / agent["experiment_name"] / f"{stamp}_{a.run_name}"
    dump_yaml(log_dir / "params/env.yaml", asdict(cfg))
    dump_yaml(log_dir / "params/agent.yaml", deepcopy(agent))
    manifest = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    manifest.update(task=TASK, action_scale=rd.ROLLER_ACTION_SCALE, stand_height=rd.ROLLER_STAND_HEIGHT,
                    stand_pitch=rd.ROLLER_STAND_PITCH, push_gain=cfg.commands["twist"].push_gain,
                    push_range=list(cfg.commands["twist"].push_range),
                    heading_max=cfg.commands["twist"].heading_max, library=rd.DANCE_LIBRARY,
                    push_piece_prob=cfg.commands["twist"].push_piece_prob,
                    zero_twist_prob=cfg.commands["twist"].zero_prob,
                    wheel_speed_weight=cfg.rewards["wheel_speed"].weight, stage=stage,
                    braking_weight=cfg.rewards["braking"].weight,
                    braking_vel_std=cfg.rewards["braking"].params["vel_std"])

    raw = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    env = RslRlVecEnvWrapper(raw, clip_actions=agent["clip_actions"])
    try:
        runner = MicroduckOnPolicyRunner(env, agent, str(log_dir), "cuda:0")
        install_optimizer(runner.alg, a.actor_lr, a.critic_lr)
        obs_manager = raw.observation_manager
        assert obs_manager.active_terms["actor"] == ["base_ang_vel", "projected_gravity", "joint_pos", "joint_vel",
                                                     "actions", "command", "head_command", "body_command"], \
            obs_manager.active_terms["actor"]
        if a.init is not None:
            init = torch.load(a.init, map_location="cuda:0", weights_only=False)
            manifest["init_infos"] = {k: v for k, v in (init.get("infos") or {}).items() if k != "onnx_metadata"}
            manifest.update(prepare_actor(runner, obs_manager, init["actor_state_dict"], rd.DANCE_LIBRARY))
            start_iter = 0
        else:
            runner.load(str(a.resume), map_location="cuda:0")
            start_iter = runner.current_learning_iteration + 1
            runner.current_learning_iteration = start_iter
        manifest["start_iteration"] = start_iter
        env.reset()
        obs = env.get_observations()
        assert obs["actor"].shape == (a.num_envs, 61), obs["actor"].shape
        assert torch.count_nonzero(obs["actor"][:, 51:55]) > 0, "head command slots are zero"
        if a.init is not None:
            err = step0_parity(runner, obs)
            manifest["step0_parity_max_abs"] = err
            print("STEP0_PARITY", err, flush=True)
            assert err < 1e-5, err

        # Critic warmup: actor rate 0 for the first warmup updates (counted from iteration 0),
        # and on resume for the first rewarm updates after the resumed iteration.
        state = {"it": start_iter}
        update = runner.alg.update
        rewarm_end = start_iter + a.rewarm_iterations

        def staged_update():
            warm = state["it"] < a.warmup_iterations or state["it"] < rewarm_end
            set_rates(runner.alg, 0.0 if warm else a.actor_lr, a.critic_lr)
            state["it"] += 1
            return update()

        runner.alg.update = staged_update
        (log_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        runner.add_git_repo_to_log(__file__)
        print("FINETUNE_START", json.dumps({"log_dir": str(log_dir), **{k: manifest[k] for k in
              ("start_iteration", "warmup_iterations", "iterations", "actor_lr", "critic_lr", "action_scale")}}),
              flush=True)
        total = a.warmup_iterations + a.iterations if a.init is not None else rewarm_end + a.iterations
        manifest["end_iteration"] = total
        (log_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        runner.learn(max(total - start_iter, 0), init_at_random_ep_len=True)
        print("FINETUNE_DONE", str(log_dir), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
