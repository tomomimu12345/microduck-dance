"""Resume the blind basketball actor/critic/Adam state with a fixed smaller LR.

--task Mjlab-BallDance-MicroDuck continues the same checkpoint on the ball dance
task. The head and body command slots, which b11 only saw as zeros, get their
obs normalizer columns reseeded with the library command statistics and
their input weight columns zeroed, so step 0 acts exactly like b11.

Derived from scripts/finetune_basketball.py of microduck-playground
(https://github.com/Vottivott/microduck-playground, commit f010ef1), Copyright 2026
Hannes von Essen, Apache License 2.0. The upstream script stays unchanged in the
submodule. Changes against the upstream file: the --task option, the helpers
come from microduck_dance.basketball_finetune, the log folder is
logs/rsl_rl/<experiment_name of the task>/ instead of logs/rsl_rl/basketball/,
the manifest is built by continuation_manifest, the ball dance branch reseeds
the normalizer and zeroes the command input columns, and the step 0 obs check
expects non-zero head and body commands on the ball dance task.
"""
import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.os import dump_yaml
from mjlab.utils.torch import configure_torch_backends
from mjlab_microduck.basketball_distillation import checkpoint_sha256
from microduck_dance.basketball_finetune import (
    BALL_DANCE_TASK, BASKETBALL_TASK, BasketballContinuationRunner, continuation_configs,
    continuation_manifest, reseed_command_normalizer, zero_command_input_columns,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint', type=Path)
    p.add_argument('--task', default=BASKETBALL_TASK, choices=(BASKETBALL_TASK, BALL_DANCE_TASK))
    p.add_argument('--run-name', required=True)
    p.add_argument('--num-envs', type=int, default=4096)
    p.add_argument('--iterations', type=int, default=1000)
    p.add_argument('--learning-rate', type=float, default=2e-5)
    p.add_argument('--action-rate-weight', type=float, default=-.2)
    p.add_argument('--command-scale', type=float, default=1.)
    p.add_argument('--episode-seconds', type=float, default=10.)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--push-interval-s', type=float, nargs=2)
    p.add_argument('--save-interval', type=int, default=250)
    a = p.parse_args()
    if min(a.num_envs,a.iterations,a.save_interval) < 1:
        p.error('num-envs, iterations and save-interval must be positive')
    configure_torch_backends()
    checkpoint = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    cfg, agent = continuation_configs(checkpoint, learning_rate=a.learning_rate,
        action_rate_weight=a.action_rate_weight, command_scale=a.command_scale,
        episode_seconds=a.episode_seconds, seed=a.seed, push_interval_s=a.push_interval_s,
        task=a.task)
    cfg.scene.num_envs = a.num_envs
    agent.update(run_name=a.run_name, max_iterations=a.iterations, save_interval=a.save_interval)
    stamp = datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%S')
    log_dir = Path('logs/rsl_rl') / agent['experiment_name'] / f'{stamp}_{a.run_name}'
    dump_yaml(log_dir/'params/env.yaml', asdict(cfg))
    dump_yaml(log_dir/'params/agent.yaml', deepcopy(agent))
    manifest = continuation_manifest(a, checkpoint, checkpoint_sha256(a.checkpoint))
    (log_dir/'manifest.json').write_text(json.dumps(manifest, indent=2))
    raw = ManagerBasedRlEnv(cfg=cfg, device='cuda:0')
    env = RslRlVecEnvWrapper(raw, clip_actions=agent['clip_actions'])
    try:
        runner = BasketballContinuationRunner(env, agent, str(log_dir), 'cuda:0')
        runner.load(str(a.checkpoint), map_location='cuda:0')
        assert runner.alg.schedule == 'fixed'
        assert all(g['lr'] == a.learning_rate for g in runner.alg.optimizer.param_groups)
        assert raw.common_step_counter >= checkpoint['iter'] * agent['num_steps_per_env']
        if a.task == BALL_DANCE_TASK:
            from microduck_dance.tasks.microduck_ball_dance_env_cfg import library_command_stats
            stats = library_command_stats()
            manifest['normalizer_reseeded'] = reseed_command_normalizer(runner, raw.observation_manager, stats)
            manifest['zeroed_input_columns'] = zero_command_input_columns(runner, raw.observation_manager, stats)
            (log_dir/'manifest.json').write_text(json.dumps(manifest, indent=2))
            print('NORMALIZER_RESEEDED', manifest['normalizer_reseeded'], flush=True)
            print('INPUT_COLUMNS_ZEROED', manifest['zeroed_input_columns'], flush=True)
        env.reset()
        obs = env.get_observations()['actor']
        assert obs.shape == (a.num_envs,61)
        if a.task == BALL_DANCE_TASK:
            assert torch.count_nonzero(obs[:,51:55]) > 0 and torch.count_nonzero(obs[:,57:60]) > 0
        else:
            assert torch.count_nonzero(obs[:,55:]) == 0
        runner.add_git_repo_to_log(__file__)
        print('FINETUNE_START', json.dumps({'log_dir':str(log_dir), **manifest}), flush=True)
        runner.learn(a.iterations, init_at_random_ep_len=True)
        assert all(g['lr'] == a.learning_rate for g in runner.alg.optimizer.param_groups)
        print('FINETUNE_DONE', str(log_dir), flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
