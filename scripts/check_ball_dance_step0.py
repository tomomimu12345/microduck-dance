"""Runtime check of the b11 continuation on Mjlab-BallDance-MicroDuck (GPU).

Builds the real BallDance env, loads b11 twice (A goes through the reseed of the
command normalizer columns and the zeroing of the command input columns, B is
plain b11), and checks on the real observation manager that

* A with the head and body commands set to 0 acts like B on the same obs,
* A with the real (non-zero) commands acts like A with zeros,
* the critic input columns zeroed are the head slot only, and the ball state
  in the critic body slot keeps b11's weights,
* the critic values follow the same two rules.

Prints the max abs differences and exits 1 if one is above the tolerance.
"""
import argparse
from copy import deepcopy
import json
import sys
import tempfile
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.torch import configure_torch_backends

from microduck_dance.basketball_finetune import (
    BALL_DANCE_TASK, BasketballContinuationRunner, continuation_configs, input_weight,
    reseed_command_normalizer, term_slice, zero_command_input_columns,
)
from microduck_dance.tasks.microduck_ball_dance_env_cfg import library_command_stats

TOL = 1e-5


def _reset_memory(model):
    if hasattr(model, 'reset'):
        model.reset()


def _act(runner, obs):
    actor = runner.alg.actor
    _reset_memory(actor)
    with torch.inference_mode():
        return runner.get_inference_policy(device='cuda:0')(obs).clone()


def _value(runner, obs):
    critic = runner.alg.critic
    _reset_memory(critic)
    with torch.inference_mode():
        return critic(obs).clone()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint', type=Path)
    p.add_argument('--num-envs', type=int, default=64)
    p.add_argument('--json', type=Path)
    a = p.parse_args()
    configure_torch_backends()
    checkpoint = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    cfg, agent = continuation_configs(checkpoint, task=BALL_DANCE_TASK, push_interval_s=(1.5, 3.0))
    cfg.scene.num_envs = a.num_envs
    log_dir = Path(tempfile.mkdtemp(prefix='ball_dance_step0_'))
    raw = ManagerBasedRlEnv(cfg=cfg, device='cuda:0')
    env = RslRlVecEnvWrapper(raw, clip_actions=agent['clip_actions'])
    try:
        runner_a = BasketballContinuationRunner(env, deepcopy(agent), str(log_dir / 'a'), 'cuda:0')
        runner_a.load(str(a.checkpoint), map_location='cuda:0')
        runner_b = BasketballContinuationRunner(env, deepcopy(agent), str(log_dir / 'b'), 'cuda:0')
        runner_b.load(str(a.checkpoint), map_location='cuda:0')
        critic_w_b11 = input_weight(runner_b.alg.critic).detach().clone()
        actor_w_b11 = input_weight(runner_b.alg.actor).detach().clone()
        stats = library_command_stats()
        reseeded = reseed_command_normalizer(runner_a, raw.observation_manager, stats)
        zeroed = zero_command_input_columns(runner_a, raw.observation_manager, stats)

        om = raw.observation_manager
        a_head, a_body = term_slice(om, 'actor', 'head_command'), term_slice(om, 'actor', 'body_command')
        c_head, c_body = term_slice(om, 'critic', 'head_command'), term_slice(om, 'critic', 'body_command')

        # Critic columns: zeroed exactly on the head slot, b11 everywhere else.
        w_c = input_weight(runner_a.alg.critic).detach()
        zero_cols = [i for i in range(w_c.shape[1]) if torch.count_nonzero(w_c[:, i]) == 0]
        changed_cols = [i for i in range(w_c.shape[1]) if not torch.equal(w_c[:, i], critic_w_b11[:, i])]
        w_a = input_weight(runner_a.alg.actor).detach()
        actor_changed = [i for i in range(w_a.shape[1]) if not torch.equal(w_a[:, i], actor_w_b11[:, i])]

        env.reset()
        obs = env.get_observations()
        assert obs['actor'].shape[-1] == 61
        nonzero_cmd = int(torch.count_nonzero(obs['actor'][:, a_head.start:a_body.stop]))
        zero_obs = obs.clone() if hasattr(obs, 'clone') else {k: v.clone() for k, v in obs.items()}
        zero_obs['actor'][:, a_head] = 0.0
        zero_obs['actor'][:, a_body] = 0.0
        zero_obs['critic'][:, c_head] = 0.0

        act_a_zero = _act(runner_a, zero_obs)
        act_b_zero = _act(runner_b, zero_obs)
        act_a_real = _act(runner_a, obs)
        val_a_zero = _value(runner_a, zero_obs)
        val_b_zero = _value(runner_b, zero_obs)
        val_a_real = _value(runner_a, obs)

        res = {
            'num_envs': a.num_envs,
            'reseeded': reseeded,
            'zeroed': zeroed,
            'actor_cmd_obs_nonzero_entries': nonzero_cmd,
            'actor_head_slice': [a_head.start, a_head.stop], 'actor_body_slice': [a_body.start, a_body.stop],
            'critic_head_slice': [c_head.start, c_head.stop], 'critic_body_slice': [c_body.start, c_body.stop],
            'actor_columns_changed': actor_changed,
            'critic_columns_zero': zero_cols,
            'critic_columns_changed': changed_cols,
            'max_abs_action_a_zero_vs_b11': float((act_a_zero - act_b_zero).abs().max()),
            'max_abs_action_a_real_vs_a_zero': float((act_a_real - act_a_zero).abs().max()),
            'max_abs_value_a_zero_vs_b11': float((val_a_zero - val_b_zero).abs().max()),
            'max_abs_value_a_real_vs_a_zero': float((val_a_real - val_a_zero).abs().max()),
            'max_abs_action': float(act_b_zero.abs().max()),
        }
        ok = (
            nonzero_cmd > 0
            and actor_changed == list(range(a_head.start, a_body.stop))
            and changed_cols == list(range(c_head.start, c_head.stop))
            and all(c_head.start <= i < c_head.stop for i in zero_cols)
            and all(res[k] <= TOL for k in res if k.startswith('max_abs_') and k != 'max_abs_action')
        )
        res['ok'] = bool(ok)
        print('STEP0_CHECK', json.dumps(res), flush=True)
        if a.json:
            a.json.parent.mkdir(parents=True, exist_ok=True)
            a.json.write_text(json.dumps(res, indent=2) + '\n')
    finally:
        env.close()
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
