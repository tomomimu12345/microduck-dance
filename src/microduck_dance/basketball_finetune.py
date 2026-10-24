"""PPO continuation of the blind basketball policy, extended to the ball dance task.

Derived from src/mjlab_microduck/basketball_finetune.py of microduck-playground
(https://github.com/Vottivott/microduck-playground, commit f010ef1), Copyright 2026
Hannes von Essen, Apache License 2.0. The upstream module stays unchanged in the
submodule. set_fixed_learning_rate and BasketballContinuationRunner are imported
from it. Changes against the upstream file:

- continuation_configs takes task=BASKETBALL_TASK or BALL_DANCE_TASK. The ball dance
  task builds make_microduck_ball_dance_env_cfg, requires command_scale 1 and loads
  the runner cfg of that task. The body is otherwise the upstream function.
- New: term_slice, reseed_normalizer, command_slots, reseed_command_normalizer,
  input_weight and zero_command_input_columns, which give the head and body command
  slots that b11 only saw as zeros a normalizer and zeroed input weights.
- New: continuation_manifest, the run manifest that scripts/finetune_basketball.py
  wrote inline upstream, with ball dance fields.
"""
from dataclasses import asdict
import math

import torch

from mjlab.tasks.registry import load_rl_cfg
from mjlab_microduck.basketball_distillation import actor_model_cfg
from mjlab_microduck.basketball_finetune import (  # noqa: F401  (re-exported for scripts and tests)
    BasketballContinuationRunner,
    set_fixed_learning_rate,
)
from mjlab_microduck.tasks.microduck_basketball_env_cfg import make_microduck_basketball_env_cfg


BASKETBALL_TASK = 'Mjlab-Basketball-MicroDuck'
BALL_DANCE_TASK = 'Mjlab-BallDance-MicroDuck'


def continuation_configs(checkpoint, *, learning_rate=2e-5, action_rate_weight=-.2,
                         command_scale=1., episode_seconds=10., seed=42, push_interval_s=None,
                         task=BASKETBALL_TASK):
    if not math.isfinite(action_rate_weight) or action_rate_weight > 0:
        raise ValueError('action_rate_weight must be finite and non-positive')
    if not all(math.isfinite(x) and x > 0 for x in (learning_rate, command_scale, episode_seconds)):
        raise ValueError('learning rate, command scale and episode seconds must be positive')
    if task == BASKETBALL_TASK:
        cfg = make_microduck_basketball_env_cfg(blind=True, history=1)
    elif task == BALL_DANCE_TASK:
        if command_scale != 1.:
            raise ValueError('the ball dance twist is the scaled library twist, command_scale must be 1')
        from microduck_dance.tasks.microduck_ball_dance_env_cfg import make_microduck_ball_dance_env_cfg
        cfg = make_microduck_ball_dance_env_cfg()
    else:
        raise ValueError(f'unknown continuation task {task!r}')
    cfg.seed = seed
    if push_interval_s is not None:
        if len(push_interval_s) != 2 or not all(math.isfinite(v) and v > 0 for v in push_interval_s) or push_interval_s[0] > push_interval_s[1]:
            raise ValueError('push interval must be two positive ordered values')
        cfg.events['push_robot'].interval_range_s = tuple(push_interval_s)
    cfg.episode_length_s = episode_seconds
    cfg.actions['ball_hold'].levels = (0.,)
    cfg.curriculum.pop('basketball_hold', None)
    cfg.rewards['action_rate_l2'].weight = action_rate_weight
    cfg.curriculum.pop('action_rate_weight', None)
    cfg.commands['twist'].ranges.lin_vel_x = (-.15*command_scale, .15*command_scale)
    cfg.commands['twist'].ranges.lin_vel_y = (-.10*command_scale, .10*command_scale)
    cfg.commands['twist'].ranges.ang_vel_z = (-.50*command_scale, .50*command_scale)
    agent = asdict(load_rl_cfg(task))
    agent['actor'] = actor_model_cfg(checkpoint)
    if agent['actor']['class_name'] != 'RNNModel':
        raise ValueError('Expected a recurrent basketball source policy')
    agent['seed'] = seed
    agent['algorithm']['learning_rate'] = learning_rate
    agent['algorithm']['schedule'] = 'fixed'
    return cfg, agent


def term_slice(obs_manager, group, term):
    """Columns of one obs term in the flat obs vector of a group."""
    start = 0
    for name, dims in zip(obs_manager.active_terms[group], obs_manager.group_obs_term_dim[group]):
        width = math.prod(dims)
        if name == term:
            return slice(start, start + width)
        start += width
    raise KeyError(f'{term} not in obs group {group}')


def reseed_normalizer(normalizer, columns, mean, std):
    """Set the running mean and std of some columns, keeping the sample count.

    b11 saw zeros in its head and body command slots, so their std is 0 and the
    normalizer would divide new commands by eps alone. Its count (about 7e8) is
    so large that the reseeded values then stay put during the continuation.
    """
    mean = torch.as_tensor(mean, dtype=normalizer._mean.dtype, device=normalizer._mean.device)
    std = torch.as_tensor(std, dtype=normalizer._std.dtype, device=normalizer._std.device)
    with torch.no_grad():
        normalizer._mean[0, columns] = mean
        normalizer._std[0, columns] = std
        normalizer._var[0, columns] = std * std


def command_slots(obs_manager, terms):
    """(group, term, columns) of each term in terms that carries a command in that group.

    A term whose slot holds something else (the critic body_command holds the
    ball state) is skipped for that group.
    """
    out = []
    for group in ('actor', 'critic'):
        for term in terms:
            if (obs_manager.get_term_cfg(group, term).params or {}).get('command_name') is None:
                continue
            out.append((group, term, term_slice(obs_manager, group, term)))
    return out


def _models(runner):
    return {'actor': runner.alg.actor, 'critic': runner.alg.critic}


def reseed_command_normalizer(runner, obs_manager, stats):
    """Reseed the actor and critic normalizer columns of the command terms in stats.

    stats maps an obs term name to (mean, std).
    """
    done = []
    models = _models(runner)
    for group, term, columns in command_slots(obs_manager, stats):
        normalizer = getattr(models[group], 'obs_normalizer', None)
        if normalizer is None or not hasattr(normalizer, '_mean'):
            continue
        mean, std = stats[term]
        reseed_normalizer(normalizer, columns, mean, std)
        done.append(f'{group}.{term}')
    return done


def input_weight(model):
    """The weight that multiplies the normalized obs: the LSTM input weight of a
    recurrent model, else the first MLP layer."""
    params = dict(model.named_parameters())
    for name in ('rnn.rnn.weight_ih_l0', 'mlp.0.weight'):
        if name in params:
            return params[name]
    raise KeyError('no obs input weight found')


def zero_command_input_columns(runner, obs_manager, terms):
    """Zero the input weight columns of the command slots that b11 never trained.

    b11 only saw zeros in these slots, so their columns got no gradient and are
    still at their random init. With them at zero the first step reproduces b11
    exactly whatever the commands are, and PPO still sends gradients into them.
    """
    done = []
    models = _models(runner)
    for group, term, columns in command_slots(obs_manager, terms):
        with torch.no_grad():
            input_weight(models[group])[:, columns] = 0.0
        done.append(f'{group}.{term}[{columns.start}:{columns.stop}]')
    return done


def continuation_manifest(args, checkpoint, source_sha256):
    """Run manifest. Fields that only mean something for one task are left out of the other."""
    manifest = {**vars(args), 'task': args.task, 'checkpoint': str(args.checkpoint.resolve()),
                'source_sha256': source_sha256, 'source_iteration': checkpoint['iter'],
                'source_optimizer_lr': [g['lr'] for g in checkpoint['optimizer_state_dict']['param_groups']],
                'ball_state_actor': False, 'actor_width': 61, 'ball_hold': 0., 'schedule': 'fixed'}
    if args.task == BALL_DANCE_TASK:
        from microduck_dance.ranges import BALL_TWIST_SCALE
        from microduck_dance.tasks import microduck_ball_dance_env_cfg as bd
        # The twist is the scaled library twist, command_scale does not apply.
        manifest.pop('command_scale', None)
        manifest.update(command_source='library', library=bd.DANCE_LIBRARY, twist_scale=BALL_TWIST_SCALE,
                        body_scale=bd.BALL_BODY_SCALE, zero_twist_prob=bd.ZERO_TWIST_PROB,
                        body_stand_height=bd.BALL_STAND_HEIGHT)
    else:
        manifest.update(command_source='uniform', head_body_slots='zero')
    return manifest
