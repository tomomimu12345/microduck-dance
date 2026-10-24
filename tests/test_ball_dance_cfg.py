"""Cfg-level tests for the ball dance task (Mjlab-BallDance-MicroDuck).

Building the env needs warp on the GPU here, so these tests check the cfg, the
pure torch command sources and the normalizer reseed. The GPU smoke test covers
the built env.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from microduck_dance.command_file import load_command_file
from microduck_dance.ranges import BALL_TWIST_RANGES, BALL_TWIST_SCALE, BODY, HEAD, TWIST

LIBRARY = Path(__file__).resolve().parents[1] / "dance" / "data" / "library.npz"
needs_library = pytest.mark.skipif(not LIBRARY.is_file(), reason="dance/data/library.npz missing")

B11_STEP_COUNTER = 168048  # 6999 x 24, restored on resume

# Reward and command names of the basketball task before this task (snapshot).
BASKETBALL_REWARDS = {
    "track_linear_velocity": 1.0,
    "track_angular_velocity": 0.5,
    "pose": 1.0,
    "body_ang_vel": -0.05,
    "angular_momentum": -0.02,
    "dof_pos_limits": -1.0,
    "action_rate_l2": -0.05,
    "self_collisions": -1.0,
    "upright": 1.0,
    "centered": 2.0,
    "feet_on_ball": 1.0,
    "height": 1.0,
    "ball_speed": -0.05,
}
BASKETBALL_COMMANDS = ["twist"]
BASKETBALL_CURRICULA = ["action_rate_weight", "standing_envs", "com_range", "head_com_range", "basketball_hold"]

# Positive weight terms that are rewards (exp kernels, tracking), not costs.
POSITIVE_REWARDS = {
    "track_linear_velocity",
    "track_angular_velocity",
    "pose",
    "upright",
    "centered",
    "feet_on_ball",
    "height",
    "head_pose_tracking",
    "body_pose_tracking",
}


@pytest.fixture(scope="module")
def cfg():
    from microduck_dance.tasks.microduck_ball_dance_env_cfg import make_microduck_ball_dance_env_cfg

    return make_microduck_ball_dance_env_cfg()


@pytest.fixture(scope="module")
def play_cfg():
    from microduck_dance.tasks.microduck_ball_dance_env_cfg import make_microduck_ball_dance_env_cfg

    return make_microduck_ball_dance_env_cfg(play=True)


def _func_name(term):
    f = term.func
    return getattr(f, "__name__", type(f).__name__)


def test_actor_obs_is_61d_layout(cfg):
    from mjlab.envs.mdp import generated_commands

    terms = cfg.observations["actor"].terms
    assert list(terms) == [
        "base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions",
        "command", "head_command", "body_command",
    ]
    # 3 + 3 + 14 + 14 + 14 = 48 proprioception, then the 13D command block.
    for name in ("base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions"):
        assert terms[name].history_length in (0, 1)
    assert terms["head_command"].func is generated_commands
    assert terms["head_command"].params == {"command_name": "head_pose"}
    assert terms["body_command"].func is generated_commands
    assert terms["body_command"].params == {"command_name": "body_pose"}


def test_critic_keeps_ball_state_and_sees_head_command(cfg):
    from mjlab.envs.mdp import generated_commands

    from mjlab_microduck.tasks import mdp

    terms = cfg.observations["critic"].terms
    assert terms["body_command"].func is mdp.basketball_state
    assert terms["head_command"].func is generated_commands
    assert terms["head_command"].params == {"command_name": "head_pose"}
    # Same term list as the b11 critic (76D), so the checkpoint loads strictly.
    from mjlab_microduck.tasks.microduck_basketball_env_cfg import make_microduck_basketball_env_cfg

    ref = make_microduck_basketball_env_cfg(blind=True, history=1)
    assert list(terms) == list(ref.observations["critic"].terms)


def test_commands_are_library_terms(cfg):
    from microduck_dance.tasks.microduck_ball_dance_env_cfg import (
        BALL_BODY_SCALE,
        BallDanceTwistCommandCfg,
        ScaledLibraryPoseCommandCfg,
    )

    assert BALL_BODY_SCALE == 0.3
    assert list(cfg.commands) == ["twist", "head_pose", "body_pose"]
    twist, head, body = (cfg.commands[k] for k in ("twist", "head_pose", "body_pose"))
    assert isinstance(twist, BallDanceTwistCommandCfg)
    assert isinstance(head, ScaledLibraryPoseCommandCfg) and isinstance(body, ScaledLibraryPoseCommandCfg)
    for c in (twist, head, body):
        assert Path(c.command_file).resolve() == LIBRARY.resolve()
    assert twist.columns == (0, 3) and head.columns == (3, 7) and body.columns == (7, 13)
    assert twist.scale == BALL_TWIST_SCALE == 0.50
    assert twist.zero_prob == 0.15
    assert head.scale == 1.0 and body.scale == BALL_BODY_SCALE
    r = twist.ranges
    assert (r.lin_vel_x, r.lin_vel_y, r.ang_vel_z) == BALL_TWIST_RANGES


@needs_library
def test_twist_source_within_ball_range_and_has_zero_intervals():
    from microduck_dance.tasks.microduck_ball_dance_env_cfg import (
        library_frames,
        make_ball_twist_source,
    )

    frames = library_frames(str(LIBRARY), (0, 3))
    raw = load_command_file(LIBRARY)[0][:, TWIST]
    assert np.allclose(frames.numpy(), raw)
    g = torch.Generator().manual_seed(0)
    n = 512
    src = make_ball_twist_source(frames, BALL_TWIST_SCALE, BALL_TWIST_RANGES, 0.15, n, "cpu", generator=g)
    lo = torch.tensor([r[0] for r in BALL_TWIST_RANGES])
    hi = torch.tensor([r[1] for r in BALL_TWIST_RANGES])
    src.resample(torch.arange(n))
    zero_mode = src.mode == src.names.index("zero")
    for _ in range(200):
        v = src.step(0.02)
        assert torch.all(v >= lo - 1e-7) and torch.all(v <= hi + 1e-7)
    # About 15% of the envs draw an exact zero interval at a resample.
    assert 0.10 < zero_mode.float().mean().item() < 0.20
    assert torch.all(src.value()[zero_mode] == 0)
    # The scaled library values are the dance twist times the scale, clipped.
    lib = src.sources[0].frames
    expected = torch.clamp(frames * BALL_TWIST_SCALE, lo, hi)
    assert torch.allclose(lib, expected)
    # Scaling alone must already fit most frames, the clip only trims the tail.
    inside = ((frames * BALL_TWIST_SCALE >= lo) & (frames * BALL_TWIST_SCALE <= hi)).all(dim=1).float().mean()
    assert inside > 0.9


@needs_library
def test_pose_sources_scale_body_and_keep_head():
    from microduck_dance.tasks.microduck_ball_dance_env_cfg import (
        BALL_BODY_SCALE,
        library_frames,
        make_scaled_pose_source,
    )

    cmds = load_command_file(LIBRARY)[0]
    n = 256
    head = make_scaled_pose_source(library_frames(str(LIBRARY), (3, 7)), 1.0, n, "cpu",
                                   generator=torch.Generator().manual_seed(1))
    body = make_scaled_pose_source(library_frames(str(LIBRARY), (7, 13)), BALL_BODY_SCALE, n, "cpu",
                                   generator=torch.Generator().manual_seed(2))
    assert torch.allclose(head.frames, torch.as_tensor(cmds[:, HEAD], dtype=torch.float32))
    assert torch.allclose(body.frames, torch.as_tensor(cmds[:, BODY], dtype=torch.float32) * BALL_BODY_SCALE)
    ids = torch.arange(n)
    head.resample(ids)
    body.resample(ids)
    h, b = head.step(0.02), body.step(0.02)
    # Non-zero after a reset (the b11 slots were all zero).
    assert torch.count_nonzero(h) > 0 and torch.count_nonzero(b[:, 2:5]) > 0


def test_rewards_keep_basketball_and_add_pose_tracking(cfg):
    from mjlab_microduck.tasks import mdp
    from microduck_dance.tasks.dance_commands import dance_body_tracking
    from microduck_dance.tasks.microduck_ball_dance_env_cfg import BALL_STAND_HEIGHT
    from mjlab_microduck.tasks.microduck_basketball_env_cfg import BALL_RADIUS

    for name, w in BASKETBALL_REWARDS.items():
        assert cfg.rewards[name].weight == w, name
    assert set(cfg.rewards) == set(BASKETBALL_REWARDS) | {"head_pose_tracking", "body_pose_tracking"}
    head = cfg.rewards["head_pose_tracking"]
    assert head.func is mdp.head_pose_tracking and head.weight == 2.0
    assert head.params == {"command_name": "head_pose", "std": 0.25}
    body = cfg.rewards["body_pose_tracking"]
    assert body.func is dance_body_tracking and body.weight == 1.0
    assert body.params["command_name"] == "body_pose"
    assert body.params["z_std"] == 0.02 and body.params["angle_std"] == 0.15
    assert body.params["nominal_height"] == BALL_STAND_HEIGHT
    # One reference with the basketball height reward. That reward targets the
    # root at ball centre + R + root_height, and the ball centre rests at R.
    h = cfg.rewards["height"].params
    assert BALL_STAND_HEIGHT == pytest.approx(BALL_RADIUS + h["ball_radius"] + h["root_height"])
    assert BALL_STAND_HEIGHT == pytest.approx(0.365)


def test_penalty_signs(cfg):
    """AGENTS.md sign rule. Costs (>= 0) take a negative weight. Self-negating
    microduck penalties (*_penalty, *_l1, <= 0) take a positive weight. Every
    other positive weight must be a reward."""
    for name, term in cfg.rewards.items():
        fn = _func_name(term)
        module = getattr(term.func, "__module__", "")
        self_negating = module.startswith(("mjlab_microduck", "microduck_dance")) and fn.endswith(("_penalty", "_l1"))
        if self_negating:
            assert term.weight >= 0, f"{name} ({fn}) is self-negating, weight {term.weight}"
        elif term.weight > 0:
            assert name in POSITIVE_REWARDS, f"{name} ({fn}) is a cost with a positive weight"
        else:
            assert name not in POSITIVE_REWARDS, f"{name} is a reward with weight {term.weight}"
    assert cfg.rewards["ball_speed"].weight < 0


def test_no_new_curriculum_and_standing_envs_removed(cfg, play_cfg):
    # The basketball curricula minus standing_envs (it zeroes the twist of some
    # envs, the zero intervals of the twist source do that here). No new term.
    assert list(cfg.curriculum) == ["action_rate_weight", "com_range", "head_com_range", "basketball_hold"]
    assert "standing_envs" not in play_cfg.curriculum


def test_start_pose_on_ball_apex(cfg):
    spawn = cfg.events["reset_basketball"]
    from mjlab_microduck.tasks.microduck_basketball_env_cfg import BALL_RADIUS, ROOT_HEIGHT

    assert spawn.params["ball_radius"] == BALL_RADIUS and spawn.params["root_height"] == ROOT_HEIGHT


def test_basketball_task_unchanged():
    from mjlab_microduck.tasks import mdp
    from mjlab_microduck.tasks.microduck_basketball_env_cfg import make_microduck_basketball_env_cfg

    import microduck_dance.tasks.microduck_ball_dance_env_cfg  # noqa: F401  building it must not leak

    ref = make_microduck_basketball_env_cfg(blind=True, history=1)
    assert {k: v.weight for k, v in ref.rewards.items()} == BASKETBALL_REWARDS
    assert list(ref.commands) == BASKETBALL_COMMANDS
    assert list(ref.curriculum) == BASKETBALL_CURRICULA
    assert ref.observations["actor"].terms["head_command"].func is mdp.basketball_head_pad
    assert ref.observations["actor"].terms["body_command"].func is mdp.basketball_body_pad


def test_registration_resolves():
    import microduck_dance.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

    from microduck_dance.tasks.microduck_ball_dance_env_cfg import BallDanceTwistCommandCfg

    env_cfg = load_env_cfg("Mjlab-BallDance-MicroDuck")
    assert isinstance(env_cfg.commands["twist"], BallDanceTwistCommandCfg)
    play = load_env_cfg("Mjlab-BallDance-MicroDuck", play=True)
    assert play.actions["ball_hold"].levels == (0.0,)
    rl = load_rl_cfg("Mjlab-BallDance-MicroDuck")
    basket = load_rl_cfg("Mjlab-Basketball-MicroDuck")
    assert rl.experiment_name == "ball_dance"
    assert rl.actor == basket.actor
    assert rl.actor.class_name == "RNNModel" and rl.actor.rnn_type == "lstm"


def test_play_dance_uses_module_body_scale():
    text = (Path(__file__).resolve().parents[1] / "dance" / "scripts" / "evaluate" / "play_dance.py").read_text()
    assert "ball_dance.BALL_BODY_SCALE" in text
    assert 'getattr(ball_dance, "BALL_BODY_SCALE"' not in text


# --------------------------------------------------------------------------- continuation


def _fake_checkpoint():
    return {'actor_state_dict': {
        'mlp.0.weight': torch.empty(512, 256), 'mlp.2.weight': torch.empty(256, 512),
        'mlp.4.weight': torch.empty(128, 256), 'mlp.6.weight': torch.empty(14, 128),
        'rnn.rnn.weight_ih_l0': torch.empty(1024, 61),
        'rnn.rnn.weight_hh_l0': torch.empty(1024, 256),
    }}


def test_continuation_for_ball_dance():
    from microduck_dance.basketball_finetune import continuation_configs
    from microduck_dance.tasks.microduck_ball_dance_env_cfg import BallDanceTwistCommandCfg

    cfg, agent = continuation_configs(_fake_checkpoint(), task="Mjlab-BallDance-MicroDuck",
                                      push_interval_s=(1.5, 3.))
    assert isinstance(cfg.commands["twist"], BallDanceTwistCommandCfg)
    r = cfg.commands["twist"].ranges
    assert (r.lin_vel_x, r.lin_vel_y, r.ang_vel_z) == BALL_TWIST_RANGES
    assert cfg.actions['ball_hold'].levels == (0.,)
    assert 'basketball_hold' not in cfg.curriculum and 'action_rate_weight' not in cfg.curriculum
    assert cfg.rewards['action_rate_l2'].weight == -.2
    assert set(cfg.rewards) >= {"head_pose_tracking", "body_pose_tracking"}
    assert agent['experiment_name'] == 'ball_dance'
    assert agent['actor']['class_name'] == 'RNNModel'
    assert agent['algorithm']['schedule'] == 'fixed'
    # No curriculum stage left that the restored step counter has not passed.
    for term in cfg.curriculum.values():
        p = term.params or {}
        for stage in p.get("weight_stages", []) + p.get("range_stages", []) + p.get("stages", []):
            assert stage["step"] <= B11_STEP_COUNTER


def test_continuation_rejects_command_scale_for_ball_dance():
    from microduck_dance.basketball_finetune import continuation_configs

    with pytest.raises(ValueError):
        continuation_configs(_fake_checkpoint(), task="Mjlab-BallDance-MicroDuck", command_scale=2.)


def test_continuation_default_task_is_basketball():
    from microduck_dance.basketball_finetune import continuation_configs
    from mjlab_microduck.tasks import mdp

    cfg, agent = continuation_configs(_fake_checkpoint())
    assert list(cfg.commands) == ["twist"]
    assert cfg.observations['actor'].terms['body_command'].func is mdp.basketball_body_pad
    assert agent['experiment_name'] == 'basketball'


def test_reseed_normalizer_slices():
    from rsl_rl.modules.normalization import EmpiricalNormalization

    from microduck_dance.basketball_finetune import reseed_normalizer

    norm = EmpiricalNormalization(10)
    norm._mean.zero_()
    norm._var.zero_()
    norm._std.zero_()
    norm.count.fill_(1000)
    mean = torch.tensor([0.1, -0.2, 0.0])
    std = torch.tensor([0.3, 0.05, 0.0])
    reseed_normalizer(norm, slice(4, 7), mean, std)
    assert torch.allclose(norm._mean[0, 4:7], mean)
    assert torch.allclose(norm._std[0, 4:7], std)
    assert torch.allclose(norm._var[0, 4:7], std ** 2)
    assert torch.count_nonzero(norm._mean[0, :4]) == 0 and torch.count_nonzero(norm._std[0, 7:]) == 0
    assert int(norm.count) == 1000


def test_term_slice_from_obs_manager():
    from microduck_dance.basketball_finetune import term_slice

    om = SimpleNamespace(active_terms={"actor": ["a", "command", "head_command", "body_command"]},
                         group_obs_term_dim={"actor": [(48,), (3,), (4,), (6,)]})
    assert term_slice(om, "actor", "head_command") == slice(51, 55)
    assert term_slice(om, "actor", "body_command") == slice(55, 61)


@needs_library
def test_library_command_stats_match_scaled_columns():
    from microduck_dance.tasks.microduck_ball_dance_env_cfg import BALL_BODY_SCALE, library_command_stats

    stats = library_command_stats(str(LIBRARY))
    cmds = load_command_file(LIBRARY)[0]
    assert np.allclose(stats["head_command"][0].numpy(), cmds[:, HEAD].mean(0), atol=1e-5)
    assert np.allclose(stats["head_command"][1].numpy(), cmds[:, HEAD].std(0), atol=1e-5)
    assert np.allclose(stats["body_command"][1].numpy(), cmds[:, BODY].std(0) * BALL_BODY_SCALE, atol=1e-5)


def test_reseed_command_normalizer_skips_critic_ball_state():
    from rsl_rl.modules.normalization import EmpiricalNormalization

    from microduck_dance.basketball_finetune import reseed_command_normalizer

    terms = {"actor": ["p", "head_command", "body_command"], "critic": ["p", "head_command", "body_command"]}
    params = {
        ("actor", "head_command"): {"command_name": "head_pose"},
        ("actor", "body_command"): {"command_name": "body_pose"},
        ("critic", "head_command"): {"command_name": "head_pose"},
        ("critic", "body_command"): {"pos_scale": 3.0, "vel_scale": 1.0},
    }
    om = SimpleNamespace(
        active_terms=terms,
        group_obs_term_dim={"actor": [(2,), (4,), (6,)], "critic": [(2,), (4,), (6,)]},
        get_term_cfg=lambda g, n: SimpleNamespace(params=params.get((g, n), {})),
    )
    actor, critic = EmpiricalNormalization(12), EmpiricalNormalization(12)
    for n in (actor, critic):
        n._std.zero_()
        n._var.zero_()
    runner = SimpleNamespace(alg=SimpleNamespace(actor=SimpleNamespace(obs_normalizer=actor),
                                                 critic=SimpleNamespace(obs_normalizer=critic)))
    stats = {"head_command": (torch.zeros(4), torch.full((4,), 0.2)),
             "body_command": (torch.zeros(6), torch.full((6,), 0.01))}
    done = reseed_command_normalizer(runner, om, stats)
    assert done == ["actor.head_command", "actor.body_command", "critic.head_command"]
    assert torch.allclose(actor._std[0, 2:6], torch.full((4,), 0.2))
    assert torch.allclose(actor._std[0, 6:], torch.full((6,), 0.01))
    assert torch.allclose(critic._std[0, 2:6], torch.full((4,), 0.2))
    assert torch.count_nonzero(critic._std[0, 6:]) == 0


def _fake_obs_manager():
    terms = {"actor": ["p", "head_command", "body_command"], "critic": ["p", "head_command", "body_command"]}
    params = {
        ("actor", "head_command"): {"command_name": "head_pose"},
        ("actor", "body_command"): {"command_name": "body_pose"},
        ("critic", "head_command"): {"command_name": "head_pose"},
        ("critic", "body_command"): {"pos_scale": 3.0, "vel_scale": 1.0},
    }
    return SimpleNamespace(
        active_terms=terms,
        group_obs_term_dim={"actor": [(2,), (4,), (6,)], "critic": [(2,), (4,), (6,)]},
        get_term_cfg=lambda g, n: SimpleNamespace(params=params.get((g, n), {})),
    )


class _Rnn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = torch.nn.LSTM(12, 8)


class _Actor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = _Rnn()
        self.mlp = torch.nn.Sequential(torch.nn.Linear(8, 3))


class _Critic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = torch.nn.Sequential(torch.nn.Linear(12, 16), torch.nn.ELU(), torch.nn.Linear(16, 1))


def test_zero_command_input_columns_after_load():
    from microduck_dance.basketball_finetune import zero_command_input_columns

    torch.manual_seed(0)
    actor, critic = _Actor(), _Critic()
    # The load: copy a "checkpoint" into fresh models.
    src_a, src_c = _Actor(), _Critic()
    actor.load_state_dict(src_a.state_dict())
    critic.load_state_dict(src_c.state_dict())
    runner = SimpleNamespace(alg=SimpleNamespace(actor=actor, critic=critic))
    done = zero_command_input_columns(runner, _fake_obs_manager(), ["head_command", "body_command"])
    assert done == ["actor.head_command[2:6]", "actor.body_command[6:12]", "critic.head_command[2:6]"]

    w_a, ref_a = actor.rnn.rnn.weight_ih_l0, src_a.rnn.rnn.weight_ih_l0
    assert torch.count_nonzero(w_a[:, 2:12]) == 0
    assert torch.equal(w_a[:, :2], ref_a[:, :2])
    w_c, ref_c = critic.mlp[0].weight, src_c.mlp[0].weight
    assert torch.count_nonzero(w_c[:, 2:6]) == 0
    assert torch.equal(w_c[:, :2], ref_c[:, :2]) and torch.equal(w_c[:, 6:], ref_c[:, 6:])
    # Every other parameter is untouched.
    for (n, p), (_, q) in zip(actor.named_parameters(), src_a.named_parameters()):
        if n != "rnn.rnn.weight_ih_l0":
            assert torch.equal(p, q), n
    for (n, p), (_, q) in zip(critic.named_parameters(), src_c.named_parameters()):
        if n != "mlp.0.weight":
            assert torch.equal(p, q), n

    # The commands then do not change the output (b11 behavior at step 0) ...
    x0 = torch.randn(1, 5, 12)
    x1 = x0.clone()
    x1[..., 2:12] = torch.randn(1, 5, 10) * 30
    assert torch.allclose(actor.rnn.rnn(x0)[0], actor.rnn.rnn(x1)[0])


# On this WSL machine torch aborts in CUDA init (torch.cuda.is_available and
# autograd backward both reach it) when CUDA_VISIBLE_DEVICES is empty.
# test_basketball_finetune.py aborts the same way. So this one runs only with
# the GPU visible.
_CUDA_HIDDEN = os.environ.get("CUDA_VISIBLE_DEVICES") in ("", "-1")


@pytest.mark.skipif(_CUDA_HIDDEN, reason="torch aborts in CUDA init here with CUDA hidden")
def test_zeroed_input_columns_still_get_gradients():
    from microduck_dance.basketball_finetune import zero_command_input_columns

    actor, critic = _Actor(), _Critic()
    runner = SimpleNamespace(alg=SimpleNamespace(actor=actor, critic=critic))
    zero_command_input_columns(runner, _fake_obs_manager(), ["head_command", "body_command"])
    x = torch.randn(1, 5, 12)
    actor.rnn.rnn(x)[0].sum().backward()
    critic.mlp(x[0]).sum().backward()
    assert torch.count_nonzero(actor.rnn.rnn.weight_ih_l0.grad[:, 2:12]) > 0
    assert torch.count_nonzero(critic.mlp[0].weight.grad[:, 2:6]) > 0


def test_continuation_manifest_fields(tmp_path):
    from microduck_dance.basketball_finetune import continuation_manifest

    ck = {"iter": 6999, "optimizer_state_dict": {"param_groups": [{"lr": 1e-4}]}}
    base = dict(checkpoint=tmp_path / "c.pt", run_name="r", num_envs=64, iterations=5, learning_rate=2e-5,
                action_rate_weight=-.2, command_scale=1., episode_seconds=10., seed=42,
                push_interval_s=None, save_interval=5)
    dance = continuation_manifest(SimpleNamespace(task="Mjlab-BallDance-MicroDuck", **base), ck, "x")
    assert dance["task"] == "Mjlab-BallDance-MicroDuck"
    assert "command_scale" not in dance and "head_body_slots" not in dance
    assert dance["command_source"] == "library" and dance["body_scale"] == 0.3
    assert dance["twist_scale"] == BALL_TWIST_SCALE and dance["body_stand_height"] == pytest.approx(0.365)
    bb = continuation_manifest(SimpleNamespace(task="Mjlab-Basketball-MicroDuck", **base), ck, "x")
    assert bb["task"] == "Mjlab-Basketball-MicroDuck" and bb["command_scale"] == 1.
    assert bb["head_body_slots"] == "zero" and "body_scale" not in bb
