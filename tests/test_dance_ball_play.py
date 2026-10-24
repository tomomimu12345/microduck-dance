"""Ball support in dance/scripts/evaluate/play_dance.py (CPU only, no env is stepped)."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

_PATH = Path(__file__).resolve().parents[1] / "dance" / "scripts" / "evaluate" / "play_dance.py"
_spec = importlib.util.spec_from_file_location("play_dance", _PATH)
play_dance = importlib.util.module_from_spec(_spec)
sys.modules["play_dance"] = play_dance  # dataclass needs its module registered
_spec.loader.exec_module(play_dance)

SONG = Path(__file__).resolve().parents[1] / "dance" / "data" / "commands" / "songs" / "swag.npz"
needs_song = pytest.mark.skipif(
    not SONG.exists(),
    reason="song command files are not in git (made from the songs by dance/scripts/retarget/retarget_all.sh)",
)
B11 = Path(__file__).resolve().parents[1] / "logs" / "rsl_rl" / "basketball" / "b11_6999" / "checkpoint.pt"


# Fall detection


def test_is_ball_fall():
    top = 0.24
    # Trunk 5 cm below the ball top: fallen off.
    assert play_dance.is_ball_fall(top - 0.05, top, False) is True
    # Upright on the apex (root 0.125 m above the ball top), no trunk contact.
    assert play_dance.is_ball_fall(top + 0.125, top, False) is False
    # Trunk touching the ball counts even when the trunk is still high.
    assert play_dance.is_ball_fall(top + 0.125, top, True) is True


def test_is_ball_fall_batched():
    z = torch.tensor([0.19, 0.365, 0.365])
    top = torch.full((3,), 0.24)
    contact = torch.tensor([False, False, True])
    assert play_dance.is_ball_fall(z, top, contact).tolist() == [True, False, True]


# Recurrent policy state


class _FakeVenv:
    """Steps return dones from a schedule. obs["actor"] is a (N, 61) counter."""

    def __init__(self, n, done_at):
        self.n = n
        self.k = 0
        self.done_at = done_at  # step -> list of env ids that finish on that step

    def step(self, actions):
        dones = torch.zeros(self.n, dtype=torch.bool)
        for i in self.done_at.get(self.k, []):
            dones[i] = True
        self.k += 1
        obs = {"actor": torch.ones(self.n, 61), "critic": torch.ones(self.n, 7)}
        return obs, torch.zeros(self.n), dones, {"time_outs": torch.zeros(self.n, dtype=torch.bool)}


class _FakeRecurrentPolicy:
    """Hidden state counts the steps since the last reset of each env."""

    def __init__(self, n):
        self.state = torch.zeros(n)
        self.seen_state = []
        self.seen_obs = []

    def __call__(self, obs):
        self.seen_state.append(self.state.clone())
        self.seen_obs.append(obs["actor"].clone())
        self.state += 1.0
        return torch.zeros(len(self.state), 14)

    def reset(self, dones):
        self.state[dones.bool()] = 0.0


def test_recurrent_state_resets_on_done():
    venv = _FakeVenv(2, done_at={2: [1]})
    policy = _FakeRecurrentPolicy(2)
    obs = {"actor": torch.ones(2, 61), "critic": torch.ones(2, 7)}
    for _ in range(5):
        obs, dones, extras = play_dance.policy_step(policy, obs, venv)
    seen = torch.stack(policy.seen_state)
    # env 0 never resets, its state is carried.
    assert seen[:, 0].tolist() == [0, 1, 2, 3, 4]
    # env 1 finished on step 2, so step 3 sees a zero state, then it is carried again.
    assert seen[:, 1].tolist() == [0, 1, 2, 0, 1]


def test_recurrent_state_resets_on_done_rsl_rl_lstm():
    """Same check on the real rsl_rl LSTM actor, run the way play_dance runs it (inference mode)."""
    from rsl_rl.models import RNNModel
    from tensordict import TensorDict

    n = 3
    obs = TensorDict({"actor": torch.randn(n, 61)}, batch_size=[n])
    model = RNNModel(obs, {"actor": ["actor"]}, "actor", 14, hidden_dims=[16], activation="elu",
                     obs_normalization=False, rnn_type="lstm", rnn_hidden_dim=8, rnn_num_layers=1)
    model.eval()

    class Venv:
        k = 0

        def step(self, actions):
            Venv.k += 1
            dones = torch.tensor([False, Venv.k == 2, False])
            return TensorDict({"actor": torch.randn(n, 61)}, batch_size=[n]), torch.zeros(n), dones, {}

    venv = Venv()
    obs, _, _ = play_dance.policy_step(model, obs, venv)
    h, c = model.get_hidden_state()
    assert h.abs().sum(dim=-1).min() > 0
    obs, dones, _ = play_dance.policy_step(model, obs, venv)
    assert dones.tolist() == [False, True, False]
    h, c = model.get_hidden_state()
    assert torch.all(h[:, 1] == 0) and torch.all(c[:, 1] == 0)
    assert torch.all(h[:, [0, 2]].abs().sum(dim=-1) > 0)


# Head and body command slots


def test_zero_pose_cmds():
    obs = {"actor": torch.randn(4, 61), "critic": torch.randn(4, 70)}
    before = obs["actor"].clone()
    out = play_dance.zero_pose_obs(obs)
    # Command block starts at 48. Its slots 3:13 (head and body) are exactly zero.
    assert torch.all(out["actor"][:, 51:61] == 0)
    assert torch.equal(out["actor"][:, :51], before[:, :51])
    # The env's own buffer is not written.
    assert torch.equal(obs["actor"], before)
    assert torch.equal(out["critic"], obs["critic"])


def test_zero_pose_cmds_reach_the_policy():
    venv = _FakeVenv(2, done_at={})
    policy = _FakeRecurrentPolicy(2)
    obs = {"actor": torch.randn(2, 61), "critic": torch.ones(2, 7)}
    for _ in range(3):
        obs, _, _ = play_dance.policy_step(policy, obs, venv, zero_pose=True)
    for seen in policy.seen_obs:
        assert torch.all(seen[:, 51:61] == 0)
        assert torch.all(seen[:, 48:51] != 0)


def test_prepare_commands_zero_pose_and_twist_scale():
    from microduck_dance.ranges import BALL_TWIST_RANGES

    rng = np.random.default_rng(0)
    cmd = rng.uniform(-1, 1, size=(200, 13)).astype(np.float32)
    out = play_dance.prepare_commands(cmd, twist_scale=0.3, zero_pose=True, twist_limits=BALL_TWIST_RANGES)
    assert np.all(out[:, 3:13] == 0)
    lim = np.array([hi for _, hi in BALL_TWIST_RANGES])
    assert np.all(np.abs(out[:, :3]) <= lim + 1e-7)
    small = np.abs(cmd[:, :3] * 0.3) < lim
    assert np.allclose(out[:, :3][small], (cmd[:, :3] * 0.3)[small])
    # Without the flag and limits, only the twist is scaled.
    plain = play_dance.prepare_commands(cmd, twist_scale=0.5, zero_pose=False)
    assert np.allclose(plain[:, :3], cmd[:, :3] * 0.5)
    assert np.array_equal(plain[:, 3:], cmd[:, 3:])
    # The input is not modified.
    assert not np.all(cmd[:, 3:13] == 0)


LIBRARY = SONG.parents[2] / "library.npz"
LIBRARY_SHA256 = "57e74da0e8d3b487f636929699b98f0a3931128dae106d5081ef2962c0ad0197"


def _rule(twist):
    from microduck_dance.ranges import BALL_TWIST_RANGES

    p95 = np.percentile(np.abs(twist), 95, axis=0)
    lim = np.array([hi for _, hi in BALL_TWIST_RANGES])
    return np.floor((lim / p95).min() / 0.05 + 1e-9) * 0.05


def test_ball_twist_scale_matches_the_library():
    """BALL_TWIST_SCALE is min over channels of ball limit / p95 |twist|, rounded down to 0.05.

    It is computed on the library that training plays (SHA pinned below). The three video
    songs built with the same retarget defaults give the same value.
    """
    import hashlib

    from microduck_dance.ranges import BALL_TWIST_RANGES, BALL_TWIST_SCALE

    lib = np.load(LIBRARY)["commands"]
    assert hashlib.sha256(np.ascontiguousarray(lib, dtype=np.float32).tobytes()).hexdigest() == LIBRARY_SHA256
    assert BALL_TWIST_SCALE == pytest.approx(_rule(lib[:, :3])) == pytest.approx(0.50)
    assert BALL_TWIST_RANGES == ((-0.15, 0.15), (-0.10, 0.10), (-0.5, 0.5))


@needs_song
def test_ball_twist_scale_matches_the_songs():
    """The three video songs built with the retarget defaults give the same BALL_TWIST_SCALE."""
    from microduck_dance.command_file import load_command_file
    from microduck_dance.ranges import BALL_TWIST_SCALE

    songs = np.concatenate([load_command_file(SONG.parent / f"{s}.npz")[0][:, :3]
                            for s in ("swag", "funky_magic", "fallin_for_you")])
    assert BALL_TWIST_SCALE == pytest.approx(_rule(songs))


# Env cfg for the ball task (built, not stepped)


def _args(**kw):
    a = play_dance.build_parser().parse_args(["--checkpoint", "x.pt", "--commands", str(SONG),
                                              "--task", "Mjlab-Basketball-MicroDuck", "--zero-pose-cmds"])
    for k, v in kw.items():
        setattr(a, k, v)
    return a


@needs_song
@pytest.mark.parametrize("play", [True, False])
def test_ball_env_cfg(play, tmp_path):
    from microduck_dance.tasks.dance_commands import DancePoseCommandCfg, DanceVelocityCommandCfg

    args = _args(twist_scale=0.3)
    cmd_file = tmp_path / "cmd.npz"
    cfg = play_dance.build_env_cfg(args, play=play, num_envs=2, seed=0, steps=100, dt=0.02, render=False,
                                   command_file=cmd_file)
    assert "ball" in cfg.scene.entities
    assert isinstance(cfg.commands["twist"], DanceVelocityCommandCfg)
    for name in ("head_pose", "body_pose"):
        assert isinstance(cfg.commands[name], DancePoseCommandCfg)
    for name in play_dance.COMMAND_TERMS:
        assert cfg.commands[name].source == "file"
        assert cfg.commands[name].command_file == str(cmd_file)
        assert cfg.commands[name].resume_on_reset
    # A free ball in eval too (the training cfg starts pinned and has a hold curriculum).
    assert cfg.actions["ball_hold"].levels == (0.0,)
    assert "basketball_hold" not in cfg.curriculum
    assert "standing_envs" not in cfg.curriculum
    # Ball-aware fall termination with a trunk / ball contact sensor.
    assert "ball_fall" in cfg.terminations and "fell" in cfg.terminations
    names = [s.name for s in cfg.scene.sensors]
    assert play_dance.TRUNK_BALL_SENSOR in names
    # Actor head/body slots stay zero pads for b11.
    terms = cfg.observations["actor"].terms
    assert terms["head_command"].func.__name__ == "basketball_head_pad"
    assert terms["body_command"].func.__name__ == "basketball_body_pad"
    # The played file has zero pose columns and the twist scaled into the ball range.
    from microduck_dance.command_file import load_command_file
    from microduck_dance.ranges import BALL_TWIST_RANGES

    played, dt, _ = load_command_file(cmd_file)
    src, _, _ = load_command_file(SONG)
    assert dt == pytest.approx(0.02) and played.shape == src.shape
    assert np.all(played[:, 3:13] == 0)
    lim = np.array([hi for _, hi in BALL_TWIST_RANGES])
    assert np.all(np.abs(played[:, :3]) <= lim + 1e-6)


def test_ball_task_needs_zero_pose_cmds(tmp_path):
    args = _args(zero_pose_cmds=False)
    with pytest.raises(SystemExit):
        play_dance.build_env_cfg(args, play=True, num_envs=1, seed=0, steps=10, dt=0.02, render=False,
                                 command_file=tmp_path / "c.npz")


@needs_song
def test_floor_dance_cfg_unchanged_by_default(tmp_path):
    """The default task still builds the floor dance cfg and plays the given file as is."""
    args = play_dance.build_parser().parse_args(["--checkpoint", "x.pt", "--commands", str(SONG)])
    assert args.task == play_dance.TASK_ID and args.twist_scale == 1.0 and not args.zero_pose_cmds
    cfg = play_dance.build_env_cfg(args, play=True, num_envs=1, seed=0, steps=10, dt=0.02, render=False,
                                   command_file=tmp_path / "c.npz")
    assert "ball" not in cfg.scene.entities
    assert "ball_fall" not in cfg.terminations
    assert cfg.commands["twist"].command_file == str(SONG)


@pytest.mark.skipif(not B11.exists(), reason="b11 checkpoint not fetched (scripts/fetch_basketball.sh)")
def test_b11_agent_cfg_matches_checkpoint():
    ckpt = torch.load(B11, map_location="cpu", weights_only=False)
    agent = play_dance.agent_cfg_for(SimpleNamespace(task="Mjlab-Basketball-MicroDuck"), ckpt)
    assert agent["actor"]["class_name"] == "RNNModel"
    assert agent["actor"]["rnn_hidden_dim"] == 256


@needs_song
def test_score_pose_cmds_keeps_ball_dance_columns_in_file(tmp_path):
    """b11 baseline scored on the ball dance head and body commands while its obs slots stay zero."""
    from microduck_dance.command_file import load_command_file
    from microduck_dance.tasks.microduck_ball_dance_env_cfg import BALL_BODY_SCALE

    src, _, _ = load_command_file(SONG)
    b11 = _args(twist_scale=0.5, score_pose_cmds=True)
    dance = _args(twist_scale=0.5, task=play_dance.BALL_DANCE_TASK, zero_pose_cmds=False)
    played, _, _, changed = play_dance.played_commands(b11)
    ref, _, _, _ = play_dance.played_commands(dance)
    assert changed and np.array_equal(played, ref)
    assert np.allclose(played[:, 3:7], src[:, 3:7]) and np.allclose(played[:, 7:13], src[:, 7:13] * BALL_BODY_SCALE)
    # The b11 actor still has zero pads in the head and body slots.
    cfg = play_dance.build_env_cfg(b11, play=False, num_envs=2, seed=0, steps=100, dt=0.02, render=False,
                                   command_file=tmp_path / "c.npz")
    terms = cfg.observations["actor"].terms
    assert terms["head_command"].func.__name__ == "basketball_head_pad"
    assert terms["body_command"].func.__name__ == "basketball_body_pad"
    # Without zero pose cmds the flag is refused.
    with pytest.raises(SystemExit):
        play_dance.build_env_cfg(_args(score_pose_cmds=True, zero_pose_cmds=False, task=play_dance.BALL_DANCE_TASK),
                                 play=False, num_envs=1, seed=0, steps=10, dt=0.02, render=False,
                                 command_file=tmp_path / "d.npz")
