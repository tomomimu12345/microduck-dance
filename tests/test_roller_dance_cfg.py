"""Cfg-level tests for the roller dance task (Mjlab-RollerDance-MicroDuck).

CPU only: the cfg, the rollers model compiled with plain MuJoCo, the twist
mapping on synthetic commands, the ONNX to actor conversion and the optimizer
helpers of the continuation script. The GPU smoke test covers the built env.
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
LIBRARY = ROOT / "dance" / "data" / "library.npz"
needs_library = pytest.mark.skipif(not LIBRARY.is_file(), reason="dance/data/library.npz missing")
ROLLER_ONNX = os.environ.get("ROLLER_ONNX", "")

SERVO_NAMES = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
]


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def cfg():
    from microduck_dance.tasks.microduck_roller_dance_env_cfg import make_microduck_roller_dance_env_cfg

    return make_microduck_roller_dance_env_cfg()


def _fname(term):
    f = term.func
    return getattr(f, "__name__", type(f).__name__)


# --------------------------------------------------------------------------- model and layout


def test_cfg_uses_the_rollers_model(cfg):
    from mjlab_microduck.robot.microduck_constants import MICRODUCK_WALK_ROLLERS_ROBOT_CFG

    assert cfg.scene.entities["robot"] is MICRODUCK_WALK_ROLLERS_ROBOT_CFG


def test_servo_joint_ids_on_the_rollers_model():
    import mujoco

    from mjlab_microduck.robot.microduck_constants import get_walk_rollers_spec

    model = get_walk_rollers_spec().compile()
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(model.njnt)
             if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE]
    assert len(names) == 18
    servo = [i for i, n in enumerate(names) if not n.startswith("passive_")]
    assert servo == [0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    assert [names[i] for i in servo] == SERVO_NAMES  # the joint order of roller.onnx
    wheels = [n for n in names if n.startswith("passive_")]
    assert sorted(wheels) == ["passive_LF_wheel", "passive_LR_wheel", "passive_RF_wheel", "passive_RR_wheel"]


def test_actor_obs_is_the_61d_layout(cfg):
    from mjlab.envs.mdp import generated_commands

    terms = cfg.observations["actor"].terms
    assert list(terms) == ["base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions", "command",
                           "head_command", "body_command"]
    widths = {"base_ang_vel": 3, "projected_gravity": 3, "joint_pos": 14, "joint_vel": 14, "actions": 14,
              "command": 3, "head_command": 4, "body_command": 6}
    assert sum(widths.values()) == 61
    for t in ("joint_pos", "joint_vel"):
        assert terms[t].params["asset_cfg"].joint_names == (r"^(?!passive_).*",)
    for group in ("actor", "critic"):
        g = cfg.observations[group].terms
        assert g["head_command"].func is generated_commands and g["head_command"].params["command_name"] == "head_pose"
        assert g["body_command"].func is generated_commands and g["body_command"].params["command_name"] == "body_pose"
    assert "wheel_vel" in cfg.observations["critic"].terms


def test_action_is_the_14_servos(cfg):
    from microduck_dance.tasks import microduck_roller_dance_env_cfg as rd

    assert cfg.actions["joint_pos"].scale == rd.ROLLER_ACTION_SCALE
    assert rd.ROLLER_ACTION_SCALE in (1.0, 0.8)


def test_task_is_registered_with_its_own_runner_cfg():
    from dataclasses import asdict

    from mjlab.tasks.registry import load_rl_cfg

    import microduck_dance.tasks  # noqa: F401

    a = asdict(load_rl_cfg("Mjlab-RollerDance-MicroDuck"))
    assert a["experiment_name"] == "roller_dance"
    assert a["experiment_name"] != asdict(load_rl_cfg("Mjlab-Velocity-Flat-MicroDuck-Rollers"))["experiment_name"]
    assert 0.2 <= a["actor"]["distribution_cfg"]["init_std"] <= 0.3
    assert a["actor"]["hidden_dims"] == (512, 256, 128) and a["actor"]["activation"] == "elu"
    assert a["algorithm"]["schedule"] == "fixed"


# --------------------------------------------------------------------------- rewards


# Costs (return >= 0) need a negative weight.
COSTS = {"body_ang_vel", "angular_momentum", "action_rate_l2", "self_collisions", "feet_flat", "neck_action_rate_l2",
         "joint_torques_l2", "action_over_limit", "hip_roll_neutral", "gait_symmetry"}
# Self negating penalties (return <= 0) need a positive weight.
SELF_NEGATING = {"head_pose_bias"}
REWARDS = {"pose", "upright", "com_height_target", "wheel_speed", "braking", "skating_air_time", "glide",
           "single_support", "forward_lean", "heading_tracking", "head_pose_tracking", "body_pose_tracking"}


def test_reward_set(cfg):
    assert set(cfg.rewards) == COSTS | SELF_NEGATING | REWARDS
    assert "neck_joint_pos_l2" not in cfg.rewards
    assert "heading_hold" not in cfg.rewards


def test_reward_signs(cfg):
    for name in COSTS:
        assert cfg.rewards[name].weight < 0, name
    for name in SELF_NEGATING | REWARDS:
        assert cfg.rewards[name].weight > 0, name
    assert _fname(cfg.rewards["head_pose_bias"]).endswith("_penalty")


def test_changed_rewards(cfg):
    from microduck_dance.tasks import microduck_roller_dance_env_cfg as rd

    r = cfg.rewards
    assert _fname(r["pose"]) == "posture_relative"
    for key in ("std_standing", "std_walking", "std_running"):
        assert r["pose"].params[key][r".*neck.*"] == rd.LOOSE_STD
        assert r["pose"].params[key][r".*head.*"] == rd.LOOSE_STD
    assert _fname(r["upright"]) == "upright_relative"
    assert _fname(r["forward_lean"]) == "forward_lean_relative_reward"
    assert r["glide"].params["stillness_std"] == rd.GLIDE_STILLNESS_STD > 5.0
    assert r["neck_action_rate_l2"].weight == rd.NECK_ACTION_RATE_WEIGHT
    assert r["head_pose_tracking"].params["std"] == 0.25
    bt = r["body_pose_tracking"].params
    assert bt["nominal_height"] == rd.ROLLER_STAND_HEIGHT and bt["nominal_pitch"] == rd.ROLLER_STAND_PITCH
    assert 0.10 < rd.ROLLER_STAND_HEIGHT < 0.16


def test_curricula_fixed_at_final_stage(cfg):
    from microduck_dance.tasks import microduck_roller_dance_env_cfg as rd

    assert dict(cfg.curriculum) == {}
    assert cfg.events["randomize_wheel_friction"].params["ranges"] == (rd.WHEEL_FRICTION_FINAL,) * 2
    assert cfg.events["randomize_com"].params["ranges"] == (-rd.COM_RANGE_FINAL, rd.COM_RANGE_FINAL)
    assert cfg.events["randomize_head_com"].params["ranges"] == (-rd.COM_RANGE_FINAL, rd.COM_RANGE_FINAL)
    assert "expand_bam_friction_fields" in cfg.events
    assert "push_robot" in cfg.events


def test_commands(cfg):
    from microduck_dance.tasks.dance_rollers import RollerDanceTwistCommandCfg
    from microduck_dance.tasks.dance_v3 import DanceV3PoseCommandCfg

    tw = cfg.commands["twist"]
    assert isinstance(tw, RollerDanceTwistCommandCfg) and tw.source == "library"
    assert tw.ranges.lin_vel_y == (0.0, 0.0)
    assert isinstance(cfg.commands["head_pose"], DanceV3PoseCommandCfg)
    assert cfg.commands["head_pose"].library_prob == 0.85
    assert cfg.commands["body_pose"].library_prob == 0.60


def test_file_mode_cfg():
    from microduck_dance.tasks.microduck_roller_dance_env_cfg import make_microduck_roller_dance_env_cfg

    song = ROOT / "dance" / "data" / "commands" / "songs" / "swag.npz"
    c = make_microduck_roller_dance_env_cfg(command_file=str(song), source="file")
    for name in ("twist", "head_pose", "body_pose"):
        assert c.commands[name].source == "file"
        assert c.commands[name].command_file == str(song)


# --------------------------------------------------------------------------- twist mapping


def test_push_mapping_and_vy_dropped():
    from microduck_dance.tasks.dance_rollers import PUSH_GAIN, PUSH_RANGE, map_dance_twist

    dance = torch.tensor([[0.04, 0.3, 0.0], [0.5, -0.3, 0.0], [-0.04, 0.1, 0.0], [-0.5, 0.0, 0.0], [0.0, 0.2, 0.0]])
    n = len(dance)
    cmd, _ = map_dance_twist(dance, torch.zeros(n), torch.zeros(n), 0.02)
    assert torch.allclose(cmd[:, 0], torch.tensor([0.04 * PUSH_GAIN, PUSH_RANGE[1], -0.04 * PUSH_GAIN,
                                                   PUSH_RANGE[0], 0.0]))
    assert torch.all(cmd[:, 1] == 0.0)
    assert torch.all(cmd[:, 2] == 0.0)


def test_heading_integrates_wz_with_the_roller_sign():
    from microduck_dance.tasks.dance_rollers import map_dance_twist

    dt = 0.02
    dance = torch.tensor([[0.0, 0.0, 0.5], [0.0, 0.0, -0.5]])
    yaw = torch.zeros(2)
    target = torch.zeros(2)
    for _ in range(50):  # 1 s, the robot does not turn
        cmd, target = map_dance_twist(dance, yaw, target, dt)
    # Positive wz (CCW) puts the target to the left: positive heading error, as RelativeHeadingVelocityCommand.
    assert cmd[0, 2] == pytest.approx(0.5, abs=1e-4)
    assert cmd[1, 2] == pytest.approx(-0.5, abs=1e-4)


def test_heading_error_saturates_without_debt():
    from microduck_dance.tasks.dance_rollers import HEADING_MAX, map_dance_twist

    dt = 0.02
    dance = torch.tensor([[0.0, 0.0, 1.0]])
    yaw, target = torch.zeros(1), torch.zeros(1)
    for _ in range(500):  # 10 s of turning the robot ignores
        cmd, target = map_dance_twist(dance, yaw, target, dt)
    assert cmd[0, 2] == pytest.approx(HEADING_MAX)
    # The turn stops: the error decays at the dance rate instead of 9 rad of debt.
    stop = torch.zeros(1, 3)
    cmd, target = map_dance_twist(stop, yaw, target, dt)
    assert cmd[0, 2] == pytest.approx(HEADING_MAX)
    back = torch.tensor([[0.0, 0.0, -1.0]])
    for _ in range(50):
        cmd, target = map_dance_twist(back, yaw, target, dt)
    assert cmd[0, 2] == pytest.approx(0.0, abs=1e-4)


def test_heading_follows_a_turning_robot():
    from microduck_dance.tasks.dance_rollers import map_dance_twist, wrap_angle

    dt = 0.02
    dance = torch.tensor([[0.0, 0.0, 0.8]])
    yaw, target = torch.tensor([3.0]), torch.tensor([3.0])
    for _ in range(200):  # the robot turns exactly as commanded, across the +-pi wrap
        cmd, target = map_dance_twist(dance, yaw, target, dt)
        yaw = wrap_angle(yaw + 0.8 * dt)
        assert abs(float(cmd[0, 2])) < 0.8 * dt + 1e-4


@needs_library
def test_library_push_distribution():
    from microduck_dance.command_file import load_command_file
    from microduck_dance.tasks.dance_rollers import map_dance_twist

    c, dt, _ = load_command_file(LIBRARY)
    d = torch.as_tensor(c[:, :3])
    cmd, _ = map_dance_twist(d, torch.zeros(len(d)), torch.zeros(len(d)), dt)
    push = cmd[:, 0].numpy()
    assert 0.3 < np.percentile(np.abs(push), 95) < 0.6
    assert np.mean(push > 0.05) > 0.2 and np.mean(push < -0.05) > 0.2


# --------------------------------------------------------------------------- held push pieces (roller002)


def test_push_incentive_cfg(cfg):
    from microduck_dance.tasks import microduck_roller_dance_env_cfg as rd

    assert cfg.rewards["wheel_speed"].weight == rd.WHEEL_SPEED_WEIGHT > 10.0
    assert cfg.rewards["braking"].weight == rd.BRAKING_WEIGHT > 1.0
    assert cfg.rewards["braking"].params["vel_std"] == rd.BRAKING_VEL_STD < 0.3
    tw = cfg.commands["twist"]
    assert tw.push_piece_prob == rd.PUSH_PIECE_PROB > 0.0
    assert tw.zero_prob + tw.push_piece_prob < 1.0


def test_file_mode_has_no_push_pieces():
    from microduck_dance.tasks.microduck_roller_dance_env_cfg import make_microduck_roller_dance_env_cfg

    song = ROOT / "dance" / "data" / "commands" / "songs" / "swag.npz"
    c = make_microduck_roller_dance_env_cfg(command_file=str(song), source="file")
    assert c.commands["twist"].push_piece_prob == 0.0


def test_push_piece_kinds_map_to_sustained_pushes():
    from microduck_dance.tasks.dance_rollers import PUSH_PIECE_KINDS, PUSH_RANGE, map_dance_twist

    assert abs(sum(k.prob for k in PUSH_PIECE_KINDS) - 1.0) < 1e-9
    pushes = [k for k in PUSH_PIECE_KINDS if k.ranges[0][0] > 0]
    assert sum(k.prob for k in pushes) >= 0.7
    for k in PUSH_PIECE_KINDS:
        assert k.ranges[1] == (0.0, 0.0)  # no vy
        lo, hi = k.ranges[0]
        d = torch.tensor([[lo, 0.0, 0.0], [hi, 0.0, 0.0]])
        cmd, _ = map_dance_twist(d, torch.zeros(2), torch.zeros(2), 0.02)
        assert PUSH_RANGE[0] <= float(cmd[:, 0].min()) and float(cmd[:, 0].max()) <= PUSH_RANGE[1]
    assert max(k.ranges[0][1] for k in pushes) * 5.0 >= PUSH_RANGE[1] - 1e-6  # reaches full push


def _const_lib(n, value):
    from microduck_dance.mix import ZeroSource

    class Const(ZeroSource):
        def __init__(self):
            super().__init__(3, n, "cpu")
            self._value[:] = torch.tensor(value)

    return Const()


def _src_cfg(push_piece_prob, zero_prob=0.15):
    from types import SimpleNamespace

    from microduck_dance.tasks import dance_rollers as dr

    return SimpleNamespace(zero_prob=zero_prob, push_piece_prob=push_piece_prob, push_piece_kinds=dr.PUSH_PIECE_KINDS,
                           push_piece_s=dr.PUSH_PIECE_S, push_piece_blend_s=dr.PUSH_PIECE_BLEND_S)


def test_training_source_mixes_library_pieces_and_zero():
    from microduck_dance.tasks.dance_rollers import build_training_source

    n = 4000
    cfg = _src_cfg(0.4)
    g = torch.Generator().manual_seed(0)
    src = build_training_source(_const_lib(n, [0.0, 0.0, 9.0]), cfg, n, "cpu", g)
    src.resample(torch.arange(n))
    for _ in range(100):  # 2 s, past the blend
        v = src.step(0.02)
    lib = (v[:, 2] == 9.0).float().mean()
    zero = (v.abs().sum(1) == 0).float().mean()
    held = 1.0 - lib - zero
    assert abs(float(lib) - 0.45) < 0.03 and abs(float(zero) - 0.15) < 0.03 and abs(float(held) - 0.4) < 0.03
    sel = v[:, 2] != 9.0
    sel &= v.abs().sum(1) > 0
    assert float((v[sel, 0] > 0).float().mean()) > 0.7  # mostly pushes
    assert float(v[sel, 1].abs().max()) == 0.0


def test_training_source_without_pieces_is_the_roller001_mix():
    from microduck_dance.tasks.dance_rollers import build_training_source

    cfg = _src_cfg(0.0)
    src = build_training_source(_const_lib(8, [0.1, 0.0, 0.0]), cfg, 8, "cpu")
    assert src.names == ("library", "zero") and src.probs == (0.85, 0.15)
    with pytest.raises(ValueError):
        build_training_source(_const_lib(8, [0.0] * 3), _src_cfg(0.9), 8, "cpu")


def test_held_push_lasts_seconds():
    from microduck_dance.mix import PiecewiseHeld
    from microduck_dance.tasks.dance_rollers import PUSH_PIECE_BLEND_S, PUSH_PIECE_KINDS, PUSH_PIECE_S, map_dance_twist

    n, dt = 256, 0.02
    held = PiecewiseHeld(PUSH_PIECE_KINDS, PUSH_PIECE_S, PUSH_PIECE_BLEND_S, n, "cpu",
                         torch.Generator().manual_seed(1))
    pushes = []
    for _ in range(1000):
        d = held.step(dt)
        cmd, _ = map_dance_twist(d, torch.zeros(n), torch.zeros(n), dt)
        pushes.append(cmd[:, 0] > 0.2)
    on = torch.stack(pushes).numpy()
    runs, cur = [], np.zeros(n, dtype=int)
    for row in on:
        ended = (~row) & (cur > 0)
        runs += list(cur[ended])
        cur = np.where(row, cur + 1, 0)
    assert np.mean(runs) * dt > 1.0  # library: 0.17 s


# --------------------------------------------------------------------------- ONNX to actor


def _fake_inits(rng, obs_dim=61, act_dim=14):
    sizes = [(512, obs_dim), (256, 512), (128, 256), (act_dim, 128)]
    inits = {"obs_normalizer._mean": rng.standard_normal((1, obs_dim)).astype(np.float32)}
    std = np.abs(rng.standard_normal((1, obs_dim))).astype(np.float32)
    std[0, 51:61] = 0.0
    inits["__divisor__"] = std + 0.01
    for layer, (o, i) in zip(("mlp.0", "mlp.2", "mlp.4", "mlp.6"), sizes):
        inits[f"{layer}.weight"] = (rng.standard_normal((o, i)) * 0.05).astype(np.float32)
        inits[f"{layer}.bias"] = (rng.standard_normal(o) * 0.05).astype(np.float32)
    return inits, std


def test_actor_state_dict_from_onnx_initializers():
    o2c = _load_script("onnx_to_checkpoint")
    inits, std = _fake_inits(np.random.default_rng(0))
    sd = o2c.actor_state_dict(inits, 0.25)
    assert torch.allclose(sd["obs_normalizer._std"], torch.as_tensor(std), atol=1e-6)
    assert torch.allclose(sd["obs_normalizer._var"], torch.as_tensor(std) ** 2, atol=1e-6)
    assert int(sd["obs_normalizer.count"]) >= 10**8
    assert torch.all(sd["distribution.std_param"] == 0.25) and sd["distribution.std_param"].shape == (14,)
    expected = {"obs_normalizer._mean", "obs_normalizer._var", "obs_normalizer._std", "obs_normalizer.count",
                "distribution.std_param"} | {f"mlp.{i}.{p}" for i in (0, 2, 4, 6) for p in ("weight", "bias")}
    assert set(sd) == expected


def test_actor_state_dict_loads_into_the_rsl_rl_actor():
    """The rebuilt state dict loads strictly into the actor of the roller dance runner cfg and matches mlp_forward."""
    from dataclasses import asdict

    from mjlab.tasks.registry import load_rl_cfg
    from rsl_rl.models import MLPModel
    from tensordict import TensorDict

    import microduck_dance.tasks  # noqa: F401

    o2c = _load_script("onnx_to_checkpoint")
    inits, _ = _fake_inits(np.random.default_rng(1))
    sd = o2c.actor_state_dict(inits, 0.25)
    a = asdict(load_rl_cfg("Mjlab-RollerDance-MicroDuck"))["actor"]
    obs = TensorDict({"actor": torch.randn(8, 61)}, batch_size=[8])
    kw = {k: a[k] for k in ("hidden_dims", "activation", "obs_normalization", "distribution_cfg")}
    model = MLPModel(obs, {"actor": ["actor"]}, "actor", 14, **kw)
    model.load_state_dict(sd, strict=True)
    model.eval()
    with torch.no_grad():
        assert torch.allclose(model(obs), o2c.mlp_forward(sd, obs["actor"]), atol=1e-5)


@pytest.mark.skipif(not Path(ROLLER_ONNX).is_file(), reason="set ROLLER_ONNX to roller.onnx")
def test_parity_with_roller_onnx():
    o2c = _load_script("onnx_to_checkpoint")
    inits, meta = o2c.read_onnx(Path(ROLLER_ONNX))
    assert meta["joint_names"].split(",") == SERVO_NAMES
    sd = o2c.actor_state_dict(inits, 0.25)
    assert o2c.parity(Path(ROLLER_ONNX), sd) < 1e-4
    # roller.onnx saw zeros in the head and body slots.
    assert torch.all(sd["obs_normalizer._std"][0, 51:61] < 1e-6)


# --------------------------------------------------------------------------- continuation helpers


def test_two_group_optimizer_and_rates():
    ft = _load_script("finetune_roller_dance")

    class Alg:
        actor = torch.nn.Linear(3, 2)
        critic = torch.nn.Linear(3, 1)

    alg = Alg()
    ft.install_optimizer(alg, 2e-5, 3e-4)
    g = alg.optimizer.param_groups
    assert [x["name"] for x in g] == ["actor", "critic"]
    assert g[0]["lr"] == 2e-5 and g[1]["lr"] == 3e-4 and alg.schedule == "fixed"
    # Warmup: actor rate 0 leaves the actor unchanged after an Adam step.
    ft.set_rates(alg, 0.0, 3e-4)
    before = alg.actor.weight.detach().clone()
    loss = alg.actor(torch.ones(1, 3)).sum() + alg.critic(torch.ones(1, 3)).sum()
    loss.backward()
    alg.optimizer.step()
    assert torch.equal(alg.actor.weight, before)
    with pytest.raises(ValueError):
        ft.install_optimizer(alg, -1.0, 1e-3)


@needs_library
def test_library_command_stats_shapes():
    ft = _load_script("finetune_roller_dance")
    s = ft.library_command_stats(str(LIBRARY))
    assert s["head_command"][0].shape == (4,) and s["body_command"][1].shape == (6,)
    assert torch.all(s["head_command"][1] > 0.05)


def test_eval_helpers():
    sys.path.insert(0, str(ROOT / "dance" / "scripts" / "evaluate"))
    import roller_dance_eval as rde

    xy = np.cumsum(np.full((101, 2), [0.03, 0.04]), axis=0)
    term = np.zeros(101, bool)
    assert rde.path_length(xy, term) == pytest.approx(100 * 0.05)
    term[50] = True  # a reset jump is not travel
    xy[50:] += 5.0
    assert rde.path_length(xy, term) == pytest.approx(98 * 0.05)
    t = np.arange(1000) * 0.02
    d = np.stack([np.sin(t), np.zeros_like(t), np.cos(t)], 1)
    cx, cz, cm = rde.twist_corr(d, np.sin(t), -np.cos(t), 0.02, 1000)
    assert cx == pytest.approx(1.0, abs=1e-6) and cz == pytest.approx(-1.0, abs=1e-6) and cm == pytest.approx(0.0, abs=1e-6)
    cmds = np.zeros((3000, 13), np.float32)
    cmds[1000:1750, 5] = np.sin(np.arange(750) * 0.3)
    assert rde.liveliest_window(cmds, 0.02) == pytest.approx(20.0)


# --------------------------------------------------------------------------- training stages


@pytest.mark.parametrize("stage,push,wheel,brake_w,brake_std", [
    ("roller001", 0.0, 10.0, 1.0, 0.3),
    ("roller002_stage1", 0.4, 20.0, 1.0, 0.3),
    ("roller002_stage2", 0.4, 20.0, 2.0, 0.1),
])
def test_stage_settings(stage, push, wheel, brake_w, brake_std):
    """The three stages differ only in push pieces, wheel_speed weight and the braking reward
    (EXPERIMENTS.md roller001 and roller002)."""
    from microduck_dance.tasks import microduck_roller_dance_env_cfg as rd

    c = rd.make_microduck_roller_dance_env_cfg(stage=stage)
    assert c.commands["twist"].push_piece_prob == pytest.approx(push)
    assert c.commands["twist"].zero_prob == pytest.approx(rd.ZERO_TWIST_PROB)
    assert c.rewards["wheel_speed"].weight == pytest.approx(wheel)
    assert c.rewards["braking"].weight == pytest.approx(brake_w)
    assert c.rewards["braking"].params["vel_std"] == pytest.approx(brake_std)


def test_default_stage_is_the_final_one(cfg):
    from microduck_dance.tasks import microduck_roller_dance_env_cfg as rd

    final = rd.make_microduck_roller_dance_env_cfg(stage="roller002_stage2")
    assert rd.FINAL_STAGE == "roller002_stage2"
    assert set(cfg.rewards) == set(final.rewards)
    for name in cfg.rewards:
        assert cfg.rewards[name].weight == final.rewards[name].weight, name
    assert cfg.rewards["braking"].params == final.rewards["braking"].params
    assert cfg.commands["twist"].push_piece_prob == final.commands["twist"].push_piece_prob


def test_stages_leave_file_mode_without_push_pieces():
    from microduck_dance.tasks import microduck_roller_dance_env_cfg as rd

    for stage in rd.ROLLER_STAGES:
        c = rd.make_microduck_roller_dance_env_cfg(play=True, source="file", stage=stage)
        assert c.commands["twist"].push_piece_prob == 0.0
    with pytest.raises(ValueError):
        rd.make_microduck_roller_dance_env_cfg(stage="roller003")


def test_finetune_script_has_the_stage_option():
    import subprocess

    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "finetune_roller_dance.py"), "--help"],
                       capture_output=True, text=True, check=True)
    assert "--stage" in r.stdout and "roller002_stage2" in r.stdout
