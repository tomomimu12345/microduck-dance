"""dance_v3 config, command mix and reward tests (v3 design report, section 4). CPU only."""

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from mjlab.tasks.velocity.mdp.rewards import upright, variable_posture

from microduck_dance.mix import PieceKind, PiecewiseHeld
from microduck_dance.ranges import BODY_RANGES, DANCE_STAND_HEIGHT, HEAD_RANGES, TWIST_RANGES
from microduck_dance.tasks import microduck_dance_v3_env_cfg as v3cfg
from microduck_dance.tasks.dance_v3 import (
    DanceV3PoseCommandCfg,
    DanceV3VelocityCommandCfg,
    commanded_gravity_b,
    dance_body_tracking_clamped,
    load_library_columns,
    make_pose_source,
    make_twist_source,
    posture_relative,
    synergy_offsets,
    track_planar_velocity,
    track_yaw_rate,
    upright_relative,
)
from microduck_dance.tasks.microduck_dance_v3_env_cfg import (
    MicroduckDanceV3RlCfg,
    make_microduck_dance_v3_env_cfg,
)
from mjlab_microduck.tasks.microduck_velocity_env_cfg import make_microduck_velocity_env_cfg

REPO = Path(__file__).resolve().parents[1]
LIBRARY = REPO / "dance" / "data" / "library.npz"
DT = 0.02


def _cfg(tmp_path=None, play=False):
    return make_microduck_dance_v3_env_cfg(play=play, command_file=str(LIBRARY))


# --------------------------------------------------------------------------- fakes


class _Data:
    def __init__(self, n):
        self.root_link_pos_w = torch.zeros(n, 3)
        self.root_link_quat_w = torch.zeros(n, 4)
        self.root_link_quat_w[:, 0] = 1.0
        self.body_link_quat_w = self.root_link_quat_w[:, None, :].clone()
        self.root_link_lin_vel_b = torch.zeros(n, 3)
        self.root_link_ang_vel_b = torch.zeros(n, 3)
        self.gravity_vec_w = torch.tensor([0.0, 0.0, -1.0]).expand(n, 3)
        self.joint_pos = None
        self.default_joint_pos = None


class _Asset:
    def __init__(self, n):
        self.data = _Data(n)


class _Scene:
    def __init__(self, n):
        self._asset = _Asset(n)
        self.terrain = type("T", (), {"env_origins": torch.zeros(n, 3)})()

    def __getitem__(self, _):
        return self._asset


class _Cmds:
    def __init__(self):
        self.cmds = {}

    def get_command(self, name):
        return self.cmds[name]


class _Env:
    def __init__(self, n):
        self.num_envs = n
        self.device = "cpu"
        self.step_dt = DT
        self.scene = _Scene(n)
        self.command_manager = _Cmds()


class _Sel:
    def __init__(self, name="robot", body_ids=None, joint_ids=None, joint_names=None):
        self.name = name
        self.body_ids = body_ids
        self.joint_ids = joint_ids
        self.joint_names = joint_names


def quat_from_rp(roll, pitch):
    """ZYX Euler quaternion with zero yaw, (w, x, y, z)."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    return torch.tensor([cr * cp, sr * cp, cr * sp, -sr * sp])


# --------------------------------------------------------------------------- 1. twist rewards


def test_twist_rewards_ignore_roll_pitch_rate_and_vz():
    env = _Env(2)
    d = env.scene._asset.data
    d.root_link_ang_vel_b[:] = torch.tensor([0.5, 0.5, 0.3])
    d.root_link_lin_vel_b[:] = torch.tensor([0.2, -0.1, 0.07])
    env.command_manager.cmds["twist"] = torch.tensor([[0.2, -0.1, 0.3], [0.2, -0.1, 0.8]])
    lin = track_planar_velocity(env, std=v3cfg.LIN_STD, command_name="twist")
    ang = track_yaw_rate(env, std=v3cfg.YAW_STD, command_name="twist")
    assert torch.allclose(lin, torch.ones(2))
    assert ang[0].item() == pytest.approx(1.0)
    assert ang[1].item() == pytest.approx(math.exp(-0.5), rel=1e-5)


def test_twist_rewards_in_cfg():
    r = _cfg().rewards
    assert r["track_linear_velocity"].func is track_planar_velocity
    assert r["track_angular_velocity"].func is track_yaw_rate
    assert r["track_linear_velocity"].weight == 2.0 and r["track_angular_velocity"].weight == 2.0
    assert r["track_linear_velocity"].params["std"] == pytest.approx(math.sqrt(0.1))
    assert r["track_angular_velocity"].params["std"] == pytest.approx(math.sqrt(0.5))


# --------------------------------------------------------------------------- 2. relative upright


def _upright_env(quats, body_cmd):
    env = _Env(len(quats))
    d = env.scene._asset.data
    d.body_link_quat_w = torch.stack(quats)[:, None, :]
    env.command_manager.cmds["body_pose"] = torch.tensor(body_cmd, dtype=torch.float32)
    return env


def test_commanded_gravity_matches_rotation():
    from mjlab.utils.lab_api.math import quat_apply_inverse

    for r, p in ((0.2, 0.0), (0.0, -0.2), (0.15, 0.1)):
        q = quat_from_rp(r, p)[None]
        g = quat_apply_inverse(q, torch.tensor([[0.0, 0.0, -1.0]]))
        assert torch.allclose(g, commanded_gravity_b(torch.tensor([r]), torch.tensor([p])), atol=1e-6)


def test_relative_upright():
    std = math.sqrt(0.05)
    sel = _Sel(body_ids=[0])
    cmd_tilt = [0, 0, 0, 0.2, -0.1, 0]
    quats = [quat_from_rp(0.2, -0.1), quat_from_rp(0.0, 0.0), quat_from_rp(0.1, 0.05)]
    env = _upright_env(quats, [cmd_tilt, cmd_tilt, [0.0] * 6])
    out = upright_relative(env, std=std, command_name="body_pose", asset_cfg=sel)
    assert out[0].item() == pytest.approx(1.0, abs=1e-6)  # matches the commanded tilt
    assert out[1].item() < 0.9  # level trunk under a 0.2 rad command
    # Zero command: identical to mjlab's upright.
    ref = upright.__new__(upright)
    expect = ref(env, std=std, asset_cfg=sel)
    assert torch.allclose(out[2], expect[2])
    env0 = _upright_env(quats, [[0.0] * 6] * 3)
    assert torch.equal(upright_relative(env0, std=std, command_name="body_pose", asset_cfg=sel), ref(env0, std=std, asset_cfg=sel))


def test_upright_in_cfg():
    v = make_microduck_velocity_env_cfg().rewards["upright"]
    u = _cfg().rewards["upright"]
    assert u.func is upright_relative and u.weight == v.weight == 2.0
    assert u.params["std"] == v.params["std"]
    assert u.params["asset_cfg"].body_names == ("trunk_base",)
    assert u.params["command_name"] == "body_pose"


# --------------------------------------------------------------------------- 3. relative posture


def _posture_pair(names, n=5, seed=0):
    """posture_relative and variable_posture on a fake asset with the given joint order."""
    g = torch.Generator().manual_seed(seed)
    J = len(names)
    std = {".*": 0.2}
    env = _Env(n)
    a = env.scene._asset
    a.data.default_joint_pos = torch.randn(n, J, generator=g) * 0.3
    a.data.joint_pos = a.data.default_joint_pos + torch.randn(n, J, generator=g) * 0.1
    a.find_joints = lambda patterns, preserve_order=False: (list(range(J)), list(names))
    sel = _Sel(joint_ids=list(range(J)), joint_names=(".*",))
    pose = v3cfg.make_microduck_dance_v3_env_cfg(command_file=str(LIBRARY)).rewards["pose"]
    params = {
        "asset_cfg": sel,
        "std_standing": {".*": 0.15},
        "std_walking": std,
        "std_running": std,
        "command_name": "twist",
        "synergy": {k: v for k, v in pose.params["synergy"].items() if k in names},
        "z_knots": pose.params["z_knots"],
    }
    rc = type("C", (), {"params": params})()
    rel = posture_relative(rc, env)
    base = variable_posture(rc, env)
    return env, rel, base, params


def test_posture_reduces_to_variable_posture_at_zero_body_command():
    from microduck_dance.posture_fk import LEG_JOINTS

    env, rel, base, p = _posture_pair(LEG_JOINTS)
    env.command_manager.cmds["twist"] = torch.tensor([[0.0, 0, 0], [0.3, 0, 0], [0, 0, 1.0], [0, 0, 0], [0.1, 0.1, 0]])
    env.command_manager.cmds["body_pose"] = torch.zeros(5, 6)
    kw = dict(std_standing=None, std_walking=None, std_running=None, asset_cfg=p["asset_cfg"], command_name="twist",
              walking_threshold=0.01)
    a = rel(env, body_command_name="body_pose", synergy=p["synergy"], z_knots=p["z_knots"], **kw)
    b = base(env, **kw)
    assert torch.equal(a, b)
    env.command_manager.cmds["body_pose"][:, 2] = -0.02
    assert not torch.allclose(rel(env, body_command_name="body_pose", synergy=p["synergy"], z_knots=p["z_knots"], **kw), b)


def test_posture_synergy_follows_joint_names_not_indices():
    from microduck_dance.posture_fk import LEG_JOINTS

    order = [LEG_JOINTS[i] for i in (5, 0, 6, 1, 7, 2, 8, 3, 9, 4)] + ["passive_backlash_x"]
    _, rel, _, p = _posture_pair(order)
    table = p["synergy"]
    zero = tuple((0.0,) * 6 for _ in p["z_knots"])
    for row, name in enumerate(order):
        expect = table.get(name, zero)
        assert torch.allclose(rel.table[row], torch.tensor(expect, dtype=torch.float32))


def test_posture_in_cfg():
    v = make_microduck_velocity_env_cfg().rewards["pose"]
    p = _cfg().rewards["pose"]
    assert p.func is posture_relative and p.weight == v.weight == 1.0
    for k in ("std_standing", "std_walking", "std_running", "walking_threshold", "command_name"):
        assert p.params[k] == v.params[k]
    assert p.params["asset_cfg"].joint_names == v.params["asset_cfg"].joint_names
    assert p.params["z_range"] == (-0.035, 0.010)
    knots = p.params["z_knots"]
    assert knots[0] <= -0.035 and knots[-1] >= 0.010 and 0.0 in knots
    for rows in p.params["synergy"].values():
        assert len(rows) == len(knots) and rows[knots.index(0.0)][0] == 0.0
    assert set(p.params["synergy"]) == {
        f"{s}_{j}" for s in ("left", "right") for j in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle")
    }


@pytest.fixture(scope="module")
def kin():
    from microduck_dance.posture_fk import FlatFootKinematics

    return FlatFootKinematics()


def _fk(kin, z, roll, pitch):
    from microduck_dance.posture_fk import LEG_JOINTS

    table = torch.tensor([v3cfg.POSTURE_SYNERGY[n] for n in LEG_JOINTS], dtype=torch.float64)
    knots = torch.tensor(v3cfg.POSTURE_Z_KNOTS, dtype=torch.float64)
    cmd = torch.tensor([[0.0, 0.0, z, roll, pitch, 0.0]], dtype=torch.float64)
    delta = synergy_offsets(cmd, table, knots, v3cfg.BODY_Z_TARGET_RANGE)[0].numpy()
    return kin.state(delta)


# Commands the flat-foot IK reaches (fit_posture_synergy.py). Roll needs a squat that
# leaves room for the long leg, which cannot extend much past HOME.
REACHABLE = (
    [(z, 0.0, 0.0) for z in (-0.035, -0.03, -0.02, -0.01, -0.005, 0.005, 0.01)]
    + [(0.0, 0.0, p) for p in (-0.2, -0.1, 0.1, 0.2)]
    + [(-0.02, r, 0.0) for r in (-0.2, -0.1, 0.1, 0.2)]
    + [(-0.025, r, p) for r in (-0.15, 0.15) for p in (-0.15, 0.15)]
    + [(-0.035, 0.0, -0.2), (-0.035, 0.2, -0.2), (-0.035, -0.2, -0.2), (-0.035, 0.2, 0.2)]
    + [(0.0, r, p) for r in (-0.2, 0.2) for p in (-0.2, 0.2)]
    + [(0.0075, -0.2, 0.2), (0.0075, 0.2, 0.2), (0.005, 0.2, -0.2), (-0.0125, 0.1, -0.05)]
)


@pytest.mark.parametrize("z,roll,pitch", REACHABLE)
def test_posture_synergy_fk(kin, z, roll, pitch):
    s = _fk(kin, z, roll, pitch)
    assert abs(s.height - z) <= 0.002
    assert abs(s.roll - roll) <= 0.02 and abs(s.pitch - pitch) <= 0.02
    # Feet stay flat and on the ground together.
    assert abs(s.foot_height_diff) <= 0.002 and s.foot_tilt_diff <= 0.02


def test_posture_synergy_signs(kin):
    S = v3cfg.POSTURE_SYNERGY
    for k in range(len(v3cfg.POSTURE_Z_KNOTS)):
        for j in ("hip_pitch", "knee", "ankle"):
            # Pitch-chain joints are mirrored between the sides for the squat offset
            # and the pitch slope, and move the same way for roll (one leg shortens,
            # the other lengthens).
            for col in (0, 2):
                assert S[f"left_{j}"][k][col] * S[f"right_{j}"][k][col] <= 0
            assert S[f"left_{j}"][k][1] * S[f"right_{j}"][k][1] >= 0
        assert S["left_hip_roll"][k][1] == pytest.approx(-1.0, abs=0.05)
        assert S["right_hip_roll"][k][1] == pytest.approx(-1.0, abs=0.05)
    assert _fk(kin, -0.02, 0.0, 0.0).height < -0.015
    assert _fk(kin, -0.02, 0.15, 0.0).roll > 0.1
    assert _fk(kin, -0.02, -0.15, 0.0).roll < -0.1
    assert _fk(kin, 0.0, 0.0, 0.15).pitch > 0.1


# --------------------------------------------------------------------------- 4. body tracking


def test_body_tracking_params():
    r = _cfg().rewards["body_pose_tracking"]
    assert r.func is dance_body_tracking_clamped and r.weight == 3.0
    assert r.params["z_std"] == 0.02 and r.params["angle_std"] == 0.15
    assert r.params["z_range"] == (-0.035, 0.010)
    assert r.params["nominal_height"] == DANCE_STAND_HEIGHT


def test_body_tracking_values():
    env = _Env(3)
    d = env.scene._asset.data
    d.root_link_pos_w[:, 2] = torch.tensor([DANCE_STAND_HEIGHT, DANCE_STAND_HEIGHT + 0.010, DANCE_STAND_HEIGHT - 0.035])
    env.command_manager.cmds["body_pose"] = torch.tensor(
        [[0, 0, 0.0, 0, 0, 0], [0, 0, 0.03, 0, 0, 0], [0, 0, -0.04, 0, 0, 0]], dtype=torch.float32
    )
    out = dance_body_tracking_clamped(env)
    assert torch.allclose(out, torch.ones(3), atol=1e-5)  # clamped targets are met exactly


# --------------------------------------------------------------------------- 5 to 7. command mix


def _frames(dim, T=3000, seed=1, scale=0.2):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(T, dim, generator=g) * 2 - 1) * scale


def _twist(n, seed=0, frames=None):
    cfg = _cfg().commands["twist"]
    g = torch.Generator().manual_seed(seed)
    frames = _frames(3) if frames is None else frames
    return make_twist_source(cfg, frames, DT, n, "cpu", g)


def test_twist_cfg_values():
    c = _cfg().commands["twist"]
    assert isinstance(c, DanceV3VelocityCommandCfg) and c.source == "mix"
    assert c.mode_probs == (0.50, 0.40, 0.10)
    assert c.piece_s == (1.0, 3.0) and c.blend_s == 0.2
    assert c.piece_probs == (0.15, 0.20, 0.65)
    assert c.turn_wz_abs == (0.4, 1.0)
    assert c.held_ranges == ((-0.35, 0.35), (-0.25, 0.25), (-1.0, 1.0))
    assert c.resampling_time_range == (4.0, 8.0)
    assert c.command_file == str(LIBRARY) and c.columns == (0, 3)


def test_twist_mode_frequencies():
    n = 4096
    src = _twist(n)
    counts = torch.zeros(3)
    ids = torch.arange(n)
    for _ in range(25):
        src.resample(ids)
        counts += torch.bincount(src.mode, minlength=3).float()
    freq = counts / counts.sum()
    assert torch.allclose(freq, torch.tensor([0.50, 0.40, 0.10]), atol=0.02)


def test_twist_held_pieces():
    n = 512
    src = _twist(n)
    src.resample(torch.arange(n))
    held = src.sources[0]
    lo = torch.tensor([r[0] for r in TWIST_RANGES])
    hi = torch.tensor([r[1] for r in TWIST_RANGES])
    kinds = torch.zeros(3)
    lengths = []
    zero_seen = False
    for _ in range(1500):
        v = src.step(DT)
        assert (v >= lo - 1e-6).all() and (v <= hi + 1e-6).all()
        lengths.append(held.length.clone())
        kinds += torch.bincount(held.kind, minlength=3).float()
        idle = src.mode == 2
        if idle.any():
            assert torch.equal(v[idle], torch.zeros_like(v[idle]))
        z = (src.mode == 0) & (held.kind == 0) & (held.t >= 0.2)
        if z.any():
            zero_seen = True
            assert torch.equal(v[z], torch.zeros_like(v[z]))
        t = (src.mode == 0) & (held.kind == 1) & (held.t >= 0.2)
        if t.any():
            assert (v[t, :2] == 0).all() and (v[t, 2].abs() >= 0.4).all() and (v[t, 2].abs() <= 1.0).all()
    L = torch.stack(lengths)
    assert L.min() >= 1.0 and L.max() <= 3.0
    assert zero_seen
    assert (src.mode == 2).any()


def test_twist_blend_is_linear_over_0p2_s():
    kinds = (PieceKind(1.0, ((-0.35, 0.35), (-0.25, 0.25), (-1.0, 1.0))),)
    h = PiecewiseHeld(kinds, (1.0, 3.0), 0.2, 64, "cpu", torch.Generator().manual_seed(3))
    h.resample(torch.arange(64))
    for _ in range(200):  # run into the second piece
        h.step(DT)
    h.t[:] = 0.0
    h.length[:] = 10.0
    prev, target = h.prev.clone(), h.target.clone()
    h._value = h._eval()
    for k in range(1, 15):
        v = h.step(DT)
        a = min(1.0, k * DT / 0.2)
        assert torch.allclose(v, prev + (target - prev) * a, atol=1e-5)
    assert torch.equal(v, target)


def test_twist_mix_reproducible():
    a, b = _twist(64, seed=7), _twist(64, seed=7)
    ids = torch.arange(64)
    a.resample(ids)
    b.resample(ids)
    for _ in range(300):
        assert torch.equal(a.step(DT), b.step(DT))


def _sustained_fraction(cmd, steps_needed):
    """cmd: (T, N, 3). Fraction of envs with a run of steps_needed where |vx| > 0.1 or |wz| > 0.4 at constant sign."""
    conds = [cmd[..., 0] > 0.1, cmd[..., 0] < -0.1, cmd[..., 2] > 0.4, cmd[..., 2] < -0.4]
    hit = torch.zeros(cmd.shape[1], dtype=torch.bool)
    for c in conds:
        run = torch.zeros(cmd.shape[1])
        best = torch.zeros(cmd.shape[1])
        for t in range(cmd.shape[0]):
            run = torch.where(c[t], run + 1, torch.zeros_like(run))
            best = torch.maximum(best, run)
        hit |= best >= steps_needed
    return hit.float().mean().item()


@pytest.mark.skipif(not LIBRARY.is_file(), reason="dance library not built")
def test_twist_sustained_fraction():
    n = 2048
    frames = load_library_columns(str(LIBRARY), (0, 3), DT)
    src = _twist(n, seed=11, frames=frames)
    g = torch.Generator().manual_seed(5)
    win = (4.0 + 4.0 * torch.rand(n, generator=g)) / DT  # window U(4, 8) s, as the resample
    src.resample(torch.arange(n))
    T = int(8.0 / DT)
    cmd = torch.stack([src.step(DT).clone() for _ in range(T)])
    mask = torch.arange(T)[:, None] < win[None, :]
    cmd = torch.where(mask[..., None], cmd, torch.zeros_like(cmd))
    assert _sustained_fraction(cmd, int(1.0 / DT)) >= 0.40


def test_head_mix():
    c = _cfg().commands["head_pose"]
    assert isinstance(c, DanceV3PoseCommandCfg)
    assert c.library_prob == 0.85 and c.alt == "held" and c.hold_s == (2.0, 5.0)
    assert c.ranges == HEAD_RANGES and c.columns == (3, 7)
    n = 4096
    src = make_pose_source(c, _frames(4), DT, n, "cpu", torch.Generator().manual_seed(0))
    counts = torch.zeros(2)
    for _ in range(20):
        src.resample(torch.arange(n))
        counts += torch.bincount(src.mode, minlength=2).float()
    assert (counts[1] / counts.sum()).item() == pytest.approx(0.15, abs=0.02)
    held = src.sources[1]
    lo = torch.tensor([r[0] for r in HEAD_RANGES])
    hi = torch.tensor([r[1] for r in HEAD_RANGES])
    for _ in range(400):
        v = src.step(DT)
        assert held.length.min() >= 2.0 and held.length.max() <= 5.0
        m = src.mode == 1
        assert (v[m] >= lo).all() and (v[m] <= hi).all()


def test_body_mix():
    c = _cfg().commands["body_pose"]
    assert isinstance(c, DanceV3PoseCommandCfg)
    assert c.library_prob == 0.60 and c.alt == "smooth"
    assert c.freq_range_hz == (0.1, 0.4) and c.hold_prob == 0.1
    assert c.alt_ranges == ((0.0, 0.0), (0.0, 0.0), (-0.03, 0.01), (-0.20, 0.20), (-0.20, 0.20), (0.0, 0.0))
    assert c.ranges == BODY_RANGES and c.columns == (7, 13)
    n = 4096
    src = make_pose_source(c, _frames(6), DT, n, "cpu", torch.Generator().manual_seed(0))
    counts = torch.zeros(2)
    for _ in range(20):
        src.resample(torch.arange(n))
        counts += torch.bincount(src.mode, minlength=2).float()
    assert (counts[1] / counts.sum()).item() == pytest.approx(0.40, abs=0.02)
    slow = src.sources[1]
    f = slow.freq[src.mode == 1]
    assert f.min() >= 0.1 and f.max() <= 0.4
    vals = torch.stack([src.step(DT).clone() for _ in range(500)])
    m = (src.mode == 1)[None, :].expand(500, n)
    sv = vals[m]
    assert sv[:, 2].min() >= -0.03 - 1e-6 and sv[:, 2].max() <= 0.01 + 1e-6
    assert sv[:, 3:5].abs().max() <= 0.20 + 1e-6
    assert sv[:, 3:5].abs().max() > 0.1
    assert (sv[:, [0, 1, 5]] == 0).all()


# --------------------------------------------------------------------------- 8. signs


def test_reward_signs():
    import mjlab_microduck.tasks.mdp as microduck_mdp

    cfg = _cfg()
    for name, term in cfg.rewards.items():
        fname = getattr(term.func, "__name__", type(term.func).__name__)
        mod = getattr(term.func, "__module__", "")
        self_negating = mod.startswith(("mjlab_microduck", "microduck_dance")) and (fname.endswith("_penalty") or fname.endswith("_l1"))
        if self_negating:
            assert term.weight >= 0, name
        elif mod.startswith("mjlab.") and fname in (
            "action_rate_l2", "joint_pos_limits", "body_angular_velocity_penalty", "angular_momentum_penalty",
            "self_collision_cost", "feet_slip",
        ):
            assert term.weight <= 0, name
    for name in ("track_linear_velocity", "track_angular_velocity", "upright", "pose", "body_pose_tracking",
                 "head_pose_tracking", "air_time"):
        assert cfg.rewards[name].weight > 0, name
    stages = cfg.curriculum["head_pose_bias_weight"].params["weight_stages"]
    assert all(s["weight"] >= 0 for s in stages)
    assert cfg.rewards["head_pose_bias"].func is microduck_mdp.head_pose_bias_penalty
    v = make_microduck_velocity_env_cfg().rewards
    for name in ("body_ang_vel", "action_rate_l2", "air_time", "head_pose_tracking"):
        assert cfg.rewards[name].weight == v[name].weight or name == "head_pose_tracking"
    assert cfg.rewards["body_ang_vel"].weight == -0.05 and cfg.rewards["air_time"].weight == 3.0
    assert cfg.rewards["head_pose_tracking"].params["std"] == 0.25


# --------------------------------------------------------------------------- 9. obs and command slots


def test_obs_layout_matches_velocity():
    d, v = _cfg(), make_microduck_velocity_env_cfg()
    for group in v.observations:
        assert list(d.observations[group].terms) == list(v.observations[group].terms)
    assert d.commands["body_pose"].ranges == BODY_RANGES


def test_every_live_command_slot_is_sampled_nonzero():
    cfg = _cfg()
    n = 512
    g = torch.Generator().manual_seed(0)
    srcs = [
        make_twist_source(cfg.commands["twist"], _frames(3), DT, n, "cpu", g),
        make_pose_source(cfg.commands["head_pose"], _frames(4), DT, n, "cpu", g),
        make_pose_source(cfg.commands["body_pose"], _frames(6), DT, n, "cpu", g),
    ]
    for s in srcs:
        s.resample(torch.arange(n))
    for s in srcs:
        for k in range(len(s.sources)):
            ids = (s.mode == k)
            vals = torch.stack([s.step(DT).clone() for _ in range(100)])[:, ids]
            dims = vals.shape[-1]
            live = [i for i in range(dims) if not (dims == 6 and i in (0, 1, 5))]  # body x, y, yaw are always 0
            for i in live:
                if s.names[k] == "idle":
                    continue
                assert vals[..., i].abs().max() > 0, (s.names[k], i)


# --------------------------------------------------------------------------- 10. curriculum steps


def test_curriculum_stages_are_past_on_resume():
    cfg = _cfg()
    v = make_microduck_velocity_env_cfg()
    for name, term in cfg.curriculum.items():
        assert name in v.curriculum, f"new curriculum {name} (any new stage must be above 120000)"
        for key in ("weight_stages", "range_stages", "standing_stages"):
            for st in term.params.get(key, []):
                assert st["step"] <= v3cfg.STEP_CEILING_INHERITED, (name, st)


# --------------------------------------------------------------------------- 11. file playback


def test_file_playback_unchanged(tmp_path):
    from microduck_dance.command_file import save_command_file
    from microduck_dance.tasks.dance_commands import DancePoseCommandCfg, DanceVelocityCommandCfg, _make_source
    from microduck_dance.tasks.microduck_dance_env_cfg import make_microduck_dance_env_cfg

    rng = np.random.default_rng(0)
    frames = (rng.uniform(-1, 1, (300, 13)) * 0.2).astype(np.float32)
    path = tmp_path / "song.npz"
    save_command_file(path, frames, DT, {})
    cfg = make_microduck_dance_env_cfg(play=True, source="file", command_file=str(path))
    env = _Env(4)
    for name, (lo, hi) in (("twist", (0, 3)), ("head_pose", (3, 7)), ("body_pose", (7, 13))):
        c = cfg.commands[name]
        assert type(c) in (DanceVelocityCommandCfg, DancePoseCommandCfg)
        src = _make_source(c, env, c.ranges if name != "twist" else ((0, 0),) * 3)
        src.resample(torch.arange(4))
        out = [src.value().clone()] + [src.step(DT).clone() for _ in range(299)]
        out = torch.stack(out)
        assert torch.equal(out, torch.as_tensor(frames[:, lo:hi])[:, None, :].expand(300, 4, hi - lo))


# --------------------------------------------------------------------------- 12. registration


def test_v3_task_registered_with_library_path():
    from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg

    import microduck_dance.tasks  # noqa: F401

    assert "Mjlab-DanceV3-Flat-MicroDuck" in list_tasks()
    cfg = load_env_cfg("Mjlab-DanceV3-Flat-MicroDuck")
    for k in ("twist", "head_pose", "body_pose"):
        p = Path(cfg.commands[k].command_file)
        assert p.is_absolute() and p.parts[-3:] == ("dance", "data", "library.npz")
    rl = load_rl_cfg("Mjlab-DanceV3-Flat-MicroDuck")
    assert rl.experiment_name == "dance_v3" and rl.max_iterations == 3000
    assert MicroduckDanceV3RlCfg.save_interval == 250


def test_missing_library_fails_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="library"):
        load_library_columns(str(tmp_path / "nope.npz"), (0, 3), DT)
    with pytest.raises(ValueError):
        make_microduck_dance_v3_env_cfg()


def test_existing_dance_tasks_unchanged():
    from mjlab.tasks.registry import load_env_cfg

    import microduck_dance.tasks  # noqa: F401
    from microduck_dance.tasks.dance_commands import dance_body_tracking

    for tid in ("Mjlab-Dance-Flat-MicroDuck", "Mjlab-DanceLib-Flat-MicroDuck"):
        c = load_env_cfg(tid)
        assert c.rewards["body_pose_tracking"].func is dance_body_tracking
        assert c.rewards["body_pose_tracking"].weight == 2.0
        assert c.commands["twist"].source in ("smooth", "library")
        assert c.rewards["track_angular_velocity"].func.__module__.startswith("mjlab.")
