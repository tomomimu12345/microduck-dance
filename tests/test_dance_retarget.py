"""Tests for SMPL to microduck command retargeting and the command file format.

All motion here is synthetic. Helpers describe the motion in a y-up SMPL world
(x left, y up, z forward, the same axes as the SMPL body frame). For up_axis=2
the whole motion is rotated by +90 deg about x, which sends y-up to z-up, so a
z-up clip has the world rotation baked into the root orientation. trans is
converted the way SMPL requires (pelvis = trans + J0, J0 not rotated by the
root): trans_z = Q (trans_y + J0) - J0, so the pelvis path is the same motion.
"""

import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from microduck_dance.command_file import load_command_file, save_command_file
from microduck_dance.ranges import ALL_RANGES
from microduck_dance.retarget import RetargetConfig, _heading, detect_up_axis, smpl_to_commands

PELVIS_Y = 0.95  # trans_y of a standing y-up clip; the pelvis sits at PELVIS_Y + J0_Y
J0 = np.array([-0.0018, -0.2233, 0.0282])  # neutral SMPL pelvis rest position
Y_TO_Z = R.from_rotvec([math.pi / 2, 0.0, 0.0])  # y-up world to z-up world


def make_motion(F, fps, up_axis, trans_fn, root_rotvec_fn, spine_rotvec_fn=None, head_rotvec_fn=None):
    """Build (poses (F, 72), trans (F, 3)) from per-time functions in the y-up world.

    spine_rotvec_fn sets joints 3, 6 and 9 (the same local rotation on each).
    head_rotvec_fn sets joint 15 (head); the neck (12) stays identity.
    """
    t = np.arange(F) / fps
    poses = np.zeros((F, 24, 3))
    trans = np.array([trans_fn(ti) for ti in t], dtype=float)
    root = R.from_rotvec(np.array([root_rotvec_fn(ti) for ti in t], dtype=float))
    if up_axis == 2:
        root = Y_TO_Z * root
        trans = Y_TO_Z.apply(trans + J0) - J0
    elif up_axis != 1:
        raise ValueError(up_axis)
    poses[:, 0] = root.as_rotvec()
    if spine_rotvec_fn is not None:
        s = np.array([spine_rotvec_fn(ti) for ti in t], dtype=float)
        poses[:, 3] = poses[:, 6] = poses[:, 9] = s
    if head_rotvec_fn is not None:
        poses[:, 15] = np.array([head_rotvec_fn(ti) for ti in t], dtype=float)
    return poses.reshape(F, 72), trans


def standing_trans(up, F=300, fps=30):
    _, trans = make_motion(
        F, fps, up,
        trans_fn=lambda t: (0.3 * math.sin(t), PELVIS_Y + 0.02 * math.sin(3 * t), 0.2 * math.cos(0.7 * t)),
        root_rotvec_fn=lambda t: (0.0, 0.0, 0.0),
    )
    return trans


def still_motion(F, fps, up_axis=1):
    return make_motion(F, fps, up_axis, lambda t: (0.0, PELVIS_Y, 0.0), lambda t: (0.0, 0.0, 0.0))


def walking_motion(speed, F, fps, up_axis=1):
    # Root identity faces SMPL +z, so walking along +z is walking forward.
    return make_motion(F, fps, up_axis, lambda t: (0.0, PELVIS_Y, speed * t), lambda t: (0.0, 0.0, 0.0))


def spinning_motion(rate, F, fps, up_axis=1):
    # Positive rotation about up is counterclockwise seen from above (turn left).
    return make_motion(F, fps, up_axis, lambda t: (0.0, PELVIS_Y, 0.0), lambda t: (0.0, rate * t, 0.0))


def leaning_motion(pitch_deg, F, fps, up_axis=1):
    # Rotation about SMPL +x (left) by a positive angle tips the forward axis down.
    a = math.radians(pitch_deg)
    return make_motion(F, fps, up_axis, lambda t: (0.0, PELVIS_Y, 0.0), lambda t: (a, 0.0, 0.0))


def lying_face_up_rotvec():
    # Rotating about +x by -90 deg sends forward (+z) to up (+y).
    return np.array([-math.pi / 2, 0.0, 0.0])


def random_motion(seed, F, fps, up_axis=1):
    rng = np.random.default_rng(seed)

    def sines(n, amp):
        f = rng.uniform(0.2, 3.0, (n, 3)); p = rng.uniform(0, 2 * math.pi, (n, 3)); a = rng.uniform(0, amp, (n, 3))
        return lambda t: (a * np.sin(2 * math.pi * f * t + p)).sum(axis=0)

    tr = sines(3, 0.6); rr = sines(3, 1.5); sp = sines(3, 0.8); hd = sines(3, 1.5)
    yaw_rate = rng.uniform(-3.0, 3.0)

    def trans_fn(t):
        v = tr(t)
        return (v[0], PELVIS_Y + 0.3 * v[1], v[2])

    def root_fn(t):
        r = R.from_rotvec([0.0, yaw_rate * t, 0.0]) * R.from_rotvec(rr(t))
        return r.as_rotvec()

    return make_motion(F, fps, up_axis, trans_fn, root_fn, sp, hd)


# command file


def test_command_file_roundtrip(tmp_path):
    c = np.random.default_rng(0).uniform(-0.1, 0.1, (50, 13)).astype(np.float32)
    save_command_file(tmp_path / "a.npz", c, 0.02, {"song": "x"})
    c2, dt, meta = load_command_file(tmp_path / "a.npz")
    assert np.array_equal(c, c2) and dt == 0.02 and meta == {"song": "x"}


def test_load_command_file_without_meta(tmp_path):
    c = np.zeros((10, 13), np.float32)
    np.savez(tmp_path / "b.npz", commands=c, dt=0.02)
    c2, dt, meta = load_command_file(tmp_path / "b.npz")
    assert np.array_equal(c, c2) and dt == 0.02 and meta == {}


def test_save_command_file_rejects_wrong_width(tmp_path):
    with pytest.raises(ValueError):
        save_command_file(tmp_path / "c.npz", np.zeros((10, 12)), 0.02, {})


# up axis


def test_detect_up_axis_y_and_z():
    assert detect_up_axis(standing_trans(up=1)) == 1 and detect_up_axis(standing_trans(up=2)) == 2


def test_detect_up_axis_rejects_ambiguous():
    with pytest.raises(ValueError):
        detect_up_axis(np.zeros((100, 3)))


def test_detect_up_axis_rejects_two_plausible_axes():
    trans = np.full((100, 3), PELVIS_Y)
    with pytest.raises(ValueError):
        detect_up_axis(trans)


# retargeting


def test_still_pose_gives_zero_commands():
    out = smpl_to_commands(*still_motion(F=90, fps=30), fps=30, cfg=RetargetConfig())
    assert out.shape == (150, 13) and np.abs(out).max() < 1e-6


def test_walking_forward_gives_positive_vx_after_ramp():
    # The person walks forward at 0.4 m/s.
    out = smpl_to_commands(*walking_motion(speed=0.4, F=150, fps=30), fps=30, cfg=RetargetConfig(vel_gain=0.5, ramp_out_s=0.0))
    assert out[100:, 0].mean() == pytest.approx(0.2, abs=0.03)  # vel_gain 0.5
    assert np.abs(out[100:, 1]).max() < 0.02


def test_walking_forward_z_up_matches_y_up():
    a = smpl_to_commands(*walking_motion(speed=0.4, F=150, fps=30, up_axis=1), fps=30, cfg=RetargetConfig())
    b = smpl_to_commands(*walking_motion(speed=0.4, F=150, fps=30, up_axis=2), fps=30, cfg=RetargetConfig())
    np.testing.assert_allclose(a, b, atol=1e-6)


@pytest.mark.parametrize("servo", [False, True])
def test_yaw_wraparound_has_no_spike(servo):
    # 0.5 rad/s for 40 s crosses +-pi several times. The servo first catches up the
    # 0.25 rad lost in the 1 s ramp-in (time constant 0.67 s), so it is checked from 5 s.
    cfg = RetargetConfig(ramp_out_s=0.0, heading_servo=servo)
    out = smpl_to_commands(*spinning_motion(rate=0.5, F=1200, fps=30), fps=30, cfg=cfg)
    assert np.abs(out[250 if servo else 60:, 2] - 0.5).max() < 0.05
    assert np.abs(np.diff(out[:, 2])).max() <= 4.0 * 0.02 + 1e-9  # wz rate limit


@pytest.mark.parametrize("servo", [False, True])
def test_spinning_z_up_gives_same_turn_direction(servo):
    cfg = RetargetConfig(ramp_out_s=0.0, heading_servo=servo)
    out = smpl_to_commands(*spinning_motion(rate=0.5, F=600, fps=30, up_axis=2), fps=30, cfg=cfg)
    assert np.abs(out[250 if servo else 60:, 2] - 0.5).max() < 0.05


def test_forward_lean_gives_positive_body_pitch():
    out = smpl_to_commands(*leaning_motion(pitch_deg=10, F=150, fps=30), fps=30, cfg=RetargetConfig(ramp_out_s=0.0))
    assert out[100:, 11].mean() == pytest.approx(math.radians(10), abs=0.02)


def test_forward_lean_z_up_gives_positive_body_pitch():
    out = smpl_to_commands(*leaning_motion(pitch_deg=10, F=150, fps=30, up_axis=2), fps=30, cfg=RetargetConfig(ramp_out_s=0.0))
    assert out[100:, 11].mean() == pytest.approx(math.radians(10), abs=0.02)


def test_forward_lean_matches_reward_pitch_convention():
    # dance_body_tracking reads pitch as asin(2 (qw qy - qz qx)) on the robot quaternion.
    qx, qy, qz, qw = R.from_rotvec([0.0, math.radians(10), 0.0]).as_quat()
    assert math.asin(2 * (qw * qy - qz * qx)) == pytest.approx(math.radians(10))


def test_head_pitch_split_between_neck_and_head():
    a = math.radians(20)
    poses, trans = make_motion(150, 30, 1, lambda t: (0.0, PELVIS_Y, 0.0), lambda t: (0.0, 0.0, 0.0),
                               head_rotvec_fn=lambda t: (a, 0.0, 0.0))
    out = smpl_to_commands(poses, trans, fps=30, cfg=RetargetConfig(ramp_out_s=0.0, head_gain=1.0))
    # Joint neck_pitch + raises the nose and head_pitch + lowers it, so looking down
    # by a is head_pitch = +a/2 and neck_pitch = -a/2 (measured on the robot kinematics).
    assert out[100:, 3].mean() == pytest.approx(-a / 2, abs=0.01)
    assert out[100:, 4].mean() == pytest.approx(a / 2, abs=0.01)
    assert np.abs(out[100:, 5:7]).max() < 1e-6


def test_head_yaw_left_and_roll_signs_match_robot_joints():
    a = math.radians(20)
    kw = dict(fps=30, cfg=RetargetConfig(ramp_out_s=0.0, head_gain=1.0))
    # SMPL x is left. A rotation about +y turns the nose toward +x, the person's left.
    poses, trans = make_motion(150, 30, 1, lambda t: (0.0, PELVIS_Y, 0.0), lambda t: (0.0, 0.0, 0.0),
                               head_rotvec_fn=lambda t: (0.0, a, 0.0))
    out = smpl_to_commands(poses, trans, **kw)
    assert out[100:, 5].mean() == pytest.approx(a, abs=0.01)  # robot head_yaw + turns left
    # A rotation about +z (forward) drops the person's right ear. Robot head_roll + tilts left.
    poses, trans = make_motion(150, 30, 1, lambda t: (0.0, PELVIS_Y, 0.0), lambda t: (0.0, 0.0, 0.0),
                               head_rotvec_fn=lambda t: (0.0, 0.0, a / 2))
    out = smpl_to_commands(poses, trans, **kw)
    assert out[100:, 6].mean() == pytest.approx(-a / 2, abs=0.01)


def test_ramp_in_starts_from_zero():
    out = smpl_to_commands(*walking_motion(speed=0.4, F=150, fps=30), fps=30, cfg=RetargetConfig())
    assert np.abs(out[0]).max() < 1e-6 and np.all(np.diff(np.abs(out[:50, 0])) >= -1e-6)


def test_ramp_out_ends_at_zero():
    cfg = RetargetConfig()
    out = smpl_to_commands(*walking_motion(speed=0.4, F=150, fps=30), fps=30, cfg=cfg)
    n = int(round(cfg.ramp_out_s / cfg.out_dt))
    assert np.all(out[-1] == 0.0)
    assert np.all(np.diff(np.abs(out[-n:]), axis=0) <= 1e-6)
    assert np.abs(out[-n - 1, 0]) > 0.05  # vx is still moving before the ramp-out


def test_outputs_within_training_ranges():
    out = smpl_to_commands(*random_motion(seed=0, F=600, fps=30), fps=30, cfg=RetargetConfig())
    lo = np.array([r[0] for r in ALL_RANGES]); hi = np.array([r[1] for r in ALL_RANGES])
    assert np.all(out >= lo - 1e-6) and np.all(out <= hi + 1e-6)


def test_zero_max_rate_dims_are_exactly_zero():
    out = smpl_to_commands(*random_motion(seed=1, F=300, fps=30), fps=30, cfg=RetargetConfig())
    assert np.all(out[:, [7, 8, 12]] == 0.0)


def test_degenerate_frames_hold_heading():
    poses, trans = spinning_motion(rate=0.5, F=300, fps=30)
    poses[100:110, :3] = lying_face_up_rotvec()  # forward vector points straight up
    out = smpl_to_commands(poses, trans, fps=30, cfg=RetargetConfig())
    assert np.isfinite(out).all() and np.abs(out[:, 2]).max() <= 1.0


def test_heading_holds_last_valid_value_on_degenerate_frames():
    # Spin at 0.5 rad/s; frames 100..109 point straight up with tiny noisy
    # horizontal parts, where atan2 alone would return arbitrary angles.
    F = 200
    psi_true = 0.5 * np.arange(F) / 30
    fwd = np.stack([np.cos(psi_true), np.sin(psi_true), np.zeros(F)], axis=1)
    rng = np.random.default_rng(0)
    fwd[100:110, :2] = rng.uniform(-1e-4, 1e-4, (10, 2))
    fwd[100:110, 2] = 1.0
    psi = _heading(fwd)
    np.testing.assert_allclose(psi[100:110], psi_true[99], atol=1e-12)
    np.testing.assert_allclose(psi[110:], psi_true[110:], atol=1e-9)


def test_heading_backfills_leading_degenerate_frames():
    fwd = np.array([[0.0, 0.0, 1.0]] * 3 + [[0.0, 1.0, 0.0]] * 3)
    np.testing.assert_allclose(_heading(fwd), math.pi / 2)


def squatting_motion(F, fps, up_axis):
    return make_motion(F, fps, up_axis, lambda t: (0.0, PELVIS_Y + 0.1 * math.sin(2 * t), 0.0),
                       lambda t: (0.0, 0.0, 0.0))


def test_squat_body_z_same_for_y_up_and_z_up():
    a = smpl_to_commands(*squatting_motion(300, 30, 1), fps=30, cfg=RetargetConfig())
    b = smpl_to_commands(*squatting_motion(300, 30, 2), fps=30, cfg=RetargetConfig())
    assert np.abs(a[:, 9]).max() > 0.005
    np.testing.assert_allclose(a[:, 9], b[:, 9], atol=1e-6)


def test_default_height_gain_uses_pelvis_height():
    # Pelvis height = trans_y + J0_y, so the default gain is 0.095 / (PELVIS_Y + J0_y).
    poses, trans = squatting_motion(300, 30, 1)
    cfg = RetargetConfig(height_gain=None, cutoff_hz=0.0, max_rate=(100.0,) * 13, ramp_in_s=0.0, ramp_out_s=0.0,
                         posture_lead_s=0.0)
    out = smpl_to_commands(poses, trans, fps=30, cfg=cfg)
    pelvis_h = trans[:, 1] + J0[1]
    gain = 0.095 / np.median(pelvis_h)
    t_in = np.arange(300) / 30
    t = np.arange(500) * 0.02
    expected = np.clip(gain * (np.interp(t, t_in, pelvis_h) - np.median(pelvis_h)), -0.04, 0.03)
    np.testing.assert_allclose(out[:, 9], expected, atol=2e-4)


# joints path (FlowerDance pkl: z-up, smpl_trans == full_pose[:, 0], floor near -0.95)

FLOOR_Z = -0.95
# Template joint offsets from the pelvis in SMPL y-up axes (x left, y up, z forward).
_SKELETON = np.zeros((24, 3))
_SKELETON[1:] = np.linspace([0.0, 0.1, 0.0], [0.0, 0.5, 0.0], 23)  # upper body filler, above the pelvis
_SKELETON[1] = (0.09, -0.08, 0.0); _SKELETON[2] = (-0.09, -0.08, 0.0)  # hips
_SKELETON[4] = (0.10, -0.47, 0.0); _SKELETON[5] = (-0.10, -0.47, 0.0)  # knees
_SKELETON[15] = (0.0, 0.55, 0.02)  # head
FEET = (7, 8, 10, 11)  # ankles and feet stay planted on the floor
_FEET_WORLD = {7: (0.09, FLOOR_Z + 0.08, 0.0), 8: (-0.09, FLOOR_Z + 0.08, 0.0),
               10: (0.10, FLOOR_Z, 0.12), 11: (-0.10, FLOOR_Z, 0.12)}  # (x, z, y) in z-up world


def flowerdance_squat(F=300, fps=30, depth=0.1):
    """z-up clip with full_pose, trans equal to the pelvis, feet on z = FLOOR_Z."""
    t = np.arange(F) / fps
    root = Y_TO_Z * R.from_rotvec(np.zeros((F, 3)))  # upright, world rotation baked in
    pelvis = np.stack([np.zeros(F), np.zeros(F), depth * np.sin(2 * t)], axis=1)
    joints = pelvis[:, None, :] + Y_TO_Z.apply(_SKELETON)[None]
    for j, (x, z, y) in _FEET_WORLD.items():
        joints[:, j] = (x, y, z)
    poses = np.zeros((F, 24, 3)); poses[:, 0] = root.as_rotvec()
    return poses.reshape(F, 72), pelvis.copy(), joints


def test_flowerdance_like_trans_alone_cannot_detect_up():
    _, trans, _ = flowerdance_squat()
    with pytest.raises(ValueError):
        detect_up_axis(trans)


def test_detect_up_axis_from_joints_z_up():
    _, trans, joints = flowerdance_squat()
    assert detect_up_axis(trans, joints=joints) == 2


def test_detect_up_axis_from_joints_y_up():
    _, trans, joints = flowerdance_squat()
    to_y = Y_TO_Z.inv()
    j_y = to_y.apply(joints.reshape(-1, 3)).reshape(joints.shape)
    assert detect_up_axis(to_y.apply(trans), joints=j_y) == 1


def test_detect_up_axis_from_joints_rejects_ambiguous():
    with pytest.raises(ValueError):
        detect_up_axis(np.zeros((100, 3)), joints=np.zeros((100, 24, 3)))


def test_flowerdance_squat_body_z_uses_floor_relative_pelvis():
    poses, trans, joints = flowerdance_squat()
    cfg = RetargetConfig(height_gain=None, cutoff_hz=0.0, max_rate=(100.0,) * 13, ramp_in_s=0.0, ramp_out_s=0.0,
                         posture_lead_s=0.0)
    out = smpl_to_commands(poses, trans, fps=30, cfg=cfg, joints=joints)
    h = trans[:, 2] - FLOOR_Z  # pelvis above the floor, about 0.95 m
    gain = 0.095 / np.median(h)
    t = np.arange(500) * 0.02
    expected = np.clip(gain * (np.interp(t, np.arange(300) / 30, h) - np.median(h)), -0.04, 0.03)
    assert np.abs(expected).max() > 0.005
    np.testing.assert_allclose(out[:, 9], expected, atol=2e-4)
    assert np.abs(out[:, [0, 1, 2, 10, 11]]).max() < 1e-6


def test_joints_path_uses_pelvis_joint_not_trans_plus_offset():
    # Walking forward: velocity comes from joints[:, 0]; a stale trans must not matter.
    poses, trans, joints = flowerdance_squat(depth=0.0)
    F = len(trans); t = np.arange(F) / 30
    fwd = Y_TO_Z.apply([0.0, 0.0, 1.0])  # body forward in the z-up world
    joints = joints + (0.4 * t)[:, None, None] * fwd
    out = smpl_to_commands(poses, np.zeros_like(trans), fps=30, cfg=RetargetConfig(vel_gain=0.5), joints=joints)
    assert out[100:, 0].mean() == pytest.approx(0.2, abs=0.03)


# trunk-line body tilt (joints path)

def _with_neck(joints, neck_offset_smpl):
    """Place joint 12 at pelvis + an offset given in SMPL y-up axes (rotated into the z-up world)."""
    j = joints.copy()
    j[:, 12] = j[:, 0] + Y_TO_Z.apply(neck_offset_smpl)
    return j


TILT_CFG = RetargetConfig(cutoff_hz=0.0, max_rate=(100.0,) * 13, ramp_in_s=0.0, ramp_out_s=0.0)


def test_trunk_line_ignores_chest_rotation_when_trunk_is_vertical():
    poses, trans, joints = flowerdance_squat(depth=0.0)
    poses = poses.reshape(-1, 24, 3)
    poses[:, [3, 6, 9], 0] = math.radians(5.0)  # chest frame pitched +15 deg in total
    poses = poses.reshape(-1, 72)
    joints = _with_neck(joints, (0.0, 0.3, 0.0))  # neck straight above the pelvis
    with_j = smpl_to_commands(poses, trans, fps=30, cfg=TILT_CFG, joints=joints)
    assert np.abs(with_j[:, 11]).max() < 1e-6 and np.abs(with_j[:, 10]).max() < 1e-6


def test_chest_path_still_reads_chest_rotation_without_joints():
    poses, trans = make_motion(150, 30, 1, lambda t: (0.0, PELVIS_Y, 0.0), lambda t: (0.0, 0.0, 0.0),
                               spine_rotvec_fn=lambda t: (math.radians(4.0), 0.0, 0.0))
    out = smpl_to_commands(poses, trans, fps=30, cfg=TILT_CFG)
    assert out[:, 11].mean() == pytest.approx(math.radians(12.0), abs=1e-3)  # under the 0.26 clip


def test_trunk_line_forward_lean_gives_positive_pitch():
    poses, trans, joints = flowerdance_squat(depth=0.0)
    a = math.radians(10.0)
    joints = _with_neck(joints, (0.0, 0.3 * math.cos(a), 0.3 * math.sin(a)))  # SMPL +z is forward
    out = smpl_to_commands(poses, trans, fps=30, cfg=TILT_CFG, joints=joints)
    np.testing.assert_allclose(out[:, 11], a, atol=1e-6)
    assert np.abs(out[:, 10]).max() < 1e-6


def test_trunk_line_roll_sign_matches_chest_roll_sign():
    # Rotating about SMPL +z (forward) by +10 deg tips the body toward its right (-y robot).
    a = math.radians(10.0)
    rot = R.from_rotvec([0.0, 0.0, a])
    poses, trans, joints = flowerdance_squat(depth=0.0)
    joints = _with_neck(joints, rot.apply([0.0, 0.3, 0.0]))
    with_j = smpl_to_commands(poses, trans, fps=30, cfg=TILT_CFG, joints=joints)
    p2, t2 = make_motion(150, 30, 1, lambda t: (0.0, PELVIS_Y, 0.0), lambda t: (0.0, 0.0, 0.0),
                         spine_rotvec_fn=lambda t: (0.0, 0.0, a / 3))
    chest = smpl_to_commands(p2, t2, fps=30, cfg=TILT_CFG)
    np.testing.assert_allclose(with_j[:, 10], a, atol=1e-6)
    assert chest[:, 10].mean() == pytest.approx(a, abs=1e-3)


# heading low-pass

def twisting_motion(F, fps, net_rate, twist_amp, twist_hz, speed=0.0):
    """y-up clip whose heading is net_rate * t + twist_amp * sin(2 pi twist_hz t).

    The pelvis moves along the slow (net) heading direction at speed.
    """
    def root(t):
        return (0.0, net_rate * t + twist_amp * math.sin(2 * math.pi * twist_hz * t), 0.0)

    def trans_fn(t):
        # Straight-line walking along the initial forward axis, only used with net_rate = 0.
        return (0.0, PELVIS_Y, speed * t)

    return make_motion(F, fps, 1, trans_fn, root)


def _component(x, hz, dt):
    t = np.arange(len(x)) * dt
    return 2 * abs(np.mean(x * np.exp(-2j * math.pi * hz * t)))


def test_heading_lowpass_removes_fast_twist_and_keeps_net_turn():
    cfg = RetargetConfig(yaw_rate_gain=0.2)
    out = smpl_to_commands(*twisting_motion(900, 30, net_rate=0.5, twist_amp=0.5, twist_hz=1.5), fps=30, cfg=cfg)
    wz = out[100:1400, 2]  # 1300 samples = 26 s = 39 full cycles at 1.5 Hz
    raw_amp = 0.2 * 0.5 * 2 * math.pi * 1.5
    assert _component(wz, 1.5, 0.02) <= 0.2 * raw_amp
    assert wz.mean() == pytest.approx(0.2 * 0.5, abs=0.01)


def test_velocity_frame_not_rotated_by_fast_twist():
    out = smpl_to_commands(*twisting_motion(600, 30, net_rate=0.0, twist_amp=0.5, twist_hz=1.5, speed=0.4),
                           fps=30, cfg=RetargetConfig(vel_gain=0.5))
    assert out[100:-50, 0].mean() == pytest.approx(0.2, abs=0.03)
    assert np.abs(out[100:-50, 1]).max() < 0.03


def test_tuned_default_gains_scale_amplitude():
    # height_gain 0.55 from exp tune001, vel_gain 0.3 from Task 19. The output stays inside the training range.
    cfg = RetargetConfig()
    assert cfg.vel_gain == 0.3 and cfg.height_gain == 0.55
    out = smpl_to_commands(*walking_motion(speed=0.4, F=150, fps=30), fps=30, cfg=RetargetConfig(ramp_out_s=0.0))
    assert out[100:, 0].mean() == pytest.approx(0.12, abs=0.02)  # 0.3 * 0.4 m/s
    # A 1 cm squat keeps body_z far from the clip limits and the rate limit, so the
    # amplitude ratio against height_gain=None is exactly 0.55 / (0.095 / median pelvis height).
    poses, trans = make_motion(300, 30, 1, lambda t: (0.0, PELVIS_Y + 0.01 * math.sin(2 * t), 0.0),
                               lambda t: (0.0, 0.0, 0.0))
    full = smpl_to_commands(poses, trans, fps=30, cfg=RetargetConfig(height_gain=None))
    big = smpl_to_commands(poses, trans, fps=30, cfg=cfg)
    assert np.abs(big[:, 9]).max() < 0.03  # not clipped
    expected = 0.55 / (0.095 / np.median(trans[:, 1] + J0[1]))
    ratio = np.abs(big[:, 9]).max() / np.abs(full[:, 9]).max()
    assert ratio == pytest.approx(expected, rel=0.02)
