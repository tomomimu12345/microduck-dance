"""Tests for the generated-dance clean-up in dance/motion_fix.py (CPU, synthetic motion)."""

import math

import numpy as np
from scipy.spatial.transform import Rotation as R

from microduck_dance.motion_fix import (
    DRIFT_MAX,
    FLOOR_Z,
    align_heading,
    fix_floor,
    floor_drift,
    foot_slide,
    lowest_z,
    start_heading,
)
from microduck_dance.retarget import P

FPS = 30.0
F = 1800
t = np.arange(F) / FPS

# A rough standing skeleton in the z-up world, pelvis at the origin. Joint 10
# and 11 are the feet, the rest sit above them.
REST = np.zeros((24, 3))
REST[:, 2] = np.linspace(-0.2, 0.6, 24)
REST[10] = [0.0, 0.1, -0.94]
REST[11] = [0.0, -0.1, -0.94]
REST[7] = [0.0, 0.1, -0.85]
REST[8] = [0.0, -0.1, -0.85]


def standing(dz=None):
    """joints (F, 24, 3), trans (F, 3) for a dancer standing still, lifted by dz(t)."""
    J = np.repeat(REST[None], F, axis=0)
    T = np.zeros((F, 3))
    if dz is not None:
        J[..., 2] += dz[:, None]
        T[:, 2] += dz
    return J, T


def test_floating_motion_is_grounded():
    # Slow float of 12 cm (0.1 Hz) plus a fast 4 cm hop at 2 Hz, as in raw EDGE.
    slow = 0.06 * np.sin(2 * math.pi * 0.1 * t) + 0.3
    hop = 0.02 * np.sin(2 * math.pi * 2.0 * t)
    J, T = standing(slow + hop)
    assert floor_drift(J) > DRIFT_MAX
    T2, J2, info = fix_floor(T, J)
    assert floor_drift(J2) < 0.05
    assert abs(np.median(lowest_z(J2)) - FLOOR_Z) < 1e-6
    # The fast hop survives (its std is 0.014 m).
    assert 0.010 < lowest_z(J2).std() < 0.02
    # trans moves with the joints.
    np.testing.assert_allclose(J2[:, 0, 2] - T2[:, 2], J[:, 0, 2] - T[:, 2], atol=1e-9)
    assert info["float_range"] > 0.1


def test_drift_bounded_for_slow_ramp():
    # A clip that sinks 20 cm over the minute ends within the drift rule.
    J, T = standing(np.linspace(0.0, -0.2, F))
    _, J2, _ = fix_floor(T, J)
    assert floor_drift(J2) < 0.01


def test_grounded_motion_unchanged():
    # Feet on the floor, pelvis bouncing fast: nothing to remove.
    J, T = standing()
    J[:, 0, 2] += 0.05 * np.sin(2 * math.pi * 2.0 * t)
    T2, J2, info = fix_floor(T, J)
    np.testing.assert_allclose(J2, J, atol=1e-6)
    np.testing.assert_allclose(T2, T, atol=1e-6)
    assert abs(info["shift"]) < 1e-9


def root_poses(yaw_rad):
    """poses whose root faces robot yaw yaw_rad in a z-up world (body frame mapped by P)."""
    poses = np.zeros((F, 72))
    rot = R.from_euler("z", np.asarray(yaw_rad)[:, None]) * R.from_matrix(P)
    poses[:, :3] = rot.as_rotvec()
    return poses


def test_heading_aligned():
    yaw = np.radians(30.0) + 0.5 * t / 60
    poses = root_poses(yaw)
    assert abs(math.degrees(start_heading(poses)) - 30.0) < 1.0
    J, T = standing()
    J[:, :, 0] += 0.3 * t[:, None]  # walk along +x
    T[:, 0] += 0.3 * t
    p2, T2, J2, th = align_heading(poses, T, J, heading_deg=-90.0)
    assert abs(math.degrees(start_heading(p2)) + 90.0) < 0.5
    # The relative turn over the clip is kept.
    end = math.degrees(start_heading(p2[-15:]) - start_heading(p2[:15]))
    assert abs(end - math.degrees(yaw[-8] - yaw[7])) < 1.0
    # Joints and trans rotate together, and heights do not change.
    np.testing.assert_allclose(J2[..., 2], J[..., 2], atol=1e-9)
    np.testing.assert_allclose(J2[:, 0], T2 + (J[:, 0] - T), atol=1e-9)
    walk = J2[-1, 0, :2] - J2[0, 0, :2]
    assert abs(math.degrees(math.atan2(walk[1], walk[0])) - math.degrees(th)) < 1e-6


def test_foot_slide():
    J, _ = standing()
    assert foot_slide(J) < 1e-9
    # The support foot slides at 0.4 m/s, the other foot is lifted.
    J[:, 10, 0] += 0.4 * t
    J[:, 11, 2] += 0.1
    assert abs(foot_slide(J) - 0.4) < 1e-6
    # Moving only the lifted foot does not count.
    J2, _ = standing()
    J2[:, 11, 2] += 0.1
    J2[:, 11, 0] += 1.0 * t
    assert foot_slide(J2) < 1e-9
