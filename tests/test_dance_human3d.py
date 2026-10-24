"""Tests for dance/scripts/video/render_human_3d.py on synthetic data (CPU only, no rendering)."""

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

_DIR = Path(__file__).resolve().parents[1] / "dance" / "scripts" / "video"
sys.path.insert(0, str(_DIR))
_spec = importlib.util.spec_from_file_location("render_human_3d", _DIR / "render_human_3d.py")
R = importlib.util.module_from_spec(_spec)
sys.modules["render_human_3d"] = R
_spec.loader.exec_module(R)

# Standard SMPL 24-joint kinematic tree.
SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]


def _standing(frames: int = 60) -> np.ndarray:
    """A rough T-free standing skeleton facing -y (left hip at +x), feet joints at z = -0.94, swaying."""
    j = np.zeros((24, 3))
    j[0] = [0, 0, 0]
    j[1], j[2] = [0.09, 0, -0.08], [-0.09, 0, -0.08]
    j[3] = [0, 0, 0.1]
    j[4], j[5] = [0.1, 0, -0.48], [-0.1, 0, -0.48]
    j[6] = [0, 0, 0.23]
    j[7], j[8] = [0.1, 0.02, -0.88], [-0.1, 0.02, -0.88]
    j[9] = [0, 0, 0.28]
    j[10], j[11] = [0.1, -0.1, -0.94], [-0.1, -0.1, -0.94]
    j[12] = [0, 0, 0.5]
    j[13], j[14] = [0.08, 0, 0.42], [-0.08, 0, 0.42]
    j[15] = [0, -0.02, 0.6]
    j[16], j[17] = [0.18, 0, 0.42], [-0.18, 0, 0.42]
    j[18], j[19] = [0.2, 0, 0.16], [-0.2, 0, 0.16]
    j[20], j[21] = [0.21, 0, -0.08], [-0.21, 0, -0.08]
    j[22], j[23] = [0.22, 0, -0.16], [-0.22, 0, -0.16]
    t = np.arange(frames) / 30.0
    pose = np.repeat(j[None], frames, axis=0)
    pose[:, :, 0] += 0.2 * np.sin(2 * np.pi * 0.5 * t)[:, None]
    return pose


def test_bones_match_the_smpl_parent_table():
    assert R.PARENTS == SMPL_PARENTS
    assert len(R.BONES) == 23
    assert R.BONES == [(SMPL_PARENTS[c], c) for c in range(1, 24)]
    assert {c for _, c in R.BONES} == set(range(1, 24))


def test_mjcf_has_one_mocap_body_per_bone_and_a_head():
    mujoco = pytest.importorskip("mujoco")
    m = mujoco.MjModel.from_xml_string(R.build_mjcf())
    assert m.nmocap == 25
    for _, c in R.BONES:
        b = m.body(f"bone{c}")
        assert m.geom_type[b.geomadr[0]] == mujoco.mjtGeom.mjGEOM_CAPSULE
        assert m.geom_size[b.geomadr[0], 0] == pytest.approx(R.bone_radius(c))
    assert m.vis.global_.offwidth == R.SIZE[0] and m.vis.global_.offheight == R.SIZE[1]


@pytest.mark.parametrize("p,q", [([0, 0, 0], [0, 0, 1]), ([1, 2, 3], [1.3, 1.6, 3.5]),
                                 ([0, 0, 1], [0, 0, 0.2]), ([0.1, -0.2, 0.5], [-0.4, 0.3, 0.5])])
def test_capsule_pose_from_two_endpoints(p, q):
    p, q = np.array(p, float), np.array(q, float)
    centre, quat, half = R.capsule_pose(p, q)
    assert np.allclose(centre, (p + q) / 2)
    assert half == pytest.approx(np.linalg.norm(q - p) / 2)
    assert np.linalg.norm(quat) == pytest.approx(1.0)
    axis = R.rotate(quat, [0, 0, 1])
    assert np.allclose(axis, (q - p) / np.linalg.norm(q - p), atol=1e-9)
    # The capsule ends land on the joints.
    assert np.allclose(centre - half * axis, p) and np.allclose(centre + half * axis, q)


def test_set_pose_places_every_capsule_on_its_bone():
    mujoco = pytest.importorskip("mujoco")
    m = mujoco.MjModel.from_xml_string(R.build_mjcf())
    d = mujoco.MjData(m)
    joints = R.shift_to_floor(_standing())[5]
    R.set_pose(m, d, joints)
    for p, c in R.BONES:
        b = m.body(f"bone{c}")
        g = b.geomadr[0]
        axis = d.geom_xmat[g].reshape(3, 3)[:, 2]
        half = m.geom_size[g, 1]
        ends = sorted([d.geom_xpos[g] - half * axis, d.geom_xpos[g] + half * axis], key=lambda e: np.linalg.norm(e - joints[p]))
        assert np.allclose(ends[0], joints[p], atol=1e-6) and np.allclose(ends[1], joints[c], atol=1e-6)


def test_visor_sits_on_the_front_of_the_head():
    mujoco = pytest.importorskip("mujoco")
    m = mujoco.MjModel.from_xml_string(R.build_mjcf())
    d = mujoco.MjData(m)
    pose = _standing()
    joints = R.shift_to_floor(pose)[5]
    R.set_pose(m, d, joints)
    fwd = np.array([math.cos(R.heading(pose)), math.sin(R.heading(pose)), 0.0])
    gap = d.geom_xpos[m.body("visor").geomadr[0]] - d.geom_xpos[m.body("head").geomadr[0]]
    assert gap @ fwd == pytest.approx(R.VISOR_FORWARD, abs=0.02)
    assert abs(gap @ np.cross(fwd, [0, 0, 1])) < 0.02


def test_floor_shift_puts_the_lowest_foot_on_the_floor():
    pose = _standing()
    shifted = R.shift_to_floor(pose)
    foot_bottom = shifted[:, R.FOOT_JOINTS, 2].min(axis=1) - R.FOOT_RADIUS
    assert abs(foot_bottom.min()) < 0.005
    assert np.allclose(foot_bottom, 0.0, atol=0.005)
    # Only z moves.
    assert np.allclose(shifted[..., :2], pose[..., :2])


def test_floor_ignores_a_few_frames_that_dip_below_the_contact_height():
    pose = _standing(90)
    pose[10:13, 10, 2] -= 0.06  # three frames with a foot 6 cm below the floor EDGE keeps
    shifted = R.shift_to_floor(pose)
    foot_bottom = shifted[:, R.FOOT_JOINTS, 2].min(axis=1) - R.FOOT_RADIUS
    assert np.median(foot_bottom) == pytest.approx(0.0, abs=0.005)


def test_heading_of_a_dancer_facing_minus_y():
    assert R.heading(_standing()) == pytest.approx(-math.pi / 2, abs=1e-6)
    # Same view as the robot: robot yaw 0, camera azimuth 150, so the human gets 150 - 90.
    assert R.ROBOT_CAM_AZIMUTH + math.degrees(R.heading(_standing())) == pytest.approx(60.0)


@pytest.mark.parametrize("start,dur", [(0.0, 10.0), (11.0, 10.0), (17.5, 30.0), (0.0, 60.0), (2.0, 0.5)])
def test_frame_count_is_duration_times_fps(start, dur):
    fr = R.frame_range(1800, start, dur)
    assert len(fr) == round(dur * 30)
    assert fr.start == round(start * 30)


def test_full_render_covers_the_clip_and_bad_windows_raise():
    assert R.frame_range(1800) == range(0, 1800)
    with pytest.raises(ValueError):
        R.frame_range(1800, 50.0, 20.0)
    with pytest.raises(ValueError):
        R.frame_range(1800, 61.0, 1.0)


def test_camera_is_the_lagged_follow_with_the_cap():
    path = np.zeros((300, 3))
    path[30:, 0] = 3.0  # jump of 3 m at 1 s
    tg = R.lagged_targets(path)
    assert np.allclose(tg[:30], 0.0)
    gap = np.abs(path[:, 0] - tg[:, 0])
    assert gap.max() <= R.CAM_MAX_OFFSET + 1e-9
    assert gap[-1] < 0.01
