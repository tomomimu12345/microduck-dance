"""Clean-up and contact checks for generated SMPL dances (pure numpy, z-up, 30 fps).

EDGE (AIST++) output floats. The whole body wanders up and down by 7 to 12 cm
over several seconds while both feet stay off the floor (alt-generator report,
2026-10-20). fix_floor removes that slow wander. align_heading turns a clip so
it starts facing the same way as the FlowerDance clips, which is what
dance/scripts/video/render_human.py assumes. retarget.py is yaw invariant, so the
heading only changes the render view.

All functions take the arrays of a dance pkl:
- poses (F, 72): SMPL axis-angle rotations, root first;
- trans (F, 3): root translation in the z-up data world;
- joints (F, 24, 3): joint positions (the full_pose key).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation as R

from microduck_dance.retarget import P, _heading

FPS = 30.0
# Median lowest joint of the FlowerDance clips (z-up). retarget.py estimates the
# floor itself, so the level is only a convention shared by all dance pkls.
FLOOR_Z = -0.94
FLOAT_CUTOFF_HZ = 0.5
# Drift rule from the alt-generator report: p98 - p2 of the lowest joint height
# above 0.10 m means the body floats.
DRIFT_MAX = 0.10
FOOT_JOINTS = (10, 11)  # SMPL left and right foot
CONTACT_HEIGHT = 0.03  # m above the per-frame lowest joint


def lowest_z(joints: np.ndarray) -> np.ndarray:
    """(F,) height of the lowest of the 24 joints in each frame."""
    return np.asarray(joints, float)[..., 2].min(axis=1)


def floor_drift(joints: np.ndarray) -> float:
    """p98 - p2 of the per-frame lowest joint height (m)."""
    low = lowest_z(joints)
    return float(np.percentile(low, 98) - np.percentile(low, 2))


def slow_component(x: np.ndarray, fps: float = FPS, cutoff_hz: float = FLOAT_CUTOFF_HZ) -> np.ndarray:
    """Part of x below cutoff_hz (2nd order Butterworth, zero phase)."""
    b, a = butter(2, cutoff_hz / (0.5 * fps))
    return filtfilt(b, a, np.asarray(x, float))


def fix_floor(trans: np.ndarray, joints: np.ndarray, fps: float = FPS,
              cutoff_hz: float = FLOAT_CUTOFF_HZ, floor_z: float = FLOOR_Z):
    """Remove the slow float and put the median lowest joint at floor_z.

    The part of the lowest-joint height below cutoff_hz, minus its median, is
    subtracted from the z of trans and of every joint. Hops faster than
    cutoff_hz stay. Then all z is shifted so the median lowest joint sits at
    floor_z. Rotations are unchanged. Returns (trans, joints, info).
    """
    J = np.asarray(joints, float).copy()
    T = np.asarray(trans, float).copy()
    slow = slow_component(lowest_z(J), fps, cutoff_hz)
    corr = slow - np.median(slow)
    J[..., 2] -= corr[:, None]
    T[:, 2] -= corr
    shift = floor_z - float(np.median(lowest_z(J)))
    J[..., 2] += shift
    T[:, 2] += shift
    info = {"float_std": float(corr.std()), "float_range": float(np.ptp(corr)), "shift": shift}
    return T, J, info


def start_heading(poses: np.ndarray, n_frames: int = 15) -> float:
    """Mean pelvis heading (rad, robot convention as in retarget.py) over the first frames."""
    root = R.from_rotvec(np.asarray(poses, float).reshape(-1, 72)[:, :3])
    fwd = (root * R.from_matrix(P).inv()).apply([1.0, 0.0, 0.0])
    return float(_heading(fwd)[:n_frames].mean())


def align_heading(poses: np.ndarray, trans: np.ndarray, joints: np.ndarray,
                  heading_deg: float = -90.0, n_frames: int = 15):
    """Rotate the clip about world z so its start heading is heading_deg.

    The FlowerDance clips start near -97 deg (dancer facing -y). Rotates trans,
    joints and the root rotation poses[:, :3] about the world origin. Returns
    (poses, trans, joints, yaw_rad).
    """
    p = np.asarray(poses, float).reshape(-1, 72).copy()
    th = np.radians(heading_deg) - start_heading(p, n_frames)
    rz = R.from_euler("z", th)
    p[:, :3] = (rz * R.from_rotvec(p[:, :3])).as_rotvec()
    J = rz.apply(np.asarray(joints, float).reshape(-1, 3)).reshape(-1, 24, 3)
    T = rz.apply(np.asarray(trans, float))
    return p, T, J, float(th)


def foot_slide(joints: np.ndarray, fps: float = FPS, contact_height: float = CONTACT_HEIGHT) -> float:
    """Mean horizontal speed (m/s) of the support foot.

    The support foot of a frame is the lower of the two foot joints, counted
    only while it is within contact_height of the per-frame lowest joint (so a
    frame where a hand or knee is lowest, or both feet are up, does not count).
    A planted foot gives 0. NaN when no frame qualifies.
    """
    J = np.asarray(joints, float)
    feet = J[:, list(FOOT_JOINTS)]  # (F, 2, 3)
    v = np.linalg.norm(np.gradient(feet[..., :2], 1.0 / fps, axis=0), axis=2)  # (F, 2)
    k = np.argmin(feet[..., 2], axis=1)
    idx = np.arange(len(J))
    support_z = feet[idx, k, 2]
    contact = support_z - lowest_z(J) < contact_height
    if not contact.any():
        return float("nan")
    return float(v[idx, k][contact].mean())
