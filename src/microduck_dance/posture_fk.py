"""Flat-foot kinematics of the walk model for the dance_v3 posture synergy.

The trunk is held at the identity pose and the leg joints are set. The ground
normal is taken as the mean of the two sole normals (the direction that is
world up for each foot at HOME). The trunk roll and pitch are read from that
normal, and the trunk height is the distance from the trunk origin to the mean
foot site along it, offset so that HOME reads DANCE_STAND_HEIGHT. The feet are
flat when both sole normals agree and both foot sites lie at the same height.

Used by dance/scripts/retarget/fit_posture_synergy.py to fit the synergy and by the
tests to check it. CPU only, no mjlab env.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

# The robot model is in the upstream package (the submodule). find_spec locates it without importing it.
WALK_XML = Path(importlib.util.find_spec("mjlab_microduck").origin).parent / "robot" / "microduck" / "scene_walk.xml"
LEG_JOINTS = tuple(
    f"{side}_{j}" for side in ("left", "right") for j in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle")
)


def home_joint_pos(names) -> np.ndarray:
    import re

    # Importing the tasks package first lets mjlab's task plugin loader finish
    # before microduck_constants is imported here (a circular import otherwise).
    import microduck_dance.tasks  # noqa: F401
    from mjlab_microduck.robot.microduck_constants import HOME_FRAME

    out = []
    for n in names:
        val = 0.0
        for pat, v in HOME_FRAME.joint_pos.items():
            if re.fullmatch(pat, n):
                val = v
        out.append(val)
    return np.asarray(out, dtype=np.float64)


@dataclass
class FlatFootState:
    height: float  # trunk height above the feet minus the HOME value, m
    roll: float
    pitch: float
    foot_height_diff: float  # left minus right foot height along the ground normal, m
    foot_tilt_diff: float  # angle between the two sole normals, rad
    com_forward: float  # CoM minus mean foot position along the trunk heading on the ground, m


class FlatFootKinematics:
    def __init__(self, xml: Path | str = WALK_XML):
        self.m = mujoco.MjModel.from_xml_path(str(xml))
        self.d = mujoco.MjData(self.m)
        self.qadr = np.array(
            [self.m.jnt_qposadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in LEG_JOINTS]
        )
        self.home = home_joint_pos(LEG_JOINTS)
        self.trunk = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
        self.sites = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, s) for s in ("left_foot", "right_foot")]
        self._set(np.zeros(len(LEG_JOINTS)))
        # Sole normal in each foot site frame: world up at HOME with a level trunk.
        self.sole = [self.d.site_xmat[s].reshape(3, 3).T @ np.array([0.0, 0.0, 1.0]) for s in self.sites]
        self.h0 = 0.0
        self.c0 = 0.0
        st = self.state(np.zeros(len(LEG_JOINTS)))
        self.h0 = st.height
        self.c0 = st.com_forward

    def _set(self, delta: np.ndarray) -> None:
        d = self.d
        d.qpos[:] = 0.0
        d.qpos[3] = 1.0
        d.qpos[self.qadr] = self.home + delta
        mujoco.mj_kinematics(self.m, d)
        mujoco.mj_comPos(self.m, d)

    def state(self, delta: np.ndarray) -> FlatFootState:
        """delta: leg joint offsets from HOME in LEG_JOINTS order."""
        self._set(np.asarray(delta, dtype=np.float64))
        d = self.d
        nl, nr = (d.site_xmat[s].reshape(3, 3) @ self.sole[i] for i, s in enumerate(self.sites))
        n = nl + nr
        n /= np.linalg.norm(n)
        pl, pr = d.site_xpos[self.sites[0]].copy(), d.site_xpos[self.sites[1]].copy()
        pm = 0.5 * (pl + pr)
        height = float(-pm @ n) - self.h0
        roll = float(np.arctan2(n[1], n[2]))
        pitch = float(np.arcsin(np.clip(-n[0], -1.0, 1.0)))
        com = d.subtree_com[self.trunk].copy()  # trunk_base subtree is the whole robot
        fwd = np.array([1.0, 0.0, 0.0]) - n[0] * n
        fwd /= np.linalg.norm(fwd)
        return FlatFootState(
            height=height,
            roll=roll,
            pitch=pitch,
            foot_height_diff=float((pl - pr) @ n),
            foot_tilt_diff=float(np.arccos(np.clip(nl @ nr, -1.0, 1.0))),
            com_forward=float((com - pm) @ fwd) - self.c0,
        )
