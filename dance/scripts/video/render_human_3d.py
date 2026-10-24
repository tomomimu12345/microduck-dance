"""Render a pkl's `full_pose` (F, 24, 3 joint positions, z-up, meters, 30 fps) as a 3D mannequin in MuJoCo.

    MUJOCO_GL=egl uv run python dance/scripts/video/render_human_3d.py --motion dance/data/motions/songs/swag.pkl \
        --out out/rec_v4/swag_human3d.mp4 [--start S --duration D] [--cam-distance M]

The output replaces the 2D stick figure of render_human.py: same 640x720 panel, 30 fps, and video
frame i shows pose frame round(S * 30) + i, so a full render (no --start) lines up with the song
from t = 0 like the robot recordings.

The mannequin is built at runtime as an MJCF with one mocap body per SMPL bone (23 capsules, the
standard SMPL parent table) and a sphere head. Each frame the capsule of bone (parent, child) is
placed at the midpoint of the two joints, turned so its axis runs from parent to child, and given
the half length of the bone. Left limbs are orange and right limbs blue, as in the 2D figure. The
floor, the gradient sky and the sun light follow the robot recordings (play_dance.py and the mjlab
plane terrain), with shadows on.

Floor: EDGE pins the lowest foot joint of most frames to the same height (about z = -0.94), so the
floor is the median over frames of the lowest foot joint. The clip is shifted so that this height
sits one foot radius above z = 0, which puts the foot capsules on the floor. Frames where a foot
dips below that level (a few cm, a few percent of frames) sink slightly into the floor.

Camera: the lagged camera of play_dance.py (first-order low-pass of the pelvis position, time
constant 1.5 s, horizontal gap capped), with the robot's elevation (-15 deg) and its azimuth
(150 deg) taken relative to the dancer's initial facing, so the human is seen from the same side
as the robot. The distance is CAM_DIST_PER_HEIGHT times the figure height, which makes the figure
fill about the same share of the frame height as the robot does at 0.5 m. The figure is generated
geometry with no face.
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_human import LEFT, PARENTS, RIGHT, load_pose  # noqa: E402

FPS = 30
SIZE = (640, 720)  # width, height, as render_human.py
# Robot recordings (play_dance.py): camera elevation and azimuth, robot initial yaw 0.
CAM_ELEVATION = -15.0
ROBOT_CAM_AZIMUTH = 150.0
CAM_TAU_S = 1.5
# The robot (about 0.25 m tall, 0.5 m away) fills 60 to 80 % of the frame height. 1.8 x height
# gives the human about 65 % and leaves room for the arms in the narrow 640 px panel.
CAM_DIST_PER_HEIGHT = 1.8
# Horizontal cap on the pelvis to camera-target gap (m). The robot uses 0.25 m, about 0.7 of the
# half width of its 1280 px frame. The human panel is narrower and the arms reach out, so the cap
# is about 0.45 of the half width here.
CAM_MAX_OFFSET = 0.6

BONES = [(PARENTS[j], j) for j in range(1, 24)]
FOOT_JOINTS = (7, 8, 10, 11)  # ankles and feet
HEAD_JOINT, NECK_JOINT = 15, 12
HEAD_RADIUS = 0.105
HEAD_OFFSET = 0.07  # head sphere centre above the head joint, along the neck direction
# Visor: a dark flat box on the front of the head sphere, so the facing direction can be read.
VISOR_HALF = (0.02, 0.075, 0.03)  # half sizes (m) along forward, left and up
VISOR_FORWARD = 0.085  # centre distance from the head sphere centre along forward
VISOR_UP = 0.015  # centre height above the head sphere centre

# Capsule radius (m) per child joint. Spine and hips are thicker (torso and pelvis).
_RADIUS = {
    1: 0.075, 2: 0.075, 3: 0.095,  # hips, lower spine
    4: 0.065, 5: 0.065, 6: 0.10,  # thighs, mid spine
    7: 0.05, 8: 0.05, 9: 0.11,  # shins, upper spine
    10: 0.04, 11: 0.04, 12: 0.05,  # feet, neck
    13: 0.055, 14: 0.055, 15: 0.045,  # collars, head joint
    16: 0.045, 17: 0.045, 18: 0.04, 19: 0.04,  # upper arms, forearms
    20: 0.035, 21: 0.035, 22: 0.03, 23: 0.03,  # wrists, hands
}
FOOT_RADIUS = 0.04

GREY = (0.86, 0.86, 0.88)
ORANGE = (1.0, 0.478, 0.349)  # #ff7a59, left in render_human.py
BLUE = (0.31, 0.765, 0.969)  # #4fc3f7, right

# play_dance.py add_sky colours (kept here so the module imports without mjlab).
SKY_TOP = (0.20, 0.42, 0.70)
SKY_HORIZON = (0.78, 0.88, 0.96)
# mjlab plane terrain checker. The robot sees 0.125 m squares; scaled to the human they are 0.5 m.
CHECKER_RGB1, CHECKER_RGB2, CHECKER_MARK = (0.2, 0.3, 0.4), (0.1, 0.2, 0.3), (0.8, 0.8, 0.8)
CHECKER_REPEAT = 1.0  # texture repeats per meter (2 squares per repeat)


def bone_radius(child: int) -> float:
    return _RADIUS[child]


def bone_color(child: int) -> tuple[float, float, float]:
    return ORANGE if child in LEFT else (BLUE if child in RIGHT else GREY)


def quat_z_to(v) -> np.ndarray:
    """Unit quaternion (w, x, y, z) that turns +z onto the direction of v."""
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])
    v = v / n
    w = 1.0 + v[2]  # 1 + dot(z, v)
    if w < 1e-9:  # v = -z: half turn about x
        return np.array([0.0, 1.0, 0.0, 0.0])
    q = np.array([w, -v[1], v[0], 0.0])  # (1 + z.v, z x v)
    return q / np.linalg.norm(q)


def rotate(q, v) -> np.ndarray:
    """Rotate vector v by unit quaternion q (w, x, y, z)."""
    w, u = q[0], np.asarray(q[1:])
    v = np.asarray(v, dtype=float)
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def capsule_pose(p, q) -> tuple[np.ndarray, np.ndarray, float]:
    """Centre, quaternion and half length of a capsule from joint p to joint q (axis = local z)."""
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    return (p + q) / 2.0, quat_z_to(q - p), float(np.linalg.norm(q - p)) / 2.0


def floor_height(pose: np.ndarray) -> float:
    """Median over frames of the lowest foot joint: the contact height EDGE keeps most frames at."""
    return float(np.median(pose[:, FOOT_JOINTS, 2].min(axis=1)))


def shift_to_floor(pose: np.ndarray) -> np.ndarray:
    """Pose shifted in z so that the floor height is one foot radius above z = 0."""
    out = np.array(pose, dtype=float, copy=True)
    out[..., 2] += FOOT_RADIUS - floor_height(pose)
    return out


def heading(pose: np.ndarray, frames: int = 30) -> float:
    """Initial facing yaw (rad). Forward = (left hip - right hip) x up, averaged over the first frames."""
    lr = (pose[:frames, 1] - pose[:frames, 2]).mean(axis=0)
    fwd = np.cross(lr, [0.0, 0.0, 1.0])
    return float(math.atan2(fwd[1], fwd[0]))


def figure_height(pose: np.ndarray) -> float:
    """Median top of the head sphere above the floor (pose already shifted)."""
    top = pose[:, HEAD_JOINT, 2] + HEAD_OFFSET + HEAD_RADIUS
    return float(np.median(top))


def frame_range(n_frames: int, start_s: float = 0.0, duration_s: float | None = None) -> range:
    """Pose frames to render: from round(start_s * FPS), duration_s * FPS of them (default: to the end)."""
    a = int(round(start_s * FPS))
    if a < 0 or a >= n_frames:
        raise ValueError(f"start {start_s} s is outside the clip ({n_frames / FPS:.2f} s)")
    b = n_frames if duration_s is None else a + int(round(duration_s * FPS))
    if b > n_frames:
        raise ValueError(f"start + duration = {b / FPS:.2f} s is past the clip end ({n_frames / FPS:.2f} s)")
    return range(a, b)


def lagged_targets(path: np.ndarray, dt: float = 1.0 / FPS, tau: float = CAM_TAU_S,
                   max_offset: float = CAM_MAX_OFFSET) -> np.ndarray:
    """Camera targets (T, 3) for the pelvis path: play_dance.lagged_follow, the robot's lagged camera."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluate"))  # play_dance.py
    from play_dance import lagged_follow

    return lagged_follow(path, dt, tau, max_offset)


def _rgb(c) -> str:
    return " ".join(f"{x:.3f}" for x in c)


def _rgba(c, a=1.0) -> str:
    return _rgb((*c, a))


def build_mjcf() -> str:
    """Scene with one mocap body per bone, one for the head and one for the visor. Capsule lengths are set per frame."""
    bodies = []
    for p, c in BONES:
        mat = "left" if c in LEFT else ("right" if c in RIGHT else "body")
        # The sun rides on the lower spine (mode="track": it moves with the body, its direction stays
        # straight down), so the shadow map stays centred on the dancer wherever the dancer goes.
        light = ('<light name="sun" mode="track" pos="0 0 5" dir="0 0 -1" directional="true" castshadow="true"/>'
                 if c == 3 else "")
        bodies.append(f'<body name="bone{c}" mocap="true"><geom type="capsule" size="{bone_radius(c)} 0.05" '
                      f'material="{mat}" contype="0" conaffinity="0"/>{light}</body>')
    bodies.append(f'<body name="head" mocap="true"><geom type="sphere" size="{HEAD_RADIUS}" material="body" '
                  'contype="0" conaffinity="0"/></body>')
    vx, vy, vz = VISOR_HALF
    bodies.append(f'<body name="visor" mocap="true"><geom type="box" size="{vx} {vy} {vz}" material="visor" '
                  'contype="0" conaffinity="0"/></body>')
    return f"""<mujoco model="human3d">
  <visual>
    <global offwidth="{SIZE[0]}" offheight="{SIZE[1]}"/>
    <quality shadowsize="4096"/>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="{_rgba(SKY_HORIZON)}"/>
    <map znear="0.05" zfar="60"/>
  </visual>
  <statistic center="0 0 0.9" extent="1.5"/>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="{_rgb(SKY_TOP)}" rgb2="{_rgb(SKY_HORIZON)}"
             width="512" height="3072"/>
    <texture name="checker" type="2d" builtin="checker" mark="edge" rgb1="{_rgb(CHECKER_RGB1)}"
             rgb2="{_rgb(CHECKER_RGB2)}" markrgb="{_rgb(CHECKER_MARK)}" width="300" height="300"/>
    <material name="floor" texture="checker" texuniform="true" texrepeat="{CHECKER_REPEAT} {CHECKER_REPEAT}"
              reflectance="0.2"/>
    <material name="body" rgba="{_rgba(GREY)}" specular="0.3" shininess="0.3"/>
    <material name="visor" rgba="0.08 0.08 0.1 1" specular="0.6" shininess="0.8"/>
    <material name="left" rgba="{_rgba(ORANGE)}" specular="0.3" shininess="0.3"/>
    <material name="right" rgba="{_rgba(BLUE)}" specular="0.3" shininess="0.3"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 0.05" material="floor"/>
    {chr(10).join('    ' + b for b in bodies)}
  </worldbody>
</mujoco>
"""


def set_pose(model, data, joints: np.ndarray) -> None:
    """Place every bone capsule and the head for one frame of (24, 3) joints, then mj_forward."""
    import mujoco

    for p, c in BONES:
        body = model.body(f"bone{c}")
        centre, quat, half = capsule_pose(joints[p], joints[c])
        data.mocap_pos[body.mocapid[0]] = centre
        data.mocap_quat[body.mocapid[0]] = quat
        # The floor is geom 0, so the capsule of a bone is found through its body, not by bone index.
        model.geom_size[body.geomadr[0], 1] = max(half, 1e-4)
    up = joints[HEAD_JOINT] - joints[NECK_JOINT]
    up = up / max(np.linalg.norm(up), 1e-9)
    centre = joints[HEAD_JOINT] + HEAD_OFFSET * up
    data.mocap_pos[model.body("head").mocapid[0]] = centre
    # Frame of the head: up along the neck, forward from the hips (left hip - right hip) x up.
    fwd = np.cross(joints[1] - joints[2], up)
    fwd = fwd / max(np.linalg.norm(fwd), 1e-9)
    left = np.cross(up, fwd)
    rot = np.stack([fwd, left, up], axis=1)  # columns are the visor box axes in world coordinates
    visor = model.body("visor").mocapid[0]
    data.mocap_pos[visor] = centre + VISOR_FORWARD * fwd + VISOR_UP * up
    mujoco.mju_mat2Quat(data.mocap_quat[visor], rot.reshape(9))
    mujoco.mj_forward(model, data)


def render(motion: Path, out: Path, start_s: float = 0.0, duration_s: float | None = None,
           cam_distance: float | None = None) -> dict:
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco

    pose = shift_to_floor(load_pose(motion))
    frames = frame_range(len(pose), start_s, duration_s)
    dist = cam_distance if cam_distance is not None else CAM_DIST_PER_HEIGHT * figure_height(pose)
    # The camera runs from frame 0 so a window render matches the same frames of a full render.
    targets = lagged_targets(pose[:, 0])
    azimuth = ROBOT_CAM_AZIMUTH + math.degrees(heading(pose))

    model = mujoco.MjModel.from_xml_string(build_mjcf())
    data = mujoco.MjData(model)
    assert model.nmocap == len(BONES) + 2
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth, cam.elevation, cam.distance = azimuth, CAM_ELEVATION, dist

    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{SIZE[0]}x{SIZE[1]}",
           "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(out)]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(model, height=SIZE[1], width=SIZE[0])
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n = 0
    try:
        for f in frames:
            set_pose(model, data, pose[f])
            cam.lookat[:] = targets[f]
            renderer.update_scene(data, camera=cam)
            proc.stdin.write(renderer.render().tobytes())
            n += 1
    finally:
        proc.stdin.close()
        rc = proc.wait()
        renderer.close()
    if rc != 0:
        raise RuntimeError(f"ffmpeg exited with {rc}")
    return {"out": str(out), "frames": n, "first_pose_frame": frames.start, "seconds": n / FPS,
            "cam_distance": round(dist, 3), "azimuth_deg": round(azimuth, 1)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--motion", type=Path, required=True, help="pkl with full_pose (F, 24, 3)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--start", type=float, default=0.0, help="first second of the clip to render")
    ap.add_argument("--duration", type=float, default=None, help="seconds to render (default: to the end)")
    ap.add_argument("--cam-distance", type=float, default=None,
                    help=f"m from the camera target (default {CAM_DIST_PER_HEIGHT} x figure height)")
    a = ap.parse_args()
    print(render(a.motion, a.out, a.start, a.duration, a.cam_distance))


if __name__ == "__main__":
    main()
