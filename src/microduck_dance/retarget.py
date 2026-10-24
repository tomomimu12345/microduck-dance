"""Retarget SMPL motion (smpl_poses, smpl_trans) to the 13D dance command stream.

Axes
- SMPL body frame: x left, y up, z forward.
- Robot frame (body and world): x forward, y left, z up.
- P maps SMPL body components (z, x, y) to robot (x, y, z). It is a cyclic
  permutation, so det P = +1 and handedness is kept.
- The data world is y-up or z-up, chosen by detect_up_axis. W is the cyclic
  permutation that sends the detected up axis to robot z (W = P for y-up and
  the identity for z-up). Both are proper rotations.
- The robot-frame body orientation is W * R_smpl * P^T. Its +x column is the
  pelvis forward vector W R0 e_z, so heading needs no hardcoded world forward.

Angles use ZYX Euler (yaw, pitch, roll), the same convention as
dance_body_tracking in tasks/dance_commands.py. With y to the left, a positive
rotation about +y lowers the nose, so a forward lean is positive pitch.

Pelvis position. SMPL places the pelvis joint at trans + J0 in the data world,
where J0 is the pelvis rest position of the body shape. The root rotation acts
about J0, so J0 is added unrotated. A y-up clip converted to z-up by baking a
world rotation Q into the root must use trans' = Q (trans + J0) - J0 to keep
the body in place, so trans + J0 is the pelvis in both conventions, while raw
trans differs from it by J0 (about 0.22 m along y-up). We use the neutral shape
J0 (SMPL_PELVIS_OFFSET) for detect_up_axis, velocity and height. The residual
is the shape dependence of J0 (a few cm across betas), which shifts the median
pelvis height and so the default height_gain by a few percent.

Joint positions. When the clip also has joint positions (FlowerDance and EDGE
write full_pose, (F, 24, 3)), they take priority over trans + J0:
- up axis: the axis along which head (15) minus mean ankle (7, 8) is above
  HEAD_ANKLE_MIN on at least HEAD_ANKLE_FRAC of frames, exactly one axis;
- pelvis: joints[:, 0] (in FlowerDance output this equals smpl_trans);
- height: pelvis up coordinate minus the floor, where the floor is the
  FLOOR_PERCENTILE percentile of the per-frame lowest joint over the clip.
  A low percentile, not the minimum, so a few bad frames do not move it.
  Joints are joint centres, so the lowest one (a foot joint) sits a few cm
  above the real floor and the floor-relative height is a few cm low. That
  shifts only the default height_gain, by a few percent.
- body roll and pitch: the trunk line, pelvis (0) to neck (12), in the
  heading frame, read as the ZYX roll and pitch that would tilt robot +z onto
  it: pitch = atan2(v_x, v_z), roll = atan2(-v_y, hypot(v_x, v_z)). A neck
  ahead of the pelvis is positive pitch and a lean toward the right (-y) is
  positive roll, the same signs as the chest-rotation path. The chest frame is
  not used here because spine flexion over a pelvis tilted back rotates it
  10 to 20 deg forward while the trunk stays upright, which saturated body
  pitch on real clips.
FlowerDance clips are z-up with the floor near z = -0.95 and the pelvis near
z = 0, which is why trans alone cannot reveal the up axis there.

Heading low-pass. The unwrapped heading is low-passed at heading_cutoff_hz
(2nd order Butterworth, zero phase via filtfilt) before it is differentiated
into wz and before it rotates the pelvis velocity into vx and vy. Dancers twist
the hips back and forth at 1 to 2 Hz, outside the 0.1 to 0.6 Hz band the
policy saw for wz, so only the slower turning is passed on. Body roll and
pitch use the unfiltered heading, so a lean is read relative to where the
pelvis faces in that frame.

Head gain and groove layer (Task 11). head_gain multiplies the four head
columns (neck_pitch, head_pitch, head_yaw, head_roll) of the human motion
before the low-pass. It is a 4-tuple or one number for all four. When beats are passed and
groove is on, the beat layer of groove.py (head_pitch nod, body_z dip,
head_yaw sway) is added after the rate limit, so its pulses keep their shape
and timing, and before the clip to the training ranges and the ramps, so it is
clipped and faded like the rest. head_gain 1.0 with groove off reproduces the
earlier output exactly.

Music-driven variety (Task 15). smpl_to_commands takes the beats json as music
(compute_beats.py with the onset and RMS keys). With it:
- phrase_gate: the twist columns (vx, vy, wz) are multiplied by the phrase gate
  of phrases.py after the rate limit, so the robot stands in the quieter
  phrases and walks in the louder ones;
- groove_pattern: the groove layer uses the per-song pattern of groove.py
  (accent nods, bounce depth from the onset strength, trunk or head sway from
  the tempo) instead of the shared one, scaled by groove_phrase_gain where the
  gate is 0 and by 1 where it is 1.
Without music (beats only) neither applies and the shared groove is used, as
before. foot_velocity (off by default) replaces the pelvis velocity by the
pelvis velocity minus the support foot's horizontal velocity. On the EDGE
clips it did not remove the slide (Task 15 measurement in EXPERIMENTS.md),
so it stays off.

Fidelity to the human (Task 19, match report of 2026-10-23). The user said the
robot does not move like the 3D mannequin next to it. The defaults changed:
- phrase_gate is off. The dancer keeps moving in every phrase (mean pelvis
  speed 0.31 to 0.49 m/s), so the gate stopped the robot where the human
  travels and turns. phrase_gate=True restores it.
- groove_scale 0.5 halves the whole beat layer (nod, bounce, sway, also the
  per-song pattern). groove_scale=1.0 restores the full layer.
- posture_lead_s (POSTURE_LEAD_S, 0.12 s) moves the posture columns 3 to 13
  of the retargeted human motion earlier in time, after the low-pass and the
  rate limit. It cancels the lag of the robot behind the human that the match
  report measured on dance_v4 (posture 0.10 to 0.16 s, of which 0.06 to 0.08 s
  is the causal retarget filter and the rest the policy). The groove layer is
  added after the shift and is not shifted. It is not filtered, so it has only
  the policy lag, which groove_lag_s (GROOVE_POLICY_LAG_COMP_S) already
  handles. The two leads therefore never add up: a human nod is 0.12 s early,
  a groove nod 0.14 s before its beat. posture_lead_s=0 restores the old timing.
- heading_servo. The commanded heading psi_c is the integral of the emitted
  wz. With the servo, wz = clip(g dpsi_h/dt + HEADING_SERVO_GAIN (g psi_h -
  psi_c), wz range), where psi_h is the low-passed human heading relative to
  its first frame and g is yaw_rate_gain. When the clip has cut a fast turn,
  the second term keeps turning afterwards until psi_c has caught up. This is
  open loop on the command itself (the robot's measured yaw is not used). vx
  and vy are the human pelvis velocity (times vel_gain) expressed in the
  commanded heading frame, so the robot walks the human's world direction
  even while its heading lags. The emitted wz carries the ramps and the gate
  inside the loop, so psi_c is exactly the integral of the file's wz.
  heading_servo=False restores the open-loop wz and the human heading frame.
- vel_gain 0.3, chosen so the robot's realized travel in body heights
  (0.5 s smoothed pelvis path, robot 0.25 m, human figure height) is within
  20 % of the human's on the 3 video songs with dance_v4 (0.89 to 0.94). At
  0.6 it was 1.38 to 1.42 (dance/docs/EXPERIMENTS.md, fid001).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation as R

from microduck_dance.groove import check_beats, choose_pattern, groove_layer, pattern_layer
from microduck_dance.phrases import FADE_S, MIN_RMS_CV, plan_phrases, twist_gate
from microduck_dance.ranges import ALL_RANGES, CMD_DIM
from microduck_dance.ranges import HEAD as HEAD_COLS

# SMPL body (x left, y up, z forward) to robot (x forward, y left, z up).
P = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
# Data world to robot world, keyed by detected up axis.
WORLD_TO_ROBOT = {1: P, 2: np.eye(3)}

# Neutral SMPL pelvis rest position J0 in template axes (x left, y up, z forward), metres.
SMPL_PELVIS_OFFSET = np.array([-0.0018, -0.2233, 0.0282])

# detect_up_axis thresholds. An axis is a plausible pelvis height when its mean
# lies in [UP_MEAN_MIN, UP_MEAN_MAX] metres and its std is at most
# UP_MAX_REL_STD times the mean. Exactly one axis must qualify.
UP_MEAN_MIN = 0.5
UP_MEAN_MAX = 2.0
UP_MAX_REL_STD = 0.2

# detect_up_axis with joints: head minus mean ankle along the up axis must exceed
# HEAD_ANKLE_MIN metres on at least HEAD_ANKLE_FRAC of the frames.
HEAD_ANKLE_MIN = 0.5
HEAD_ANKLE_FRAC = 0.9
HEAD, L_ANKLE, R_ANKLE = 15, 7, 8
FLOOR_PERCENTILE = 2.0

# Frames whose forward vector has a horizontal norm below this hold heading.
MIN_HORIZONTAL_FORWARD = 1e-3

SPINE_CHAIN = (0, 3, 6, 9)  # pelvis to chest
NECK_HEAD_CHAIN = (12, 15)  # chest to head
NECK = 12
FOOT_JOINTS = (10, 11)  # SMPL left and right foot

# Keys of the beats json that the phrase gate and the groove pattern read.
MUSIC_KEYS = ("beat_times_s", "tempo_bpm", "onset_mean", "onset_at_beat", "beat_onset", "beat_rms")


# Default groove lag: the command leads the beat by the measured head_pitch tracking lag.
GROOVE_POLICY_LAG_COMP_S = -0.14

# Posture columns (3 to 13) lead the human by this much. Match report 2026-10-23:
# with dance_v4 every posture channel trails the human by 0.10 to 0.16 s (best lag of
# the cross-correlation, 3 songs), 0.06 to 0.08 s from the causal 2 Hz low-pass and
# rate limit here and the rest from the policy. 0.12 s raised r(human, robot) for
# height 0.52 to 0.71 and head pitch 0.73 to 0.90 in the report's sim check.
POSTURE_LEAD_S = 0.12
POSTURE_COLS = slice(3, 13)

# Heading servo gain (1/s), the value of the match report's sim check (P2). A heading
# error decays with a time constant of 1 / 1.5 = 0.67 s when wz is not clipped.
HEADING_SERVO_GAIN = 1.5
WZ = 2

# RetargetConfig overrides that give the Task 15 defaults back (the commands dance_v4
# was trained and recorded on): RetargetConfig(**TASK15_OPTIONS).
TASK15_OPTIONS = {"phrase_gate": True, "groove_scale": 1.0, "posture_lead_s": 0.0, "heading_servo": False,
                  "vel_gain": 1.3}


@dataclass(frozen=True)
class RetargetConfig:
    # height_gain chosen in exp tune001 (dance/docs/EXPERIMENTS.md): 0.55 puts body_z p95
    # near 0.02 m. The physical scale (robot over human height) would be about 0.1.
    # vel_gain was 2.0 there (vx p95 about 0.24 m/s on FlowerDance). The EDGE clips
    # move the pelvis faster, so Task 15 lowered it to 1.3, the largest value on a
    # 0.1 grid with vx and vy at the clip limit on at most 8 % of the frames of the
    # 5 songs and 33 library clips, phrase gate off (vy 7.6 %, vx 1.3 %). None for
    # height_gain means nominal_height / median human pelvis height.
    # Task 19 lowered vel_gain to 0.3 (module docstring): with the gate off the robot
    # walks all the time, and at 0.6 to 1.0 it travelled 1.4 to 1.9 times the human.
    vel_gain: float = 0.3
    yaw_rate_gain: float = 1.0
    height_gain: float | None = 0.55
    nominal_height: float = 0.095
    neck_pitch_share: float = 0.5
    # Per-column sign from the human convention (positive pitch is nose down,
    # positive yaw is left, positive roll is the right ear down) to the robot joint
    # convention, measured on microduck kinematics. Joint neck_pitch + raises the
    # nose and head_roll + tilts the head to the robot's left, so both are
    # flipped. head_pitch + lowers the nose and head_yaw + turns left, as retargeted.
    signs: tuple[float, ...] = (1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    cutoff_hz: float = 2.0
    # Units per second. A 0 holds that dimension at exactly 0.
    max_rate: tuple[float, ...] = (1.0, 1.0, 4.0, 6.0, 6.0, 6.0, 6.0, 0, 0, 0.15, 2.0, 2.0, 0)
    ramp_in_s: float = 1.0
    out_dt: float = 0.02
    heading_cutoff_hz: float = 0.5  # zero-phase low-pass on heading; <= 0 disables it
    ramp_out_s: float = 1.0  # linear fade of all dims to exactly 0 at the last frame
    # Task 11. Gain on the human head columns 3 to 6 (neck_pitch, head_pitch, head_yaw,
    # head_roll), or one number for all four. 1.0 is the earlier behavior. head_roll
    # stays at 1.0: at 2.0 its +-0.31 range clipped 12 to 36 % of frames per song.
    head_gain: float | tuple[float, ...] = (2.0, 2.0, 2.0, 1.0)
    # Beat groove layer, applied only when smpl_to_commands gets beats (see groove.py).
    groove: bool = True
    groove_nod: float = 0.35  # rad added to head_pitch at the pulse peak (look down)
    groove_bounce: float = -0.015  # m added to body_z at the pulse peak
    groove_sway: float = 0.5  # rad amplitude of the half-beat head_yaw sine
    groove_width: float = 0.6  # pulse width as a fraction of the local beat period
    # Pulse peak and sway extreme sit this long after the beat. Negative compensates the
    # measured policy lag: dance_v2 realizes head_pitch about 0.14 s after the command
    # (motion-amp report, swag, best-lag correlation), so the command peaks 0.14 s early
    # and the realized nod lands on the beat.
    groove_lag_s: float = GROOVE_POLICY_LAG_COMP_S
    # Task 15, active only when smpl_to_commands gets music (see the module docstring).
    # Off since Task 19: the dancer keeps moving in every phrase.
    phrase_gate: bool = False
    phrase_fade_s: float = FADE_S
    phrase_min_cv: float = MIN_RMS_CV
    groove_pattern: bool = True
    groove_phrase_gain: float = 1.3  # groove amplitude while the twist is gated off
    # Pelvis minus support-foot velocity (joints only). Off: on the EDGE clips the
    # support foot slides with the pelvis, so it removed almost none of the slide.
    foot_velocity: bool = False
    # Task 19 (module docstring). Factor on the whole groove layer, 1.0 is the Task 15 layer.
    groove_scale: float = 0.5
    # Posture columns 3 to 13 of the human motion move this much earlier; 0 disables it.
    posture_lead_s: float = POSTURE_LEAD_S
    # wz pulls the commanded heading toward the human heading, vx and vy in that heading.
    heading_servo: bool = True
    heading_servo_gain: float = HEADING_SERVO_GAIN


def _detect_up_axis_joints(joints: np.ndarray) -> int:
    span = joints[:, HEAD] - 0.5 * (joints[:, L_ANKLE] + joints[:, R_ANKLE])
    frac = (span > HEAD_ANKLE_MIN).mean(axis=0)
    cand = [i for i in range(3) if frac[i] >= HEAD_ANKLE_FRAC]
    if len(cand) != 1:
        raise ValueError(
            f"cannot detect up axis from joints: head-ankle median={np.median(span, axis=0).round(3).tolist()}, "
            f"fraction above {HEAD_ANKLE_MIN} m={frac.round(3).tolist()}, candidates={cand}"
        )
    if cand[0] == 0:
        raise ValueError("x-up motion is not supported")
    return cand[0]


def _check_joints(joints, F: int) -> np.ndarray:
    joints = np.asarray(joints, dtype=float)
    if joints.shape != (F, 24, 3):
        raise ValueError(f"joints must have shape ({F}, 24, 3), got {joints.shape}")
    return joints


def detect_up_axis(trans: np.ndarray, joints: np.ndarray | None = None) -> int:
    """Return 1 (y-up) or 2 (z-up), else raise ValueError.

    With joints (F, 24, 3) the head-over-ankle test decides. Without them the
    test runs on the pelvis position trans + SMPL_PELVIS_OFFSET.
    """
    if joints is not None:
        return _detect_up_axis_joints(_check_joints(joints, len(np.asarray(trans))))
    trans = np.asarray(trans, dtype=float)
    if trans.ndim != 2 or trans.shape[1] != 3 or len(trans) == 0:
        raise ValueError(f"trans must have shape (F, 3), got {trans.shape}")
    pelvis = trans + SMPL_PELVIS_OFFSET
    mean = pelvis.mean(axis=0)
    std = pelvis.std(axis=0)
    cand = [
        i for i in range(3)
        if UP_MEAN_MIN <= mean[i] <= UP_MEAN_MAX and std[i] <= UP_MAX_REL_STD * mean[i]
    ]
    if len(cand) != 1:
        raise ValueError(
            f"cannot detect up axis: mean={mean.round(3).tolist()} std={std.round(3).tolist()}, "
            f"plausible pelvis-height axes={cand}"
        )
    if cand[0] == 0:
        raise ValueError("x-up motion is not supported")
    return cand[0]


def _chain(poses: np.ndarray, joints: tuple[int, ...]) -> R:
    rot = R.from_rotvec(poses[:, joints[0]])
    for j in joints[1:]:
        rot = rot * R.from_rotvec(poses[:, j])
    return rot


def _heading(fwd: np.ndarray) -> np.ndarray:
    """Unwrapped heading of robot-world forward vectors; holds through degenerate frames."""
    horiz = np.linalg.norm(fwd[:, :2], axis=1)
    valid = horiz >= MIN_HORIZONTAL_FORWARD
    psi = np.zeros(len(fwd))
    if not valid.any():
        return psi
    raw = np.arctan2(fwd[:, 1], fwd[:, 0])
    last = raw[np.argmax(valid)]  # frames before the first valid one take its value
    for i in range(len(fwd)):
        if valid[i]:
            last = raw[i]
        psi[i] = last
    return np.unwrap(psi)


def _lowpass_zero_phase(x: np.ndarray, fps: float, cutoff_hz: float) -> np.ndarray:
    """2nd order Butterworth with filtfilt. Returns x unchanged if disabled or too short."""
    nyq = 0.5 * fps
    if cutoff_hz <= 0 or cutoff_hz >= nyq:
        return x.copy()
    b, a = butter(2, cutoff_hz / nyq)
    if len(x) <= 3 * max(len(a), len(b)):
        return x.copy()
    # Odd extension over about 3 time constants, so a steady turn (a ramp)
    # extends linearly and the start-up transient dies out inside the padding.
    padlen = min(len(x) - 1, int(math.ceil(3.0 * fps / cutoff_hz)))
    return filtfilt(b, a, x, padtype="odd", padlen=padlen)


def _trunk_roll_pitch(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ZYX roll and pitch (yaw 0) that tilt robot +z onto v, (F, 3) in the heading frame."""
    pitch = np.arctan2(v[:, 0], v[:, 2])
    roll = np.arctan2(-v[:, 1], np.hypot(v[:, 0], v[:, 2]))
    return roll, pitch


def _euler_zyx(rot: R) -> np.ndarray:
    """(F, 3) yaw, pitch, roll, each unwrapped over time."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # gimbal lock notices
        e = rot.as_euler("ZYX")
    return np.unwrap(e, axis=0)


def _lowpass(x: np.ndarray, dt: float, cutoff_hz: float) -> np.ndarray:
    if cutoff_hz <= 0:
        return x.copy()
    alpha = dt / (dt + 1.0 / (2.0 * math.pi * cutoff_hz))
    y = np.empty_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = y[i - 1] + alpha * (x[i] - y[i - 1])
    return y


def _rate_limit(x: np.ndarray, dt: float, max_rate: np.ndarray) -> np.ndarray:
    step = max_rate * dt
    y = np.empty_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = y[i - 1] + np.clip(x[i] - y[i - 1], -step, step)
    y[:, max_rate == 0] = 0.0
    return y


def _shift_earlier(x: np.ndarray, n: int) -> np.ndarray:
    """x moved n samples earlier along axis 0; the last value fills the end."""
    if n <= 0 or len(x) == 0:
        return x.copy()
    n = min(n, len(x) - 1)
    return np.concatenate([x[n:], np.repeat(x[-1:], n, axis=0)])


def heading_servo(psi_h: np.ndarray, dt: float, gain: float, wz_range: tuple[float, float],
                  max_rate: float, scale: np.ndarray | None = None,
                  ff: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """wz (T,) whose integral psi_c follows the target heading psi_h (T,), and psi_c (T,).

    w_k = clip(ff_k + gain (psi_h_k - psi_c_k), wz_range), rate limited to max_rate
    (rad/s^2, 0 means no limit), and wz_k = scale_k w_k (ramps and gate, default 1).
    The limit acts on w, before scale. So wz is rate limited only where scale is
    constant. A change of scale passes to wz at once (a gate that opens or closes
    steps wz), and the slow ramps keep that effect small. The feed-forward ff
    defaults to the derivative of psi_h.
    With the phrase gate on (scale 0 in closed phrases) psi_c stands still while psi_h
    keeps moving, so the heading error builds up during every closed phrase. w keeps
    following that error and sits at the clip, so the servo spins at the clip as soon
    as the phrase opens, until the error is gone. This is why the gate is off by
    default and TASK15_OPTIONS leaves the servo off.
    psi_c_0 = psi_h_0 and psi_c_{k+1} = psi_c_k + wz_k dt, so psi_c_k is the heading the
    robot has been commanded to before step k, exactly the integral of the returned wz.
    """
    psi_h = np.asarray(psi_h, dtype=float)
    T = len(psi_h)
    scale = np.ones(T) if scale is None else np.asarray(scale, dtype=float)
    if ff is None:
        ff = np.gradient(psi_h, dt) if T > 1 else np.zeros(T)
    wz = np.zeros(T)
    psi_c = np.zeros(T)
    c, prev = (psi_h[0] if T else 0.0), 0.0
    step = max_rate * dt if max_rate > 0 else np.inf
    for k in range(T):
        psi_c[k] = c
        w = float(np.clip(ff[k] + gain * (psi_h[k] - c), wz_range[0], wz_range[1]))
        w = prev + float(np.clip(w - prev, -step, step))
        prev = w
        wz[k] = scale[k] * w
        c += wz[k] * dt
    return wz, psi_c


def check_music(music: dict) -> dict:
    missing = [k for k in MUSIC_KEYS if k not in music]
    if missing:
        raise ValueError(f"beats json lacks {missing}; rerun dance/scripts/generate/compute_beats.py")
    n = len(music["beat_times_s"])
    for k in ("onset_at_beat", "beat_onset", "beat_rms"):
        if len(music[k]) != n:
            raise ValueError(f"music[{k!r}] has {len(music[k])} values for {n} beats")
    return music


def support_foot_velocity(joints: np.ndarray, up: int, dt: float) -> np.ndarray:
    """(F, 3) robot-world velocity of the lower SMPL foot joint, horizontal part only."""
    feet = joints[:, list(FOOT_JOINTS)] @ WORLD_TO_ROBOT[up].T  # (F, 2, 3) robot world
    v = np.gradient(feet, dt, axis=0)
    k = np.argmin(feet[..., 2], axis=1)
    vs = v[np.arange(len(joints)), k]
    vs[:, 2] = 0.0
    return vs


def smpl_to_commands(
    poses: np.ndarray,
    trans: np.ndarray,
    fps: float,
    cfg: RetargetConfig,
    joints: np.ndarray | None = None,
    beats: np.ndarray | None = None,
    music: dict | None = None,
    return_info: bool = False,
):
    """Convert SMPL poses (F, 72) and trans (F, 3) at fps to commands (T, 13) at cfg.out_dt.

    joints (F, 24, 3), when given, supplies the up axis, pelvis and floor (see
    the module docstring). beats (seconds from the clip start), when given and
    cfg.groove is on, adds the beat groove layer. music is the beats json
    (MUSIC_KEYS); it supplies the beats when beats is None and turns on the
    phrase gate and the per-song groove pattern. With return_info the result
    is (commands, info), where info records the pattern and the phrases.
    """
    poses = np.asarray(poses, dtype=float)
    trans = np.asarray(trans, dtype=float)
    F = len(poses)
    if poses.shape != (F, 72) or trans.shape != (F, 3) or F < 2:
        raise ValueError(f"expected poses (F, 72) and trans (F, 3) with F >= 2, got {poses.shape}, {trans.shape}")
    if len(cfg.signs) != CMD_DIM or len(cfg.max_rate) != CMD_DIM:
        raise ValueError(f"signs and max_rate must have {CMD_DIM} entries")
    poses = poses.reshape(F, 24, 3)
    dt_in = 1.0 / fps

    if joints is not None:
        joints = _check_joints(joints, F)
    up = detect_up_axis(trans, joints)
    Wr = R.from_matrix(WORLD_TO_ROBOT[up])
    Pr = R.from_matrix(P)

    # Pelvis heading from the forward vector W R0 e_z.
    pelvis = Wr * R.from_rotvec(poses[:, 0]) * Pr.inv()
    psi = _heading(pelvis.apply([1.0, 0.0, 0.0]))
    psi_slow = _lowpass_zero_phase(psi, fps, cfg.heading_cutoff_hz)
    wz = cfg.yaw_rate_gain * np.gradient(psi_slow, dt_in)

    # Horizontal pelvis velocity in the heading frame.
    if joints is not None:
        pelvis_data = joints[:, 0]
        floor = float(np.percentile(joints[..., up].min(axis=1), FLOOR_PERCENTILE))
    else:
        pelvis_data = trans + SMPL_PELVIS_OFFSET
        floor = 0.0
    pos = pelvis_data @ WORLD_TO_ROBOT[up].T  # pelvis in robot world
    vel = np.gradient(pos, dt_in, axis=0)
    if cfg.foot_velocity and joints is not None:
        vel = vel - support_foot_velocity(joints, up, dt_in)
    vel_world = cfg.vel_gain * vel[:, :2]  # for the heading servo
    c, s = np.cos(psi_slow), np.sin(psi_slow)
    vx = cfg.vel_gain * (c * vel[:, 0] + s * vel[:, 1])
    vy = cfg.vel_gain * (-s * vel[:, 0] + c * vel[:, 1])

    # Body roll and pitch in the heading frame: trunk line with joints, chest rotation without.
    to_heading = R.from_euler("z", -psi[:, None])
    if joints is not None:
        trunk = (joints[:, NECK] - joints[:, 0]) @ WORLD_TO_ROBOT[up].T
        body_roll, body_pitch = _trunk_roll_pitch(to_heading.apply(trunk))
    else:
        chest = Wr * _chain(poses, SPINE_CHAIN) * Pr.inv()
        chest_e = _euler_zyx(to_heading * chest)
        body_pitch, body_roll = chest_e[:, 1], chest_e[:, 2]

    # Height delta, scaled from human pelvis height to the robot.
    height = pos[:, 2] - floor
    med = float(np.median(height))
    gain = cfg.height_gain if cfg.height_gain is not None else cfg.nominal_height / med
    body_z = gain * (height - med)

    # Head relative to chest (neck * head), expressed in robot axes.
    head = Pr * _chain(poses, NECK_HEAD_CHAIN) * Pr.inv()
    head_e = _euler_zyx(head)
    head_yaw, head_pitch_total, head_roll = head_e[:, 0], head_e[:, 1], head_e[:, 2]
    neck_pitch = cfg.neck_pitch_share * head_pitch_total
    head_pitch = (1.0 - cfg.neck_pitch_share) * head_pitch_total

    zeros = np.zeros(F)
    raw = np.stack(
        [vx, vy, wz, neck_pitch, head_pitch, head_yaw, head_roll,
         zeros, zeros, body_z, body_roll, body_pitch, zeros],
        axis=1,
    )
    raw = raw * np.asarray(cfg.signs, dtype=float)
    head_gain = np.broadcast_to(np.asarray(cfg.head_gain, dtype=float).reshape(-1), (4,))
    if np.any(head_gain != 1.0):
        raw[:, HEAD_COLS] *= head_gain

    # Resample to the control rate.
    T = int(round(F * dt_in / cfg.out_dt))
    t_in = np.arange(F) * dt_in
    t_out = np.arange(T) * cfg.out_dt
    out = np.stack([np.interp(t_out, t_in, raw[:, k]) for k in range(CMD_DIM)], axis=1)

    max_rate = np.asarray(cfg.max_rate, dtype=float)
    out = _lowpass(out, cfg.out_dt, cfg.cutoff_hz)
    out = _rate_limit(out, cfg.out_dt, max_rate)
    # Posture lead (module docstring). Only the human part; the groove is added below.
    n_lead = int(round(cfg.posture_lead_s / cfg.out_dt))
    out[:, POSTURE_COLS] = _shift_earlier(out[:, POSTURE_COLS], n_lead)

    info: dict = {"foot_velocity": bool(cfg.foot_velocity and joints is not None),
                  "groove": None, "pattern": None, "phrases": None}
    if music is not None:
        check_music(music)
        if beats is None:
            beats = music["beat_times_s"]
        elif len(beats) != len(music["beat_times_s"]):
            raise ValueError("beats and music['beat_times_s'] differ in length")
    groove_gain = np.ones(T)
    gate = np.ones(T)
    if music is not None and cfg.phrase_gate:
        plan = plan_phrases(beats, music["beat_rms"], music["beat_onset"], min_cv=cfg.phrase_min_cv)
        gate = twist_gate(t_out, np.asarray(plan["start_times_s"][1:]), plan["travel"], cfg.phrase_fade_s)
        groove_gain = 1.0 + (cfg.groove_phrase_gain - 1.0) * (1.0 - gate)
        plan["travel_time_frac"] = float(gate.mean())
        info["phrases"] = plan
    ramp = np.ones(T)
    if cfg.ramp_in_s > 0:
        ramp = ramp * np.clip(t_out / cfg.ramp_in_s, 0.0, 1.0)
    if cfg.ramp_out_s > 0 and T > 0:
        # The clip ends at rest, and the filtfilt edge transient in wz is faded out.
        ramp = ramp * np.clip((t_out[-1] - t_out) / cfg.ramp_out_s, 0.0, 1.0)
    if cfg.heading_servo:
        # Target heading relative to the first frame, and the human pelvis velocity
        # relative to the initial heading, both at the control rate.
        psi_h = cfg.yaw_rate_gain * np.interp(t_out, t_in, psi_slow - psi_slow[0])
        c0, s0 = math.cos(psi_slow[0]), math.sin(psi_slow[0])
        vw = vel_world @ np.array([[c0, -s0], [s0, c0]])  # rows rotated by -psi_slow[0]
        vw = np.stack([np.interp(t_out, t_in, vw[:, k]) for k in range(2)], axis=1)
        vw = _lowpass(vw, cfg.out_dt, cfg.cutoff_hz)
        # Feed-forward is the open-loop wz (the human yaw rate), resampled like the other columns.
        ff = np.interp(t_out, t_in, wz)
        wz_s, psi_c = heading_servo(psi_h, cfg.out_dt, cfg.heading_servo_gain, ALL_RANGES[WZ],
                                    float(max_rate[WZ]), scale=ramp * gate, ff=ff)
        cc, sc = np.cos(psi_c), np.sin(psi_c)
        vb = np.stack([cc * vw[:, 0] + sc * vw[:, 1], -sc * vw[:, 0] + cc * vw[:, 1]], axis=1)
        out[:, 0:2] = _rate_limit(vb, cfg.out_dt, max_rate[0:2]) * gate[:, None]
        out[:, WZ] = wz_s
        info["heading_servo"] = {"final_target_rad": float(psi_h[-1]) if T else 0.0,
                                 "final_command_rad": float(psi_c[-1] + wz_s[-1] * cfg.out_dt) if T else 0.0}
    else:
        out[:, 0:3] *= gate[:, None]
    if beats is not None and cfg.groove:
        b = check_beats(beats)
        if music is not None and cfg.groove_pattern:
            pattern = choose_pattern(music, cfg.groove_nod, cfg.groove_sway)
            out = out + cfg.groove_scale * pattern_layer(t_out, b, pattern, cfg.groove_width, cfg.groove_lag_s,
                                                         groove_gain)
            info["groove"], info["pattern"] = "pattern", pattern
        else:
            layer = groove_layer(t_out, b, cfg.groove_nod, cfg.groove_bounce,
                                 cfg.groove_sway, cfg.groove_width, cfg.groove_lag_s)
            out = out + cfg.groove_scale * layer * groove_gain[:, None]
            info["groove"] = "uniform"
    lo = np.array([r[0] for r in ALL_RANGES])
    hi = np.array([r[1] for r in ALL_RANGES])
    out = np.clip(out, lo, hi)

    # With the servo, wz already carries the ramps (inside its loop), so psi_c stays its integral.
    cols = [k for k in range(CMD_DIM) if not (cfg.heading_servo and k == WZ)]
    out[:, cols] *= ramp[:, None]
    return (out, info) if return_info else out
