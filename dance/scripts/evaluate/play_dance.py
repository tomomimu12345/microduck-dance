"""Play a dance command file with a trained policy: record an mp4 or measure tracking.

Record (one env, play cfg with randomization removed, 1280x720 at 30 fps, camera tracks trunk_base):
    uv run dance/scripts/evaluate/play_dance.py --checkpoint <model.pt> \
        --commands dance/data/commands/songs/<song>.npz --out out/<song>.mp4

Evaluate (training cfg with its randomization, num_envs envs per seed):
    uv run dance/scripts/evaluate/play_dance.py --checkpoint <model.pt> \
        --commands dance/data/commands/songs/<song>.npz --eval --seeds 0 1 2 --num-envs 16 \
        --json out/<song>_eval.json

State dump for motion statistics (record setup, no rendering without --out):
    uv run dance/scripts/evaluate/play_dance.py --checkpoint <model.pt> \
        --commands dance/data/commands/songs/<song>.npz --dump-states out/<song>_states.npz

The command terms play the file from frame 0 (source="file"). The episode is
set longer than the clip, so only falls end it. A fallen env resets and
resumes the clip at the global step (ClipPlayer resume_on_reset), fading its
command in over --reset-ramp-s, so the video stays in sync with the music and
eval scores every second of the song once per env. The recorded command is
read from the command manager each step, so it includes that fade.

Ball tasks (--task Mjlab-Basketball-MicroDuck or Mjlab-BallDance-MicroDuck)
put the robot upright on the apex of a free ball, as in the basketball task:
    uv run dance/scripts/evaluate/play_dance.py --task Mjlab-Basketball-MicroDuck \
        --checkpoint logs/rsl_rl/basketball/b11_6999/checkpoint.pt \
        --commands dance/data/commands/songs/<song>.npz --zero-pose-cmds --twist-scale 0.5 --out out/<song>_ball.mp4
The ball is free (hold 0) in record and eval mode. A fall is the task's own
termination or is_ball_fall (trunk below the ball top, or trunk touching the
ball). The actor is recurrent, so its hidden state is carried between steps
and zeroed for an env when it resets. --twist-scale multiplies the twist
columns of the file, and on a ball task the result is clipped to
BALL_TWIST_RANGES. --zero-pose-cmds zeroes the head and body columns of the
file and the actor obs slots 51:61 (b11 was trained with zeros there, so it is
required for Mjlab-Basketball-MicroDuck).

Lagged camera (--cam-follow lagged): the camera looks at a first-order
low-pass of the trunk position (time constant --cam-tau, 1.5 s) instead of the
trunk itself, with a fixed azimuth, so travel shows as the robot moving across
the frame and drifting back to the centre when it stops. The horizontal gap
between the trunk and the camera target is capped at CAM_MAX_OFFSET so the
robot stays inside the frame. The default distance in this mode is
LAGGED_CAM_DISTANCE. The default (--cam-follow trunk) keeps the tracking camera.

Rough terrain (--terrain rough): the floor is the Velocity rough terrain
(MICRODUCK_ROUGH_TERRAINS_CFG, 5 x 5 random tiles, generator seed = run seed).
Each tile spawns the robot on a flat centre platform (1.5 to 2 m wide), and a
dancer that travels about a body height per minute would never leave it. So by
default (--rough-spawn off-platform) every env origin is moved to a point on its
tile at Chebyshev distance OFF_PLATFORM_D from the tile centre, outside every
platform, at the ground height found by a ray cast, and the spawn xy noise is
set to 0. The terrain_levels curriculum is removed, so a fall reset respawns at
the same point. --tile-type puts env 0 on the hardest tile of that type (record mode). Eval
JSON gets a "terrain" block: tile type and difficulty per env, and the fraction
of steps with an ankle outside the tile's flat platform. On rough ground the
measured body height is taken from the ground under the feet (foot_height_scan),
so a step up or down does not read as a squat.

Video terrain (--terrain video): the rough physics settings with the rough
dance training geometry at full difficulty (video_terrain_cfg). Pyramid stairs
of 1.5 cm steps with 0.15 m treads and a 0.5 m platform, and a random grid of
0.45 m cells up to 1 cm high, coloured in two alternating tones
(color_video_terrain). --spawn-u places env 0
at a chosen point of the off-platform ring. The training terrain is not changed.

Test-only switches for the rough A/B: --ground box replaces the plane by one
large box with the plane's contact parameters, --solver-iters sets the solver
iterations, --no-dr removes the randomization in eval mode, and --dump-eval
writes the per-env eval rollout (cmd, meas, term, pos, yaw) to an npz.

Rendering needs MUJOCO_GL=egl (or osmesa) in the environment.

compute_stats is a pure function and imports nothing heavy, so the tests run
without a GPU or an env.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from microduck_dance.ranges import BALL_TWIST_RANGES
from microduck_dance.ranges import DANCE_STAND_HEIGHT

TASK_ID = "Mjlab-Dance-Flat-MicroDuck"
BASKETBALL_TASK = "Mjlab-Basketball-MicroDuck"
BALL_DANCE_TASK = "Mjlab-BallDance-MicroDuck"
BALL_TASKS = (BASKETBALL_TASK, BALL_DANCE_TASK)
TRUNK_BALL_SENSOR = "trunk_ball_contact"
# Actor obs columns of the head and body command slots (command block 48:61, its slots 3:13).
POSE_OBS = slice(51, 61)
ACTOR_OBS_DIM = 61
# Camera distance from trunk_base (m) when --cam-distance is not given. On the
# ball the robot stands 0.24 m higher, so the camera backs off to keep the ball in frame.
FLOOR_CAM_DISTANCE = 0.55
BALL_CAM_DISTANCE = 0.9
# Lagged camera: low-pass time constant (s), cap on the trunk to target gap (m,
# horizontal) and the floor distance (m). Closer than the tracking camera because
# the robot can be up to about 0.15 m farther away than the target while it walks
# away from the camera, and it should still fill 40 % of the frame height.
CAM_TAU_S = 1.5
CAM_MAX_OFFSET = 0.25
LAGGED_CAM_DISTANCE = 0.5
FEET_BODIES = ["ankle_left", "ankle_right"]
HEAD_JOINT_PATTERNS = [r".*neck_pitch.*", r".*head_pitch.*", r".*head_yaw.*", r".*head_roll.*"]
# Rough terrain: spawn distance from the tile centre (Chebyshev, m). The widest
# flat platform of MICRODUCK_ROUGH_TERRAINS_CFG is 2.0 m (half width 1.0).
OFF_PLATFORM_D = 1.6
ROUGH_TERRAINS = ("rough", "video")
# Video terrain (task 30): the training geometry of the rough dance tiles at full difficulty, so the
# video shows the floor the policy was trained and measured on. The task 28 video terrain had 0.25 m
# treads and 0.30 m cells, and dance_v4 fell 0.69 to 1.38 times per minute there.
VIDEO_STEP_HEIGHT = 0.015  # the training maximum (step_height_range (0, 0.015))
VIDEO_STEP_WIDTH = 0.15
VIDEO_PLATFORM_WIDTH = 0.5  # DANCE_ROUGH_PLATFORM_WIDTH
VIDEO_GRID_WIDTH = 0.45  # 17 cells = 7.65 m, does not divide the 8 m tile
VIDEO_GRID_HEIGHT = 0.010
VIDEO_CREAM = (0.92, 0.88, 0.78)
VIDEO_SLATE = (0.25, 0.32, 0.42)
VIDEO_PLATFORM_GREY = (0.6, 0.6, 0.6)
VIDEO_TOP_BRIGHTEN = 0.15
# A/B ground box (half sizes, m). Its top is at z = 0, where the plane is. About a terrain tile:
# with 50 m half sides the box against foot mesh contact failed (dance_v4 fell 15 times in 20 s
# per env, also with nconmax 200), with 4.5 m it held. 16 envs at 2 m spacing sit within 3.5 m.
GROUND_BOX_HALF = (4.5, 4.5, 0.05)
VIDEO_FPS = 30
VIDEO_SIZE = (1280, 720)  # width, height

# Columns of the 13D command layout scored by each RMSE. Body x, y and yaw are
# always 0 in both command and measurement, so only z, roll and pitch count.
TWIST_COLS = [0, 1, 2]
HEAD_COLS = [3, 4, 5, 6]
BODY_COLS = [9, 10, 11]
# Twist correlation is taken on signals smoothed by a moving average of this length (s).
TWIST_CORR_SMOOTH_S = 0.5


@dataclass
class EpisodeStats:
    falls: int
    duration_s: float
    twist_rmse: float
    head_rmse: float
    body_rmse: float
    twist_corr_vx: float = float("nan")
    twist_corr_vy: float = float("nan")
    twist_corr_wz: float = float("nan")
    twist_corr: float = float("nan")


def smoothed_corr(cmd: np.ndarray, meas: np.ndarray, window: int) -> float:
    """Pearson correlation of two 1D signals after a centred moving average of window samples.

    Returns nan if either smoothed signal is constant or there are fewer than
    window samples.
    """
    cmd, meas = np.asarray(cmd, float), np.asarray(meas, float)
    if len(cmd) < max(window, 2):
        return float("nan")
    k = np.ones(window) / window
    a, b = np.convolve(cmd, k, mode="valid"), np.convolve(meas, k, mode="valid")
    a, b = a - a.mean(), b - b.mean()
    den = np.sqrt(np.sum(a**2) * np.sum(b**2))
    return float(np.sum(a * b) / den) if den > 1e-12 else float("nan")


def compute_stats(cmd: np.ndarray, measured: np.ndarray, terminated: np.ndarray, dt: float) -> EpisodeStats:
    """Tracking stats for one env.

    cmd and measured are (T, 13) in the dance command layout. terminated is
    (T,) bool, True on steps that ended in a non-timeout termination. A fall is
    a rising edge of terminated. RMSEs pool every scored column over the steps
    where terminated is False (the reset step's state is not tracking data).
    twist_corr_* is the correlation of command and measured twist per channel
    after a TWIST_CORR_SMOOTH_S moving average, over the steps up to the first
    fall (the whole run if none). twist_corr is their mean, ignoring nan.
    """
    cmd = np.asarray(cmd, dtype=float)
    measured = np.asarray(measured, dtype=float)
    terminated = np.asarray(terminated, dtype=bool)
    T = len(cmd)
    if cmd.shape != measured.shape or cmd.ndim != 2 or cmd.shape[1] != 13 or terminated.shape != (T,):
        raise ValueError(f"shape mismatch: cmd {cmd.shape}, measured {measured.shape}, terminated {terminated.shape}")
    prev = np.concatenate([[False], terminated[:-1]])
    falls = int(np.sum(terminated & ~prev))
    keep = ~terminated
    err = measured - cmd

    def rmse(cols: list[int]) -> float:
        e = err[keep][:, cols]
        return float(np.sqrt(np.mean(e**2))) if e.size else float("nan")

    first = int(np.argmax(terminated)) if terminated.any() else T
    window = max(1, int(round(TWIST_CORR_SMOOTH_S / dt)))
    corr = [smoothed_corr(cmd[:first, c], measured[:first, c], window) for c in TWIST_COLS]
    finite = [c for c in corr if np.isfinite(c)]
    return EpisodeStats(
        falls=falls,
        duration_s=T * dt,
        twist_rmse=rmse(TWIST_COLS),
        head_rmse=rmse(HEAD_COLS),
        body_rmse=rmse(BODY_COLS),
        twist_corr_vx=corr[0],
        twist_corr_vy=corr[1],
        twist_corr_wz=corr[2],
        twist_corr=float(np.mean(finite)) if finite else float("nan"),
    )


def fall_steps(terminated: np.ndarray) -> list[int]:
    """Control steps at which a fall starts (rising edges of terminated)."""
    terminated = np.asarray(terminated, dtype=bool)
    prev = np.concatenate([[False], terminated[:-1]])
    return [int(k) for k in np.flatnonzero(terminated & ~prev)]


def is_ball_fall(trunk_z, ball_top_z, trunk_ball_contact):
    """True when the robot is off the ball. The trunk is below the ball top, or the trunk touches the ball.

    Works on floats and bools (returns a bool) and elementwise on arrays or tensors.
    """
    out = (trunk_z < ball_top_z) | trunk_ball_contact
    return bool(out) if isinstance(out, (bool, np.bool_)) else out


def lowpass_follow(target, pos, dt: float, tau: float = CAM_TAU_S, max_offset: float | None = CAM_MAX_OFFSET):
    """One step of the lagged camera. Returns the new camera target (3,).

    target moves toward pos by the exact discretization of a first-order
    low-pass, a = 1 - exp(-dt / tau). If max_offset is given, the target is then
    pulled along the horizontal line to pos until the xy gap is at most max_offset.
    """
    target = np.asarray(target, dtype=float)
    pos = np.asarray(pos, dtype=float)
    a = 1.0 - math.exp(-dt / tau)
    out = target + a * (pos - target)
    if max_offset is not None:
        gap = pos[:2] - out[:2]
        n = float(np.hypot(gap[0], gap[1]))
        if n > max_offset:
            out[:2] = pos[:2] - gap * (max_offset / n)
    return out


def lagged_follow(pos, dt: float, tau: float = CAM_TAU_S, max_offset: float | None = CAM_MAX_OFFSET) -> np.ndarray:
    """Camera targets (T, 3) for a trunk path pos (T, 3). The first target is pos[0]."""
    pos = np.asarray(pos, dtype=float)
    out = np.empty_like(pos)
    out[0] = pos[0]
    for k in range(1, len(pos)):
        out[k] = lowpass_follow(out[k - 1], pos[k], dt, tau, max_offset)
    return out


class LaggedCamera:
    """Points a free MjvCamera at the lagged trunk position. Azimuth, elevation and distance stay as set."""

    def __init__(self, cam, dt: float, tau: float = CAM_TAU_S, max_offset: float | None = CAM_MAX_OFFSET):
        import mujoco

        self.cam = cam
        self.dt, self.tau, self.max_offset = dt, tau, max_offset
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE.value
        cam.trackbodyid = -1
        cam.fixedcamid = -1
        self.target = None

    def update(self, pos) -> None:
        """Called once per control step with the trunk position in world coordinates."""
        pos = np.asarray(pos, dtype=float)
        if self.target is None:
            self.target = pos.copy()
        else:
            self.target = lowpass_follow(self.target, pos, self.dt, self.tau, self.max_offset)
        self.cam.lookat[:] = self.target


def prepare_commands(commands: np.ndarray, twist_scale: float = 1.0, zero_pose: bool = False,
                     twist_limits=None, body_scale: float = 1.0) -> np.ndarray:
    """The command rows that are played. Returns a new (T, 13) float32 array.

    The twist columns are multiplied by twist_scale and, if twist_limits
    ((lo, hi) per channel) is given, clipped to them. The body columns are
    multiplied by body_scale. zero_pose sets the head and body columns to 0.
    """
    out = np.array(commands, dtype=np.float32, copy=True)
    out[:, 0:3] *= twist_scale
    if twist_limits is not None:
        lo = np.array([r[0] for r in twist_limits], dtype=np.float32)
        hi = np.array([r[1] for r in twist_limits], dtype=np.float32)
        out[:, 0:3] = np.clip(out[:, 0:3], lo, hi)
    out[:, 7:13] *= body_scale
    if zero_pose:
        out[:, 3:13] = 0.0
    return out


def zero_pose_obs(obs):
    """A shallow copy of obs whose actor head and body slots (51:61) are exactly zero. obs is not written."""
    actor = obs["actor"]
    if actor.shape[-1] != ACTOR_OBS_DIM:
        raise ValueError(f"actor obs has {actor.shape[-1]} columns, expected {ACTOR_OBS_DIM}")
    actor = actor.clone()
    actor[..., POSE_OBS] = 0.0
    out = obs.copy()
    out["actor"] = actor
    return out


def policy_step(policy, obs, venv, zero_pose: bool = False):
    """One control step. Returns the next obs, dones and extras.

    A recurrent policy keeps its hidden state between calls. After the step,
    policy.reset(dones) zeroes the state of the envs that just reset, so they
    start the next step from zero like at activation. The reset runs in
    inference mode because the state tensors were made there.
    """
    import torch

    if zero_pose:
        obs = zero_pose_obs(obs)
    with torch.inference_mode():
        actions = policy(obs)
    obs, _, dones, extras = venv.step(actions)
    if hasattr(policy, "reset"):
        with torch.inference_mode():
            policy.reset(dones)
    return obs, dones, extras


def platform_half_width(sub_cfg) -> float:
    """Half side of the flat centre platform of a sub-terrain cfg, 0 for a flat tile (no rough part)."""
    w = getattr(sub_cfg, "platform_width", None)
    return 0.0 if w is None else float(w) / 2.0


def ring_point(center, d: float, u: float) -> np.ndarray:
    """Point at Chebyshev distance d from center (xy), at fraction u in [0, 1) of the square's perimeter."""
    c = np.asarray(center, dtype=float)
    t = (float(u) % 1.0) * 8.0 * d  # arc length along the perimeter, starting at (+d, -d)
    side, a = int(t // (2.0 * d)), t % (2.0 * d) - d
    dx, dy = [(d, a), (-a, d), (-d, -a), (a, -d)][side]
    return np.array([c[0] + dx, c[1] + dy])


def off_platform_fraction(feet_xy, center_xy, half_width: float) -> float:
    """Fraction of steps with at least one ankle outside the square platform. feet_xy (T, 2, 2) in world xy."""
    if half_width <= 0.0:
        return float("nan")
    rel = np.abs(np.asarray(feet_xy, dtype=float) - np.asarray(center_xy, dtype=float)[None, None, :])
    out = rel.max(axis=-1) > half_width  # (T, 2)
    return float(np.mean(out.any(axis=-1)))


# Env side. Everything below imports torch and mjlab lazily.

COMMAND_TERMS = ("twist", "head_pose", "body_pose")

# Events that randomize the robot or push it. Removed for the recording
# (brief: "no randomization"). foot_friction is removed so the model's nominal
# foot friction (1.0, set in the robot cfg) stays. Required events such as
# expand_bam_friction_fields and reset_action_history, and the spawn
# (reset_base, reset_robot_joints), are kept.
DR_EVENTS = (
    "push_robot",
    "foot_friction",
    "encoder_bias",
    "base_com",
    "randomize_com",
    "randomize_head_com",
    "randomize_motor_gains",
    "randomize_mass_inertia",
    "randomize_joint_friction",
    "randomize_joint_damping",
    "randomize_armature",
    "randomize_base_orientation",
)


def strip_randomization(cfg) -> None:
    """Make a play cfg nominal: no DR events, no obs noise, no IMU misalignment."""
    removed = [name for name in DR_EVENTS if cfg.events.pop(name, None) is not None]
    # Curricula that widen a removed event (com_range, head_com_range) would fail to find it.
    orphans = [n for n, c in cfg.curriculum.items() if (c.params or {}).get("event_name") in removed]
    for n in orphans:
        cfg.curriculum.pop(n)
    print(f"[play_dance] removed curricula of removed events: {orphans}")
    for group in cfg.observations.values():
        group.enable_corruption = False
    imu = []
    for gname, group in cfg.observations.items():
        for tname, term in group.terms.items():
            if term is not None and "max_angle_deg" in (term.params or {}):
                term.params["max_angle_deg"] = 0.0
                imu.append(f"{gname}.{tname}")
    print(f"[play_dance] removed DR events: {removed}")
    print(f"[play_dance] kept events: {list(cfg.events)}")
    print(f"[play_dance] obs corruption off for {list(cfg.observations)}; IMU misalignment 0 deg on {imu}")


def fix_spawn_pose(cfg) -> None:
    """Record mode: spawn at the origin with yaw 0 (facing +x, toward the camera), also after a fall reset."""
    if "reset_basketball" in cfg.events:
        # Ball tasks: ball at the env origin, robot on its apex. Keep the small tilt and joint noise.
        params = cfg.events["reset_basketball"].params
        params["xy_noise"] = 0.0
        params["yaw_range"] = (0.0, 0.0)
        return
    pose_range = cfg.events["reset_base"].params["pose_range"]
    for k in ("x", "y", "yaw"):
        pose_range[k] = (0.0, 0.0)


SKY_TOP = (0.20, 0.42, 0.70)
SKY_HORIZON = (0.78, 0.88, 0.96)


def add_sky(spec) -> None:
    """Add a vertical-gradient skybox (blue overhead, pale at the horizon) so the video has no black band."""
    import mujoco

    tex = spec.add_texture(name="dance_sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                           builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT, rgb1=list(SKY_TOP),
                           rgb2=list(SKY_HORIZON), width=512, height=3072)
    del tex
    spec.visual.rgba.haze = [*SKY_HORIZON, 1.0]


def is_ball_task(task: str) -> bool:
    return task in BALL_TASKS


def ball_fall_termination(env, ball_radius: float, sensor_name: str = TRUNK_BALL_SENSOR):
    """Termination term for the ball tasks. is_ball_fall on the trunk height, ball top and trunk / ball contact."""
    import torch

    trunk_z = torch.nan_to_num(env.scene["robot"].data.root_link_pos_w[:, 2], nan=0.0)
    ball_top = torch.nan_to_num(env.scene["ball"].data.root_link_pos_w[:, 2], nan=0.0) + ball_radius
    found = env.scene.sensors[sensor_name].data.found
    contact = (found.reshape(env.num_envs, -1) > 0).any(dim=-1)
    return is_ball_fall(trunk_z, ball_top, contact)


def played_commands(args) -> tuple[np.ndarray, float, dict, bool]:
    """The command rows this run plays, their dt and meta, and whether they differ from the file."""
    from microduck_dance.command_file import load_command_file

    commands, dt, meta = load_command_file(args.commands)
    ball = is_ball_task(args.task)
    body_scale = 1.0
    score_pose = getattr(args, "score_pose_cmds", False)
    if args.task == BALL_DANCE_TASK or score_pose:
        from microduck_dance.tasks import microduck_ball_dance_env_cfg as ball_dance

        body_scale = float(ball_dance.BALL_BODY_SCALE)
    changed = ball or args.twist_scale != 1.0 or args.zero_pose_cmds or body_scale != 1.0
    if not changed:
        return commands, dt, meta, False
    # With score_pose the head and body columns stay in the file (so they are scored), and only the
    # actor obs slots are zeroed (policy_step).
    out = prepare_commands(commands, twist_scale=args.twist_scale, zero_pose=args.zero_pose_cmds and not score_pose,
                           twist_limits=BALL_TWIST_RANGES if ball else None, body_scale=body_scale)
    return out, dt, meta, True


def _ball_base_cfg(task: str, play: bool):
    if task == BASKETBALL_TASK:
        from mjlab_microduck.tasks.microduck_basketball_env_cfg import make_microduck_basketball_env_cfg

        # The released b11 contract. Blind actor, one frame of proprioception.
        return make_microduck_basketball_env_cfg(play=play, blind=True, history=1)
    from microduck_dance.tasks.microduck_ball_dance_env_cfg import make_microduck_ball_dance_env_cfg

    return make_microduck_ball_dance_env_cfg(play=play)


def _make_ball_cfg(task: str, play: bool, command_file: str):
    """Ball task cfg that plays command_file on the global clock, with a free ball and the ball fall check."""
    from mjlab.managers import TerminationTermCfg
    from mjlab.sensor import ContactMatch, ContactSensorCfg

    from mjlab_microduck.tasks.microduck_basketball_env_cfg import BALL_RADIUS
    from microduck_dance.tasks.microduck_dance_env_cfg import make_microduck_dance_env_cfg

    cfg = _ball_base_cfg(task, play)
    # The file command terms, built the same way as on the floor dance task.
    dance = make_microduck_dance_env_cfg(play=play, command_file=command_file, source="file")
    for name in COMMAND_TERMS:
        cfg.commands[name] = dance.commands[name]
    # standing_envs zeroes twist commands of some envs. The file says when to stand.
    cfg.curriculum.pop("standing_envs", None)
    # Free ball in eval too. The training cfg starts pinned (hold 1) and lowers it by curriculum.
    cfg.actions["ball_hold"].levels = (0.0,)
    cfg.curriculum.pop("basketball_hold", None)
    trunk_ball = ContactSensorCfg(
        name=TRUNK_BALL_SENSOR,
        primary=ContactMatch(mode="body", pattern="trunk_base", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="ball", entity="ball"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )
    cfg.scene.sensors = (*cfg.scene.sensors, trunk_ball)
    cfg.terminations["ball_fall"] = TerminationTermCfg(
        func=ball_fall_termination, params={"ball_radius": BALL_RADIUS, "sensor_name": TRUNK_BALL_SENSOR}
    )
    return cfg


def build_env_cfg(args, *, play: bool, num_envs: int, seed: int, steps: int, dt: float, render: bool,
                  command_file: Path):
    """The env cfg of this run. If the played rows differ from args.commands they are written to command_file."""
    import mjlab.tasks  # noqa: F401  (populates the registry)
    import microduck_dance.tasks  # noqa: F401
    from microduck_dance.command_file import save_command_file
    from microduck_dance.tasks.microduck_dance_env_cfg import make_microduck_dance_env_cfg

    ball = is_ball_task(args.task)
    if args.task == BASKETBALL_TASK and not args.zero_pose_cmds:
        raise SystemExit(f"{BASKETBALL_TASK} has no head or body command inputs (b11 was trained with zeros). "
                         "Pass --zero-pose-cmds")
    if getattr(args, "score_pose_cmds", False) and not args.zero_pose_cmds:
        raise SystemExit("--score-pose-cmds only applies with --zero-pose-cmds")
    commands, file_dt, meta, changed = played_commands(args)
    if changed:
        save_command_file(command_file, commands, file_dt,
                          {**meta, "play_dance": {"source": str(args.commands), "twist_scale": args.twist_scale,
                                                  "zero_pose_cmds": args.zero_pose_cmds, "task": args.task}})
        path = str(command_file)
    else:
        path = str(args.commands)

    if ball:
        cfg = _make_ball_cfg(args.task, play, path)
    else:
        terrain = getattr(args, "terrain", "flat")
        rough = terrain in ROUGH_TERRAINS
        cfg = make_microduck_dance_env_cfg(play=play, command_file=path, source="file", rough=rough)
        if getattr(args, "ground", "plane") == "box":
            if rough:
                raise SystemExit("--ground box is for the flat terrain")
            add_spec_fn(cfg, replace_plane_by_box)
        if terrain == "video":
            cfg.scene.terrain.terrain_generator = video_terrain_cfg()
        if rough:
            gen = cfg.scene.terrain.terrain_generator
            gen.curriculum, gen.num_rows, gen.num_cols, gen.seed = False, 5, 5, seed
            # A fall reset must respawn on the same tile, at the same point.
            cfg.curriculum.pop("terrain_levels", None)
            if args.rough_spawn == "off-platform":
                pose_range = cfg.events["reset_base"].params["pose_range"]
                pose_range["x"] = pose_range["y"] = (0.0, 0.0)
    iters = getattr(args, "solver_iters", None)
    if iters is not None:
        cfg.sim.mujoco.iterations, cfg.sim.mujoco.ls_iterations = int(iters[0]), int(iters[1])
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    # Longer than the clip, so a time-out never cuts the dance.
    cfg.episode_length_s = steps * dt + 2.0
    # Keep the clip on the global clock: a reset env resumes at the current step
    # (in sync with the music) and fades its command in, instead of restarting.
    for name in COMMAND_TERMS:
        cfg.commands[name].resume_on_reset = True
        cfg.commands[name].reset_ramp_s = args.reset_ramp_s
    if (play and not args.keep_dr) or (not play and getattr(args, "no_dr", False)):
        strip_randomization(cfg)
    if render or getattr(args, "dump_states", None) is not None:
        # Same start as the video (origin, yaw 0) for the state dump too.
        fix_spawn_pose(cfg)
    if render:
        cfg.viewer.body_name = "trunk_base"
        cfg.viewer.width, cfg.viewer.height = VIDEO_SIZE
        cam_distance = args.cam_distance
        if cam_distance is None:
            if ball:
                cam_distance = BALL_CAM_DISTANCE
            else:
                cam_distance = LAGGED_CAM_DISTANCE if args.cam_follow == "lagged" else FLOOR_CAM_DISTANCE
        cfg.viewer.distance = cam_distance
        cfg.viewer.elevation = args.cam_elevation
        cfg.viewer.azimuth = args.cam_azimuth
        cfg.viewer.max_extra_envs = 0
        # With MUJOCO_GL=egl on WSL2 the shadow map smears long black bars over the floor.
        cfg.viewer.enable_shadows = args.shadows
        # The default scene has no skybox, so everything above the horizon renders black.
        add_spec_fn(cfg, add_sky)
    return cfg


def add_spec_fn(cfg, fn) -> None:
    """Chain fn after the cfg's existing spec_fn."""
    prev_spec_fn = cfg.scene.spec_fn

    def spec_fn(spec):
        if prev_spec_fn is not None:
            prev_spec_fn(spec)
        fn(spec)

    cfg.scene.spec_fn = spec_fn


def replace_plane_by_box(spec) -> None:
    """A/B test: the plane stops colliding and a large box with the plane's contact parameters takes over.

    The box is a geom of the terrain body (so the foot contact sensor and the ray sensors see it), group 0,
    transparent, with its top at z = 0.
    """
    import mujoco

    body = spec.body("terrain")
    planes = [g for g in body.geoms if g.type == mujoco.mjtGeom.mjGEOM_PLANE]
    if len(planes) != 1:
        raise RuntimeError(f"expected one ground plane, found {len(planes)}")
    plane = planes[0]
    plane.contype, plane.conaffinity = 0, 0
    hx, hy, hz = GROUND_BOX_HALF
    body.add_geom(name="ab_ground_box", type=mujoco.mjtGeom.mjGEOM_BOX, size=[hx, hy, hz], pos=[0.0, 0.0, -hz],
                  friction=list(plane.friction), solref=list(plane.solref), solimp=list(plane.solimp),
                  priority=plane.priority, condim=plane.condim, contype=1, conaffinity=1, group=plane.group,
                  rgba=[0.0, 0.0, 0.0, 0.0])
    print(f"[play_dance] ground plane replaced by a box {GROUND_BOX_HALF} (friction {list(plane.friction)}, "
          f"solref {list(plane.solref)}, priority {plane.priority}, condim {plane.condim})")


def video_terrain_cfg():
    """The terrain for the video. Stairs and random grid only, larger treads and cells, same height caps.

    A new cfg, MICRODUCK_ROUGH_TERRAINS_CFG (training) is not touched.
    """
    from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
    import mjlab.terrains as terrain_gen

    return TerrainGeneratorCfg(
        size=(8.0, 8.0),
        border_width=20.0,
        num_rows=5,
        num_cols=5,
        curriculum=False,
        sub_terrains={
            "pyramid_stairs": terrain_gen.BoxPyramidStairsTerrainCfg(
                proportion=0.5,
                step_height_range=(VIDEO_STEP_HEIGHT, VIDEO_STEP_HEIGHT),
                step_width=VIDEO_STEP_WIDTH,
                platform_width=VIDEO_PLATFORM_WIDTH,
                border_width=1.0,
            ),
            "random_grid": terrain_gen.BoxRandomGridTerrainCfg(
                proportion=0.5,
                grid_width=VIDEO_GRID_WIDTH,
                grid_height_range=(0.0, VIDEO_GRID_HEIGHT),
                platform_width=VIDEO_PLATFORM_WIDTH,
                merge_similar_heights=False,
            ),
        },
        add_lights=False,
    )


def video_box_color(kind: str, z_top: float, cell: tuple[int, int] = (0, 0), frac: float = 0.0):
    """Render colour of one video terrain box.

    kind "step": cream and slate alternating by step index round(z_top / VIDEO_STEP_HEIGHT).
    kind "cell": checkerboard by cell parity, brightened by up to VIDEO_TOP_BRIGHTEN for the highest
    cells (frac in [0, 1] is the cell height within the tile's range). kind "platform": grey.
    """
    if kind == "platform":
        return VIDEO_PLATFORM_GREY
    if kind == "step":
        k = int(round(z_top / VIDEO_STEP_HEIGHT)) % 2
        return VIDEO_CREAM if k == 0 else VIDEO_SLATE
    if kind == "cell":
        base = VIDEO_CREAM if (cell[0] + cell[1]) % 2 == 0 else VIDEO_SLATE
        s = 1.0 + VIDEO_TOP_BRIGHTEN * min(max(frac, 0.0), 1.0)
        return tuple(min(1.0, c * s) for c in base)
    raise ValueError(kind)


def color_video_terrain(model, tiles: dict, terrain_origins) -> int:
    """Recolour the video terrain (rendering only). Returns the number of recoloured boxes.

    Each box is assigned to the tile whose centre is nearest. The box centred on the tile centre is the
    platform (grey). On stairs tiles every other box is coloured by its step index, on grid tiles by
    cell parity, with the grid's border boxes treated as steps.
    """
    import mujoco

    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    body = model.body("terrain").id
    centers = {rc: np.asarray(terrain_origins[rc[0], rc[1]].cpu().numpy(), float) for rc in tiles}
    boxes = [g for g in range(model.ngeom)
             if model.geom_bodyid[g] == body and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX]
    by_tile: dict = {}
    for g in boxes:
        xy = data.geom_xpos[g, :2]
        rc = min(centers, key=lambda k: float(np.max(np.abs(centers[k][:2] - xy))))
        by_tile.setdefault(rc, []).append(g)
    n = 0
    for rc, gs in by_tile.items():
        name = tiles[rc][0]
        c = centers[rc]
        tops = {g: float(data.geom_xpos[g, 2] + model.geom_size[g, 2]) for g in gs}
        cells = [g for g in gs if name == "random_grid"
                 and abs(model.geom_size[g, 0] - VIDEO_GRID_WIDTH / 2) < 1e-4
                 and abs(model.geom_size[g, 1] - VIDEO_GRID_WIDTH / 2) < 1e-4]
        if cells:
            lo, hi = min(tops[g] for g in cells), max(tops[g] for g in cells)
        for g in gs:
            xy = data.geom_xpos[g, :2]
            if np.max(np.abs(xy - c[:2])) < 1e-3:
                rgb = video_box_color("platform", tops[g])
            elif g in cells:
                ij = tuple(int(v) for v in np.floor((xy - c[:2]) / VIDEO_GRID_WIDTH))
                frac = 0.0 if hi - lo < 1e-9 else (tops[g] - lo) / (hi - lo)
                rgb = video_box_color("cell", tops[g], ij, frac)
            else:
                rgb = video_box_color("step", tops[g])
            model.geom_rgba[g, :3] = rgb
            model.geom_rgba[g, 3] = 1.0
            n += 1
    print(f"[play_dance] video terrain: recoloured {n} boxes (steps cream/slate, grid checkerboard, platform grey)")
    return n


def agent_cfg_for(args, checkpoint: dict | None = None) -> dict:
    """Runner cfg dict for args.task. On a ball task the actor architecture is read from the checkpoint weights."""
    from mjlab.tasks.registry import load_rl_cfg

    import mjlab.tasks  # noqa: F401
    import microduck_dance.tasks  # noqa: F401

    agent = asdict(load_rl_cfg(args.task))
    if is_ball_task(args.task):
        from mjlab_microduck.basketball_distillation import actor_model_cfg

        if checkpoint is None:
            import torch

            checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        agent["actor"] = actor_model_cfg(checkpoint)
    return agent


def _build(args, *, play: bool, num_envs: int, seed: int, steps: int, dt: float, render: bool):
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_runner_cls
    from rsl_rl.runners import OnPolicyRunner

    tmp = Path(tempfile.mkdtemp(prefix="play_dance_"))
    try:
        cfg = build_env_cfg(args, play=play, num_envs=num_envs, seed=seed, steps=steps, dt=dt, render=render,
                            command_file=tmp / "commands.npz")
        device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        # The command terms read the file here, so it can go afterwards.
        with _record_tiles() as tiles:
            env = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode="rgb_array" if render else None)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    terrain = getattr(args, "terrain", "flat")
    if terrain in ROUGH_TERRAINS:
        env._dance_terrain = place_on_terrain(env, tiles, seed=seed, spawn=args.rough_spawn,
                                              tile_type=args.tile_type, spawn_u=getattr(args, "spawn_u", None))
        if render:
            if terrain == "video":
                color_video_terrain(env.sim.mj_model, tiles, env.scene.terrain.terrain_origins)
            else:
                color_terrain_by_height(env.sim.mj_model)
    if abs(env.step_dt - dt) > 1e-9:
        raise ValueError(f"command file dt={dt} differs from env step_dt={env.step_dt}")

    agent = agent_cfg_for(args)
    venv = RslRlVecEnvWrapper(env, clip_actions=agent["clip_actions"])
    runner_cls = load_runner_cls(args.task) or OnPolicyRunner
    runner = runner_cls(venv, agent, device=device)
    runner.load(str(args.checkpoint), map_location=device)
    policy = runner.get_inference_policy(device=device)
    actor_dim = venv.get_observations()["actor"].shape[-1]
    if not is_ball_task(args.task) and actor_dim != ACTOR_OBS_DIM:
        raise RuntimeError(f"actor obs is {actor_dim}D, expected {ACTOR_OBS_DIM}")
    return env, venv, policy


class _record_tiles:
    """Context manager: records (row, col) -> (sub-terrain name, difficulty) while the terrain generator runs."""

    def __enter__(self):
        from mjlab.terrains.terrain_generator import TerrainGenerator

        self._cls, self._orig, self.tiles = TerrainGenerator, TerrainGenerator._create_terrain_geom, {}
        tiles, orig = self.tiles, self._orig

        def wrapped(gen, spec, world_position, difficulty, cfg, sub_row, sub_col):
            name = next(k for k, v in gen.cfg.sub_terrains.items() if v is cfg)
            tiles[(int(sub_row), int(sub_col))] = (name, float(difficulty), cfg)
            return orig(gen, spec, world_position, difficulty, cfg, sub_row, sub_col)

        TerrainGenerator._create_terrain_geom = wrapped
        return self.tiles

    def __exit__(self, *exc):
        self._cls._create_terrain_geom = self._orig
        return False


# 7 colours of 2.5 mm bands repeat every 17.5 mm, so two surfaces 2.5 to 15 mm apart (any step of the
# rough terrain) never share a colour.
TERRAIN_BAND_M = 0.0025
TERRAIN_PALETTE = ((0.85, 0.85, 0.80), (0.35, 0.55, 0.85), (0.90, 0.65, 0.30), (0.45, 0.70, 0.45),
                   (0.75, 0.45, 0.60), (0.55, 0.55, 0.55), (0.95, 0.85, 0.45))


def terrain_band_color(z_top: float, band: float = TERRAIN_BAND_M) -> tuple[float, float, float]:
    """Render colour of a terrain box whose top is at z_top: one palette entry per band of height."""
    return TERRAIN_PALETTE[int(math.floor(z_top / band + 0.5)) % len(TERRAIN_PALETTE)]


def color_terrain_by_height(model) -> int:
    """Recolour the terrain boxes in height bands so 1 cm steps and grid cells show on video.

    Rendering only (geom_rgba). The generator's own height colours differ too little at these heights.
    Returns the number of recoloured geoms.
    """
    import mujoco

    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    body = model.body("terrain").id
    n = 0
    for g in range(model.ngeom):
        if model.geom_bodyid[g] != body or model.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX:
            continue
        top = float(data.geom_xpos[g, 2] + model.geom_size[g, 2])
        model.geom_rgba[g, :3] = terrain_band_color(top)
        n += 1
    print(f"[play_dance] recoloured {n} terrain boxes in {TERRAIN_BAND_M * 1000:.1f} mm height bands")
    return n


def _ground_z(model, data, x: float, y: float, z_top: float, terrain_body: int) -> float:
    import mujoco

    geomid = np.array([-1], dtype=np.int32)
    dist = mujoco.mj_ray(model, data, np.array([x, y, z_top]), np.array([0.0, 0.0, -1.0]), None, 1, -1, geomid)
    if dist < 0 or model.geom_bodyid[geomid[0]] != terrain_body:
        raise RuntimeError(f"ray at ({x:.3f}, {y:.3f}) hit no terrain geom (geom {geomid[0]}, dist {dist})")
    return z_top - dist


def place_on_terrain(env, tiles: dict, seed: int, spawn: str = "off-platform", tile_type: str | None = None,
                     spawn_u: float | None = None) -> dict:
    """Rough terrain: optionally move each env origin off its tile's platform, reset, and return the tile info.

    tile_type moves env 0 to the tile of that type with the highest difficulty. spawn_u fixes env 0's
    point on the off-platform ring (fraction of the perimeter, see ring_point; 0.125 is +x of the centre). The returned dict has, per env,
    the tile (row, col), type, difficulty, tile centre (the generator's origin), platform half width
    and the origin used.
    """
    import mujoco
    import torch

    terrain = env.scene.terrain
    if terrain.terrain_origins is None or not tiles:
        raise RuntimeError("rough terrain expected a generated terrain")
    rows, cols = terrain.terrain_levels.clone(), terrain.terrain_types.clone()
    if tile_type is not None:
        # The hardest tile of that type (highest difficulty), ties in row-major order.
        hit = sorted((rc for rc, t in tiles.items() if t[0] == tile_type), key=lambda rc: (-tiles[rc][1], rc))
        if not hit:
            raise SystemExit(f"no {tile_type} tile in this terrain (seed {seed}): {sorted(set(t[0] for t in tiles.values()))}")
        rows[0], cols[0] = hit[0]
        terrain.terrain_levels[0], terrain.terrain_types[0] = int(hit[0][0]), int(hit[0][1])
    model = env.sim.mj_model
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    terrain_body = model.body("terrain").id
    rng = np.random.default_rng(seed)
    envs = []
    origins = terrain.env_origins
    for i in range(env.num_envs):
        r, c = int(rows[i]), int(cols[i])
        name, difficulty, sub_cfg = tiles[(r, c)]
        center = terrain.terrain_origins[r, c].cpu().numpy().astype(float)
        u = float(rng.uniform())
        if i == 0 and spawn_u is not None:
            u = float(spawn_u)
        if spawn == "off-platform":
            xy = ring_point(center[:2], OFF_PLATFORM_D, u)
            z = _ground_z(model, data, xy[0], xy[1], center[2] + 1.0, terrain_body)
            origin = np.array([xy[0], xy[1], z])
        else:
            origin = center
        origins[i] = torch.as_tensor(origin, dtype=origins.dtype, device=origins.device)
        envs.append({"env": i, "tile": [r, c], "type": name, "difficulty": round(difficulty, 3),
                     "center": [round(float(v), 4) for v in center], "origin": [round(float(v), 4) for v in origin],
                     "platform_half": platform_half_width(sub_cfg)})
    env.reset()
    print(f"[play_dance] rough terrain, spawn {spawn}: " + ", ".join(
        f"env{e['env']} {e['type']} d={e['difficulty']:.2f} dz={e['origin'][2] - e['center'][2]:+.3f}" for e in envs))
    return {"spawn": spawn, "off_platform_d": OFF_PLATFORM_D if spawn == "off-platform" else 0.0, "envs": envs}


def terrain_report(info: dict, feet_world_xy: np.ndarray, falls: list[int]) -> dict:
    """Per-env terrain block for the eval JSON. feet_world_xy (T, N, 2, 2)."""
    envs = []
    for e in info["envs"]:
        i = e["env"]
        frac = off_platform_fraction(feet_world_xy[:, i], e["center"][:2], e["platform_half"])
        envs.append({**e, "off_platform_frac": None if math.isnan(frac) else round(frac, 4), "falls": int(falls[i])})
    mix: dict[str, int] = {}
    for e in envs:
        mix[e["type"]] = mix.get(e["type"], 0) + 1
    by_type = {t: float(np.mean([e["falls"] for e in envs if e["type"] == t])) for t in mix}
    fr = [e["off_platform_frac"] for e in envs if e["off_platform_frac"] is not None]
    return {"spawn": info["spawn"], "off_platform_d": info["off_platform_d"], "mix": mix,
            "falls_per_env_by_type": by_type, "mean_off_platform_frac_rough_tiles": float(np.mean(fr)) if fr else None,
            "envs": envs}


class _Probe:
    """Reads the 13D command and the matching 13D measurement from the env."""

    def __init__(self, env):
        self.env = env
        self.robot = env.scene["robot"]
        ids, names = self.robot.find_joints(HEAD_JOINT_PATTERNS, preserve_order=True)
        if len(ids) != 4:
            raise RuntimeError(f"expected 4 head joints, found {names}")
        self.head_ids = ids
        self.head_names = names
        # Ball tasks measure body z against standing on the apex (root this high above the ball centre).
        self.ball = env.scene["ball"] if "ball" in env.scene.entities else None
        # Rough ground: body height above the ground under the feet, not above the spawn point.
        self.ground_from_feet = getattr(env, "_dance_terrain", None) is not None
        if self.ball is not None:
            from mjlab_microduck.tasks.microduck_basketball_env_cfg import BALL_RADIUS, ROOT_HEIGHT

            self.ball_stand = BALL_RADIUS + ROOT_HEIGHT

    def command(self):
        import torch

        cm = self.env.command_manager
        return torch.cat(
            [cm.get_command("twist")[:, :3], cm.get_command("head_pose"), cm.get_command("body_pose")], dim=1
        )

    def measured(self):
        import torch

        d = self.robot.data
        q = d.root_link_quat_w
        qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        # ZYX Euler, the convention of dance_body_tracking.
        yaw = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        roll = torch.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
        pitch = torch.asin(torch.clamp(2.0 * (qw * qy - qz * qx), -1.0, 1.0))
        v = d.root_link_lin_vel_w
        c, s = torch.cos(yaw), torch.sin(yaw)
        vx = c * v[:, 0] + s * v[:, 1]
        vy = -s * v[:, 0] + c * v[:, 1]
        wz = d.root_link_ang_vel_w[:, 2]
        head = d.joint_pos[:, self.head_ids] - d.default_joint_pos[:, self.head_ids]
        if self.ball is None:
            if self.ground_from_feet:
                from microduck_dance.tasks.dance_v3 import ground_z_under_feet

                ground = ground_z_under_feet(self.env)
            else:
                ground = self.env.scene.terrain.env_origins[:, 2]
            z = d.root_link_pos_w[:, 2] - ground - DANCE_STAND_HEIGHT
        else:
            z = d.root_link_pos_w[:, 2] - self.ball.data.root_link_pos_w[:, 2] - self.ball_stand
        zeros = torch.zeros_like(z)
        return torch.stack(
            [vx, vy, wz, head[:, 0], head[:, 1], head[:, 2], head[:, 3], zeros, zeros, z, roll, pitch, zeros],
            dim=1,
        )


def _clip_index(env):
    """(N,) frame index the next control step plays, checked equal across the three terms."""
    idx = [env.command_manager.get_term(name).source.index() for name in COMMAND_TERMS]
    if not all(bool((i == idx[0]).all()) for i in idx[1:]):
        raise RuntimeError(f"command terms out of step: {[i.tolist() for i in idx]}")
    return idx[0]


def _rollout(env, venv, policy, steps: int, on_state=None, pose: dict | None = None, zero_pose: bool = False):
    """Run steps control steps.

    Returns cmd, measured (steps, N, 13), terminated (steps, N) and clip_idx
    (steps, N), the clip frame each env plays at the next step. On the global
    clock clip_idx[k] is min(k + 1, T - 1) for every env, fallen or not.
    If pose is a dict, it receives "pos" (steps, N, 3), the trunk position
    relative to the env origin, "yaw" (steps, N) and "feet" (steps, N, 2, 3),
    the left and right ankle positions relative to the env origin, all after each step.
    zero_pose zeroes the actor head and body obs slots before every policy call.
    """
    import torch

    probe = _Probe(env)
    robot = env.scene["robot"]
    obs = venv.get_observations()
    cmds, meas, terms, idxs, poss, yaws, feet = [], [], [], [], [], [], []
    feet_ids = robot.find_bodies(FEET_BODIES, preserve_order=True)[0] if pose is not None else None
    for k in range(steps):
        if on_state is not None:
            on_state(k)
        cmds.append(probe.command().cpu().numpy().copy())  # the command the policy acts on now
        obs, dones, extras = policy_step(policy, obs, venv, zero_pose=zero_pose)
        time_outs = extras.get("time_outs", torch.zeros_like(dones, dtype=torch.bool))
        terms.append((dones.bool() & ~time_outs.bool()).cpu().numpy())
        meas.append(probe.measured().cpu().numpy().copy())
        idxs.append(_clip_index(env).cpu().numpy().copy())
        if pose is not None:
            d = robot.data
            org = env.scene.terrain.env_origins
            poss.append((d.root_link_pos_w - org).cpu().numpy().copy())
            feet.append((d.body_link_pos_w[:, feet_ids] - org[:, None]).cpu().numpy().copy())
            q = d.root_link_quat_w
            yaws.append(torch.atan2(2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                                    1.0 - 2.0 * (q[:, 2] ** 2 + q[:, 3] ** 2)).cpu().numpy().copy())
    if pose is not None:
        pose["pos"] = np.stack(poss)
        pose["yaw"] = np.stack(yaws)
        pose["feet"] = np.stack(feet)
    return np.stack(cmds), np.stack(meas), np.stack(terms), np.stack(idxs)


def _check_clock(clip_idx: np.ndarray, T: int) -> int:
    """Number of (step, env) entries off the global clock."""
    expected = np.minimum(np.arange(1, len(clip_idx) + 1), T - 1)[:, None]
    return int(np.sum(clip_idx != expected))


def _report_falls(term: np.ndarray, clip_idx: np.ndarray, dt: float, env_id: int = 0) -> list[dict]:
    """Fall times of one env, with the clip index played right after each reset."""
    out = []
    for k in fall_steps(term[:, env_id]):
        out.append({"step": k, "t_s": round((k + 1) * dt, 3), "clip_index_after_reset": int(clip_idx[k, env_id]),
                    "global_step_after": k + 1})
    return out


def save_states(path, cmd, meas, term, pos, yaw, dt: float, feet=None, **meta) -> Path:
    """Write the per-step states of one env (cmd, meas (T, 13), term (T,), pos (T, 3), yaw (T,)) to an npz.

    The command is the one the policy acted on at step k, and meas, term, pos
    and yaw are read after that step. feet (T, 2, 3), the ankle positions, is
    stored if given. meta values are stored as strings.
    """
    cmd, meas = np.asarray(cmd, np.float32), np.asarray(meas, np.float32)
    term, pos, yaw = np.asarray(term, bool), np.asarray(pos, np.float32), np.asarray(yaw, np.float32)
    T = len(cmd)
    if cmd.shape != (T, 13) or meas.shape != (T, 13) or term.shape != (T,) or pos.shape != (T, 3) \
            or yaw.shape != (T,):
        raise ValueError(f"bad shapes: cmd {cmd.shape} meas {meas.shape} term {term.shape} pos {pos.shape} "
                         f"yaw {yaw.shape}")
    arrays = {}
    if feet is not None:
        feet = np.asarray(feet, np.float32)
        if feet.shape != (T, 2, 3):
            raise ValueError(f"bad shape: feet {feet.shape}")
        arrays["feet"] = feet
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, cmd=cmd, meas=meas, term=term, pos=pos, yaw=yaw, dt=np.float64(dt), **arrays,
             **{k: np.array(str(v)) for k, v in meta.items()})
    return path


def _steps_for(args, T: int, dt: float) -> int:
    if args.max_seconds is None:
        return T
    return min(T, int(round(args.max_seconds / dt)))


def record(args) -> dict:
    import mediapy as media

    commands, dt, _, _ = played_commands(args)
    steps = _steps_for(args, len(commands), dt)
    render = args.out is not None
    env, venv, policy = _build(args, play=True, num_envs=1, seed=args.seeds[0], steps=steps, dt=dt, render=render)
    q = getattr(env, "_imu_misalign_quat", None)
    if q is not None:
        print(f"[play_dance] IMU misalignment quat (w, x, y, z): {q[0].tolist()}")

    duration = steps * dt
    n_frames = int(math.ceil(duration * VIDEO_FPS - 1e-9)) if render else 0
    pose = {} if args.dump_states is not None else None
    written = 0
    lagged = None
    if render and args.cam_follow == "lagged":
        lagged = LaggedCamera(env._offline_renderer._cam, dt, tau=args.cam_tau)
    robot = env.scene["robot"]
    if render:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with media.VideoWriter(out, shape=(VIDEO_SIZE[1], VIDEO_SIZE[0]), fps=VIDEO_FPS) as writer:

            cam = env._offline_renderer._cam
            close_view = (float(cam.distance), float(cam.elevation))
            wide_s = float(getattr(args, "wide_intro_s", 0.0) or 0.0)

            def on_state(k: int) -> None:
                # Video frame j shows the state at time j / VIDEO_FPS; state k is at k * dt.
                nonlocal written
                if lagged is not None and k < steps:
                    lagged.update(robot.data.root_link_pos_w[0].cpu().numpy())
                while written < n_frames and written / VIDEO_FPS <= k * dt + 1e-9:
                    if wide_s > 0.0:
                        wide = written / VIDEO_FPS < wide_s - 1e-9
                        cam.distance, cam.elevation = (args.wide_distance, args.wide_elevation) if wide else close_view
                    writer.add_image(env.render())
                    written += 1

            cmd, meas, term, clip_idx = _rollout(env, venv, policy, steps, on_state=on_state, pose=pose,
                                                 zero_pose=args.zero_pose_cmds)
            on_state(steps)  # flush: frames whose time falls in the last control step
    else:
        cmd, meas, term, clip_idx = _rollout(env, venv, policy, steps, pose=pose, zero_pose=args.zero_pose_cmds)
    env.close()
    if written != n_frames:
        raise RuntimeError(f"wrote {written} frames, expected {n_frames}")
    if pose is not None:
        save_states(args.dump_states, cmd[:, 0], meas[:, 0], term[:, 0], pose["pos"][:, 0], pose["yaw"][:, 0], dt,
                    feet=pose["feet"][:, 0], checkpoint=args.checkpoint, cam_follow=args.cam_follow, commands=args.commands, task=args.task,
                    terrain=args.terrain, tile=None if getattr(env, "_dance_terrain", None) is None else
                    json.dumps(env._dance_terrain["envs"][0]),
                    twist_scale=args.twist_scale, zero_pose_cmds=args.zero_pose_cmds)
        print(f"wrote {args.dump_states}")
    stats = compute_stats(cmd[:, 0], meas[:, 0], term[:, 0], dt)
    falls = _report_falls(term, clip_idx, dt)
    for f in falls:
        # Recorded command over the file frame for the first steps after the reset: the fade-in.
        ks = range(f["step"] + 1, min(f["step"] + 4, steps))
        f["cmd_scale_after_reset"] = [
            round(float(np.linalg.norm(cmd[k, 0]) / max(np.linalg.norm(commands[k]), 1e-9)), 4) for k in ks
        ]
    off_clock = _check_clock(clip_idx, len(commands))
    info = getattr(env, "_dance_terrain", None)
    result = {"mode": "record", "task": args.task, "terrain": args.terrain,
              "tile": None if info is None else info["envs"][0], "twist_scale": args.twist_scale,
              "zero_pose_cmds": args.zero_pose_cmds,
              "score_pose_cmds": getattr(args, "score_pose_cmds", False), "out": str(args.out), "frames": written, "fps": VIDEO_FPS,
              "video_s": written / VIDEO_FPS, "stats": asdict(stats), "falls": falls,
              "steps_off_clock": off_clock}
    print(json.dumps(result, indent=2))
    if off_clock:
        raise RuntimeError(f"{off_clock} steps off the global clock")
    if falls:
        times = ", ".join(f"{f['t_s']:.2f}" for f in falls)
        print(f"WARNING: the robot fell {len(falls)} times in the recording, at t = {times} s. "
              "Each fall resets the robot mid-song.", file=sys.stderr)
        if args.fail_on_fall:
            sys.exit(3)
    return result


def evaluate(args) -> dict:
    commands, dt, _, _ = played_commands(args)
    steps = _steps_for(args, len(commands), dt)
    per_seed = []
    for seed in args.seeds:
        env, venv, policy = _build(args, play=False, num_envs=args.num_envs, seed=seed, steps=steps, dt=dt,
                                   render=False)
        info = getattr(env, "_dance_terrain", None)
        dump = getattr(args, "dump_eval", None)
        pose = {} if (info is not None or dump is not None) else None
        cmd, meas, term, clip_idx = _rollout(env, venv, policy, steps, zero_pose=args.zero_pose_cmds, pose=pose)
        if dump is not None:
            path = eval_dump_path(dump, seed, len(args.seeds))
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, cmd=cmd.astype(np.float32), meas=meas.astype(np.float32), term=term.astype(bool),
                     pos=pose["pos"].astype(np.float32), yaw=pose["yaw"].astype(np.float32), dt=np.float64(dt),
                     seed=np.int64(seed), checkpoint=np.array(str(args.checkpoint)),
                     commands=np.array(str(args.commands)), terrain=np.array(str(args.terrain)))
            print(f"wrote {path}")
        if info is not None:
            org = env.scene.terrain.env_origins.cpu().numpy()
            feet_xy = pose["feet"][..., :2] + org[None, :, None, :2]
        env.close()
        off_clock = _check_clock(clip_idx, len(commands))
        if off_clock:
            raise RuntimeError(f"seed {seed}: {off_clock} (step, env) entries off the global clock")
        envs = [asdict(compute_stats(cmd[:, i], meas[:, i], term[:, i], dt)) for i in range(cmd.shape[1])]
        first_falls = [(_report_falls(term, clip_idx, dt, i) or [None])[0] for i in range(cmd.shape[1])]
        entry = {"seed": seed, "envs": envs, "mean": _mean(envs), "first_fall": first_falls}
        if info is not None:
            entry["terrain"] = terrain_report(info, feet_xy, [e["falls"] for e in envs])
            t = entry["terrain"]
            print(f"[play_dance] seed {seed} terrain mix {t['mix']}, falls per env by type "
                  f"{ {k: round(v, 3) for k, v in t['falls_per_env_by_type'].items()} }, mean off-platform "
                  f"fraction on rough tiles {t['mean_off_platform_frac_rough_tiles']}")
        per_seed.append(entry)
    all_envs = [e for s in per_seed for e in s["envs"]]
    result = {
        "mode": "eval",
        "task": args.task,
        "terrain": args.terrain,
        "no_dr": bool(getattr(args, "no_dr", False)),
        "ground": getattr(args, "ground", "plane"),
        "solver_iters": getattr(args, "solver_iters", None),
        "twist_scale": args.twist_scale,
        "zero_pose_cmds": args.zero_pose_cmds,
        "score_pose_cmds": getattr(args, "score_pose_cmds", False),
        "checkpoint": str(args.checkpoint),
        "commands": str(args.commands),
        "steps": steps,
        "dt": dt,
        "num_envs": args.num_envs,
        "seeds": list(args.seeds),
        "summary": {**_mean(all_envs), "falls_total": int(sum(e["falls"] for e in all_envs)),
                    "envs_with_fall": int(sum(e["falls"] > 0 for e in all_envs)), "n_envs": len(all_envs)},
        "per_seed": per_seed,
    }
    print(json.dumps(result["summary"], indent=2))
    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n")
        print(f"wrote {path}")
    return result


def eval_dump_path(dump, seed: int, n_seeds: int) -> Path:
    """The --dump-eval file of one seed. With several seeds the seed goes into the name."""
    dump = Path(dump)
    if n_seeds == 1:
        return dump
    return dump.with_name(f"{dump.stem}_seed{seed}{dump.suffix}")


def _mean(envs: list[dict]) -> dict:
    keys = ("falls", "duration_s", "twist_rmse", "head_rmse", "body_rmse")
    out = {k: float(np.mean([e[k] for e in envs])) for k in keys}
    for k in ("twist_corr_vx", "twist_corr_vy", "twist_corr_wz", "twist_corr"):
        v = [e[k] for e in envs if k in e and np.isfinite(e[k])]
        out[k] = float(np.mean(v)) if v else float("nan")
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--commands", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None, help="mp4 path (record mode)")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--num-envs", type=int, default=16, help="envs per seed (eval mode)")
    ap.add_argument("--json", type=Path, default=None, help="eval result path")
    ap.add_argument("--max-seconds", type=float, default=None, help="play only the first part of the clip")
    ap.add_argument("--task", default=TASK_ID,
                    help=f"registry id for the RL cfg and runner. {BASKETBALL_TASK} and {BALL_DANCE_TASK} "
                         "also switch the scene to the ball")
    ap.add_argument("--zero-pose-cmds", action="store_true",
                    help="zero the head and body columns of the file and the actor obs slots 51:61. "
                         f"Required for {BASKETBALL_TASK}")
    ap.add_argument("--score-pose-cmds", action="store_true",
                    help="with --zero-pose-cmds, keep the head columns and the body columns times "
                         "BALL_BODY_SCALE in the file, so head and body RMSE are measured against the ball "
                         "dance commands. The actor obs slots 51:61 stay zero")
    ap.add_argument("--twist-scale", type=float, default=1.0,
                    help="multiply the twist columns of the file. On a ball task the result is clipped to "
                         "BALL_TWIST_RANGES")
    ap.add_argument("--device", default=None)
    ap.add_argument("--cam-distance", type=float, default=None,
                    help=f"m from trunk_base. Default {FLOOR_CAM_DISTANCE} on the floor (the robot is about half "
                         f"the 720 px frame height) and {BALL_CAM_DISTANCE} on the ball")
    ap.add_argument("--cam-elevation", type=float, default=-15.0)
    ap.add_argument("--cam-azimuth", type=float, default=150.0,
                    help="degrees. 90 is a side view with the robot facing right, 180 looks at it head on")
    ap.add_argument("--cam-follow", choices=("trunk", "lagged"), default="trunk",
                    help="trunk: the camera tracks trunk_base. lagged: it looks at a low-pass of the trunk "
                         f"position (time constant --cam-tau), so travel shows; default distance {LAGGED_CAM_DISTANCE}")
    ap.add_argument("--cam-tau", type=float, default=CAM_TAU_S, help="lagged camera time constant (s)")
    ap.add_argument("--shadows", action="store_true", help="cast shadows (bar artifacts under EGL on WSL2)")
    ap.add_argument("--reset-ramp-s", type=float, default=0.5,
                    help="after a fall, fade the resumed command in from 0 over this time")
    ap.add_argument("--keep-dr", action="store_true", help="record mode: keep the play cfg's randomization")
    ap.add_argument("--fail-on-fall", action="store_true", help="record mode: exit 3 if the robot fell")
    ap.add_argument("--terrain", choices=("flat", "rough", "video"), default="flat",
                    help="rough: the Velocity rough terrain (5 x 5 random tiles, generator seed = run seed). "
                         "video: the rough physics with the video terrain (video_terrain_cfg)")
    ap.add_argument("--rough-spawn", choices=("off-platform", "origin"), default="off-platform",
                    help=f"rough: spawn at Chebyshev distance {OFF_PLATFORM_D} m from the tile centre, off the flat "
                         "platform (default), or at the tile origin on the platform")
    ap.add_argument("--tile-type", default=None,
                    help="rough: put env 0 on the hardest tile of this type (pyramid_stairs, random_grid, "
                         "pyramid_slope, flat)")
    ap.add_argument("--spawn-u", type=float, default=None,
                    help="rough or video: env 0 at this fraction of the off-platform ring (0.125 is +x of the centre)")
    ap.add_argument("--ground", choices=("plane", "box"), default="plane",
                    help="flat only, A/B test: box replaces the plane by one large box with the same contact params")
    ap.add_argument("--solver-iters", type=int, nargs=2, default=None, metavar=("ITER", "LS_ITER"),
                    help="A/B test: solver iterations and line search iterations")
    ap.add_argument("--no-dr", action="store_true", help="eval mode: remove the randomization (as in record mode)")
    ap.add_argument("--dump-eval", type=Path, default=None,
                    help="eval mode: write per-env cmd, meas, term, pos, yaw (T, N, ...) to this npz "
                         "(with several seeds, one file per seed)")
    ap.add_argument("--wide-intro-s", type=float, default=0.0,
                    help="record mode: render the first seconds with the wide camera below")
    ap.add_argument("--wide-distance", type=float, default=1.5)
    ap.add_argument("--wide-elevation", type=float, default=-40.0)
    ap.add_argument("--dump-states", type=Path, default=None,
                    help="record mode: write per-step cmd, meas, term, trunk pos and yaw to this npz. "
                         "Without --out, nothing is rendered")
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    if args.eval:
        evaluate(args)
    else:
        if args.out is None and args.dump_states is None:
            raise SystemExit("record mode needs --out or --dump-states")
        record(args)


if __name__ == "__main__":
    main()
