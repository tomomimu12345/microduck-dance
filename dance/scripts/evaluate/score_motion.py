"""Robot-visible energy score for dance pkls (full_pose, smpl_poses, z-up, 30 fps).

Usage (repo root, EDGE venv, it has librosa):
    python dance/scripts/evaluate/score_motion.py OUT.csv PKL:WAV [PKL:WAV ...]

Ported from out/gen_candidates/score_motion.py (source-dance report, 2026-10-20)
with the foot slide metric and the composite energy added. Components, at
human scale and only what the retarget passes to the robot:
  v_rms     RMS horizontal pelvis speed, pelvis velocity low-passed at 2 Hz (m/s)
  turn      sum |d heading| per minute, heading low-passed at 0.5 Hz as in retarget (deg/min)
  z_std     std of pelvis height above the per-frame lowest joint (cm)
  tilt      RMS deviation of trunk roll and pitch about their means (deg)
  head      RMS deviation of neck*head yaw/pitch/roll about their means (deg)
  energy    geometric mean of the five above, each divided by its median over the
            38 FlowerDance clips scored on 2026-10-20 (REF_MEDIANS), 1.0 = typical
  bas       beat alignment: mean over music beats of exp(-d^2 / 2 s^2), s = 0.1 s,
            d = distance to the nearest local minimum of pelvis vertical velocity
  bas_lift  bas minus its value with the beat grid shifted by random offsets (chance level)
  drift     p98 - p2 of the per-frame lowest joint z (m); > 0.10 means the body floats
  foot_slide mean horizontal speed of the support foot (the lower foot while within
            3 cm of the lowest joint), m/s, 0 for a planted foot (motion_fix.foot_slide)
  arm       share of joint kinetic energy (relative to pelvis) in elbows, wrists, hands
  cmd       mean over the 9 active command dims of RMS/half-range after the real retarget
"""

import csv
import pickle
import sys
from pathlib import Path

import librosa
import numpy as np
from scipy.signal import argrelmin, butter, filtfilt
from scipy.spatial.transform import Rotation as R

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
from microduck_dance.motion_fix import floor_drift, foot_slide  # noqa: E402
from microduck_dance.ranges import ALL_RANGES  # noqa: E402
from microduck_dance.retarget import (  # noqa: E402
    NECK_HEAD_CHAIN, P, RetargetConfig, _chain, _heading, _lowpass_zero_phase, smpl_to_commands,
)

FPS = 30.0
AMP = ["v_rms", "turn", "z_std", "tilt", "head"]
# Medians of out/gen_candidates/existing_scores.csv (38 FlowerDance clips: the 5
# songs and 33 library clips as of 2026-10-20, before the Breaking switch).
REF_MEDIANS = {"v_rms": 0.11027231936215035, "turn": 418.2369249081544, "z_std": 1.3397971550010261,
               "tilt": 2.787602968005264, "head": 9.269794367770015}


def lp(x, hz):
    b, a = butter(2, hz / (FPS / 2))
    return filtfilt(b, a, x, axis=0)


def energy(row: dict) -> float:
    return float(np.exp(np.mean([np.log(row[k] / REF_MEDIANS[k]) for k in AMP])))


def beats_of(wav, dur):
    y, sr = librosa.load(wav, sr=22050, mono=True, duration=dur)
    tempo, bt = librosa.beat.beat_track(y=y, sr=sr, hop_length=512)
    return float(np.atleast_1d(tempo)[0]), librosa.frames_to_time(bt, sr=sr, hop_length=512)


def bas(music_beats, kin_beats, sigma=0.1):
    if len(kin_beats) == 0 or len(music_beats) == 0:
        return 0.0
    d = np.abs(music_beats[:, None] - kin_beats[None, :]).min(axis=1)
    return float(np.mean(np.exp(-d ** 2 / (2 * sigma ** 2))))


def score(pkl, wav, rng):
    with open(pkl, "rb") as fh:
        d = pickle.load(fh)
    J = np.asarray(d["full_pose"], float)
    poses = np.asarray(d["smpl_poses"], float).reshape(len(J), 72)
    trans = np.asarray(d["smpl_trans"], float)
    F = len(J)
    dur = F / FPS
    pel = J[:, 0]
    vel = lp(np.gradient(pel, 1 / FPS, axis=0), 2.0)
    v_rms = float(np.sqrt(np.mean(vel[:, 0] ** 2 + vel[:, 1] ** 2)))

    rot = R.from_rotvec(poses[:, :3]) * R.from_matrix(P).inv()
    psi = _heading(rot.apply([1.0, 0, 0]))
    psi_slow = _lowpass_zero_phase(psi, FPS, 0.5)
    turn = float(np.degrees(np.abs(np.diff(psi_slow)).sum()) / (dur / 60))

    low = J[..., 2].min(axis=1)
    z_std = float(100 * (pel[:, 2] - low).std())

    trunk = J[:, 12] - J[:, 0]
    c, s = np.cos(-psi), np.sin(-psi)
    tx = c * trunk[:, 0] - s * trunk[:, 1]
    ty = s * trunk[:, 0] + c * trunk[:, 1]
    pitch = np.arctan2(tx, trunk[:, 2])
    roll = np.arctan2(-ty, np.hypot(tx, trunk[:, 2]))
    tilt = float(np.degrees(np.sqrt(roll.var() + pitch.var())))

    head_rot = R.from_matrix(P) * _chain(poses.reshape(F, 24, 3), NECK_HEAD_CHAIN) * R.from_matrix(P).inv()
    he = np.unwrap(head_rot.as_euler("ZYX"), axis=0)
    head = float(np.degrees(np.sqrt(he.var(axis=0).sum())))

    tempo, mb = beats_of(wav, dur)
    mb = mb[(mb > 1.0) & (mb < dur - 1.0)]
    vz = lp(np.gradient(pel[:, 2], 1 / FPS), 6.0)
    mins = argrelmin(vz, order=3)[0]
    mins = mins[vz[mins] < -0.05]  # ignore flat wobble
    kin = mins / FPS
    b = bas(mb, kin)
    period = 60.0 / tempo
    chance = np.mean([bas(mb + rng.uniform(0.2, 0.8) * period, kin) for _ in range(64)])

    jv = np.gradient(J - pel[:, None], 1 / FPS, axis=0)
    e = (jv ** 2).sum(axis=2).mean(axis=0)
    arm_share = float(e[16:24].sum() / e[1:].sum())  # elbows+wrists+hands, relative to pelvis
    cmds = smpl_to_commands(poses, trans, fps=FPS, cfg=RetargetConfig(), joints=J)
    half = np.array([(hi - lo) / 2 for lo, hi in ALL_RANGES])
    active = half > 0
    cmd = float(np.mean(np.sqrt((cmds[:, active] ** 2).mean(axis=0)) / half[active]))
    row = dict(v_rms=v_rms, turn=turn, z_std=z_std, tilt=tilt, head=head)
    row["energy"] = energy(row)
    row.update(bas=b, bas_lift=b - chance, drift=floor_drift(J), foot_slide=foot_slide(J),
               arm=arm_share, cmd=cmd, tempo=tempo)
    return row


def main():
    out = sys.argv[1]
    rng = np.random.default_rng(0)
    rows = []
    for arg in sys.argv[2:]:
        pkl, wav = arg.rsplit(":", 1)
        r = score(pkl, wav, rng)
        r["pkl"] = pkl
        rows.append(r)
        print(Path(pkl).name, {k: round(v, 3) for k, v in r.items() if k != "pkl"}, flush=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
