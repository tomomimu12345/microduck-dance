# Dance motions

SMPL clips (z-up metres, 30 fps, 60 s, 1800 frames, keys smpl_poses (F, 72),
smpl_trans (F, 3), full_pose (F, 24, 3)) generated from the first 60 s of each
song. Since 2026-10-20 (Task 14) the clips come from EDGE (AIST++), before that
from FlowerDance (FineDance). Credits and licenses: `dance/data/CREDITS.md`.

## Video and spare songs (current, EDGE, 2026-10-20)

`bash dance/scripts/generate/generate_motion_edge.sh dance/data/music/opentracks/<song>.wav dance/data/motions/songs/<song>.pkl <seed>`
(install with `dance/scripts/setup/setup_edge.sh`). The script runs the Jukebox
features, EDGE sampling, the floor correction (the lowest-joint height below
0.5 Hz is removed, floor at z = -0.94) and the heading alignment (start facing
-y), all in `dance/scripts/generate/edge_generate.py` and
`src/microduck_dance/motion_fix.py`.

Seeds 0, 1 and 2 were generated for each song. The kept seed has the highest
robot-visible energy among the seeds with drift under 0.10 m
(`dance/scripts/evaluate/score_motion.py`; energy 1.0 is a typical FlowerDance clip of
the 2026-10-20 set). Drift is p98 minus p2 of the per-frame lowest joint height
after the floor correction. Foot slide is the mean horizontal speed of the
support foot (the lower foot while within 3 cm of the lowest joint).

| song | use | generator | seed | energy | drift (m) | foot slide (m/s) | energy of the seeds 0 / 1 / 2 | FlowerDance Breaking energy |
|---|---|---|---|---|---|---|---|---|
| swag | video | EDGE | 0 | 2.65 | 0.072 | 0.40 | 2.65 / 2.37 / 2.20 | 1.78 |
| funky_magic | video | EDGE | 0 | 2.86 | 0.079 | 0.50 | 2.86 / 2.34 / 2.61 | 1.76 |
| fallin_for_you | video | EDGE | 1 | 3.00 | 0.073 | 0.46 | 2.73 / 3.00 / 2.81 | 2.28 |
| redwing_beats | spare | EDGE | 0 | 3.03 | 0.099 | 0.51 | 3.03 / 2.67 / 2.80 | 2.00 |
| pastel_house | spare | EDGE | 0 | 2.71 | 0.085 | 0.45 | 2.71 / 2.21 / 2.63 | 2.14 |

The FlowerDance Breaking clips have drift 0.037 to 0.047 m and foot slide 0.19
to 0.42 m/s. redwing_beats passes the drift rule by 1 mm (all its seeds are
0.092 to 0.099 m).

The command files in `dance/data/commands/` and `dance/data/library.npz` come
from these clips since Task 15 (`bash dance/scripts/retarget/retarget_all.sh`: default
RetargetConfig with the phrase gate and the per-song groove pattern, vel_gain
1.3). The earlier commands were made from FlowerDance clips. They are not in the
public repository.

## FlowerDance clips (until 2026-10-20, replaced by EDGE)

Before Task 14 the song clips, the library clips and the commands made from them
came from FlowerDance (https://github.com/XulongT/FlowerDance). FlowerDance has
no license, so the scripts that call it and every file made with it are left out
of the public repository. The experiment log (dance/docs/, not in git) keeps the history.

## Library clips (`dance/data/motions/training/`, EDGE, 2026-10-20)

`bash dance/scripts/generate/generate_library_edge.sh` on the synthetic tracks
`dance/data/music/synthetic/synthetic_XX.wav` (XX = 00 to 32). Clip i uses seed i,
with the same floor correction and heading alignment as the songs. A clip with
drift 0.10 m or more would be generated again with the next seed. None needed
it (drift 0.031 to 0.063 m). Energy 1.60 to 2.46, mean 1.93 (the earlier FlowerDance
library: 0.34 to 2.09, mean 1.32). Foot slide 0.18 to 0.39 m/s.
`dance/data/library.npz` and the library command files are built from these
clips since Task 15.

## Beats (`dance/data/beats/songs/`)

`~/.venvs/edge/bin/python dance/scripts/generate/compute_beats.py <wav> ... --out-dir dance/data/beats/songs`
(librosa beat_track). The beats of the five songs are used only to make the
video command files, never for training data (CREDITS.md, license clause 11).
