"""Command block layout and per-dimension ranges for the dance tasks.

The 13D command block is [twist(3), head_pose(4), body_pose(6)], the same
layout as the rest of the policy family.
"""

CMD_DIM = 13
TWIST = slice(0, 3)
HEAD = slice(3, 7)
BODY = slice(7, 13)

TWIST_RANGES = ((-0.4, 0.4), (-0.3, 0.3), (-1.0, 1.0))
HEAD_RANGES = ((-1.10, 1.10), (-1.10, 1.10), (-1.40, 1.40), (-0.31, 0.31))
BODY_RANGES = ((0.0, 0.0), (0.0, 0.0), (-0.04, 0.03), (-0.26, 0.26), (-0.26, 0.26), (0.0, 0.0))
ALL_RANGES = TWIST_RANGES + HEAD_RANGES + BODY_RANGES

# Trunk height (root_link z above the env origin, m) that body_z = 0 refers to.
# Measured on dance_v1 (model_4998) standing with zero commands: median 0.1158 m
# over 5 s after a 2 s settle, with and without DR (exp ref001). The earlier
# 0.095 sat 21 mm below the real stance, so the body_z Gaussian (std 0.01) was
# near 0 when standing and gave no height gradient.
DANCE_STAND_HEIGHT = 0.116

# Twist range of the b11 basketball continuation training (vx, vy in m/s, wz in rad/s).
BALL_TWIST_RANGES = ((-0.15, 0.15), (-0.10, 0.10), (-0.5, 0.5))

# Factor on the dance twist for dancing on the ball. Per channel, ball limit
# over the p95 of |twist|, then the minimum over channels, rounded down to 0.05.
# Computed on the library (dance/data/library.npz, command SHA-256 57e74da0...0197,
# 99000 frames), which is what training plays.
#   vx  0.15 / 0.0839 = 1.79
#   vy  0.10 / 0.1157 = 0.86
#   wz  0.50 / 1.0000 = 0.50  (binding. wz sits at its 1.0 clip)
# The 3 video songs (swag, funky_magic, fallin_for_you, 9000 frames pooled) give
# 1.01, 0.65 and 0.50, so the value is 0.50 there too.
# The Task 15 commands gave 0.30 (vy binding). ball001 (b11 baseline) used 0.30.
BALL_TWIST_SCALE = 0.50
