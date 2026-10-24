"""Fit the dance_v3 posture synergy S by flat-foot IK on the walk model.

The leg joints are solved so that the trunk reaches a body command with both
feet flat, level with each other and the CoM forward offset unchanged. S is a
table over body_z knots: at each knot the joint offset of the squat and the
coefficients of roll, pitch, roll^2, pitch^2 and roll*pitch (least squares on
IK solutions at that height). Between knots S is linear in z.
Prints the table and the FK error of HOME + S(cmd) on a command grid.

Run: uv run python dance/scripts/retarget/fit_posture_synergy.py
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from microduck_dance.posture_fk import LEG_JOINTS, FlatFootKinematics

K = FlatFootKinematics()
YAW = [i for i, n in enumerate(LEG_JOINTS) if "hip_yaw" in n]


def residual(delta, z, roll, pitch, com_w):
    s = K.state(delta)
    r = [
        (s.height - z) / 0.0005,
        (s.roll - roll) / 0.005,
        (s.pitch - pitch) / 0.005,
        s.foot_height_diff / 0.0005,
        s.foot_tilt_diff / 0.005,
        s.com_forward / 0.005 * com_w,
    ]
    r += [delta[i] / 0.01 for i in YAW]
    r += list(delta / 2.0)  # weak regularizer picks the smallest solution
    return np.asarray(r)


def solve(z, roll, pitch, x0=None, com_w=1.0):
    x0 = np.zeros(len(LEG_JOINTS)) if x0 is None else x0
    sol = least_squares(residual, x0, args=(z, roll, pitch, com_w), method="lm", max_nfev=4000)
    return sol.x, K.state(sol.x)


Z_KNOTS = (-0.035, -0.025, -0.015, -0.005, 0.0, 0.005, 0.010)
FIT_RP = [(r, p) for r in (-0.1, -0.05, 0.0, 0.05, 0.1) for p in (-0.2, -0.1, 0.0, 0.1, 0.2) if (r, p) != (0.0, 0.0)]
COLS = ("offset", "roll", "pitch", "roll^2", "pitch^2", "roll*pitch")


def basis(r, p):
    return np.array([r, p, r * r, p * p, r * p])


def main():
    """Table (J, K, 6): per z knot the joint offset and the coefficients of roll,
    pitch, roll^2, pitch^2 and roll*pitch (least squares on IK solutions)."""
    T = np.zeros((len(LEG_JOINTS), len(Z_KNOTS), 6))
    for k, z in enumerate(Z_KNOTS):
        if z == 0.0:
            dz, s = np.zeros(len(LEG_JOINTS)), K.state(np.zeros(len(LEG_JOINTS)))
        else:
            dz, s = solve(z, 0.0, 0.0)
        print(f"z {z:+.3f}: IK h={s.height:+.4f} com={s.com_forward:+.4f}", flush=True)
        A = np.stack([basis(r, p) for r, p in FIT_RP])
        D = np.stack([solve(z, r, p, x0=dz)[0] - dz for r, p in FIT_RP])
        coef, *_ = np.linalg.lstsq(A, D, rcond=None)  # (5, J)
        T[:, k, 0] = dz
        T[:, k, 1:] = coef.T
    T = np.round(T, 4)
    T[:, Z_KNOTS.index(0.0), 0] = 0.0
    print("\nZ_KNOTS =", Z_KNOTS)
    print("POSTURE_SYNERGY = {  # per knot:", COLS)
    for i, n in enumerate(LEG_JOINTS):
        row = ", ".join("(" + ", ".join(f"{x:+.4f}" for x in T[i, k]) + ")" for k in range(len(Z_KNOTS)))
        print(f"    {n!r}: ({row}),")
    print("}")
    evaluate(T)
    return T


def apply(T, z, roll, pitch):
    zk = np.array(Z_KNOTS)
    z = float(np.clip(z, zk[0], zk[-1]))
    j = int(np.clip(np.searchsorted(zk, z, side="right") - 1, 0, len(zk) - 2))
    w = (z - zk[j]) / (zk[j + 1] - zk[j])
    row = (1 - w) * T[:, j] + w * T[:, j + 1]
    return row[:, 0] + row[:, 1:] @ basis(roll, pitch)


def evaluate(T):
    zs = (-0.035, -0.03, -0.025, -0.02, -0.015, -0.01, -0.005, 0.0, 0.0025, 0.005, 0.0075, 0.01)
    angs = (-0.2, -0.1, 0.0, 0.1, 0.2)
    print("\nFK of HOME + S(cmd). feasible = flat-foot IK reaches the command within 1 mm / 0.01 rad")
    print("   z      roll  pitch | feasible | err_h(mm) err_r  err_p  dfoot(mm) tilt")
    worst = {True: [0, 0, 0, 0, 0], False: [0, 0, 0, 0, 0]}
    for z in zs:
        for r in angs:
            for p in angs:
                _, si = solve(z, r, p)
                feas = abs(si.height - z) < 1e-3 and abs(si.roll - r) < 0.01 and abs(si.pitch - p) < 0.01 \
                    and abs(si.foot_height_diff) < 1e-3 and si.foot_tilt_diff < 0.01
                s = K.state(apply(T, z, r, p))
                e = [abs(s.height - z) * 1e3, abs(s.roll - r), abs(s.pitch - p), abs(s.foot_height_diff) * 1e3, s.foot_tilt_diff]
                worst[feas] = [max(a, b) for a, b in zip(worst[feas], e)]
                print(f"{z:+.4f} {r:+.2f} {p:+.2f} | {str(feas):5s} | {e[0]:6.2f} {e[1]:.4f} {e[2]:.4f} {e[3]:6.2f} {e[4]:.4f}")
    for k, v in worst.items():
        print(f"worst over {'feasible' if k else 'infeasible'} grid points: h {v[0]:.2f} mm, roll {v[1]:.4f}, pitch {v[2]:.4f}, "
              f"foot dh {v[3]:.2f} mm, foot tilt {v[4]:.4f}")


if __name__ == "__main__":
    main()
