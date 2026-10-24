"""Draw a pkl's `full_pose` (F, 24, 3 joint positions, z-up, meters, 30 fps) as a stick figure mp4.

    uv run python dance/scripts/video/render_human.py dance/data/motions/songs/swag.pkl out/swag_human.mp4

The pkl's SMPL joint positions are used as they are (no forward kinematics). The camera is fixed in
front of the dancer (the clip starts with the dancer facing -y), turned 20 degrees for depth, and it
follows the pelvis sideways. The floor line is the lowest joint of the whole clip. The figure is
generated geometry, with no face.
"""

from __future__ import annotations

import argparse
import pickle
import subprocess
from pathlib import Path

import numpy as np

FPS = 30
SIZE = (640, 720)  # width, height
AZIMUTH = np.deg2rad(20.0)
# SMPL 24-joint parents.
PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
LEFT = {1, 4, 7, 10, 13, 16, 18, 20, 22}
RIGHT = {2, 5, 8, 11, 14, 17, 19, 21, 23}


def project(pose: np.ndarray) -> tuple[np.ndarray, float]:
    """(F, 24, 3) world joints (z up) -> (F, 24, 2) screen coords (x right, y up), and the floor height.

    The camera looks along +y, so a dancer facing -y faces the camera. It is rotated by AZIMUTH about z.
    """
    c, s = np.cos(AZIMUTH), np.sin(AZIMUTH)
    xy = np.stack([pose[..., 0] * c + pose[..., 1] * s, pose[..., 2]], axis=-1)
    return xy, float(pose[..., 2].min())


def load_pose(pkl: Path) -> np.ndarray:
    with open(pkl, "rb") as f:
        d = pickle.load(f)
    if "full_pose" not in d:
        raise ValueError(f"{pkl} has no 'full_pose'")
    pose = np.asarray(d["full_pose"], dtype=float)
    if pose.ndim != 3 or pose.shape[1:] != (24, 3):
        raise ValueError(f"full_pose must be (F, 24, 3), got {pose.shape}")
    return pose


def render(pkl: Path, out: Path) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pose = load_pose(pkl)
    xy, floor = project(pose)
    top = float(xy[..., 1].max()) + 0.3  # room for the head circle above the head joint
    h_world = (top - floor) * 1.08
    w_world = h_world * SIZE[0] / SIZE[1]
    x_center = np.convolve(np.pad(xy[:, 0, 0], 7, mode="edge"), np.ones(15) / 15, mode="valid")

    dpi = 100
    fig = plt.figure(figsize=(SIZE[0] / dpi, SIZE[1] / dpi), dpi=dpi, facecolor="#1b1f27")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("#1b1f27")
    ax.set_ylim(floor - 0.04 * h_world, floor - 0.04 * h_world + h_world)
    ax.axis("off")
    ax.plot([-100, 100], [floor, floor], color="#8a93a3", lw=3, zorder=1)
    lines = []
    for j in range(1, 24):
        col = "#ff7a59" if j in LEFT else ("#4fc3f7" if j in RIGHT else "#f2f2f2")
        lines.append(ax.plot([], [], color=col, lw=7, solid_capstyle="round", zorder=3)[0])
    head = plt.Circle((0, 0), 0.11, color="#f2f2f2", zorder=4)
    ax.add_patch(head)
    ax.set_aspect("equal", adjustable="datalim")

    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{SIZE[0]}x{SIZE[1]}",
           "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(out)]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for f in range(len(pose)):
            ax.set_xlim(x_center[f] - w_world / 2, x_center[f] + w_world / 2)
            for j in range(1, 24):
                p, q = xy[f, PARENTS[j]], xy[f, j]
                lines[j - 1].set_data([p[0], q[0]], [p[1], q[1]])
            head.center = tuple(xy[f, 15] + np.array([0.0, 0.12]))
            fig.canvas.draw()
            proc.stdin.write(np.asarray(fig.canvas.buffer_rgba()).tobytes())
    finally:
        proc.stdin.close()
        rc = proc.wait()
    plt.close(fig)
    if rc != 0:
        raise RuntimeError(f"ffmpeg exited with {rc}")
    return len(pose)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pkl", type=Path)
    ap.add_argument("out", type=Path)
    a = ap.parse_args()
    n = render(a.pkl, a.out)
    print(f"{a.out}: {n} frames, {n / FPS:.2f} s")


if __name__ == "__main__":
    main()
