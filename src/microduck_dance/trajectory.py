"""Pure-torch command trajectory sources used by the dance command terms."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class SmoothTrajectorySpec:
    ranges: tuple[tuple[float, float], ...]
    freq_range_hz: tuple[float, float]
    num_harmonics: int = 3
    hold_prob: float = 0.2


class SmoothTrajectory:
    """Zero-centred sum of sines per dimension, scaled into asymmetric ranges.

    raw = sum_k a_k sin(2 pi f_k t + phi_k) / sum_k a_k lies in [-1, 1].
    The output is s * raw * hi for raw >= 0 and s * raw * (-lo) for raw < 0,
    with s ~ U(0, 1) drawn per dimension, so it stays inside [lo, hi].
    """

    def __init__(self, spec: SmoothTrajectorySpec, num_envs: int, device, generator: torch.Generator | None = None):
        self.spec = spec
        self.num_envs = num_envs
        self.device = device
        self.generator = generator
        dim, k = len(spec.ranges), spec.num_harmonics
        self.dim = dim
        self.lo = torch.tensor([r[0] for r in spec.ranges], dtype=torch.float32, device=device)
        self.hi = torch.tensor([r[1] for r in spec.ranges], dtype=torch.float32, device=device)
        self.amp = torch.ones(num_envs, dim, k, device=device)
        self.freq = torch.zeros(num_envs, dim, k, device=device)
        self.phase = torch.zeros(num_envs, dim, k, device=device)
        self.scale = torch.zeros(num_envs, dim, device=device)
        self.hold = torch.ones(num_envs, dtype=torch.bool, device=device)
        self.t = torch.zeros(num_envs, dtype=torch.float64, device=device)
        self._value = torch.zeros(num_envs, dim, device=device)

    def _rand(self, *shape) -> Tensor:
        return torch.rand(*shape, generator=self.generator, device=self.device)

    def _eval(self) -> Tensor:
        arg = 2 * math.pi * self.freq * self.t.float()[:, None, None] + self.phase
        raw = (self.amp * torch.sin(arg)).sum(-1) / self.amp.sum(-1).clamp_min(1e-8)
        out = self.scale * raw * torch.where(raw >= 0, self.hi, -self.lo)
        return torch.where(self.hold[:, None], torch.zeros_like(out), out)

    def resample(self, env_ids: Tensor) -> None:
        env_ids = env_ids.to(self.device)
        n, d, k = len(env_ids), self.dim, self.spec.num_harmonics
        f_lo, f_hi = self.spec.freq_range_hz
        self.amp[env_ids] = self._rand(n, d, k)
        self.freq[env_ids] = f_lo + (f_hi - f_lo) * self._rand(n, d, k)
        self.phase[env_ids] = 2 * math.pi * self._rand(n, d, k)
        self.scale[env_ids] = self._rand(n, d)
        self.hold[env_ids] = self._rand(n) < self.spec.hold_prob
        self.t[env_ids] = 0.0
        self._value[env_ids] = self._eval()[env_ids]

    def step(self, dt: float) -> Tensor:
        self.t += dt
        self._value = self._eval()
        return self._value

    def value(self) -> Tensor:
        return self._value


class ClipPlayer:
    """Plays a (T, dim) clip sampled every `dt` seconds, one frame per step.

    clock counts step() calls, which is the global control step. With
    resume_on_reset (playback only, random_start False), a resample after the
    first one seeks the env to the clock instead of frame 0, so a respawned env
    stays in sync with the music, and fades its command in over
    reset_ramp_steps. A resample happens inside the env step before the command
    update, so the next step() lands on the same frame as every other env.
    """

    def __init__(self, frames: Tensor, dt: float, num_envs: int, device, random_start: bool,
                 generator: torch.Generator | None = None, resume_on_reset: bool = False,
                 reset_ramp_steps: int = 0):
        if resume_on_reset and random_start:
            raise ValueError("resume_on_reset requires random_start=False")
        self.frames = frames.to(device)
        self.dt = dt
        self.num_envs = num_envs
        self.device = device
        self.random_start = random_start
        self.generator = generator
        self.resume_on_reset = resume_on_reset
        self.reset_ramp_steps = int(reset_ramp_steps)
        self.clock = 0
        self.start = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.n = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.ramp = torch.zeros(num_envs, dtype=torch.long, device=device)  # 0 means no ramp

    def index(self) -> Tensor:
        """Frame index each env plays now."""
        T = self.frames.shape[0]
        idx = self.start + self.n
        return idx % T if self.random_start else idx.clamp(max=T - 1)

    def seek(self, env_ids: Tensor, index: int, ramp_steps: int = 0) -> None:
        """Play frame `index` now for env_ids, scaled by n / ramp_steps until n reaches ramp_steps."""
        env_ids = env_ids.to(self.device)
        self.start[env_ids] = int(index)
        self.n[env_ids] = 0
        self.ramp[env_ids] = int(ramp_steps)

    def resample(self, env_ids: Tensor) -> None:
        env_ids = env_ids.to(self.device)
        T = self.frames.shape[0]
        if self.random_start:
            self.start[env_ids] = torch.randint(0, T, (len(env_ids),), generator=self.generator, device=self.device)
        elif self.resume_on_reset and self.clock > 0:
            self.seek(env_ids, self.clock, self.reset_ramp_steps)
            return
        else:
            self.start[env_ids] = 0
        self.n[env_ids] = 0
        self.ramp[env_ids] = 0

    def step(self, dt: float) -> Tensor:
        # env.reset() calls the command update with dt=0; that is not a control step.
        if dt > 0:
            self.n += 1
            self.clock += 1
        return self.value()

    def value(self) -> Tensor:
        v = self.frames[self.index()]
        if bool((self.ramp > 0).any()):
            scale = torch.where(
                self.ramp > 0,
                (self.n.float() / self.ramp.clamp(min=1).float()).clamp(max=1.0),
                torch.ones_like(self.n, dtype=torch.float32),
            )
            v = v * scale.unsqueeze(1)
        return v


TrajectorySource = SmoothTrajectory | ClipPlayer
