"""Mixed command sources for dance_v3 training (pure torch).

A MixedSource picks one sub-source per env at every resample and streams that
sub-source's value. All sub-sources advance every step for all envs, so a
sub-source keeps its own clock and the mix is a per-env gather.

PiecewiseHeld produces piecewise-constant commands: pieces of U(piece_s) s,
each piece one of several kinds (exact zero, turn in place, uniform box), with
a linear blend of blend_s seconds from the previous value into the new piece.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


class ZeroSource:
    """Exact zero command."""

    def __init__(self, dim: int, num_envs: int, device):
        self._value = torch.zeros(num_envs, dim, device=device)

    def resample(self, env_ids: Tensor) -> None:
        pass

    def step(self, dt: float) -> Tensor:
        return self._value

    def value(self) -> Tensor:
        return self._value


@dataclass(frozen=True)
class PieceKind:
    """One kind of held piece.

    prob: probability of this kind per piece.
    ranges: per-dimension uniform (lo, hi); (0, 0) gives an exact zero.
    abs_ranges: dims whose magnitude is drawn from (lo, hi) with a random sign
    (turn in place: |wz| in (0.4, 1.0)). Overrides ranges on those dims.
    """

    prob: float
    ranges: tuple[tuple[float, float], ...]
    abs_ranges: tuple[tuple[int, float, float], ...] = ()


class PiecewiseHeld:
    def __init__(self, kinds: tuple[PieceKind, ...], piece_s: tuple[float, float], blend_s: float,
                 num_envs: int, device, generator: torch.Generator | None = None):
        if abs(sum(k.prob for k in kinds) - 1.0) > 1e-6:
            raise ValueError("piece kind probabilities must sum to 1")
        dims = {len(k.ranges) for k in kinds}
        if len(dims) != 1:
            raise ValueError("all piece kinds need the same dimension")
        self.dim = dims.pop()
        self.kinds = kinds
        self.piece_s = piece_s
        self.blend_s = float(blend_s)
        self.num_envs = num_envs
        self.device = device
        self.generator = generator
        self.cum = torch.cumsum(torch.tensor([k.prob for k in kinds], device=device), 0)
        self.lo = torch.tensor([k.ranges for k in kinds], dtype=torch.float32, device=device)[..., 0]
        self.hi = torch.tensor([k.ranges for k in kinds], dtype=torch.float32, device=device)[..., 1]
        self.prev = torch.zeros(num_envs, self.dim, device=device)
        self.target = torch.zeros(num_envs, self.dim, device=device)
        self.kind = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.t = torch.zeros(num_envs, device=device)
        self.length = torch.ones(num_envs, device=device)
        self._value = torch.zeros(num_envs, self.dim, device=device)
        # Every env starts inside a valid piece (blending in from zero).
        self._new_piece(torch.arange(num_envs, device=device))

    def _rand(self, *shape) -> Tensor:
        return torch.rand(*shape, generator=self.generator, device=self.device)

    def _new_piece(self, env_ids: Tensor) -> None:
        n = len(env_ids)
        if n == 0:
            return
        kind = torch.searchsorted(self.cum, self._rand(n).clamp(max=1.0 - 1e-7), right=True).clamp(max=len(self.kinds) - 1)
        lo, hi = self.lo[kind], self.hi[kind]
        val = lo + (hi - lo) * self._rand(n, self.dim)
        for k, spec in enumerate(self.kinds):
            sel = kind == k
            for d, a, b in spec.abs_ranges:
                mag = a + (b - a) * self._rand(n)
                sign = torch.where(self._rand(n) < 0.5, -1.0, 1.0)
                val[:, d] = torch.where(sel, sign * mag, val[:, d])
        self.prev[env_ids] = self._value[env_ids]
        self.target[env_ids] = val
        self.kind[env_ids] = kind
        self.t[env_ids] = 0.0
        p0, p1 = self.piece_s
        self.length[env_ids] = p0 + (p1 - p0) * self._rand(n)

    def _eval(self) -> Tensor:
        if self.blend_s <= 0.0:
            return self.target.clone()
        a = (self.t / self.blend_s).clamp(max=1.0)[:, None]
        blended = self.prev + (self.target - self.prev) * a
        # After the blend the value is the piece target bit for bit (exact zeros stay zero).
        return torch.where(a >= 1.0, self.target, blended)

    def resample(self, env_ids: Tensor) -> None:
        env_ids = env_ids.to(self.device)
        self._new_piece(env_ids)
        self._value[env_ids] = self._eval()[env_ids]

    def step(self, dt: float) -> Tensor:
        if dt > 0:
            self.t += dt
            done = (self.t >= self.length).nonzero(as_tuple=False).squeeze(-1)
            if len(done):
                self._new_piece(done)
        self._value = self._eval()
        return self._value

    def value(self) -> Tensor:
        return self._value


class MixedSource:
    """Per-env choice among sub-sources, redrawn at every resample."""

    def __init__(self, sources: tuple, probs: tuple[float, ...], num_envs: int, device,
                 generator: torch.Generator | None = None, names: tuple[str, ...] | None = None):
        if len(sources) != len(probs) or abs(sum(probs) - 1.0) > 1e-6:
            raise ValueError("need one probability per source, summing to 1")
        self.sources = sources
        self.probs = probs
        self.names = names or tuple(str(i) for i in range(len(sources)))
        self.num_envs = num_envs
        self.device = device
        self.generator = generator
        self.cum = torch.cumsum(torch.tensor(probs, device=device), 0)
        self.mode = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._value = torch.zeros_like(sources[0].value())

    def _gather(self) -> Tensor:
        vals = torch.stack([s.value() for s in self.sources], 0)  # (S, N, D)
        idx = self.mode[None, :, None].expand(1, self.num_envs, vals.shape[-1])
        return torch.gather(vals, 0, idx)[0]

    def resample(self, env_ids: Tensor) -> None:
        env_ids = env_ids.to(self.device)
        if len(env_ids) == 0:
            return
        r = torch.rand(len(env_ids), generator=self.generator, device=self.device).clamp(max=1.0 - 1e-7)
        mode = torch.searchsorted(self.cum, r, right=True).clamp(max=len(self.sources) - 1)
        self.mode[env_ids] = mode
        for k, src in enumerate(self.sources):
            ids = env_ids[mode == k]
            if len(ids):
                src.resample(ids)
        self._value = self._gather()

    def step(self, dt: float) -> Tensor:
        for s in self.sources:
            s.step(dt)
        self._value = self._gather()
        return self._value

    def value(self) -> Tensor:
        return self._value
