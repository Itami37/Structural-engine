"""Fiber section built from a Profile.

Concrete fibers fill the profile; steel fibers are the longitudinal bars.
Bar areas are subtracted from the concrete fibers they overlap, so the
sum of fiber areas ≈ the net concrete area + the bar areas.
"""
from __future__ import annotations
from dataclasses import dataclass
import math

from .profiles import Profile
from .rebar import RebarLayout, longitudinal_bar_positions


@dataclass
class Fiber:
    y: float
    z: float
    area: float
    material: str


@dataclass
class FiberSection:
    profile: Profile
    rebar: RebarLayout
    n_fiber_y: int = 30
    n_fiber_z: int = 30

    def fibers(self):
        x0, y0, x1, y1 = self.profile.bbox()
        dx = (x1 - x0) / self.n_fiber_y
        dy = (y1 - y0) / self.n_fiber_z

        fibs = []
        for i in range(self.n_fiber_y):
            y = x0 + (i + 0.5) * dx
            for j in range(self.n_fiber_z):
                z = y0 + (j + 0.5) * dy
                if self._inside(y, z):
                    fibs.append(Fiber(y, z, dx * dy, "concrete"))

        bar_r = self.rebar.db / 2.0
        bar_a = math.pi * bar_r ** 2
        bars = longitudinal_bar_positions(
            self.profile.width(), self.profile.height(), self.rebar)

        remaining = [bar_a] * len(bars)
        for fib in fibs:
            for k, (yb, zb) in enumerate(bars):
                if remaining[k] <= 0.0:
                    continue
                if math.hypot(fib.y - yb, fib.z - zb) <= bar_r:
                    take = min(fib.area, remaining[k])
                    fib.area -= take
                    remaining[k] -= take
                    if fib.area <= 1e-12:
                        break

        fibs = [f for f in fibs if f.area > 1e-12]
        for (yb, zb) in bars:
            fibs.append(Fiber(yb, zb, bar_a, "steel"))
        return fibs

    def _inside(self, y, z) -> bool:
        if not _point_in_poly(y, z, self.profile.outer):
            return False
        for h in self.profile.holes:
            if _point_in_poly(y, z, h):
                return False
        return True


def _point_in_poly(x, y, poly) -> bool:
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and \
           (x < (xj - xi) * (y - yi) / (yj - yi + 1e-30) + xi):
            inside = not inside
        j = i
    return inside