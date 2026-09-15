"""Fiber section definition (rectangular RC).

Concrete fibers represent the NET concrete: the gross section minus the
area occupied by the longitudinal bars that fall inside each fiber.  This
avoids counting the same area twice when summing stiffness contributions
from concrete and steel.

The steel fibers are the discrete longitudinal bars.
"""
from __future__ import annotations
from dataclasses import dataclass
import math

from .rebar import RebarLayout, longitudinal_bar_positions


@dataclass
class Fiber:
    y: float
    z: float
    area: float
    material: str  # 'concrete' | 'steel'


@dataclass
class FiberSection:
    b: float
    h: float
    rebar: RebarLayout
    n_fiber_y: int = 20
    n_fiber_z: int = 20

    def fibers(self) -> list[Fiber]:
        fibs: list[Fiber] = []
        dy = self.h / self.n_fiber_y
        dz = self.b / self.n_fiber_z
        A_c = dy * dz
        for i in range(self.n_fiber_y):
            y = -self.h / 2 + (i + 0.5) * dy
            for j in range(self.n_fiber_z):
                z = -self.b / 2 + (j + 0.5) * dz
                fibs.append(Fiber(y, z, A_c, "concrete"))

        bar_radius = self.rebar.db / 2.0
        bar_area = math.pi * bar_radius ** 2
        bar_positions = longitudinal_bar_positions(self.b, self.h, self.rebar)

        # Subtract the bar area from concrete fibers the bar overlaps.
        # Each bar is assigned to the concrete fibers whose centroid lies
        # within bar_radius of the bar centre.  If a bar spans multiple
        # concrete fibers, the subtraction is distributed among them.
        remaining = [bar_area] * len(bar_positions)
        for fib in fibs:
            if fib.area <= 0.0:
                continue
            for k, (yb, zb) in enumerate(bar_positions):
                if remaining[k] <= 0.0:
                    continue
                dist = math.hypot(fib.y - yb, fib.z - zb)
                if dist <= bar_radius:
                    take = min(fib.area, remaining[k])
                    fib.area -= take
                    remaining[k] -= take
                    if fib.area <= 1e-12:
                        break

        # Drop concrete fibers that lost all their area
        fibs = [f for f in fibs if f.area > 1e-12]

        # Append steel fibers
        for (y, z) in bar_positions:
            fibs.append(Fiber(y, z, bar_area, "steel"))

        return fibs

    def geometric_properties(self):
        fibs = self.fibers()
        A = sum(f.area for f in fibs)
        yb = sum(f.area * f.y for f in fibs) / A
        zb = sum(f.area * f.z for f in fibs) / A
        Iy = sum(f.area * (f.y - yb) ** 2 for f in fibs)
        Iz = sum(f.area * (f.z - zb) ** 2 for f in fibs)
        return A, Iy, Iz, yb, zb