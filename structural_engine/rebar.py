"""Automatic reinforcement layout.

Section local axes: y = height direction, z = width direction,
section centered at origin.
"""
from __future__ import annotations
from dataclasses import dataclass
import math


@dataclass
class RebarLayout:
    cover: float
    db: float
    ds: float
    n_bars_x: int
    n_bars_y: int
    stirrup_spacing: float
    stirrup_end_offset: float = 0.05

    def validate(self) -> None:
        if self.n_bars_x < 2 or self.n_bars_y < 2:
            raise ValueError("n_bars_x and n_bars_y must be >= 2")
        if min(self.cover, self.db, self.ds, self.stirrup_spacing) <= 0:
            raise ValueError("cover, db, ds, stirrup_spacing must be > 0")
        if self.stirrup_end_offset < 0:
            raise ValueError("stirrup_end_offset must be >= 0")


def longitudinal_bar_positions(b: float, h: float, layout: RebarLayout
                               ) -> list[tuple[float, float]]:
    """Return [(y, z), ...] bar centers in the section frame."""
    layout.validate()
    c = layout.cover + layout.ds + layout.db / 2.0
    z0, z1 = -b / 2 + c, b / 2 - c
    y0, y1 = -h / 2 + c, h / 2 - c
    if z1 <= z0 or y1 <= y0:
        raise ValueError("Section too small for cover + stirrup + bar")

    seen: set[tuple[float, float]] = set()
    bars: list[tuple[float, float]] = []

    def _add(y, z):
        k = (round(y, 9), round(z, 9))
        if k not in seen:
            seen.add(k)
            bars.append((y, z))

    for i in range(layout.n_bars_x):
        z = z0 + i * (z1 - z0) / (layout.n_bars_x - 1)
        _add(y0, z)
        _add(y1, z)

    if layout.n_bars_y > 2:
        for i in range(1, layout.n_bars_y - 1):
            y = y0 + i * (y1 - y0) / (layout.n_bars_y - 1)
            _add(y, z0)
            _add(y, z1)

    return bars


def stirrup_positions(member_length: float, spacing: float,
                      end_offset: float = 0.05) -> list[float]:
    usable = member_length - 2.0 * end_offset
    if usable <= 0:
        return []
    n = max(1, int(math.floor(usable / spacing)))
    actual = usable / n
    return [end_offset + i * actual for i in range(n + 1)]