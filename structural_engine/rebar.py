"""Reinforcement layout: longitudinal bars, stirrups, slab grids."""
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


@dataclass
class SlabRebar:
    """Two-way slab reinforcement (top and bottom mats)."""
    cover: float
    bar_dia: float
    spacing_x: float         # spacing of bars running along local x
    spacing_y: float         # spacing of bars running along local y
    top_mat: bool = True
    bottom_mat: bool = True

    def validate(self) -> None:
        if min(self.cover, self.bar_dia, self.spacing_x, self.spacing_y) <= 0:
            raise ValueError("slab rebar parameters must be > 0")


def longitudinal_bar_positions(b, h, layout: RebarLayout):
    layout.validate()
    c = layout.cover + layout.ds + layout.db / 2.0
    z0, z1 = -b / 2 + c, b / 2 - c
    y0, y1 = -h / 2 + c, h / 2 - c
    if z1 <= z0 or y1 <= y0:
        raise ValueError("Section too small for cover + stirrup + bar")

    seen, bars = set(), []

    def _add(y, z):
        k = (round(y, 9), round(z, 9))
        if k not in seen:
            seen.add(k); bars.append((y, z))

    for i in range(layout.n_bars_x):
        z = z0 + i * (z1 - z0) / (layout.n_bars_x - 1)
        _add(y0, z); _add(y1, z)
    if layout.n_bars_y > 2:
        for i in range(1, layout.n_bars_y - 1):
            y = y0 + i * (y1 - y0) / (layout.n_bars_y - 1)
            _add(y, z0); _add(y, z1)
    return bars


def stirrup_positions(L, spacing, end_offset=0.05):
    usable = L - 2.0 * end_offset
    if usable <= 0:
        return []
    n = max(1, int(math.floor(usable / spacing)))
    actual = usable / n
    return [end_offset + i * actual for i in range(n + 1)]


def slab_bar_grid(width, height, layout: SlabRebar):
    """Return (bars_x, bars_y) where each is a list of (coord, layer_z)."""
    def _run(length, spacing):
        n = max(2, int(math.ceil(length / spacing)) + 1)
        return [(-length / 2) + i * (length / (n - 1)) for i in range(n)]

    xs = _run(width, layout.spacing_x)
    ys = _run(height, layout.spacing_y)

    z_layers = []
    if layout.bottom_mat:
        z_layers.append(-layout.cover - layout.bar_dia / 2.0)
    if layout.top_mat:
        z_layers.append(+layout.cover + layout.bar_dia / 2.0)

    bars_x, bars_y = [], []
    for z in z_layers:
        for y in ys:
            bars_x.append((y, z))       # runs along x
        for x in xs:
            bars_y.append((x, z))       # runs along y
    return bars_x, bars_y