"""Parametric Column and Beam."""
from __future__ import annotations
import math

from .materials import Concrete, Steel
from .rebar import RebarLayout
from .param import Param, Derived


_DEFAULT_REBAR = RebarLayout(cover=0.04, db=0.020, ds=0.010,
                             n_bars_x=3, n_bars_y=3,
                             stirrup_spacing=0.150)


class Column:
    def __init__(self, name, x=0.0, y=0.0, z=0.0,
                 b=0.40, h=0.40, length=3.0,
                 concrete=None, steel=None, rebar=None):
        self.name = name
        self.concrete = concrete or Concrete()
        self.steel = steel or Steel()
        self.rebar = rebar or _DEFAULT_REBAR

        self.px = Param(f"{name}.px", x, {"unit": "m", "group": "position"})
        self.py = Param(f"{name}.py", y, {"unit": "m", "group": "position"})
        self.pz = Param(f"{name}.pz", z, {"unit": "m", "group": "position"})
        self.b = Param(f"{name}.b", b, {"unit": "m", "group": "section"})
        self.h = Param(f"{name}.h", h, {"unit": "m", "group": "section"})
        self.length = Param(f"{name}.length", length,
                            {"unit": "m", "group": "section"})

        self.base = Derived(f"{name}.base", self._base,
                            [self.px, self.py, self.pz])
        self.top = Derived(f"{name}.top", self._top,
                           [self.px, self.py, self.pz, self.length])

        self._node_b: int | None = None
        self._node_t: int | None = None
        self._ele_tag: int | None = None

    def _base(self):
        return (self.px.value, self.py.value, self.pz.value)

    def _top(self):
        return (self.px.value, self.py.value, self.pz.value + self.length.value)

    @property
    def all_params(self):
        return [self.px, self.py, self.pz, self.b, self.h, self.length]


class Beam:
    def __init__(self, name, col_a, col_b, b=0.30, h=0.50, z_offset=0.0,
                 concrete=None, steel=None, rebar=None):
        self.name = name
        self.col_a = col_a
        self.col_b = col_b
        self.concrete = concrete or col_a.concrete
        self.steel = steel or col_a.steel
        self.rebar = rebar or _DEFAULT_REBAR

        self.b = Param(f"{name}.b", b, {"unit": "m", "group": "section"})
        self.h = Param(f"{name}.h", h, {"unit": "m", "group": "section"})
        self.z_offset = Param(f"{name}.z_offset", z_offset,
                              {"unit": "m", "group": "section"})

        self.start = Derived(f"{name}.start", self._start,
                             [col_a.top, self.z_offset])
        self.end = Derived(f"{name}.end", self._end,
                           [col_b.top, self.z_offset])
        self.length = Derived(f"{name}.length", self._length,
                              [self.start, self.end])

        self._node_s: int | None = None
        self._node_e: int | None = None
        self._ele_tag: int | None = None

    def _start(self):
        x, y, z = self.col_a.top.value
        return (x, y, z + self.z_offset.value)

    def _end(self):
        x, y, z = self.col_b.top.value
        return (x, y, z + self.z_offset.value)

    def _length(self):
        (x1, y1, z1) = self.start.value
        (x2, y2, z2) = self.end.value
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)

    @property
    def all_params(self):
        return [self.b, self.h, self.z_offset]