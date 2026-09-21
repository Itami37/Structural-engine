"""Nodes, members, slabs — everything is editable after placement.

A Node is a first-class entity: (x, y, z) OR (master_node + offset).
Members reference two nodes.  Slabs reference a list of nodes.

Changing ANY parameter recomputes downstream geometry via
Model.recompute_geometry(), which is called on every change.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import math

from .templates import ComponentTemplate, SlabTemplate, COLUMN, BEAM
from .profiles import Profile


# ---------------------------------------------------------------- Node
@dataclass
class Node:
    name: str
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    # Optional link: if master is set, the node follows master + offset
    master: str | None = None
    offset_x: float = 0.0
    offset_y: float = 0.0
    offset_z: float = 0.0

    # resolved world coordinates (updated by Model.recompute_geometry)
    _resolved: tuple = field(default=(0.0, 0.0, 0.0), init=False)

    def update(self, model) -> bool:
        if self.master is None:
            new = (self.x, self.y, self.z)
        else:
            m = model.nodes.get(self.master)
            if m is None:
                new = (0.0, 0.0, 0.0)
            else:
                mx, my, mz = m._resolved
                new = (mx + self.offset_x, my + self.offset_y, mz + self.offset_z)
        changed = new != self._resolved
        self._resolved = new
        return changed

    @property
    def xyz(self):
        return self._resolved

    def describe(self):
        if self.master is None:
            return f"fixed ({self.x:.3f}, {self.y:.3f}, {self.z:.3f})"
        return (
            f"→ {self.master} "
            f"(+{self.offset_x:.3f},+{self.offset_y:.3f},"
            f"+{self.offset_z:.3f})"
        )


# ---------------------------------------------------------------- Member
class Member:
    """Line member: references two nodes and a ComponentTemplate."""

    def __init__(
        self, name: str, template: ComponentTemplate, start_node: str, end_node: str
    ):
        self.name = name
        self.template = template
        self.start_node = start_node
        self.end_node = end_node
        self._ele_tag: int | None = None

        # Editable after placement — override any template value here.
        self._override_profile: Profile | None = None
        self._override_rebar = None
        self._override_concrete = None
        self._override_steel = None
        self._local_axis = None  # optional: force a member axis direction
        self._roll = 0.0  # rotation of the section about the axis

    # ----- template-aware accessors (override-aware)
    @property
    def category(self):
        return self.template.category

    @property
    def profile(self) -> Profile:
        return self._override_profile or self.template.profile

    @property
    def b(self):
        return self.profile.width()

    @property
    def h(self):
        return self.profile.height()

    @property
    def rebar(self):
        return self._override_rebar or self.template.rebar

    @property
    def concrete(self):
        return self._override_concrete or self.template.concrete

    @property
    def steel(self):
        return self._override_steel or self.template.steel

    def length(self, model) -> float:
        s = model.nodes[self.start_node].xyz
        e = model.nodes[self.end_node].xyz
        return math.sqrt(sum((e[i] - s[i]) ** 2 for i in range(3)))

    def node_names(self):
        return (self.start_node, self.end_node)

    def set_profile(self, p: Profile):
        ok, msg = p.is_valid()
        if not ok:
            raise ValueError(f"Invalid profile: {msg}")
        self._override_profile = p

    def clear_overrides(self):
        self._override_profile = None
        self._override_rebar = None
        self._override_concrete = None
        self._override_steel = None
        self._local_axis = None
        self._roll = 0.0


# ---------------------------------------------------------------- Slab
class Slab:
    """Area member: a set of corner node names + a SlabTemplate.

    Editable: thickness, corner node references, mesh density.
    """

    def __init__(
        self,
        name: str,
        template: SlabTemplate,
        corners: list[str],
        mesh_nx: int = 6,
        mesh_ny: int = 6,
    ):
        if len(corners) < 3:
            raise ValueError("Slab needs at least 3 corner nodes")
        self.name = name
        self.template = template
        self.corners = list(corners)
        self.mesh_nx = mesh_nx
        self.mesh_ny = mesh_ny
        self._override_thickness = None
        self._override_rebar = None

    @property
    def thickness(self):
        return (
            self._override_thickness
            if self._override_thickness is not None
            else self.template.thickness
        )

    @property
    def concrete(self):
        return self.template.concrete

    @property
    def steel(self):
        return self.template.steel

    @property
    def rebar(self):
        return self._override_rebar or self.template.rebar

    def set_thickness(self, t: float):
        if t <= 0:
            raise ValueError("thickness must be > 0")
        self._override_thickness = t
