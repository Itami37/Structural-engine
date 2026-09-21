"""Component and slab templates — reusable definitions."""
from __future__ import annotations
from dataclasses import dataclass
import copy

from .materials import Concrete, Steel
from .profiles import Profile, rectangle
from .rebar import RebarLayout, SlabRebar


COLUMN = "column"
BEAM = "beam"
SLAB = "slab"


@dataclass
class ComponentTemplate:
    name: str
    category: str                 # COLUMN or BEAM
    profile: Profile
    concrete: Concrete
    steel: Steel
    rebar: RebarLayout

    def __post_init__(self):
        if self.category not in (COLUMN, BEAM):
            raise ValueError("category must be 'column' or 'beam'")
        ok, msg = self.profile.is_valid()
        if not ok:
            raise ValueError(f"Invalid profile: {msg}")

    def clone(self, new_name):
        return ComponentTemplate(
            name=new_name, category=self.category,
            profile=self.profile.clone(new_name + "_p"),
            concrete=copy.deepcopy(self.concrete),
            steel=copy.deepcopy(self.steel),
            rebar=copy.deepcopy(self.rebar))

    @property
    def b(self): return self.profile.width()

    @property
    def h(self): return self.profile.height()


@dataclass
class SlabTemplate:
    name: str
    thickness: float
    concrete: Concrete
    steel: Steel
    rebar: SlabRebar

    def __post_init__(self):
        if self.thickness <= 0:
            raise ValueError("thickness must be > 0")
        self.rebar.validate()


class TemplateLibrary:
    def __init__(self):
        self._components: dict[str, ComponentTemplate] = {}
        self._slabs: dict[str, SlabTemplate] = {}

    # ---- components
    def add(self, tpl: ComponentTemplate):
        if tpl.name in self._components:
            raise ValueError(f"Template '{tpl.name}' already exists")
        self._components[tpl.name] = tpl

    def replace(self, tpl: ComponentTemplate):
        self._components[tpl.name] = tpl

    def remove(self, name):
        self._components.pop(name, None)

    def get(self, name) -> ComponentTemplate:
        return self._components[name]

    def all(self): return list(self._components.values())
    def by_category(self, cat): return [t for t in self._components.values()
                                        if t.category == cat]
    def names(self): return list(self._components.keys())
    def __contains__(self, n): return n in self._components

    # ---- slabs
    def add_slab(self, tpl: SlabTemplate):
        if tpl.name in self._slabs:
            raise ValueError(f"Slab template '{tpl.name}' already exists")
        self._slabs[tpl.name] = tpl

    def replace_slab(self, tpl: SlabTemplate):
        self._slabs[tpl.name] = tpl

    def remove_slab(self, name):
        self._slabs.pop(name, None)

    def get_slab(self, name) -> SlabTemplate:
        return self._slabs[name]

    def all_slabs(self): return list(self._slabs.values())
    def slab_names(self): return list(self._slabs.keys())