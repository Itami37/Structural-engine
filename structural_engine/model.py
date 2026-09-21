"""Model: nodes + members + slabs + restraints + loads + UDLs."""
from __future__ import annotations

from .templates import TemplateLibrary, ComponentTemplate, SlabTemplate
from .components import Node, Member, Slab


class Model:
    def __init__(self):
        self.templates = TemplateLibrary()
        self.nodes: dict[str, Node] = {}
        self.members: dict[str, Member] = {}
        self.slabs: dict[str, Slab] = {}

        self.restraints: dict[str, tuple] = {}          # node -> 6 bools
        self.point_loads: list = []                     # (node, 6 forces)
        self.member_udls: list = []                     # (member, direction, w)
        self.slab_udls: list = []                       # (slab, w)

        self._listeners: list = []

    # ---------------- templates
    def add_template(self, tpl):        self.templates.add(tpl);       self._notify()
    def replace_template(self, tpl):    self.templates.replace(tpl);   self._notify()
    def remove_template(self, name):
        for m in self.members.values():
            if m.template.name == name:
                raise ValueError(f"Template '{name}' in use by '{m.name}'")
        self.templates.remove(name); self._notify()

    def add_slab_template(self, tpl):   self.templates.add_slab(tpl);  self._notify()
    def replace_slab_template(self, tpl): self.templates.replace_slab(tpl); self._notify()
    def remove_slab_template(self, name):
        for s in self.slabs.values():
            if s.template.name == name:
                raise ValueError(f"Slab template '{name}' in use by '{s.name}'")
        self.templates.remove_slab(name); self._notify()

    # ---------------- nodes
    def add_node(self, node: Node):
        if node.name in self.nodes:
            raise ValueError(f"Node '{node.name}' exists")
        self.nodes[node.name] = node
        self._notify()
        return node
    def find_or_create_node(self, x: float, y: float, z: float,
                            tol: float = 1e-4,
                            name_hint: str | None = None) -> str:
        """Return the name of an existing free node within `tol` of
        (x, y, z), or create a new one.  Nodes with a master link are
        skipped (they are not free)."""
        for n in self.nodes.values():
            if n.master is not None:
                continue
            if (abs(n.x - x) < tol and abs(n.y - y) < tol
                    and abs(n.z - z) < tol):
                return n.name
        if name_hint is None or name_hint in self.nodes:
            i = 1
            while f"N{i}" in self.nodes:
                i += 1
            name_hint = f"N{i}"
        self.add_node(Node(name=name_hint, x=x, y=y, z=z))
        return name_hint
    def remove_node(self, name: str):
        self.nodes.pop(name, None)
        self.restraints.pop(name, None)
        self.point_loads = [pl for pl in self.point_loads if pl[0] != name]
        self.members = {k: v for k, v in self.members.items()
                        if v.start_node != name and v.end_node != name}
        self.slabs = {k: v for k, v in self.slabs.items()
                      if name not in v.corners}
        self._notify()

    # ---------------- members
    def add_member(self, m: Member):
        if m.name in self.members:
            raise ValueError(f"Member '{m.name}' exists")
        self.members[m.name] = m
        self._notify()
        return m

    def remove_member(self, name: str):
        self.members.pop(name, None)
        self.member_udls = [u for u in self.member_udls if u[0] != name]
        self._notify()

    # ---------------- slabs
    def add_slab(self, s: Slab):
        if s.name in self.slabs:
            raise ValueError(f"Slab '{s.name}' exists")
        self.slabs[s.name] = s
        self._notify()
        return s

    def remove_slab(self, name: str):
        self.slabs.pop(name, None)
        self.slab_udls = [u for u in self.slab_udls if u[0] != name]
        self._notify()

    # ---------------- restraints
    def add_restraint(self, node_name, fixed):
        if node_name not in self.nodes:
            raise KeyError(node_name)
        self.restraints[node_name] = tuple(bool(f) for f in fixed)
        self._notify()

    def remove_restraint(self, node_name):
        self.restraints.pop(node_name, None)
        self._notify()

    # ---------------- loads
    def add_point_load(self, node_name, forces):
        if node_name not in self.nodes:
            raise KeyError(node_name)
        f = tuple(list(forces) + [0.0] * (6 - len(forces)))
        self.point_loads.append((node_name, f))
        self._notify()

    def add_member_udl(self, member_name, direction, magnitude):
        """direction: 'z' (gravity), 'y', or 'x' — in world axes."""
        if member_name not in self.members:
            raise KeyError(member_name)
        self.member_udls.append((member_name, direction, magnitude))
        self._notify()

    def add_slab_udl(self, slab_name, magnitude):
        if slab_name not in self.slabs:
            raise KeyError(slab_name)
        self.slab_udls.append((slab_name, magnitude))
        self._notify()

    # ---------------- geometry
    def recompute_geometry(self):
        for _ in range(len(self.nodes) + 2):
            changed = False
            for n in self.nodes.values():
                if n.update(self):
                    changed = True
            if not changed:
                break

    # ---------------- observers
    def on_change(self, cb): self._listeners.append(cb)
    def _notify(self):
        for cb in list(self._listeners):
            cb()