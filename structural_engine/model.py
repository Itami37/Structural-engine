"""Structural model container.

Restraints and point loads are keyed by (component_name, node_name).
node_name is 'base'/'top' for columns and 'start'/'end' for beams.
"""
from __future__ import annotations


class Model:
    def __init__(self):
        self.columns: dict = {}
        self.beams: dict = {}
        # restraints: {(comp_name, node_name): (ux, uy, uz, rx, ry, rz)}
        self.restraints: dict = {}
        # point loads: list of (comp_name, node_name, (Fx,Fy,Fz,Mx,My,Mz))
        self.point_loads: list = []
        self._listeners: list = []

    # ---- components
    def add_column(self, col):
        self.columns[col.name] = col
        self._notify()
        return col

    def add_beam(self, beam):
        self.beams[beam.name] = beam
        self._notify()
        return beam

    # ---- restraints
    def add_restraint(self, comp_name, node_name, fixed):
        self.restraints[(comp_name, node_name)] = tuple(bool(f) for f in fixed)
        self._notify()

    def remove_restraint(self, comp_name, node_name):
        self.restraints.pop((comp_name, node_name), None)
        self._notify()

    # ---- loads
    def add_point_load(self, comp_name, node_name, forces):
        f = tuple(list(forces) + [0.0] * (6 - len(forces)))
        self.point_loads.append((comp_name, node_name, f))
        self._notify()

    def remove_point_load(self, index):
        if 0 <= index < len(self.point_loads):
            self.point_loads.pop(index)
            self._notify()

    # ---- deletion
    def remove(self, name):
        self.columns.pop(name, None)
        self.beams.pop(name, None)
        self.restraints = {k: v for k, v in self.restraints.items()
                           if k[0] != name}
        self.point_loads = [pl for pl in self.point_loads if pl[0] != name]
        self._notify()

    # ---- observers
    def on_change(self, cb):
        self._listeners.append(cb)

    def _notify(self):
        for cb in list(self._listeners):
            cb()