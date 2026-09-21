"""Entry point — GUI or headless demo."""
from __future__ import annotations
import sys


def main():
    try:
        from PySide6 import QtWidgets  # noqa
        from vtkmodules.qt.QVTKRenderWindowInteractor import (  # noqa
            QVTKRenderWindowInteractor)
        from . import geometry as geom
        if not geom.has_occ():
            raise ImportError("pythonocc-core not available")
        from .app import run_gui
        return run_gui()
    except ImportError as e:
        print(f"[!] GUI dependencies missing: {e}")
        print("    Running headless demo instead.\n")
        return _headless_demo()


def _headless_demo():
    from .materials import Concrete, Steel
    from .profiles import rectangle
    from .rebar import RebarLayout
    from .templates import ComponentTemplate, COLUMN, BEAM
    from .components import Node, Member
    from .model import Model
    from .solver import solve, _find_dangling_anchors

    C = Concrete("C30", fc=30e6, Ec=30e9)
    S = Steel("Fe500", fy=500e6, Es=200e9)
    R = RebarLayout(0.04, 0.020, 0.010, 3, 3, 0.15)

    m = Model()
    m.add_template(ComponentTemplate("Col", COLUMN, rectangle(0.3, 0.3), C, S, R))
    m.add_template(ComponentTemplate("Bm", BEAM, rectangle(0.3, 0.5), C, S, R))

    m.add_node(Node("N1", 0, 0, 0))
    m.add_node(Node("N2", 0, 0, 3.0))
    m.add_node(Node("N3", 0, 5.0, 0))
    m.add_node(Node("N4", 0, 5.0, 3.0))
    m.recompute_geometry()

    m.add_member(Member("C1", m.templates.get("Col"), "N1", "N2"))
    m.add_member(Member("C2", m.templates.get("Col"), "N3", "N4"))
    m.add_member(Member("B1", m.templates.get("Bm"), "N2", "N4"))
    m.add_point_load("N4", (0.0, 0.0, -10e3, 0.0, 0.0, 0.0))
    m.add_member_udl("B1", "z", -5e3)

    print("Dangling nodes:", _find_dangling_anchors(m))
    res = solve(m)
    print(f"solve(): used_opensees={res.used_opensees}  {res.message}")


if __name__ == "__main__":
    sys.exit(main())