"""python -m structural_engine -- tries GUI, falls back to headless demo."""
from __future__ import annotations
import sys


def main():
    try:
        from PySide6 import QtWidgets          # noqa: F401
        from vtkmodules.qt.QVTKRenderWindowInteractor import (  # noqa: F401
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
    from .rebar import (RebarLayout, longitudinal_bar_positions,
                        stirrup_positions)
    from .section import FiberSection
    from .components import Column, Beam
    from .model import Model
    from .solver import solve, fiber_stresses_from_section_forces

    C = Concrete("C30", fc=30e6, Ec=30e9)
    S = Steel("Fe500", fy=500e6, Es=200e9)
    R = RebarLayout(cover=0.04, db=0.020, ds=0.010,
                    n_bars_x=3, n_bars_y=3, stirrup_spacing=0.15)

    bars = longitudinal_bar_positions(0.4, 0.4, R)
    print(f"Longitudinal bars ({len(bars)}):")
    for (y, z) in bars:
        print(f"  y={y:+.4f}  z={z:+.4f}")

    st = stirrup_positions(3.0, 0.15, 0.05)
    print(f"Stirrups (L=3.0, s=0.15): {len(st)} positions, "
          f"first={st[0]:.4f}, last={st[-1]:.4f}")

    model = Model()
    c1 = model.add_column(Column("C1", x=0.0, y=0.0, z=0.0,
                                 concrete=C, steel=S, rebar=R))
    c2 = model.add_column(Column("C2", x=4.0, y=0.0, z=0.0,
                                 concrete=C, steel=S, rebar=R))
    b1 = model.add_beam(Beam("B1", c1, c2, concrete=C, steel=S, rebar=R))
    print(f"\nBeam length before move : {b1.length.value:.4f} m")

    c2.px.value = 6.0
    print(f"Beam length after C2.x=6 : {b1.length.value:.4f} m")

    sec = FiberSection(0.4, 0.4, R)
    stresses, fibs = fiber_stresses_from_section_forces(
        sec, C.Ec, S.Es, N=0.0, My=0.0, Mz=100e3)
    c_s = [s for s, f in zip(stresses, fibs) if f.material == "concrete"]
    s_s = [s for s, f in zip(stresses, fibs) if f.material == "steel"]
    print(f"\nSection reconstruction: {len(fibs)} fibers, "
          f"max|sigma_c| = {max(abs(s) for s in c_s)/1e6:.3f} MPa, "
          f"max|sigma_s| = {max(abs(s) for s in s_s)/1e6:.3f} MPa")

    res = solve(model)
    print(f"\nsolve(): used_opensees={res.used_opensees}  {res.message}")


if __name__ == "__main__":
    sys.exit(main())