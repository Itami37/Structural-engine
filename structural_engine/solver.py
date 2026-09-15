"""OpenSees fiber-section static analysis + fiber stress reconstruction.

Uses elasticBeamColumn elements with equivalent transformed-section
properties derived from the fiber discretization.  This is mathematically
equivalent to a linear-elastic fiber-section beam for linear static
analysis, and it avoids a transform-lookup bug in OpenSees 3.8's
dispBeamColumn element.

Fiber stresses are reconstructed from element end forces via the
linear-elastic section stiffness matrix:
    [N ]   [S00 S01 S02] [eps0]
    [Mz] = [S10 S11 S12] [ a  ]
    [My]   [S20 S21 S22] [ b  ]
    strain(y,z) = eps0 + a*y + b*z ; stress_i = E_i * strain(y_i, z_i)
"""
from __future__ import annotations
from dataclasses import dataclass, field
import math
import os
import sys

import numpy as np

# Windows: openseespy ships its runtime DLLs in a sibling 'lib' folder that
# Python's loader does not search by default.  Register it before importing.
if sys.platform == "win32":
    _opensees_lib = os.path.join(
        sys.prefix, "Lib", "site-packages", "openseespywin", "lib")
    if os.path.isdir(_opensees_lib):
        try:
            os.add_dll_directory(_opensees_lib)
        except Exception:
            pass

from .model import Model
from .section import FiberSection
from .rebar import longitudinal_bar_positions


@dataclass
class SolutionResult:
    node_displacements: dict = field(default_factory=dict)
    element_fiber_stress: dict = field(default_factory=dict)
    element_meta: dict = field(default_factory=dict)
    used_opensees: bool = False
    message: str = ""


# ----------------------------------------------------------------- helpers
def _section_matrix(section: FiberSection, E_c: float, E_s: float):
    fibs = section.fibers()
    S = np.zeros((3, 3))
    E_list = np.empty(len(fibs))
    for k, f in enumerate(fibs):
        E = E_c if f.material == "concrete" else E_s
        EA = E * f.area
        y, z = f.y, f.z
        S[0, 0] += EA
        S[0, 1] += EA * y
        S[0, 2] += EA * z
        S[1, 1] += EA * y * y
        S[1, 2] += EA * y * z
        S[2, 2] += EA * z * z
        E_list[k] = E
    S[1, 0] = S[0, 1]
    S[2, 0] = S[0, 2]
    S[2, 1] = S[1, 2]
    return S, fibs, E_list


def fiber_stresses_from_section_forces(section, E_c, E_s, N, Mz, My):
    """Return (stresses, fibers) for a section under axial force N,
    moment about local z (Mz) and moment about local y (My)."""
    S, fibs, E_list = _section_matrix(section, E_c, E_s)
    try:
        eps0, a, b = np.linalg.solve(S, np.array([N, Mz, My], dtype=float))
    except np.linalg.LinAlgError:
        eps0 = a = b = 0.0
    stresses = [E * (eps0 + a * f.y + b * f.z) for f, E in zip(fibs, E_list)]
    return stresses, fibs


def _equiv_props(comp):
    """Return equivalent elastic section properties for elasticBeamColumn.

    Uses transformed-section properties derived from the fiber
    discretization with linear-elastic materials, so that the beam element
    stiffness matches what a fiber-section element would produce for a
    linear analysis.
    """
    sec = FiberSection(b=comp.b.value, h=comp.h.value, rebar=comp.rebar)
    fibs = sec.fibers()
    Ec = comp.concrete.Ec
    Es = comp.steel.Es

    EA = 0.0
    Ey = 0.0
    Ez = 0.0
    for f in fibs:
        E = Ec if f.material == "concrete" else Es
        EA += E * f.area
        Ey += E * f.area * f.y
        Ez += E * f.area * f.z
    ybar = Ey / EA
    zbar = Ez / EA

    EIy = 0.0   # ∫ y_section² dA * E
    EIz = 0.0   # ∫ z_section² dA * E
    for f in fibs:
        E = Ec if f.material == "concrete" else Es
        EIy += E * f.area * (f.y - ybar) ** 2
        EIz += E * f.area * (f.z - zbar) ** 2

    A_eff = EA / Ec
    Iy_eff = EIy / Ec
    Iz_eff = EIz / Ec
    J_eff = Iy_eff + Iz_eff
    G = Ec / (2.0 * (1.0 + comp.concrete.nu))
    return A_eff, Ec, G, J_eff, Iy_eff, Iz_eff


# ------------------------------------------------------------------- entry
def solve(model: Model, use_opensees: bool = True) -> SolutionResult:
    if use_opensees:
        try:
            import openseespy.opensees  # noqa: F401
            return _solve_opensees(model)
        except ImportError:
            pass
    return _solve_mock(model)


def _solve_mock(model: Model) -> SolutionResult:
    res = SolutionResult(
        used_opensees=False,
        message="openseespy unavailable; returning zero results.")
    for col in model.columns.values():
        res.node_displacements[hash(("col", col.name))] = np.zeros(6)
    for beam in model.beams.values():
        res.node_displacements[hash(("beam", beam.name))] = np.zeros(6)
    return res


def _solve_opensees(model: Model) -> SolutionResult:
    import openseespy.opensees as ops

    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 6)

    # ---------- nodes
    node_map: dict = {}
    next_node = [1]

    def _snap(x, y, z):
        return (round(x, 6), round(y, 6), round(z, 6))

    def _get_node(x, y, z):
        k = _snap(x, y, z)
        if k not in node_map:
            node_map[k] = next_node[0]
            ops.node(next_node[0], x, y, z)
            next_node[0] += 1
        return node_map[k]

    # ---------- equivalent section properties (computed lazily, per component)
    section_objs: dict = {}    # comp_name -> (FiberSection, Ec, Es)
    section_props: dict = {}   # comp_name -> (A, E, G, J, Iy, Iz)

    def _get_props(comp):
        key = comp.name
        if key not in section_props:
            props = _equiv_props(comp)
            section_props[key] = props
            section_objs[key] = (
                FiberSection(b=comp.b.value, h=comp.h.value, rebar=comp.rebar),
                comp.concrete.Ec, comp.steel.Es)
        return section_props[key]

    # ---------- transformations
    transf_tags: dict = {}
    next_tr = [1]

    def _get_transf(vecxz):
        k = tuple(round(c, 6) for c in vecxz)
        if k not in transf_tags:
            transf_tags[k] = next_tr[0]
            ops.geomTransf("Linear", next_tr[0], *vecxz)
            next_tr[0] += 1
        return transf_tags[k]

    def _vecxz(axis):
        ax, ay, az = axis
        n = math.sqrt(ax * ax + ay * ay + az * az)
        ax, ay, az = ax / n, ay / n, az / n
        return (1.0, 0.0, 0.0) if abs(az) > 0.9 else (0.0, 0.0, 1.0)

    # ---------- create nodes (shared between columns and beams)
    for col in model.columns.values():
        bx, by, bz = col.base.value
        tx, ty, tz = col.top.value
        col._node_b = _get_node(bx, by, bz)
        col._node_t = _get_node(tx, ty, tz)

    for beam in model.beams.values():
        sx, sy, sz = beam.start.value
        ex, ey, ez = beam.end.value
        beam._node_s = _get_node(sx, sy, sz)
        beam._node_e = _get_node(ex, ey, ez)

    # ---------- elements (elasticBeamColumn with transformed props)
    next_ele = [1]
    ele_meta: dict = {}

    for col in model.columns.values():
        A, E, G, J, Iy, Iz = _get_props(col)
        tr = _get_transf((1.0, 0.0, 0.0))
        tag = next_ele[0]; next_ele[0] += 1
        ops.element("elasticBeamColumn", tag, col._node_b, col._node_t,
                    A, E, G, J, Iy, Iz, tr)
        col._ele_tag = tag
        ele_meta[tag] = (col._node_b, col._node_t, col.name, col, True)

    for beam in model.beams.values():
        A, E, G, J, Iy, Iz = _get_props(beam)
        sx, sy, sz = beam.start.value
        ex, ey, ez = beam.end.value
        tr = _get_transf(_vecxz((ex - sx, ey - sy, ez - sz)))
        tag = next_ele[0]; next_ele[0] += 1
        ops.element("elasticBeamColumn", tag, beam._node_s, beam._node_e,
                    A, E, G, J, Iy, Iz, tr)
        beam._ele_tag = tag
        ele_meta[tag] = (beam._node_s, beam._node_e, beam.name, beam, False)

    # ---------- restraints (default: fully fix every column base)
    for col in model.columns.values():
        fixed = model.restraints.get(col.name, (True,) * 6)
        ops.fix(col._node_b, *[int(f) for f in fixed])

    # ---------- loads
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    g = 9.81

    for col in model.columns.values():
        W = col.concrete.rho * col.b.value * col.h.value * col.length.value * g
        ops.load(col._node_b, 0.0, 0.0, -W / 2.0, 0.0, 0.0, 0.0)
        ops.load(col._node_t, 0.0, 0.0, -W / 2.0, 0.0, 0.0, 0.0)

    for beam in model.beams.values():
        W = (beam.concrete.rho * beam.b.value * beam.h.value
             * beam.length.value * g)
        ops.load(beam._node_s, 0.0, 0.0, -W / 2.0, 0.0, 0.0, 0.0)
        ops.load(beam._node_e, 0.0, 0.0, -W / 2.0, 0.0, 0.0, 0.0)

    for (target, node_sel, forces) in model.point_loads:
        node = None
        if target in model.columns:
            col = model.columns[target]
            node = col._node_t if node_sel == "top" else col._node_b
        elif target in model.beams:
            beam = model.beams[target]
            node = beam._node_s if node_sel == "start" else beam._node_e
        if node is not None:
            ops.load(node, *forces)

    # ---------- analysis
    ops.system("BandGeneral")
    ops.numberer("RCM")
    ops.constraints("Plain")
    ops.integrator("LoadControl", 1.0)
    ops.algorithm("Linear")
    ops.analysis("Static")

    if ops.analyze(1) != 0:
        raise RuntimeError("OpenSees analysis failed")

    # ---------- extract
    res = SolutionResult(used_opensees=True)
    res.element_meta = ele_meta

    for _key, tag in node_map.items():
        res.node_displacements[tag] = np.asarray(ops.nodeDisp(tag),
                                                 dtype=float)

    for ele_tag, (_n1, _n2, comp_name, _comp, _is_col) in ele_meta.items():
        sec, Ec, Es = section_objs[comp_name]

        # Element end forces in local coordinates
        N_i = My_i = Mz_i = 0.0
        N_j = My_j = Mz_j = 0.0
        try:
            forces = ops.eleResponse(ele_tag, 'localForces')
            if forces and len(forces) >= 12:
                # [P_i, Vy_i, Vz_i, T_i, My_i, Mz_i,
                #  P_j, Vy_j, Vz_j, T_j, My_j, Mz_j]
                N_i = -float(forces[0])
                My_i = float(forces[4])
                Mz_i = float(forces[5])
                N_j = float(forces[6])
                My_j = float(forces[10])
                Mz_j = float(forces[11])
        except Exception:
            pass

        per_ip = []
        for (N, Mz, My) in [(N_i, Mz_i, My_i), (N_j, Mz_j, My_j)]:
            stresses, fibs = fiber_stresses_from_section_forces(
                sec, Ec, Es, N, Mz, My)
            per_ip.append([(f.material, f.y, f.z, f.area, s)
                           for f, s in zip(fibs, stresses)])
        res.element_fiber_stress[ele_tag] = per_ip

    return res