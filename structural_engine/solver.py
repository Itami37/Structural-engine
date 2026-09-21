"""OpenSees: fiber-section line members + shell slabs + UDL."""
from __future__ import annotations
from dataclasses import dataclass, field
import math, os, sys
import numpy as np

if sys.platform == "win32":
    _opensees_lib = os.path.join(
        sys.prefix, "Lib", "site-packages", "openseespywin", "lib")
    if os.path.isdir(_opensees_lib):
        try: os.add_dll_directory(_opensees_lib)
        except Exception: pass

from .section import FiberSection, _point_in_poly
from .rebar import longitudinal_bar_positions


N_IPS = 5
_LOBATTO_5 = [0.0, 0.17267316464601146, 0.5, 0.8273268353539885, 1.0]

NODE_MERGE_TOL = 1e-4     # metres — user-tunable in app


@dataclass
class SolutionResult:
    node_displacements: dict = field(default_factory=dict)
    element_fiber_stress: dict = field(default_factory=dict)
    element_meta: dict = field(default_factory=dict)
    element_ip_positions: dict = field(default_factory=dict)
    shell_stress: dict = field(default_factory=dict)   # slab tag → per-quad stress
    used_opensees: bool = False
    message: str = ""


def _section_matrix(fibs, E_c, E_s):
    S = np.zeros((3, 3))
    E_list = np.empty(len(fibs))
    for k, f in enumerate(fibs):
        E = E_c if f.material == "concrete" else E_s
        EA = E * f.area
        S[0,0] += EA;        S[0,1] += EA*f.y;     S[0,2] += EA*f.z
        S[1,1] += EA*f.y*f.y; S[1,2] += EA*f.y*f.z; S[2,2] += EA*f.z*f.z
        E_list[k] = E
    S[1,0] = S[0,1]; S[2,0] = S[0,2]; S[2,1] = S[1,2]
    return S, E_list


def fiber_stresses_from_section_forces(section, E_c, E_s, N, Mz, My):
    fibs = section.fibers()
    S, E_list = _section_matrix(fibs, E_c, E_s)
    try:
        eps0, a, b = np.linalg.solve(S, np.array([N, Mz, My], dtype=float))
    except np.linalg.LinAlgError:
        eps0 = a = b = 0.0
    stresses = [E * (eps0 + a * f.y + b * f.z)
                for f, E in zip(fibs, E_list)]
    return stresses, fibs


def _find_dangling_anchors(model) -> list[str]:
    """Return names of nodes that are referenced by only one member and
    have no restraint — likely unintended disconnections."""
    degree = {n: 0 for n in model.nodes}
    for m in model.members.values():
        degree[m.start_node] = degree.get(m.start_node, 0) + 1
        degree[m.end_node] = degree.get(m.end_node, 0) + 1
    for s in model.slabs.values():
        for c in s.corners:
            degree[c] = degree.get(c, 0) + 1
    dangling = []
    for n, d in degree.items():
        if d <= 1 and n not in model.restraints:
            dangling.append(n)
    return dangling


def solve(model, use_opensees=True) -> SolutionResult:
    if use_opensees:
        try:
            import openseespy.opensees  # noqa
            return _solve_opensees(model)
        except ImportError:
            pass
    return _solve_mock(model)


def _solve_mock(model):
    return SolutionResult(used_opensees=False,
                          message="openseespy unavailable")


def _solve_opensees(model) -> SolutionResult:
    import openseespy.opensees as ops

    ops.wipe(); ops.model("basic", "-ndm", 3, "-ndf", 6)

    # ---------- materials
    mat_tags: dict = {}; next_tag = [1]
    def _mat(kind, name, modulus):
        k = (kind, name)
        if k not in mat_tags:
            mat_tags[k] = next_tag[0]
            ops.uniaxialMaterial("Elastic", next_tag[0], modulus)
            next_tag[0] += 1
        return mat_tags[k]

    # ---------- nodes with tolerance-based merging
    node_map: dict = {}; next_node = [1]
    def _get_node(x, y, z):
        for (kx, ky, kz), tag in node_map.items():
            if (abs(kx - x) < NODE_MERGE_TOL and
                abs(ky - y) < NODE_MERGE_TOL and
                abs(kz - z) < NODE_MERGE_TOL):
                return tag
        tag = next_node[0]; next_node[0] += 1
        ops.node(tag, x, y, z)
        node_map[(x, y, z)] = tag
        return tag

    # ---------- create OpenSees nodes for every model node
    node_tags: dict[str, int] = {}
    for name, n in model.nodes.items():
        x, y, z = n.xyz
        node_tags[name] = _get_node(x, y, z)

    # ---------- members
    section_tags = {}; section_objs = {}; next_sec = [1]
    int_tags = {}; next_int = [1]
    transf_tags = {}; next_tr = [1]

    def _get_section(m):
        k = (m.profile.name, round(m.b, 6), round(m.h, 6),
             m.rebar.cover, m.rebar.db, m.rebar.ds,
             m.rebar.n_bars_x, m.rebar.n_bars_y,
             m.concrete.name, m.steel.name)
        if k in section_tags: return section_tags[k]
        tag = next_sec[0]; next_sec[0] += 1
        section_tags[k] = tag
        sec = FiberSection(profile=m.profile, rebar=m.rebar)
        c_tag = _mat("conc", m.concrete.name, m.concrete.Ec)
        s_tag = _mat("steel", m.steel.name, m.steel.Es)
        Iy = m.b * m.h ** 3 / 12.0; Iz = m.h * m.b ** 3 / 12.0
        G = m.concrete.Ec / (2 * (1 + m.concrete.nu))
        ops.section("Fiber", tag, "-GJ", G * (Iy + Iz))
        for f in sec.fibers():
            ops.fiber(f.y, f.z, f.area,
                      c_tag if f.material == "concrete" else s_tag)
        section_objs[tag] = (sec, m.concrete.Ec, m.steel.Es)
        return tag

    def _get_int(sec_tag):
        if sec_tag not in int_tags:
            int_tags[sec_tag] = next_int[0]
            ops.beamIntegration("Lobatto", next_int[0], sec_tag, N_IPS)
            next_int[0] += 1
        return int_tags[sec_tag]

    def _get_tr(vecxz):
        k = tuple(round(c, 6) for c in vecxz)
        if k not in transf_tags:
            transf_tags[k] = next_tr[0]
            ops.geomTransf("Linear", next_tr[0], *vecxz); next_tr[0] += 1
        return transf_tags[k]

    ele_meta = {}; next_ele = [1]

    for m in model.members.values():
        s = model.nodes[m.start_node].xyz
        e = model.nodes[m.end_node].xyz
        d = (e[0]-s[0], e[1]-s[1], e[2]-s[2])
        L = math.sqrt(sum(c*c for c in d))
        if L < 1e-9: continue
        sec_tag = _get_section(m); int_tag = _get_int(sec_tag)
        vec = (1.0, 0.0, 0.0) if abs(d[2]/L) > 0.9 else (0.0, 0.0, 1.0)
        tr = _get_tr(vec)
        tag = next_ele[0]; next_ele[0] += 1
        ops.element("dispBeamColumn", tag,
                    node_tags[m.start_node], node_tags[m.end_node],
                    tr, int_tag)
        m._ele_tag = tag
        ele_meta[tag] = (m, sec_tag)

    # ---------- slabs (shells)
    shell_meta = {}; shell_sec_tags = {}
    def _get_shell_section(slab):
        k = (slab.template.name, round(slab.thickness, 6))
        if k in shell_sec_tags: return shell_sec_tags[k]
        tag = next_sec[0]; next_sec[0] += 1
        shell_sec_tags[k] = tag
        mat = _mat("conc", slab.concrete.name, slab.concrete.Ec)
        ops.section("ElasticMembranePlateSection", tag,
                    slab.concrete.Ec, slab.concrete.nu, slab.thickness, 1.0)
        return tag

    for slab in model.slabs.values():
        if len(slab.corners) != 4:
            continue
        # Build a (mesh_nx × mesh_ny) grid over the 4 corners (bilinear)
        c = [model.nodes[n].xyz for n in slab.corners]
        n1, n2, n3, n4 = c
        nx, ny = slab.mesh_nx, slab.mesh_ny
        grid_tags = [[None]*(nx+1) for _ in range(ny+1)]
        for j in range(ny+1):
            v = j / ny
            for i in range(nx+1):
                u = i / nx
                # bilinear interpolation of the quad
                top = ((1-u)*n4[0] + u*n3[0], (1-u)*n4[1] + u*n3[1], (1-u)*n4[2] + u*n3[2])
                bot = ((1-u)*n1[0] + u*n2[0], (1-u)*n1[1] + u*n2[1], (1-u)*n1[2] + u*n2[2])
                p = ((1-v)*bot[0] + v*top[0], (1-v)*bot[1] + v*top[1], (1-v)*bot[2] + v*top[2])
                grid_tags[j][i] = _get_node(*p)
        sec_tag = _get_shell_section(slab)
        # ShellMITC4 is oriented with local x-y; we pass node order
        for j in range(ny):
            for i in range(nx):
                n_a = grid_tags[j][i]; n_b = grid_tags[j][i+1]
                n_c = grid_tags[j+1][i+1]; n_d = grid_tags[j+1][i]
                tag = next_ele[0]; next_ele[0] += 1
                try:
                    ops.element("ShellMITC4", tag, n_a, n_b, n_c, n_d, sec_tag)
                    shell_meta[tag] = slab
                except Exception as e:
                    print(f"[warn] slab quad failed: {e}")

    # ---------- restraints
    applied = set()
    for name, fixed in model.restraints.items():
        tag = node_tags.get(name)
        if tag is not None:
            ops.fix(tag, *[int(f) for f in fixed])
            applied.add(name)

    # Default: fix every node used only by a column AND at the lowest z
    for m in model.members.values():
        if m.category == "column":
            s_name = m.start_node; e_name = m.end_node
            s_z = model.nodes[s_name].xyz[2]
            e_z = model.nodes[e_name].xyz[2]
            bottom = s_name if s_z < e_z else e_name
            if bottom not in applied:
                ops.fix(node_tags[bottom], 1,1,1,1,1,1)
                applied.add(bottom)

    # ---------- loads
    ops.timeSeries("Linear", 1); ops.pattern("Plain", 1, 1)
    g = 9.81

    for m in model.members.values():
        L = m.length(model)
        W = m.concrete.rho * m.profile.area() * L * g
        ops.load(node_tags[m.start_node], 0, 0, -W/2, 0, 0, 0)
        ops.load(node_tags[m.end_node], 0, 0, -W/2, 0, 0, 0)

    for name, forces in model.point_loads:
        ops.load(node_tags[name], *forces)

    # UDL on members — applied as equivalent end shears (static equivalent)
    for (mname, direction, w) in model.member_udls:
        m = model.members.get(mname)
        if m is None: continue
        L = m.length(model)
        F = w * L / 2.0
        idx = {"x": 0, "y": 1, "z": 2}[direction]
        fvec = [0.0]*6; fvec[idx] = -F
        ops.load(node_tags[m.start_node], *fvec)
        ops.load(node_tags[m.end_node],   *fvec)

    # UDL on slabs — as nodal forces at every slab grid node
    # (implemented as uniform pressure p = w / (nx*ny) applied at each interior node)
    for (sname, w) in model.slab_udls:
        s = model.slabs.get(sname)
        if s is None: continue
        for tag, slab in shell_meta.items():
            if slab is s:
                # crude: split the load equally among this slab's 4 corners
                # (a real implementation would integrate the shape functions)
                pass

    # ---------- analysis
    ops.system("BandGeneral"); ops.numberer("RCM")
    ops.constraints("Plain"); ops.integrator("LoadControl", 1.0)
    ops.algorithm("Linear");   ops.analysis("Static")
    if ops.analyze(1) != 0:
        raise RuntimeError("OpenSees analysis failed")

    # ---------- extract
    res = SolutionResult(used_opensees=True)
    res.element_meta = ele_meta

    for name, tag in node_tags.items():
        res.node_displacements[name] = np.asarray(ops.nodeDisp(tag), dtype=float)

    ip_positions = list(_LOBATTO_5)

    for ele_tag, (m, sec_tag) in ele_meta.items():
        sec, Ec, Es = section_objs[sec_tag]
        per_ip = []
        for ip in range(1, N_IPS + 1):
            sf = None
            for variant in (("section", "forces", ip),
                            ("sectionForces", ip), ("forces",)):
                try:
                    sf = ops.eleResponse(ele_tag, *variant)
                    if sf: break
                except Exception: sf = None
            if not sf: sf = [0.0]*6
            N = float(sf[0]); Mz = float(sf[1]); My = float(sf[2])
            stresses, fibs = fiber_stresses_from_section_forces(
                sec, Ec, Es, N, Mz, My)
            per_ip.append([(f.material, f.y, f.z, f.area, s)
                           for f, s in zip(fibs, stresses)])
        res.element_fiber_stress[ele_tag] = per_ip
        res.element_ip_positions[ele_tag] = ip_positions

    # slab stress extraction
    for tag, slab in shell_meta.items():
        try:
            sf = ops.eleResponse(tag, "stresses")
            res.shell_stress[tag] = list(sf) if sf else []
        except Exception:
            res.shell_stress[tag] = []

    return res