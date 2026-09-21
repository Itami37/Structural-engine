"""OCCT geometry built by extruding a 2D Profile along the member axis.

Every member is:
    - a 2D Profile in the local Y-Z plane (section frame)
    - linearly extruded along the member's local X axis
    - transformed to world coords with a single gp_Trsf

Rebar and slab surface follow the same transform, so alignment is
guaranteed.

pythonocc-core is imported lazily.
"""
from __future__ import annotations
import math
import numpy as np

try:
    from OCC.Core.gp import (gp_Pnt, gp_Dir, gp_Ax2, gp_Ax3, gp_Trsf, gp_Vec,
                             gp_Circ, gp_Mat)
    from OCC.Core.BRepPrimAPI import (BRepPrimAPI_MakeBox,
                                      BRepPrimAPI_MakeCylinder,
                                      BRepPrimAPI_MakePrism)
    from OCC.Core.BRepBuilderAPI import (BRepBuilderAPI_Transform,
                                         BRepBuilderAPI_MakeEdge,
                                         BRepBuilderAPI_MakeWire,
                                         BRepBuilderAPI_MakeFace)
    from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_MakePipe
    from OCC.Core.GC import GC_MakeArcOfCircle
    from OCC.Core.TopoDS import TopoDS_Compound, topods
    from OCC.Core.BRep import BRep_Builder, BRep_Tool
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.TopLoc import TopLoc_Location
    from OCC.Core.BRepCheck import BRepCheck_Analyzer
    _HAS_OCC = True
except ImportError:
    _HAS_OCC = False


def has_occ() -> bool:
    return _HAS_OCC


def _require_occ():
    if not _HAS_OCC:
        raise ImportError("pythonocc-core is required for geometry")


# ---------------------------------------------------------------- frames
def _local_to_world_trsf(origin, X, Y, Z):
    """Local coords → world: world = origin + lx·X + ly·Y + lz·Z."""
    m = gp_Mat(float(X[0]), float(Y[0]), float(Z[0]),
               float(X[1]), float(Y[1]), float(Z[1]),
               float(X[2]), float(Y[2]), float(Z[2]))
    trsf = gp_Trsf()
    trsf.SetValues(
        m.Value(1, 1), m.Value(1, 2), m.Value(1, 3), float(origin[0]),
        m.Value(2, 1), m.Value(2, 2), m.Value(2, 3), float(origin[1]),
        m.Value(3, 1), m.Value(3, 2), m.Value(3, 3), float(origin[2]),
    )
    return trsf


def _member_frame(model, member):
    """Return (origin, X, Y, Z) as numpy arrays for a member."""
    origin = np.array(model.nodes[member.start_node].xyz, dtype=float)
    end = np.array(model.nodes[member.end_node].xyz, dtype=float)
    d = end - origin
    L = float(np.linalg.norm(d))
    if L < 1e-12:
        raise ValueError(f"Degenerate member '{member.name}'")
    X = d / L
    if abs(X[2]) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    else:
        ref = np.array([0.0, 0.0, 1.0])
    Y = ref - (ref @ X) * X
    Y /= np.linalg.norm(Y)
    Z = np.cross(X, Y)
    return origin, X, Y, Z, L


# ---------------------------------------------------------------- profile
def _profile_to_occ_face(profile):
    from OCC.Core.gp import gp_Pnt

    def loop(pts):
        pts = list(pts) + [pts[0]]
        wb = BRepBuilderAPI_MakeWire()
        for i in range(len(pts) - 1):
            x1, y1 = pts[i]; x2, y2 = pts[i + 1]
            if abs(x2 - x1) < 1e-12 and abs(y2 - y1) < 1e-12:
                continue
            wb.Add(BRepBuilderAPI_MakeEdge(gp_Pnt(x1, y1, 0.0),
                                           gp_Pnt(x2, y2, 0.0)).Edge())
        return wb.Wire()

    outer = loop(profile.outer)
    face = BRepBuilderAPI_MakeFace(outer).Face()
    for h in profile.holes:
        face = BRepBuilderAPI_MakeFace(face, loop(h)).Face()
    return face


def _extrude_profile_local(profile, length):
    """Extrude the profile along local X.

    The profile face is built in OCC's XY plane, with the extrusion
    going along OCC's Z.  We then apply the permutation matrix
        face X (height)  → local Y
        face Y (width)   → local Z
        face Z (extrusion) → local X
    so that the resulting solid lives in the member's local frame:
        local X = member axis
        local Y = section height direction
        local Z = section width direction
    """
    face = _profile_to_occ_face(profile)
    prism = BRepPrimAPI_MakePrism(
        face, gp_Vec(gp_Pnt(0, 0, 0), gp_Pnt(0, 0, length))).Shape()

    # Rotation matrix: (fx, fy, fz) -> (fz, fx, fy)
    trsf = gp_Trsf()
    trsf.SetValues(0.0, 0.0, 1.0, 0.0,
                   1.0, 0.0, 0.0, 0.0,
                   0.0, 1.0, 0.0, 0.0)
    return BRepBuilderAPI_Transform(prism, trsf, True).Shape()
# ---------------------------------------------------------------- members
def make_member_solid(model, member):
    _require_occ()
    origin, X, Y, Z, L = _member_frame(model, member)
    prism_local = _extrude_profile_local(member.profile, L)
    trsf = _local_to_world_trsf(origin, X, Y, Z)
    shape = BRepBuilderAPI_Transform(prism_local, trsf, True).Shape()
    if not BRepCheck_Analyzer(shape).IsValid():
        raise ValueError(f"OCCT produced an invalid shape for '{member.name}'")
    return shape


# ---------------------------------------------------------------- rebar
def _stirrup_local(x_pos, b, h, r_bar, r_corner):
    h_2, b_2 = h / 2.0, b / 2.0
    r = min(r_corner, h_2 * 0.9, b_2 * 0.9)

    def P(y, z): return gp_Pnt(x_pos, y, z)

    p0 = P(-h_2, -b_2 + r); p1 = P(-h_2,  b_2 - r)
    p2 = P(-h_2 + r, b_2);  p3 = P( h_2 - r, b_2)
    p4 = P( h_2,  b_2 - r); p5 = P( h_2, -b_2 + r)
    p6 = P( h_2 - r, -b_2); p7 = P(-h_2 + r, -b_2)
    inv = 1.0 / math.sqrt(2.0)
    m12 = P(-h_2 + r * (1 - inv),  b_2 - r * (1 - inv))
    m34 = P( h_2 - r * (1 - inv),  b_2 - r * (1 - inv))
    m56 = P( h_2 - r * (1 - inv), -b_2 + r * (1 - inv))
    m70 = P(-h_2 + r * (1 - inv), -b_2 + r * (1 - inv))

    edges = [
        BRepBuilderAPI_MakeEdge(p0, p1).Edge(),
        BRepBuilderAPI_MakeEdge(GC_MakeArcOfCircle(p1, m12, p2).Value()).Edge(),
        BRepBuilderAPI_MakeEdge(p2, p3).Edge(),
        BRepBuilderAPI_MakeEdge(GC_MakeArcOfCircle(p3, m34, p4).Value()).Edge(),
        BRepBuilderAPI_MakeEdge(p4, p5).Edge(),
        BRepBuilderAPI_MakeEdge(GC_MakeArcOfCircle(p5, m56, p6).Value()).Edge(),
        BRepBuilderAPI_MakeEdge(p6, p7).Edge(),
        BRepBuilderAPI_MakeEdge(GC_MakeArcOfCircle(p7, m70, p0).Value()).Edge(),
    ]
    wb = BRepBuilderAPI_MakeWire()
    for e in edges: wb.Add(e)
    wire = wb.Wire()
    circ = gp_Circ(gp_Ax2(p0, gp_Dir(0.0, 0.0, 1.0)), r_bar)
    profile_edge = BRepBuilderAPI_MakeEdge(circ).Edge()
    pipe = BRepOffsetAPI_MakePipe(wire, profile_edge); pipe.Build()
    return pipe.Shape()


def make_member_rebar(model, member):
    _require_occ()
    from .rebar import longitudinal_bar_positions, stirrup_positions

    try:
        origin, X, Y, Z, L = _member_frame(model, member)
    except Exception:
        return None
    trsf = _local_to_world_trsf(origin, X, Y, Z)

    rebar = member.rebar
    b = member.b
    h = member.h

    compound = TopoDS_Compound()
    builder = BRep_Builder(); builder.MakeCompound(compound)

    bar_r = rebar.db / 2.0
    for (y_c, z_c) in longitudinal_bar_positions(b, h, rebar):
        p1 = gp_Pnt(0.0, y_c, z_c); p2 = gp_Pnt(L, y_c, z_c)
        v = gp_Vec(p1, p2)
        if v.Magnitude() < 1e-12: continue
        cyl = BRepPrimAPI_MakeCylinder(
            gp_Ax2(p1, gp_Dir(v)), bar_r, v.Magnitude()).Shape()
        builder.Add(compound,
                    BRepBuilderAPI_Transform(cyl, trsf, True).Shape())

    stirrup_r = rebar.ds / 2.0
    cage_b = b - 2.0 * rebar.cover - rebar.ds
    cage_h = h - 2.0 * rebar.cover - rebar.ds
    corner_r = rebar.ds * 2.0
    for w in stirrup_positions(L, rebar.stirrup_spacing,
                               rebar.stirrup_end_offset):
        try:
            loop = _stirrup_local(w, cage_b, cage_h, stirrup_r, corner_r)
            builder.Add(compound,
                        BRepBuilderAPI_Transform(loop, trsf, True).Shape())
        except Exception:
            pass
    return compound


# ---------------------------------------------------------------- slab
def make_slab_solid(model, slab):
    _require_occ()
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakePolygon
    corners = [model.nodes[n].xyz for n in slab.corners]
    if len(corners) == 4:
        poly = BRepBuilderAPI_MakePolygon()
        for (x, y, z) in corners:
            poly.Add(gp_Pnt(x, y, z))
        poly.Close()
        face = BRepBuilderAPI_MakeFace(poly.Wire()).Face()
        # Offset in +Z (slab thickness normal assumed +Z; refine later)
        vec = gp_Vec(0.0, 0.0, slab.thickness)
        return BRepPrimAPI_MakePrism(face, vec).Shape()
    return None


# ---------------------------------------------------------------- tessellate
def tessellate(shape, linear_deflection=0.005, angular_deflection=0.5):
    _require_occ()
    BRepMesh_IncrementalMesh(shape, linear_deflection, False,
                             angular_deflection, True)
    verts, tris = [], []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = topods.Face(exp.Current())
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation(face, loc)
        if tri is not None:
            trsf = loc.Transformation()
            off = len(verts)
            for i in range(1, tri.NbNodes() + 1):
                p = tri.Node(i).Transformed(trsf)
                verts.append((p.X(), p.Y(), p.Z()))
            rev = (face.Orientation() == 1)
            for i in range(1, tri.NbTriangles() + 1):
                a, b, c = tri.Triangle(i).Get()
                a -= 1; b -= 1; c -= 1
                if rev: b, c = c, b
                tris.append((off + a, off + b, off + c))
        exp.Next()
    return (np.asarray(verts, dtype=float).reshape(-1, 3),
            np.asarray(tris, dtype=int).reshape(-1, 3))


def bbox(shape):
    v, _ = tessellate(shape)
    if len(v) == 0: return None
    return (float(v[:,0].min()), float(v[:,1].min()), float(v[:,2].min()),
            float(v[:,0].max()), float(v[:,1].max()), float(v[:,2].max()))