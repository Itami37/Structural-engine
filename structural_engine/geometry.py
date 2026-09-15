"""OCCT geometry for columns, beams and rebar cages.

pythonocc-core is imported lazily so the analytical core works without it.
"""
from __future__ import annotations
import math

try:
    from OCC.Core.gp import (gp_Pnt, gp_Dir, gp_Ax2, gp_Ax3, gp_Trsf, gp_Vec,
                             gp_Circ)
    from OCC.Core.BRepPrimAPI import (BRepPrimAPI_MakeBox,
                                      BRepPrimAPI_MakeCylinder)
    from OCC.Core.BRepBuilderAPI import (BRepBuilderAPI_Transform,
                                         BRepBuilderAPI_MakeEdge,
                                         BRepBuilderAPI_MakeWire)
    from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_MakePipe
    from OCC.Core.GC import GC_MakeArcOfCircle
    from OCC.Core.TopoDS import TopoDS_Compound
    from OCC.Core.BRep import BRep_Builder, BRep_Tool
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopoDS import topods
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.TopLoc import TopLoc_Location
    _HAS_OCC = True
except ImportError:
    _HAS_OCC = False


def has_occ() -> bool:
    return _HAS_OCC


def _require_occ():
    if not _HAS_OCC:
        raise ImportError(
            "pythonocc-core is required for geometry. "
            "Install with: conda install -c conda-forge pythonocc-core"
        )


def _normalize(v):
    n = math.sqrt(sum(c * c for c in v))
    if n < 1e-15:
        raise ValueError("zero vector")
    return tuple(c / n for c in v)


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


# ------------------------------------------------------------------- frames
def _column_frame(base):
    x, y, z = base
    return gp_Ax3(gp_Pnt(x, y, z), gp_Dir(0, 1, 0), gp_Dir(0, 0, 1))


def _beam_frame(start, end):
    sx, sy, sz = start
    ex, ey, ez = end
    d = (ex - sx, ey - sy, ez - sz)
    x_dir = gp_Dir(*_normalize(d))
    ref = gp_Dir(1, 0, 0) if abs(x_dir.Z()) > 0.9 else gp_Dir(0, 0, 1)
    z_raw = gp_Vec(ref).Crossed(gp_Vec(x_dir))
    if z_raw.Magnitude() < 1e-9:
        z_raw = gp_Vec(gp_Dir(0, 1, 0)).Crossed(gp_Vec(x_dir))
    return gp_Ax3(gp_Pnt(sx, sy, sz), gp_Dir(z_raw), x_dir)


# ------------------------------------------------------------------- solids
def _box_along_x(L, b, h):
    return BRepPrimAPI_MakeBox(gp_Pnt(0.0, -b / 2, -h / 2), L, b, h).Shape()


def make_column_solid(col):
    _require_occ()
    box = _box_along_x(col.length.value, col.b.value, col.h.value)
    trsf = gp_Trsf()
    trsf.SetTransformation(
        gp_Ax3(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1), gp_Dir(1, 0, 0)),
        _column_frame(col.base.value))
    return BRepBuilderAPI_Transform(box, trsf, True).Shape()


def make_beam_solid(beam):
    _require_occ()
    box = _box_along_x(beam.length.value, beam.b.value, beam.h.value)
    trsf = gp_Trsf()
    trsf.SetTransformation(
        gp_Ax3(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1), gp_Dir(1, 0, 0)),
        _beam_frame(beam.start.value, beam.end.value))
    return BRepBuilderAPI_Transform(box, trsf, True).Shape()


# ------------------------------------------------------------------- rebar
def _cyl(p1, p2, radius):
    v = gp_Vec(gp_Pnt(*p1), gp_Pnt(*p2))
    if v.Magnitude() < 1e-12:
        return None
    return BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(*p1), gp_Dir(v)),
                                    radius, v.Magnitude()).Shape()


def _stirrup_loop(origin, axis, u, v, width, height, r_bar, r_corner):
    """Closed rounded-rectangle wire swept with a circular profile."""
    ox, oy, oz = origin
    hu, hv = height / 2.0, width / 2.0
    r = min(r_corner, hu * 0.9, hv * 0.9)

    def P(a, b, c):
        return gp_Pnt(ox + a * axis[0] + b * u[0] + c * v[0],
                      oy + a * axis[1] + b * u[1] + c * v[1],
                      oz + a * axis[2] + b * u[2] + c * v[2])

    p0 = P(0.0, -hu, -hv + r)
    p1 = P(0.0, -hu,  hv - r)
    p2 = P(0.0, -hu + r, hv)
    p3 = P(0.0,  hu - r, hv)
    p4 = P(0.0,  hu,  hv - r)
    p5 = P(0.0,  hu, -hv + r)
    p6 = P(0.0,  hu - r, -hv)
    p7 = P(0.0, -hu + r, -hv)

    inv = 1.0 / math.sqrt(2.0)
    m12 = P(0.0, -hu + r * (1 - inv),  hv - r * (1 - inv))
    m34 = P(0.0,  hu - r * (1 - inv),  hv - r * (1 - inv))
    m56 = P(0.0,  hu - r * (1 - inv), -hv + r * (1 - inv))
    m70 = P(0.0, -hu + r * (1 - inv), -hv + r * (1 - inv))

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
    for e in edges:
        wb.Add(e)
    wire = wb.Wire()

    # Profile: circle in the plane normal to the wire tangent at p0
    circ = gp_Circ(gp_Ax2(p0, gp_Dir(v[0], v[1], v[2])), r_bar)
    profile_edge = BRepBuilderAPI_MakeEdge(circ).Edge()

    pipe = BRepOffsetAPI_MakePipe(wire, profile_edge)
    pipe.Build()
    return pipe.Shape()


def make_rebar_cage(component, is_column: bool):
    _require_occ()
    from .rebar import longitudinal_bar_positions, stirrup_positions

    rebar = component.rebar
    L = component.length.value
    b = component.b.value
    h = component.h.value

    if is_column:
        origin = component.base.value
        frame_axis = (0.0, 0.0, 1.0)
        frame_u = (0.0, 1.0, 0.0)
        frame_v = (1.0, 0.0, 0.0)
    else:
        origin = component.start.value
        sx, sy, sz = component.start.value
        ex, ey, ez = component.end.value
        frame_axis = _normalize((ex - sx, ey - sy, ez - sz))
        ref = (1.0, 0.0, 0.0) if abs(frame_axis[2]) > 0.9 else (0.0, 0.0, 1.0)
        frame_u = _normalize(_cross(ref, frame_axis))
        frame_v = _cross(frame_axis, frame_u)

    def world(uu, vv, ww):
        return (origin[0] + ww * frame_axis[0] + uu * frame_u[0] + vv * frame_v[0],
                origin[1] + ww * frame_axis[1] + uu * frame_u[1] + vv * frame_v[1],
                origin[2] + ww * frame_axis[2] + uu * frame_u[2] + vv * frame_v[2])

    builder = BRep_Builder()
    compound = TopoDS_Compound()
    builder.MakeCompound(compound)

    bar_r = rebar.db / 2.0
    for (y_c, z_c) in longitudinal_bar_positions(b, h, rebar):
        p1 = world(y_c, z_c, 0.0)
        p2 = world(y_c, z_c, L)
        c = _cyl(p1, p2, bar_r)
        if c is not None:
            builder.Add(compound, c)

    stirrup_r = rebar.ds / 2.0
    cage_b = b - 2.0 * rebar.cover - rebar.ds
    cage_h = h - 2.0 * rebar.cover - rebar.ds
    corner_r = rebar.ds * 2.0
    for w in stirrup_positions(L, rebar.stirrup_spacing, rebar.stirrup_end_offset):
        try:
            loop = _stirrup_loop(world(0.0, 0.0, w), frame_axis, frame_u,
                                 frame_v, cage_b, cage_h, stirrup_r, corner_r)
            builder.Add(compound, loop)
        except Exception:
            pass

    return compound


# --------------------------------------------------------------- tessellate
def tessellate(shape, linear_deflection: float = 0.005,
               angular_deflection: float = 0.5):
    _require_occ()
    import numpy as np
    BRepMesh_IncrementalMesh(shape, linear_deflection, False,
                             angular_deflection, True)

    verts: list[tuple[float, float, float]] = []
    tris: list[tuple[int, int, int]] = []

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
            reversed_face = (face.Orientation() == 1)
            for i in range(1, tri.NbTriangles() + 1):
                a, b, c = tri.Triangle(i).Get()
                a -= 1; b -= 1; c -= 1
                if reversed_face:
                    b, c = c, b
                tris.append((off + a, off + b, off + c))
        exp.Next()

    return (np.asarray(verts, dtype=float).reshape(-1, 3),
            np.asarray(tris, dtype=int).reshape(-1, 3))