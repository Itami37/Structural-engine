"""2D cross-section profiles with OCCT boolean operations.

Local section coordinates:
    X = height direction (h)
    Y = width direction  (b)
    centred on (0, 0)

A profile is a closed polygon with optional holes.  Boolean ops
(union / difference / intersection) are performed through OCCT on the
corresponding planar faces, then the resulting boundary loops are
extracted back as polygons.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import math
import numpy as np


@dataclass
class Profile:
    name: str
    outer: list[tuple[float, float]]
    holes: list[list[tuple[float, float]]] = field(default_factory=list)

    # ---------------- validation / properties
    def is_valid(self) -> tuple[bool, str]:
        if len(self.outer) < 3:
            return False, "Outer boundary has fewer than 3 vertices"
        if abs(self._signed_area(self.outer)) < 1e-12:
            return False, "Outer boundary has zero area"
        for h in self.holes:
            if len(h) < 3:
                return False, "A hole has fewer than 3 vertices"
            if abs(self._signed_area(h)) < 1e-12:
                return False, "A hole has zero area"
        return True, ""

    @staticmethod
    def _signed_area(pts):
        n = len(pts)
        s = 0.0
        for i in range(n):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % n]
            s += x1 * y2 - x2 * y1
        return 0.5 * s

    def area(self) -> float:
        A = abs(self._signed_area(self.outer))
        for h in self.holes:
            A -= abs(self._signed_area(h))
        return A

    def perimeter(self) -> float:
        def _plen(pts):
            s = 0.0
            n = len(pts)
            for i in range(n):
                x1, y1 = pts[i]
                x2, y2 = pts[(i + 1) % n]
                s += math.hypot(x2 - x1, y2 - y1)
            return s
        return _plen(self.outer) + sum(_plen(h) for h in self.holes)

    def bbox(self):
        xs = [p[0] for p in self.outer]
        ys = [p[1] for p in self.outer]
        return min(xs), min(ys), max(xs), max(ys)

    def width(self) -> float:
        _, y0, _, y1 = self.bbox()
        return y1 - y0

    def height(self) -> float:
        x0, _, x1, _ = self.bbox()
        return x1 - x0

    def centroid(self) -> tuple[float, float]:
        pts = self.outer
        n = len(pts)
        A = self._signed_area(pts)
        if abs(A) < 1e-15:
            return (0.0, 0.0)
        cx = cy = 0.0
        for i in range(n):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % n]
            cross = x1 * y2 - x2 * y1
            cx += (x1 + x2) * cross
            cy += (y1 + y2) * cross
        return (cx / (6 * A), cy / (6 * A))

    def clone(self, name: str) -> "Profile":
        return Profile(name=name,
                       outer=[p for p in self.outer],
                       holes=[[p for p in h] for h in self.holes])


# ------------------------------------------------ factories
def rectangle(b: float, h: float, name="Rect") -> Profile:
    hb, hh = b / 2.0, h / 2.0
    return Profile(name=name, outer=[
        (-hh, -hb), (+hh, -hb), (+hh, +hb), (-hh, +hb)])


def circle(r: float, n: int = 32, name="Circle") -> Profile:
    return Profile(name=name, outer=[
        (r * math.cos(2 * math.pi * i / n),
         r * math.sin(2 * math.pi * i / n)) for i in range(n)])


def box_tube(b_out: float, h_out: float, t: float, name="BoxTube") -> Profile:
    hb, hh = b_out / 2.0, h_out / 2.0
    ih, iv = hb - t, hh - t
    if ih <= 0 or iv <= 0:
        raise ValueError("Wall thickness too large")
    return Profile(
        name=name,
        outer=[(-hh, -hb), (+hh, -hb), (+hh, +hb), (-hh, +hb)],
        holes=[[(-iv, -ih), (+iv, -ih), (+iv, +ih), (-iv, +ih)]])


# ------------------------------------------------ OCCT boolean ops
def _profile_to_face(profile: Profile):
    """Build an OCCT planar face from a profile (X→U, Y→V, face at Z=0)."""
    from OCC.Core.gp import gp_Pnt
    from OCC.Core.BRepBuilderAPI import (BRepBuilderAPI_MakeEdge,
                                         BRepBuilderAPI_MakeWire,
                                         BRepBuilderAPI_MakeFace)

    def loop_to_wire(pts):
        pts = list(pts) + [pts[0]]
        wb = BRepBuilderAPI_MakeWire()
        for i in range(len(pts) - 1):
            x1, y1 = pts[i]; x2, y2 = pts[i + 1]
            if abs(x2 - x1) < 1e-12 and abs(y2 - y1) < 1e-12:
                continue
            wb.Add(BRepBuilderAPI_MakeEdge(
                gp_Pnt(x1, y1, 0.0), gp_Pnt(x2, y2, 0.0)).Edge())
        return wb.Wire()

    outer_wire = loop_to_wire(profile.outer)
    face = BRepBuilderAPI_MakeFace(outer_wire).Face()
    for h in profile.holes:
        inner = loop_to_wire(h)
        face = BRepBuilderAPI_MakeFace(face, inner).Face()
    return face


def _face_to_profile(face, name="Result") -> Profile:
    """Extract boundary loops (at Z ≈ 0) from an OCCT face as polygons."""
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_WIRE, TopAbs_EDGE, TopAbs_REVERSED
    from OCC.Core.TopoDS import topods
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
    from OCC.Core.GCPnts import GCPnts_QuasiUniformDeflection

    def wire_to_points(wire):
        pts = []
        exp = TopExp_Explorer(wire, TopAbs_EDGE)
        seen = set()
        while exp.More():
            edge = topods.Edge(exp.Current())
            c = BRepAdaptor_Curve(edge)
            try:
                sampler = GCPnts_QuasiUniformDeflection(c, 1e-4)
                if sampler.IsDone():
                    for i in range(1, sampler.NbPoints() + 1):
                        p = sampler.Value(i)
                        key = (round(p.X(), 6), round(p.Y(), 6))
                        if key not in seen:
                            seen.add(key)
                            pts.append((p.X(), p.Y()))
            except Exception:
                pass
            exp.Next()
        return pts

    outer = []
    holes = []
    exp = TopExp_Explorer(face, TopAbs_WIRE)
    first = True
    while exp.More():
        wire = topods.Wire(exp.Current())
        pts = wire_to_points(wire)
        if len(pts) >= 3:
            if first:
                outer = pts
                first = False
            else:
                holes.append(pts)
        exp.Next()
    return Profile(name=name, outer=outer, holes=holes)


def boolean_union(a: Profile, b: Profile, name="Union") -> Profile:
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse
    op = BRepAlgoAPI_Fuse(_profile_to_face(a), _profile_to_face(b))
    op.Build()
    if not op.IsDone():
        raise RuntimeError("OCCT union failed")
    return _face_to_profile(op.Shape(), name)


def boolean_difference(a: Profile, b: Profile, name="Difference") -> Profile:
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
    op = BRepAlgoAPI_Cut(_profile_to_face(a), _profile_to_face(b))
    op.Build()
    if not op.IsDone():
        raise RuntimeError("OCCT difference failed")
    return _face_to_profile(op.Shape(), name)


def boolean_intersection(a: Profile, b: Profile, name="Intersection") -> Profile:
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Common
    op = BRepAlgoAPI_Common(_profile_to_face(a), _profile_to_face(b))
    op.Build()
    if not op.IsDone():
        raise RuntimeError("OCCT intersection failed")
    return _face_to_profile(op.Shape(), name)


def import_profile_from_step(path: str, name="Imported") -> Profile:
    """Read a planar face from a STEP file."""
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopoDS import topods

    reader = STEPControl_Reader()
    if reader.ReadFile(path) != IFSelect_RetDone:
        raise IOError(f"Cannot read {path}")
    reader.TransferRoots()
    shape = reader.OneShape()
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    if not exp.More():
        raise ValueError("STEP file contains no faces")
    face = topods.Face(exp.Current())
    return _face_to_profile(face, name)