"""Core analytical tests."""
import math, unittest

from structural_engine.materials import Concrete, Steel
from structural_engine.profiles import (rectangle, circle, box_tube,
                                        Profile, boolean_union,
                                        boolean_difference)
from structural_engine.rebar import (RebarLayout, SlabRebar,
                                     longitudinal_bar_positions,
                                     stirrup_positions, slab_bar_grid)
from structural_engine.section import FiberSection
from structural_engine.param import Param, Derived
from structural_engine.templates import ComponentTemplate, COLUMN, BEAM
from structural_engine.components import Node, Member
from structural_engine.model import Model
from structural_engine.solver import (fiber_stresses_from_section_forces,
                                      solve, _find_dangling_anchors)


def _C(): return Concrete("C30", fc=30e6, Ec=30e9)
def _S(): return Steel("Fe500", fy=500e6, Es=200e9)
def _R(): return RebarLayout(0.04, 0.02, 0.01, 3, 3, 0.15)


class TestProfiles(unittest.TestCase):
    def test_rectangle(self):
        p = rectangle(0.4, 0.6)
        self.assertAlmostEqual(p.area(), 0.24, places=9)
        self.assertAlmostEqual(p.width(), 0.4, places=9)
        self.assertAlmostEqual(p.height(), 0.6, places=9)
        ok, msg = p.is_valid(); self.assertTrue(ok, msg)

    def test_circle(self):
        p = circle(0.3, n=64)
        self.assertGreater(p.area(), 0.27)
        self.assertLess(p.area(), 0.29)

    def test_box_tube(self):
        p = box_tube(0.4, 0.4, 0.05)
        self.assertGreater(p.area(), 0)
        self.assertLess(p.area(), 0.16)

    def test_union(self):
        a = rectangle(0.3, 0.3)
        b = rectangle(0.3, 0.3)
        # shift b in y so they overlap
        b.outer = [(x, y + 0.1) for (x, y) in b.outer]
        u = boolean_union(a, b, "U")
        self.assertGreater(u.area(), 0.09)


class TestRebar(unittest.TestCase):
    def test_long_bars(self):
        self.assertEqual(len(longitudinal_bar_positions(0.4, 0.4, _R())), 8)
    def test_stirrups(self):
        self.assertEqual(len(stirrup_positions(3.0, 0.15, 0.05)), 20)
    def test_slab_grid(self):
        sr = SlabRebar(0.025, 0.012, 0.15, 0.15)
        bx, by = slab_bar_grid(4.0, 3.0, sr)
        self.assertGreater(len(bx), 10)
        self.assertGreater(len(by), 10)


class TestFiberSection(unittest.TestCase):
    def test_net_area_close_to_profile(self):
        p = rectangle(0.4, 0.4)
        sec = FiberSection(p, _R())
        A = sum(f.area for f in sec.fibers())
        self.assertLess(abs(A - 0.16) / 0.16, 0.05)

    def test_elastic_equilibrium(self):
        p = rectangle(0.4, 0.6)
        sec = FiberSection(p, _R())
        N, Mz, My = 500e3, 120e3, 60e3
        stresses, fibs = fiber_stresses_from_section_forces(
            sec, _C().Ec, _S().Es, N, Mz, My)
        N_r = sum(s * f.area for s, f in zip(stresses, fibs))
        Mz_r = sum(s * f.area * f.y for s, f in zip(stresses, fibs))
        My_r = sum(s * f.area * f.z for s, f in zip(stresses, fibs))
        self.assertLess(abs(N_r - N) / abs(N), 1e-6)
        self.assertLess(abs(Mz_r - Mz) / abs(Mz), 1e-6)
        self.assertLess(abs(My_r - My) / abs(My), 1e-6)


class TestModelAndLinks(unittest.TestCase):
    def _mk(self):
        m = Model()
        m.add_template(ComponentTemplate("Col", COLUMN,
                                         rectangle(0.3, 0.3),
                                         _C(), _S(), _R()))
        m.add_template(ComponentTemplate("Bm", BEAM,
                                         rectangle(0.3, 0.5),
                                         _C(), _S(), _R()))
        m.add_node(Node("N1", 0, 0, 0))
        m.add_node(Node("N2", 0, 0, 3))
        m.add_node(Node("N3", 0, 4, 0))
        m.add_node(Node("N4", 0, 4, 3))
        m.recompute_geometry()
        m.add_member(Member("C1", m.templates.get("Col"), "N1", "N2"))
        m.add_member(Member("C2", m.templates.get("Col"), "N3", "N4"))
        m.add_member(Member("B1", m.templates.get("Bm"), "N2", "N4"))
        return m

    def test_linked_node_follows_master(self):
        m = self._mk()
        m.add_node(Node("N5", master="N4", offset_z=0.5))
        m.recompute_geometry()
        self.assertEqual(m.nodes["N5"].xyz, (0, 4, 3.5))
        # move N4
        m.nodes["N4"].y = 6.0
        m.recompute_geometry()
        self.assertEqual(m.nodes["N5"].xyz, (0, 6, 3.5))

    def test_dangling_detection(self):
        m = self._mk()
        # N4 has two members (C2 + B1), so not dangling
        self.assertNotIn("N4", _find_dangling_anchors(m))
        # add an isolated node
        m.add_node(Node("ISO", 99, 99, 99))
        self.assertIn("ISO", _find_dangling_anchors(m))


class TestSolver(unittest.TestCase):
    def test_solve_returns_result(self):
        m = Model()
        m.add_template(ComponentTemplate("Col", COLUMN,
                                         rectangle(0.3, 0.3),
                                         _C(), _S(), _R()))
        m.add_node(Node("N1", 0, 0, 0))
        m.add_node(Node("N2", 0, 0, 3))
        m.recompute_geometry()
        m.add_member(Member("C1", m.templates.get("Col"), "N1", "N2"))
        res = solve(m)
        self.assertTrue(hasattr(res, "used_opensees"))


if __name__ == "__main__":
    unittest.main()