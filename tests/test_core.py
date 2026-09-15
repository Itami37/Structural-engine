"""Pure-Python tests for the analytical core (no OCCT/Qt/VTK needed)."""
import math
import unittest

from structural_engine.materials import Concrete, Steel
from structural_engine.rebar import (RebarLayout, longitudinal_bar_positions,
                                     stirrup_positions)
from structural_engine.section import FiberSection
from structural_engine.param import Param, Derived
from structural_engine.components import Column, Beam
from structural_engine.model import Model
from structural_engine.solver import (fiber_stresses_from_section_forces,
                                      solve)


class TestRebar(unittest.TestCase):
    def test_bar_count_3x3(self):
        L = RebarLayout(0.04, 0.02, 0.01, 3, 3, 0.15)
        self.assertEqual(len(longitudinal_bar_positions(0.4, 0.4, L)), 8)

    def test_bar_count_2x2(self):
        L = RebarLayout(0.04, 0.02, 0.01, 2, 2, 0.15)
        self.assertEqual(len(longitudinal_bar_positions(0.4, 0.4, L)), 4)

    def test_bar_count_4x3(self):
        L = RebarLayout(0.04, 0.02, 0.01, 4, 3, 0.15)
        self.assertEqual(len(longitudinal_bar_positions(0.5, 0.6, L)), 10)

    def test_bar_inside_section(self):
        L = RebarLayout(0.04, 0.02, 0.01, 3, 3, 0.15)
        for (y, z) in longitudinal_bar_positions(0.4, 0.4, L):
            self.assertLess(abs(y), 0.2)
            self.assertLess(abs(z), 0.2)

    def test_invalid_layout_raises(self):
        L = RebarLayout(0.04, 0.02, 0.01, 1, 2, 0.15)
        with self.assertRaises(ValueError):
            longitudinal_bar_positions(0.4, 0.4, L)

    def test_stirrup_count(self):
        pos = stirrup_positions(3.0, 0.15, 0.05)
        self.assertEqual(len(pos), 20)
        self.assertAlmostEqual(pos[0], 0.05, places=10)
        self.assertAlmostEqual(pos[-1], 2.95, places=10)


class TestFiberSection(unittest.TestCase):
    def test_fibers_include_steel(self):
        R = RebarLayout(0.04, 0.02, 0.01, 3, 3, 0.15)
        fibs = FiberSection(0.4, 0.4, R).fibers()
        self.assertEqual(sum(1 for f in fibs if f.material == "steel"), 8)

    def test_fiber_area_matches_gross(self):
        """Net concrete + steel must equal the gross rectangular area
        (to within a small tolerance from the fiber discretisation)."""
        R = RebarLayout(0.04, 0.02, 0.01, 3, 3, 0.15)
        sec = FiberSection(0.4, 0.4, R)
        A_fibers = sum(f.area for f in sec.fibers())
        A_gross = 0.4 * 0.4
        # The discretisation removes a ring of bar-overlapping concrete,
        # so the fiber sum is slightly less than gross.  Tolerance ~5%.
        self.assertLess(abs(A_fibers - A_gross) / A_gross, 0.05)

    def test_elastic_equilibrium(self):
        C, S = Concrete(), Steel()
        R = RebarLayout(0.04, 0.02, 0.01, 3, 3, 0.15)
        sec = FiberSection(0.4, 0.6, R)
        N, Mz, My = 500e3, 120e3, 60e3
        stresses, fibs = fiber_stresses_from_section_forces(
            sec, C.Ec, S.Es, N, Mz, My)
        N_rec = sum(s * f.area for s, f in zip(stresses, fibs))
        Mz_rec = sum(s * f.area * f.y for s, f in zip(stresses, fibs))
        My_rec = sum(s * f.area * f.z for s, f in zip(stresses, fibs))
        self.assertLess(abs(N_rec - N) / abs(N), 1e-6)
        self.assertLess(abs(Mz_rec - Mz) / abs(Mz), 1e-6)
        self.assertLess(abs(My_rec - My) / abs(My), 1e-6)

    def test_zero_load_zero_stress(self):
        C, S = Concrete(), Steel()
        R = RebarLayout(0.04, 0.02, 0.01, 3, 3, 0.15)
        sec = FiberSection(0.4, 0.4, R)
        stresses, _ = fiber_stresses_from_section_forces(sec, C.Ec, S.Es,
                                                         0.0, 0.0, 0.0)
        for s in stresses:
            self.assertAlmostEqual(s, 0.0, places=8)


class TestMaterials(unittest.TestCase):
    def test_ft_units(self):
        """For C30, ft should be ~3.4 MPa, not ~3.4 kPa."""
        C = Concrete("C30", fc=30e6, Ec=30e9)
        self.assertGreater(C.ft, 2.5e6)
        self.assertLess(C.ft, 4.5e6)


class TestParam(unittest.TestCase):
    def test_propagation(self):
        a, b = Param("a", 1.0), Param("b", 2.0)
        c = Derived("c", lambda: a.value + b.value, [a, b])
        self.assertEqual(c.value, 3.0)
        a.value = 5.0
        self.assertEqual(c.value, 7.0)

    def test_chain(self):
        a = Param("a", 1.0)
        b = Derived("b", lambda: a.value * 2, [a])
        c = Derived("c", lambda: b.value + 1, [b])
        self.assertEqual(c.value, 3.0)
        a.value = 10.0
        self.assertEqual(c.value, 21.0)

    def test_no_fire_if_unchanged(self):
        a = Param("a", 1.0)
        fired = []
        d = Derived("d", lambda: a.value, [a])
        d.subscribe(lambda: fired.append(1))
        a.value = 1.0
        self.assertEqual(fired, [])
        a.value = 2.0
        self.assertEqual(len(fired), 1)


class TestComponents(unittest.TestCase):
    def _mk(self):
        C, S = Concrete(), Steel()
        R = RebarLayout(0.04, 0.02, 0.01, 3, 3, 0.15)
        c1 = Column("C1", x=0, y=0, z=0, length=3, concrete=C, steel=S, rebar=R)
        c2 = Column("C2", x=4, y=0, z=0, length=3, concrete=C, steel=S, rebar=R)
        b = Beam("B1", c1, c2, concrete=C, steel=S, rebar=R)
        return c1, c2, b

    def test_beam_length_updates(self):
        c1, c2, b = self._mk()
        self.assertAlmostEqual(b.length.value, 4.0, places=9)
        c2.px.value = 6.0
        self.assertAlmostEqual(b.length.value, 6.0, places=9)
        c2.pz.value = 1.0
        self.assertAlmostEqual(b.length.value, math.sqrt(36 + 1), places=9)

    def test_beam_tracks_column_height(self):
        c1, c2, b = self._mk()
        c1.length.value = 5.0
        c2.length.value = 5.0
        self.assertAlmostEqual(b.length.value, 4.0, places=9)
        self.assertAlmostEqual(b.start.value[2], 5.0, places=9)


class TestSolver(unittest.TestCase):
    def test_solve_returns_result(self):
        C, S = Concrete(), Steel()
        R = RebarLayout(0.04, 0.02, 0.01, 3, 3, 0.15)
        m = Model()
        m.add_column(Column("C1", concrete=C, steel=S, rebar=R))
        res = solve(m)
        self.assertTrue(hasattr(res, "used_opensees"))


if __name__ == "__main__":
    unittest.main()