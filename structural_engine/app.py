"""Engine wrapper + PySide6 main window."""
from __future__ import annotations
import numpy as np

from .model import Model
from .components import Column, Beam
from .materials import Concrete, Steel
from .rebar import RebarLayout
from .solver import solve
from . import geometry as geom


DEFAULT_CONCRETE = Concrete("C30", fc=30e6, Ec=30e9)
DEFAULT_STEEL = Steel("Fe500", fy=500e6, Es=200e9)
DEFAULT_REBAR = RebarLayout(cover=0.04, db=0.020, ds=0.010,
                            n_bars_x=3, n_bars_y=3, stirrup_spacing=0.15)


class Engine:
    def __init__(self, model: Model | None = None):
        self.model = model or Model()
        self._scene = None
        self._last_result = None
        self._change_cbs: list = []
        self.model.on_change(self._on_model_change)

    # -------------------------------------------------------------- callbacks
    def on_change(self, cb):
        self._change_cbs.append(cb)

    def _on_model_change(self):
        for cb in list(self._change_cbs):
            cb()

    def attach_scene(self, scene):
        self._scene = scene
        self.refresh()

    # -------------------------------------------------------------- creators
    def add_column(self, name: str, **kw) -> Column:
        col = Column(name, concrete=DEFAULT_CONCRETE, steel=DEFAULT_STEEL,
                     rebar=DEFAULT_REBAR, **kw)
        self.model.add_column(col)
        for p in col.all_params:
            p.subscribe(self.refresh)
        col.top.subscribe(self.refresh)
        col.base.subscribe(self.refresh)
        self.refresh()
        return col

    def add_beam(self, name: str, col_a: Column, col_b: Column, **kw) -> Beam:
        beam = Beam(name, col_a, col_b, concrete=DEFAULT_CONCRETE,
                    steel=DEFAULT_STEEL, rebar=DEFAULT_REBAR, **kw)
        self.model.add_beam(beam)
        for p in beam.all_params:
            p.subscribe(self.refresh)
        beam.start.subscribe(self.refresh)
        beam.end.subscribe(self.refresh)
        beam.length.subscribe(self.refresh)
        self.refresh()
        return beam

    def delete_component(self, name: str):
        self.model.remove(name)
        self.refresh()

    # --------------------------------------------------------------- refresh
    def refresh(self, *_):
        if self._scene is None or not geom.has_occ():
            return
        self._scene.clear()
        for col in self.model.columns.values():
            try:
                v, t = geom.tessellate(geom.make_column_solid(col))
                self._scene.set_shape(f"col:{col.name}", v, t,
                                      color=(0.78, 0.78, 0.82))
            except Exception as e:
                print(f"[warn] column {col.name}: {e}")
            try:
                v, t = geom.tessellate(geom.make_rebar_cage(col, True),
                                       linear_deflection=0.010)
                self._scene.set_shape(f"rebar:{col.name}", v, t,
                                      color=(0.78, 0.35, 0.15))
            except Exception as e:
                print(f"[warn] rebar {col.name}: {e}")

        for beam in self.model.beams.values():
            try:
                v, t = geom.tessellate(geom.make_beam_solid(beam))
                self._scene.set_shape(f"beam:{beam.name}", v, t,
                                      color=(0.72, 0.72, 0.78))
            except Exception as e:
                print(f"[warn] beam {beam.name}: {e}")
            try:
                v, t = geom.tessellate(geom.make_rebar_cage(beam, False),
                                       linear_deflection=0.010)
                self._scene.set_shape(f"rebar:{beam.name}", v, t,
                                      color=(0.78, 0.35, 0.15))
            except Exception as e:
                print(f"[warn] rebar {beam.name}: {e}")

        self._scene.fit_all()

    # -------------------------------------------------------------- analysis
    def run_analysis(self, mode: str = "concrete"):
        res = solve(self.model)
        self._last_result = res
        if self._scene is not None and res.used_opensees:
            self._overlay(res, mode)
        return res

    def _member_endpoints(self, comp, is_col):
        if is_col:
            return np.asarray(comp.base.value, dtype=float), \
                   np.asarray(comp.top.value, dtype=float)
        return np.asarray(comp.start.value, dtype=float), \
               np.asarray(comp.end.value, dtype=float)

    def _overlay(self, res, mode: str):
        # Compute the global range first for a consistent colour scale
        all_vals = []
        for ele_tag, per_ip in res.element_fiber_stress.items():
            for fib_list in per_ip:
                for (m, _y, _z, _A, s) in fib_list:
                    if m == mode:
                        all_vals.append(abs(s))
        if not all_vals:
            return
        vmax = max(all_vals)
        vmin = 0.0

        for ele_tag, per_ip in res.element_fiber_stress.items():
            meta = res.element_meta[ele_tag]
            _n1, _n2, _name, comp, is_col, _sec_tag = meta
            key = f"col:{comp.name}" if is_col else f"beam:{comp.name}"
            if key not in self._scene.actors:
                continue

            ip_vals = []
            for fib_list in per_ip:
                vals = [abs(s) for (m, _y, _z, _A, s) in fib_list if m == mode]
                ip_vals.append(max(vals) if vals else 0.0)
            ip_positions = res.element_ip_positions.get(
                ele_tag, [i / max(1, len(ip_vals) - 1)
                          for i in range(len(ip_vals))])

            p_start, p_end = self._member_endpoints(comp, is_col)
            axis = p_end - p_start
            L = float(np.linalg.norm(axis))
            if L < 1e-12:
                continue
            axis /= L

            verts, tris = self._scene.raw_mesh.get(key, (None, None))
            if verts is None or len(tris) == 0:
                continue
            centroids = verts[tris].mean(axis=1)
            proj = (centroids - p_start) @ axis
            t_par = np.clip(proj / L, 0.0, 1.0)

            cell_vals = np.interp(t_par, ip_positions, ip_vals)

            self._scene.color_by_cell_scalars(
                key, cell_vals, vmin=vmin, vmax=vmax,
                label=f"{mode.capitalize()} |stress| (Pa)")


# ------------------------------------------------------------------ GUI
def run_gui():
    from PySide6 import QtWidgets, QtCore

    from .viewer import Scene, has_vtk
    if not has_vtk():
        raise ImportError("VTK not available")
    from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

    from .param import Param

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    engine = Engine()

    # -------- Scene + widget
    scene = Scene()
    vtk_widget = QVTKRenderWindowInteractor()
    rw = vtk_widget.GetRenderWindow()
    rw.AddRenderer(scene.renderer)
    scene.set_window(rw)
    engine.attach_scene(scene)

    # -------- Main window
    win = QtWidgets.QMainWindow()
    win.setWindowTitle("Structural Engine")
    central = QtWidgets.QWidget()
    hl = QtWidgets.QHBoxLayout(central)

    # -------- Side panel
    side = QtWidgets.QWidget()
    side.setFixedWidth(380)
    sv = QtWidgets.QVBoxLayout(side)

    # Add-buttons row
    btn_row = QtWidgets.QHBoxLayout()
    btn_add_col = QtWidgets.QPushButton("Add Column")
    btn_add_beam = QtWidgets.QPushButton("Add Beam")
    btn_add_restr = QtWidgets.QPushButton("Add Restraint")
    btn_add_load = QtWidgets.QPushButton("Add Point Load")
    for b in (btn_add_col, btn_add_beam, btn_add_restr, btn_add_load):
        b.setFixedHeight(28)
    btn_row.addWidget(btn_add_col)
    btn_row.addWidget(btn_add_beam)
    sv.addLayout(btn_row)
    btn_row2 = QtWidgets.QHBoxLayout()
    btn_row2.addWidget(btn_add_restr)
    btn_row2.addWidget(btn_add_load)
    sv.addLayout(btn_row2)

    # -------- Tree view (rebuilt on change)
    tree = QtWidgets.QTreeWidget()
    tree.setHeaderLabels(["Parameter", "Value"])
    tree.setColumnWidth(0, 230)

    def _add_param(parent_item, p: Param):
        it = QtWidgets.QTreeWidgetItem(
            parent_item, [p.name.split(".")[-1], str(p.value)])
        it.setData(0, QtCore.Qt.UserRole, p)
        it.setFlags(it.flags() | QtCore.Qt.ItemIsEditable)
        return it

    def _rebuild_tree():
        tree.blockSignals(True)
        tree.clear()

        # Columns
        for col in engine.model.columns.values():
            top = QtWidgets.QTreeWidgetItem(tree, [f"Column {col.name}", ""])
            for p in col.all_params:
                _add_param(top, p)
            top.setExpanded(True)

        # Beams
        for beam in engine.model.beams.values():
            top = QtWidgets.QTreeWidgetItem(tree, [f"Beam {beam.name}", ""])
            for p in beam.all_params:
                _add_param(top, p)
            top.setExpanded(True)

        # Restraints
        if engine.model.restraints:
            top = QtWidgets.QTreeWidgetItem(tree, ["Restraints", ""])
            for (comp_name, node_name), fixed in engine.model.restraints.items():
                flagstr = "".join("F" if f else "-" for f in fixed)
                QtWidgets.QTreeWidgetItem(top, [f"{comp_name}.{node_name}", flagstr])
            top.setExpanded(True)

        # Point loads
        if engine.model.point_loads:
            top = QtWidgets.QTreeWidgetItem(tree, ["Point loads", ""])
            for (comp_name, node_name, forces) in engine.model.point_loads:
                QtWidgets.QTreeWidgetItem(
                    top,
                    [f"{comp_name}.{node_name}",
                     f"F=({forces[0]:.0f},{forces[1]:.0f},{forces[2]:.0f})"])
            top.setExpanded(True)

        tree.blockSignals(False)

    def _on_item_changed(it, col):
        if col != 1:
            return
        p = it.data(0, QtCore.Qt.UserRole)
        if p is None:
            return
        try:
            new = float(it.text(1))
        except ValueError:
            it.setText(1, str(p.value))
            return
        try:
            p.value = new
        except Exception as e:
            QtWidgets.QMessageBox.warning(win, "Cannot set", str(e))
            it.setText(1, str(p.value))

    tree.itemChanged.connect(_on_item_changed)
    sv.addWidget(tree)

    engine.on_change(_rebuild_tree)

    # -------- Dialogs
    class AddColumnDialog(QtWidgets.QDialog):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setWindowTitle("Add Column")
            form = QtWidgets.QFormLayout(self)
            self.name = QtWidgets.QLineEdit(
                f"C{len(engine.model.columns) + 1}")
            self.x = QtWidgets.QDoubleSpinBox(); self.x.setRange(-1e4, 1e4)
            self.y = QtWidgets.QDoubleSpinBox(); self.y.setRange(-1e4, 1e4)
            self.z = QtWidgets.QDoubleSpinBox(); self.z.setRange(-1e4, 1e4)
            self.b = QtWidgets.QDoubleSpinBox(); self.b.setRange(0.01, 10)
            self.b.setValue(0.40)
            self.h = QtWidgets.QDoubleSpinBox(); self.h.setRange(0.01, 10)
            self.h.setValue(0.40)
            self.length = QtWidgets.QDoubleSpinBox()
            self.length.setRange(0.01, 100); self.length.setValue(3.0)
            form.addRow("Name", self.name)
            form.addRow("X (m)", self.x)
            form.addRow("Y (m)", self.y)
            form.addRow("Z (m)", self.z)
            form.addRow("Width b (m)", self.b)
            form.addRow("Depth h (m)", self.h)
            form.addRow("Length (m)", self.length)
            bb = QtWidgets.QDialogButtonBox(
                QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
            bb.accepted.connect(self.accept)
            bb.rejected.connect(self.reject)
            form.addRow(bb)

    class AddBeamDialog(QtWidgets.QDialog):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setWindowTitle("Add Beam")
            form = QtWidgets.QFormLayout(self)
            self.name = QtWidgets.QLineEdit(
                f"B{len(engine.model.beams) + 1}")
            self.col_a = QtWidgets.QComboBox()
            self.col_b = QtWidgets.QComboBox()
            for c in engine.model.columns.keys():
                self.col_a.addItem(c)
                self.col_b.addItem(c)
            if self.col_b.count() > 1:
                self.col_b.setCurrentIndex(1)
            self.b = QtWidgets.QDoubleSpinBox(); self.b.setRange(0.05, 5)
            self.b.setValue(0.30)
            self.h = QtWidgets.QDoubleSpinBox(); self.h.setRange(0.05, 5)
            self.h.setValue(0.50)
            self.z_off = QtWidgets.QDoubleSpinBox()
            self.z_off.setRange(-10, 10); self.z_off.setValue(0.0)
            form.addRow("Name", self.name)
            form.addRow("Column A", self.col_a)
            form.addRow("Column B", self.col_b)
            form.addRow("Width b (m)", self.b)
            form.addRow("Depth h (m)", self.h)
            form.addRow("Vertical offset (m)", self.z_off)
            bb = QtWidgets.QDialogButtonBox(
                QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
            bb.accepted.connect(self.accept)
            bb.rejected.connect(self.reject)
            form.addRow(bb)

    class AddRestraintDialog(QtWidgets.QDialog):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setWindowTitle("Add Restraint")
            form = QtWidgets.QFormLayout(self)
            self.comp = QtWidgets.QComboBox()
            for c in list(engine.model.columns) + list(engine.model.beams):
                self.comp.addItem(c)
            self.node = QtWidgets.QComboBox()
            self.node.addItems(["base", "top", "start", "end"])
            self.checks = {}
            for k, label in [("ux", "UX"), ("uy", "UY"), ("uz", "UZ"),
                             ("rx", "RX"), ("ry", "RY"), ("rz", "RZ")]:
                cb = QtWidgets.QCheckBox(label)
                cb.setChecked(True)
                self.checks[k] = cb
            row = QtWidgets.QHBoxLayout()
            for cb in self.checks.values():
                row.addWidget(cb)
            wrapper = QtWidgets.QWidget(); wrapper.setLayout(row)
            form.addRow("Component", self.comp)
            form.addRow("Node", self.node)
            form.addRow("Fix", wrapper)
            bb = QtWidgets.QDialogButtonBox(
                QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
            bb.accepted.connect(self.accept)
            bb.rejected.connect(self.reject)
            form.addRow(bb)

    class AddLoadDialog(QtWidgets.QDialog):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setWindowTitle("Add Point Load")
            form = QtWidgets.QFormLayout(self)
            self.comp = QtWidgets.QComboBox()
            for c in list(engine.model.columns) + list(engine.model.beams):
                self.comp.addItem(c)
            self.node = QtWidgets.QComboBox()
            self.node.addItems(["base", "top", "start", "end"])
            self.spins = {}
            for k, label in [("Fx", "Fx (N)"), ("Fy", "Fy (N)"),
                             ("Fz", "Fz (N)"), ("Mx", "Mx (N.m)"),
                             ("My", "My (N.m)"), ("Mz", "Mz (N.m)")]:
                sp = QtWidgets.QDoubleSpinBox()
                sp.setRange(-1e12, 1e12); sp.setDecimals(1)
                self.spins[k] = sp
                form.addRow(label, sp)
            form.addRow("Component", self.comp)
            form.addRow("Node", self.node)
            bb = QtWidgets.QDialogButtonBox(
                QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
            bb.accepted.connect(self.accept)
            bb.rejected.connect(self.reject)
            form.addRow(bb)

    # -------- Button handlers
    def _on_add_column():
        dlg = AddColumnDialog(win)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        name = dlg.name.text().strip()
        if not name or name in engine.model.columns:
            QtWidgets.QMessageBox.warning(win, "Invalid", "Name missing or in use.")
            return
        engine.add_column(name, x=dlg.x.value(), y=dlg.y.value(),
                          z=dlg.z.value(), b=dlg.b.value(), h=dlg.h.value(),
                          length=dlg.length.value())
        rw.Render()

    def _on_add_beam():
        dlg = AddBeamDialog(win)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        name = dlg.name.text().strip()
        if not name or name in engine.model.beams:
            QtWidgets.QMessageBox.warning(win, "Invalid", "Name missing or in use.")
            return
        if dlg.col_a.count() < 2:
            QtWidgets.QMessageBox.warning(win, "Invalid",
                                          "Need at least two columns.")
            return
        ca = engine.model.columns[dlg.col_a.currentText()]
        cb = engine.model.columns[dlg.col_b.currentText()]
        if ca is cb:
            QtWidgets.QMessageBox.warning(win, "Invalid",
                                          "Pick two different columns.")
            return
        engine.add_beam(name, ca, cb, b=dlg.b.value(), h=dlg.h.value(),
                        z_offset=dlg.z_off.value())
        rw.Render()

    def _on_add_restraint():
        dlg = AddRestraintDialog(win)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        comp = dlg.comp.currentText()
        node = dlg.node.currentText()
        # Validate node name for the chosen component
        valid = (node in ("base", "top")) if comp in engine.model.columns \
                else (node in ("start", "end"))
        if not valid:
            QtWidgets.QMessageBox.warning(win, "Invalid",
                                          f"'{node}' is not a valid node for {comp}.")
            return
        fixed = tuple(dlg.checks[k].isChecked()
                      for k in ["ux", "uy", "uz", "rx", "ry", "rz"])
        engine.model.add_restraint(comp, node, fixed)

    def _on_add_load():
        dlg = AddLoadDialog(win)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        comp = dlg.comp.currentText()
        node = dlg.node.currentText()
        valid = (node in ("base", "top")) if comp in engine.model.columns \
                else (node in ("start", "end"))
        if not valid:
            QtWidgets.QMessageBox.warning(win, "Invalid",
                                          f"'{node}' is not a valid node for {comp}.")
            return
        forces = tuple(dlg.spins[k].value()
                       for k in ["Fx", "Fy", "Fz", "Mx", "My", "Mz"])
        engine.model.add_point_load(comp, node, forces)

    btn_add_col.clicked.connect(_on_add_column)
    btn_add_beam.clicked.connect(_on_add_beam)
    btn_add_restr.clicked.connect(_on_add_restraint)
    btn_add_load.clicked.connect(_on_add_load)

    # -------- Analysis buttons
    def _make_run_btn(label, mode):
        btn = QtWidgets.QPushButton(label)
        def _go():
            try:
                res = engine.run_analysis(mode)
                if not res.used_opensees:
                    QtWidgets.QMessageBox.information(
                        win, "Analysis", f"openseespy unavailable.\n{res.message}")
                else:
                    rw.Render()
            except Exception as e:
                QtWidgets.QMessageBox.critical(win, "Analysis failed", str(e))
        btn.clicked.connect(_go)
        return btn

    sv.addWidget(_make_run_btn("Run analysis: concrete stress", "concrete"))
    sv.addWidget(_make_run_btn("Run analysis: rebar stress", "steel"))

    # -------- Initial demo model
    c1 = engine.add_column("C1", x=0.0, y=0.0, z=0.0,
                           b=0.40, h=0.40, length=3.0)
    c2 = engine.add_column("C2", x=4.0, y=0.0, z=0.0,
                           b=0.40, h=0.40, length=3.0)
    engine.add_beam("B1", c1, c2, b=0.30, h=0.50)
    engine.model.add_point_load("B1", "end", (0.0, 0.0, -50e3))
    _rebuild_tree()

    # -------- Layout
    hl.addWidget(vtk_widget, 1)
    hl.addWidget(side)
    win.setCentralWidget(central)
    win.resize(1450, 880)
    win.show()

    vtk_widget.Initialize()
    vtk_widget.Start()

    return app.exec()


if __name__ == "__main__":
    import sys
    sys.exit(run_gui())