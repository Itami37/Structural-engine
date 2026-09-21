"""Engine + PySide6 GUI (Design / Model / Analyze)."""
from __future__ import annotations
import numpy as np

from .model import Model
from .templates import ComponentTemplate, SlabTemplate, COLUMN, BEAM
from .components import Node, Member, Slab
from .profiles import (Profile, rectangle, circle, box_tube,
                       boolean_union, boolean_difference)
from .materials import Concrete, Steel
from .rebar import RebarLayout, SlabRebar, longitudinal_bar_positions
from .solver import solve, _find_dangling_anchors
from . import geometry as geom


# ================================================================ Engine
class Engine:
    def __init__(self, model=None):
        self.model = model or Model()
        self._scene = None
        self._change_cbs = []
        self.model.on_change(self._on_change)

    def on_change(self, cb):
        self._change_cbs.append(cb)

    def _on_change(self):
        self.model.recompute_geometry()
        for cb in list(self._change_cbs):
            cb()

    def attach_scene(self, scene):
        self._scene = scene
        self.refresh()

    def refresh(self, *_):
        if self._scene is None or not geom.has_occ():
            return
        self.model.recompute_geometry()
        self._scene.clear()

        for m in self.model.members.values():
            try:
                v, t = geom.tessellate(geom.make_member_solid(self.model, m))
                self._scene.set_shape(f"member:{m.name}", v, t,
                                      color=(0.78, 0.78, 0.82))
            except Exception as e:
                print(f"[warn] solid {m.name}: {e}")
            try:
                cage = geom.make_member_rebar(self.model, m)
                if cage is not None:
                    v, t = geom.tessellate(cage, linear_deflection=0.010)
                    self._scene.set_shape(f"rebar:{m.name}", v, t,
                                          color=(0.85, 0.35, 0.15))
            except Exception as e:
                print(f"[warn] rebar {m.name}: {e}")

        for s in self.model.slabs.values():
            try:
                shape = geom.make_slab_solid(self.model, s)
                if shape is not None:
                    v, t = geom.tessellate(shape)
                    self._scene.set_shape(f"slab:{s.name}", v, t,
                                          color=(0.55, 0.68, 0.85))
            except Exception as e:
                print(f"[warn] slab {s.name}: {e}")

        self._scene.fit_all()

    def run_analysis(self, mode="concrete"):
        res = solve(self.model)
        if self._scene is not None and res.used_opensees:
            self._overlay(res, mode)
        return res

    def _overlay(self, res, mode):
        all_vals = []
        for per_ip in res.element_fiber_stress.values():
            for fib_list in per_ip:
                for (mat, _y, _z, _A, s) in fib_list:
                    if mat == mode:
                        all_vals.append(abs(s))
        if not all_vals:
            return
        vmax = max(all_vals)

        for ele_tag, per_ip in res.element_fiber_stress.items():
            m, _ = res.element_meta[ele_tag]
            key = f"member:{m.name}"
            if key not in self._scene.actors:
                continue
            ip_vals = []
            for fib_list in per_ip:
                vals = [abs(s) for (mat, _y, _z, _A, s) in fib_list
                        if mat == mode]
                ip_vals.append(max(vals) if vals else 0.0)
            ip_positions = res.element_ip_positions.get(
                ele_tag, [i / max(1, len(ip_vals) - 1)
                          for i in range(len(ip_vals))])

            p_start = np.array(self.model.nodes[m.start_node].xyz)
            p_end = np.array(self.model.nodes[m.end_node].xyz)
            axis = p_end - p_start
            L = float(np.linalg.norm(axis))
            if L < 1e-12:
                continue
            axis /= L

            verts, tris = self._scene.raw_mesh.get(key, (None, None))
            if verts is None or len(tris) == 0:
                continue
            centroids = verts[tris].mean(axis=1)
            t_par = np.clip((centroids - p_start) @ axis / L, 0, 1)
            cell_vals = np.interp(t_par, ip_positions, ip_vals)
            self._scene.color_by_cell_scalars(
                key, cell_vals, vmin=0.0, vmax=vmax,
                label=f"{mode.capitalize()} |stress| (Pa)")


# ================================================================ GUI
def run_gui():
    from PySide6 import QtWidgets, QtCore, QtGui
    from .viewer import Scene, has_vtk
    if not has_vtk():
        raise ImportError("VTK required")
    from vtkmodules.qt.QVTKRenderWindowInteractor import (
        QVTKRenderWindowInteractor)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    engine = Engine()

    # --- Scene + VTK widget
    scene = Scene()
    vtk_widget = QVTKRenderWindowInteractor()
    rw = vtk_widget.GetRenderWindow()
    rw.AddRenderer(scene.renderer)
    scene.set_window(rw)
    engine.attach_scene(scene)

    # --- Main window
    win = QtWidgets.QMainWindow()
    win.setWindowTitle("Structural Engine")
    central = QtWidgets.QWidget()
    hl = QtWidgets.QHBoxLayout(central)
    tabs = QtWidgets.QTabWidget()
    tabs.setFixedWidth(430)

    # ============================================================ DESIGN TAB
    design = QtWidgets.QWidget()
    dv = QtWidgets.QVBoxLayout(design)
    dv.addWidget(QtWidgets.QLabel("<b>Component templates</b>"))

    tpl_tree = QtWidgets.QTreeWidget()
    tpl_tree.setHeaderLabels(["Name", "Cat", "b×h"])
    tpl_tree.setColumnWidth(0, 150)
    dv.addWidget(tpl_tree)

    preview = QtWidgets.QLabel("(2D preview)")
    preview.setMinimumHeight(180)
    preview.setStyleSheet(
        "background:#101418;color:#ccc;border:1px solid #333;")
    preview.setAlignment(QtCore.Qt.AlignCenter)
    dv.addWidget(preview)

    def _render_preview():
        sel = tpl_tree.selectedItems()
        if not sel:
            preview.setText("(no profile selected)")
            return
        name = sel[0].data(0, QtCore.Qt.UserRole)
        if name not in engine.model.templates:
            return
        tpl = engine.model.templates.get(name)
        p = tpl.profile

        w = max(preview.width(), 260)
        h_px = max(preview.height(), 180)
        pm = QtGui.QPixmap(w, h_px)
        pm.fill(QtCore.Qt.black)
        painter = QtGui.QPainter(pm)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)

        x0, y0, x1, y1 = p.bbox()
        W = max(1e-6, x1 - x0)
        H = max(1e-6, y1 - y0)
        s = min(w / (W * 1.3), h_px / (H * 1.3))

        def _px(x_prof, y_prof):
            return (w / 2 + (y_prof - (y0 + y1) / 2) * s,
                    h_px / 2 - (x_prof - (x0 + x1) / 2) * s)

        try:
            # 1. Profile outline
            pen = QtGui.QPen(QtGui.QColor(180, 180, 190))
            pen.setWidth(2)
            painter.setPen(pen)
            painter.setBrush(QtCore.Qt.NoBrush)
            pts = [QtCore.QPointF(*_px(*v)) for v in p.outer]
            painter.drawPolygon(QtGui.QPolygonF(pts))
            for hole in p.holes:
                hp = QtGui.QPolygonF(
                    [QtCore.QPointF(*_px(*v)) for v in hole])
                painter.drawPolygon(hp)

            # 2. Stirrup cage
            rebar = tpl.rebar
            sc = rebar.cover + rebar.ds / 2.0
            hh = tpl.h / 2 - sc
            hb = tpl.b / 2 - sc
            if hh > 0 and hb > 0:
                pen = QtGui.QPen(QtGui.QColor(80, 200, 220))
                pen.setWidth(1)
                painter.setPen(pen)
                sp = [_px(-hh, -hb), _px(+hh, -hb),
                      _px(+hh, +hb), _px(-hh, +hb)]
                painter.drawPolygon(QtGui.QPolygonF(
                    [QtCore.QPointF(*pt) for pt in sp]))

            # 3. Longitudinal bars
            try:
                bars = longitudinal_bar_positions(tpl.b, tpl.h, rebar)
            except Exception:
                bars = []
            r_bar_px = max(2.5, rebar.db / 2.0 * s)
            painter.setPen(QtGui.QPen(QtGui.QColor(240, 140, 40)))
            painter.setBrush(QtGui.QColor(240, 140, 40))
            for (y_b, z_b) in bars:
                cx, cy = _px(y_b, z_b)
                painter.drawEllipse(QtCore.QPointF(cx, cy),
                                    r_bar_px, r_bar_px)
        except Exception as e:
            painter.drawText(10, 20, f"Preview error: {e}")
        finally:
            painter.end()
        preview.setPixmap(pm)

    tpl_tree.itemSelectionChanged.connect(_render_preview)

    def _refresh_tpl_tree():
        tpl_tree.clear()
        for t in engine.model.templates.all():
            it = QtWidgets.QTreeWidgetItem(
                tpl_tree, [t.name, t.category, f"{t.b:.2f}×{t.h:.2f}"])
            it.setData(0, QtCore.Qt.UserRole, t.name)
        tpl_tree.expandAll()
        if tpl_tree.topLevelItemCount() > 0:
            tpl_tree.setCurrentItem(tpl_tree.topLevelItem(0))

    class TemplateDialog(QtWidgets.QDialog):
        def __init__(self, parent=None, existing=None):
            super().__init__(parent)
            self.setWindowTitle(
                "Edit template" if existing else "New template")
            form = QtWidgets.QFormLayout(self)
            self.name = QtWidgets.QLineEdit(
                existing.name if existing else "")
            self.category = QtWidgets.QComboBox()
            self.category.addItems([COLUMN, BEAM])
            if existing:
                self.category.setCurrentText(existing.category)
                self.category.setEnabled(False)
            self.shape = QtWidgets.QComboBox()
            self.shape.addItems(["rectangle", "circle", "box tube"])
            self.b = QtWidgets.QDoubleSpinBox()
            self.b.setRange(0.05, 5); self.b.setValue(0.4)
            self.b.setSingleStep(0.05); self.b.setDecimals(3)
            self.h = QtWidgets.QDoubleSpinBox()
            self.h.setRange(0.05, 5); self.h.setValue(0.4)
            self.h.setSingleStep(0.05); self.h.setDecimals(3)
            self.t = QtWidgets.QDoubleSpinBox()
            self.t.setRange(0.005, 0.5); self.t.setValue(0.02)
            self.t.setSingleStep(0.005); self.t.setDecimals(3)
            self.conc_fc = QtWidgets.QDoubleSpinBox()
            self.conc_fc.setRange(10, 200); self.conc_fc.setValue(30)
            self.conc_fc.setSuffix(" MPa")
            self.conc_Ec = QtWidgets.QDoubleSpinBox()
            self.conc_Ec.setRange(5, 100); self.conc_Ec.setValue(30)
            self.conc_Ec.setSuffix(" GPa")
            self.st_fy = QtWidgets.QDoubleSpinBox()
            self.st_fy.setRange(200, 1000); self.st_fy.setValue(500)
            self.st_fy.setSuffix(" MPa")
            self.st_Es = QtWidgets.QDoubleSpinBox()
            self.st_Es.setRange(100, 300); self.st_Es.setValue(200)
            self.st_Es.setSuffix(" GPa")
            self.cover = QtWidgets.QDoubleSpinBox()
            self.cover.setRange(0.01, 0.1); self.cover.setDecimals(3)
            self.cover.setValue(0.04)
            self.db = QtWidgets.QDoubleSpinBox()
            self.db.setRange(0.006, 0.05); self.db.setDecimals(3)
            self.db.setValue(0.02)
            self.ds = QtWidgets.QDoubleSpinBox()
            self.ds.setRange(0.004, 0.03); self.ds.setDecimals(3)
            self.ds.setValue(0.01)
            self.nx = QtWidgets.QSpinBox()
            self.nx.setRange(2, 20); self.nx.setValue(3)
            self.ny = QtWidgets.QSpinBox()
            self.ny.setRange(2, 20); self.ny.setValue(3)
            self.spacing = QtWidgets.QDoubleSpinBox()
            self.spacing.setRange(0.05, 0.5); self.spacing.setDecimals(3)
            self.spacing.setValue(0.15)
            if existing:
                self.shape.setCurrentIndex(0)
                self.b.setValue(existing.b)
                self.h.setValue(existing.h)
                self.conc_fc.setValue(existing.concrete.fc / 1e6)
                self.conc_Ec.setValue(existing.concrete.Ec / 1e9)
                self.st_fy.setValue(existing.steel.fy / 1e6)
                self.st_Es.setValue(existing.steel.Es / 1e9)
                self.cover.setValue(existing.rebar.cover)
                self.db.setValue(existing.rebar.db)
                self.ds.setValue(existing.rebar.ds)
                self.nx.setValue(existing.rebar.n_bars_x)
                self.ny.setValue(existing.rebar.n_bars_y)
                self.spacing.setValue(existing.rebar.stirrup_spacing)
            for lbl, w in [("Name", self.name), ("Category", self.category),
                           ("Profile shape", self.shape),
                           ("b (m)", self.b), ("h (m)", self.h),
                           ("Wall t (m)", self.t),
                           ("Concrete f'c", self.conc_fc),
                           ("Concrete Ec", self.conc_Ec),
                           ("Steel fy", self.st_fy),
                           ("Steel Es", self.st_Es),
                           ("Cover", self.cover),
                           ("Long. bar Ø", self.db),
                           ("Stirrup Ø", self.ds),
                           ("Bars top/bottom", self.nx),
                           ("Bars left/right", self.ny),
                           ("Stirrup spacing", self.spacing)]:
                form.addRow(lbl, w)
            bb = QtWidgets.QDialogButtonBox(
                QtWidgets.QDialogButtonBox.Ok
                | QtWidgets.QDialogButtonBox.Cancel)
            bb.accepted.connect(self.accept)
            bb.rejected.connect(self.reject)
            form.addRow(bb)

        def to_template(self):
            sh = self.shape.currentText()
            if sh == "rectangle":
                p = rectangle(self.b.value(), self.h.value())
            elif sh == "circle":
                p = circle(self.b.value() / 2.0)
            else:
                p = box_tube(self.b.value(), self.h.value(), self.t.value())
            return ComponentTemplate(
                name=self.name.text().strip(),
                category=self.category.currentText(),
                profile=p,
                concrete=Concrete(name="C",
                                  fc=self.conc_fc.value() * 1e6,
                                  Ec=self.conc_Ec.value() * 1e9),
                steel=Steel(name="S",
                            fy=self.st_fy.value() * 1e6,
                            Es=self.st_Es.value() * 1e9),
                rebar=RebarLayout(cover=self.cover.value(),
                                  db=self.db.value(),
                                  ds=self.ds.value(),
                                  n_bars_x=self.nx.value(),
                                  n_bars_y=self.ny.value(),
                                  stirrup_spacing=self.spacing.value()))

    def _new_tpl():
        dlg = TemplateDialog(win)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        try:
            engine.model.add_template(dlg.to_template())
        except Exception as e:
            QtWidgets.QMessageBox.warning(win, "Error", str(e))
        _refresh_tpl_tree()

    def _edit_tpl():
        sel = tpl_tree.selectedItems()
        if not sel:
            return
        name = sel[0].data(0, QtCore.Qt.UserRole)
        tpl = engine.model.templates.get(name)
        dlg = TemplateDialog(win, existing=tpl)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        try:
            engine.model.replace_template(dlg.to_template())
            engine.refresh()
        except Exception as e:
            QtWidgets.QMessageBox.warning(win, "Error", str(e))
        _refresh_tpl_tree()

    def _del_tpl():
        sel = tpl_tree.selectedItems()
        if not sel:
            return
        name = sel[0].data(0, QtCore.Qt.UserRole)
        try:
            engine.model.remove_template(name)
        except Exception as e:
            QtWidgets.QMessageBox.warning(win, "In use", str(e))
        _refresh_tpl_tree()

    drow = QtWidgets.QHBoxLayout()
    for lbl, fn in [("New", _new_tpl), ("Edit", _edit_tpl),
                    ("Delete", _del_tpl)]:
        b = QtWidgets.QPushButton(lbl)
        b.clicked.connect(fn)
        drow.addWidget(b)
    dv.addLayout(drow)
    dv.addStretch(1)

    # ============================================================ MODEL TAB
    model_panel = QtWidgets.QWidget()
    mv = QtWidgets.QVBoxLayout(model_panel)
    mv.addWidget(QtWidgets.QLabel(
        "<b>Members &amp; slabs</b> — nodes are created automatically"))

    model_tree = QtWidgets.QTreeWidget()
    model_tree.setHeaderLabels(["Entity", "Details"])
    model_tree.setColumnWidth(0, 140)
    mv.addWidget(model_tree)

    def _refresh_model_tree():
        model_tree.clear()
        mem_root = QtWidgets.QTreeWidgetItem(model_tree, ["Members", ""])
        for m in engine.model.members.values():
            info = (f"{m.start_node}→{m.end_node} "
                    f"({m.template.name}, L={m.length(engine.model):.2f})")
            it = QtWidgets.QTreeWidgetItem(mem_root, [m.name, info])
            it.setData(0, QtCore.Qt.UserRole, ("member", m.name))
        mem_root.setExpanded(True)

        slab_root = QtWidgets.QTreeWidgetItem(model_tree, ["Slabs", ""])
        for s in engine.model.slabs.values():
            it = QtWidgets.QTreeWidgetItem(
                slab_root,
                [s.name, f"{len(s.corners)} corners, t={s.thickness:.3f}"])
            it.setData(0, QtCore.Qt.UserRole, ("slab", s.name))
        slab_root.setExpanded(True)

        node_root = QtWidgets.QTreeWidgetItem(
            model_tree, [f"Nodes ({len(engine.model.nodes)})", ""])
        for n in engine.model.nodes.values():
            it = QtWidgets.QTreeWidgetItem(node_root,
                                           [n.name, n.describe()])
            it.setData(0, QtCore.Qt.UserRole, ("node", n.name))
        node_root.setExpanded(False)

        dang = _find_dangling_anchors(engine.model)
        if dang:
            warn = QtWidgets.QTreeWidgetItem(
                model_tree,
                [f"⚠ {len(dang)} dangling node(s)", ", ".join(dang[:6])])
            warn.setForeground(0, QtCore.Qt.yellow)

    class AddMemberDialog(QtWidgets.QDialog):
        def __init__(self, parent=None, existing=None):
            super().__init__(parent)
            self.setWindowTitle("Member")
            form = QtWidgets.QFormLayout(self)
            self.name = QtWidgets.QLineEdit(
                existing.name if existing
                else f"M{len(engine.model.members) + 1}")
            self.tpl = QtWidgets.QComboBox()
            for t in engine.model.templates.all():
                self.tpl.addItem(t.name)
            if existing:
                self.tpl.setCurrentText(existing.template.name)

            self.mode_start = QtWidgets.QComboBox()
            self.mode_start.addItems(["Coordinates", "Existing node"])
            self.mode_end = QtWidgets.QComboBox()
            self.mode_end.addItems(["Coordinates", "Existing node"])

            self.sx = QtWidgets.QDoubleSpinBox()
            self.sy = QtWidgets.QDoubleSpinBox()
            self.sz = QtWidgets.QDoubleSpinBox()
            self.ex = QtWidgets.QDoubleSpinBox()
            self.ey = QtWidgets.QDoubleSpinBox()
            self.ez = QtWidgets.QDoubleSpinBox()
            for sp in (self.sx, self.sy, self.sz,
                       self.ex, self.ey, self.ez):
                sp.setRange(-1e4, 1e4)
                sp.setDecimals(4)
                sp.setSingleStep(0.1)
            self.sx.setValue(0.0); self.sy.setValue(0.0); self.sz.setValue(0.0)
            self.ex.setValue(3.0); self.ey.setValue(0.0); self.ez.setValue(0.0)

            self.node_start = QtWidgets.QComboBox()
            self.node_end = QtWidgets.QComboBox()
            for n in engine.model.nodes:
                self.node_start.addItem(n)
                self.node_end.addItem(n)

            if existing:
                ns = engine.model.nodes[existing.start_node]
                ne = engine.model.nodes[existing.end_node]
                self.sx.setValue(ns.x); self.sy.setValue(ns.y); self.sz.setValue(ns.z)
                self.ex.setValue(ne.x); self.ey.setValue(ne.y); self.ez.setValue(ne.z)

            form.addRow("Name", self.name)
            form.addRow("Template", self.tpl)
            form.addRow("— Start —", QtWidgets.QLabel(""))
            form.addRow("  Mode", self.mode_start)
            form.addRow("  X", self.sx)
            form.addRow("  Y", self.sy)
            form.addRow("  Z", self.sz)
            form.addRow("  Or pick node", self.node_start)
            form.addRow("— End —", QtWidgets.QLabel(""))
            form.addRow("  Mode", self.mode_end)
            form.addRow("  X", self.ex)
            form.addRow("  Y", self.ey)
            form.addRow("  Z", self.ez)
            form.addRow("  Or pick node", self.node_end)

            bb = QtWidgets.QDialogButtonBox(
                QtWidgets.QDialogButtonBox.Ok
                | QtWidgets.QDialogButtonBox.Cancel)
            bb.accepted.connect(self.accept)
            bb.rejected.connect(self.reject)
            form.addRow(bb)

        def resolve_start(self):
            if self.mode_start.currentIndex() == 1 and self.node_start.count():
                return self.node_start.currentText()
            return engine.model.find_or_create_node(
                self.sx.value(), self.sy.value(), self.sz.value())

        def resolve_end(self):
            if self.mode_end.currentIndex() == 1 and self.node_end.count():
                return self.node_end.currentText()
            return engine.model.find_or_create_node(
                self.ex.value(), self.ey.value(), self.ez.value())

    class SlabTemplateDialog(QtWidgets.QDialog):
        def __init__(self, parent=None, existing=None):
            super().__init__(parent)
            self.setWindowTitle("Slab template")
            form = QtWidgets.QFormLayout(self)
            self.name = QtWidgets.QLineEdit(
                existing.name if existing else "Slab1")
            self.t = QtWidgets.QDoubleSpinBox()
            self.t.setRange(0.05, 1.0); self.t.setValue(0.2)
            self.t.setSingleStep(0.02); self.t.setDecimals(3)
            self.cover = QtWidgets.QDoubleSpinBox()
            self.cover.setRange(0.01, 0.05); self.cover.setDecimals(3)
            self.cover.setValue(0.025)
            self.db = QtWidgets.QDoubleSpinBox()
            self.db.setRange(0.006, 0.025); self.db.setDecimals(3)
            self.db.setValue(0.012)
            self.sx = QtWidgets.QDoubleSpinBox()
            self.sx.setRange(0.05, 0.5); self.sx.setDecimals(3)
            self.sx.setValue(0.15)
            self.sy = QtWidgets.QDoubleSpinBox()
            self.sy.setRange(0.05, 0.5); self.sy.setDecimals(3)
            self.sy.setValue(0.15)
            if existing:
                self.name.setText(existing.name)
                self.t.setValue(existing.thickness)
                self.cover.setValue(existing.rebar.cover)
                self.db.setValue(existing.rebar.bar_dia)
                self.sx.setValue(existing.rebar.spacing_x)
                self.sy.setValue(existing.rebar.spacing_y)
            for lbl, w in [("Name", self.name), ("Thickness", self.t),
                           ("Cover", self.cover), ("Bar Ø", self.db),
                           ("Spacing X", self.sx), ("Spacing Y", self.sy)]:
                form.addRow(lbl, w)
            bb = QtWidgets.QDialogButtonBox(
                QtWidgets.QDialogButtonBox.Ok
                | QtWidgets.QDialogButtonBox.Cancel)
            bb.accepted.connect(self.accept)
            bb.rejected.connect(self.reject)
            form.addRow(bb)

        def to_template(self):
            return SlabTemplate(
                name=self.name.text().strip(),
                thickness=self.t.value(),
                concrete=Concrete("C30", fc=30e6, Ec=30e9),
                steel=Steel("Fe500", fy=500e6, Es=200e9),
                rebar=SlabRebar(cover=self.cover.value(),
                                bar_dia=self.db.value(),
                                spacing_x=self.sx.value(),
                                spacing_y=self.sy.value()))

    class AddSlabDialog(QtWidgets.QDialog):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setWindowTitle("Slab")
            form = QtWidgets.QFormLayout(self)
            self.name = QtWidgets.QLineEdit(
                f"S{len(engine.model.slabs) + 1}")
            self.tpl = QtWidgets.QComboBox()
            for t in engine.model.templates.all_slabs():
                self.tpl.addItem(t.name)

            self.corner_spins = []
            for i in range(4):
                row = QtWidgets.QHBoxLayout()
                x = QtWidgets.QDoubleSpinBox(); x.setRange(-1e4, 1e4)
                x.setDecimals(4); x.setSingleStep(0.1)
                y = QtWidgets.QDoubleSpinBox(); y.setRange(-1e4, 1e4)
                y.setDecimals(4); y.setSingleStep(0.1)
                z = QtWidgets.QDoubleSpinBox(); z.setRange(-1e4, 1e4)
                z.setDecimals(4); z.setSingleStep(0.1)
                row.addWidget(QtWidgets.QLabel("X")); row.addWidget(x)
                row.addWidget(QtWidgets.QLabel("Y")); row.addWidget(y)
                row.addWidget(QtWidgets.QLabel("Z")); row.addWidget(z)
                w = QtWidgets.QWidget(); w.setLayout(row)
                form.addRow(f"Corner {i + 1}", w)
                self.corner_spins.append((x, y, z))

            self.corner_spins[0][0].setValue(0); self.corner_spins[0][1].setValue(0); self.corner_spins[0][2].setValue(3)
            self.corner_spins[1][0].setValue(3); self.corner_spins[1][1].setValue(0); self.corner_spins[1][2].setValue(3)
            self.corner_spins[2][0].setValue(3); self.corner_spins[2][1].setValue(3); self.corner_spins[2][2].setValue(3)
            self.corner_spins[3][0].setValue(0); self.corner_spins[3][1].setValue(3); self.corner_spins[3][2].setValue(3)

            self.nx = QtWidgets.QSpinBox()
            self.nx.setRange(2, 40); self.nx.setValue(6)
            self.ny = QtWidgets.QSpinBox()
            self.ny.setRange(2, 40); self.ny.setValue(6)
            form.addRow("Name", self.name)
            form.addRow("Template", self.tpl)
            form.addRow("Mesh nx", self.nx)
            form.addRow("Mesh ny", self.ny)

            bb = QtWidgets.QDialogButtonBox(
                QtWidgets.QDialogButtonBox.Ok
                | QtWidgets.QDialogButtonBox.Cancel)
            bb.accepted.connect(self.accept)
            bb.rejected.connect(self.reject)
            form.addRow(bb)

        def resolve_corners(self):
            names = []
            for (x, y, z) in self.corner_spins:
                names.append(engine.model.find_or_create_node(
                    x.value(), y.value(), z.value()))
            return names

    def _add_member():
        if not engine.model.templates.names():
            QtWidgets.QMessageBox.warning(
                win, "No templates", "Design a template first.")
            return
        dlg = AddMemberDialog(win)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        tpl = engine.model.templates.get(dlg.tpl.currentText())
        try:
            s_name = dlg.resolve_start()
            e_name = dlg.resolve_end()
            if s_name == e_name:
                QtWidgets.QMessageBox.warning(
                    win, "Invalid", "Start and end nodes are the same.")
                return
            m = Member(dlg.name.text().strip(), tpl, s_name, e_name)
            engine.model.add_member(m)
        except Exception as e:
            QtWidgets.QMessageBox.warning(win, "Error", str(e))
        engine.refresh()
        _refresh_model_tree()
        rw.Render()

    def _edit_member():
        sel = model_tree.selectedItems()
        if not sel:
            return
        kind, name = sel[0].data(0, QtCore.Qt.UserRole) or (None, None)
        if kind != "member":
            return
        m = engine.model.members[name]
        dlg = AddMemberDialog(win, existing=m)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        try:
            m.template = engine.model.templates.get(dlg.tpl.currentText())
            m.start_node = dlg.resolve_start()
            m.end_node = dlg.resolve_end()
        except Exception as e:
            QtWidgets.QMessageBox.warning(win, "Error", str(e))
        engine.refresh()
        _refresh_model_tree()
        rw.Render()

    def _add_slab_tpl():
        dlg = SlabTemplateDialog(win)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        try:
            engine.model.add_slab_template(dlg.to_template())
        except Exception as e:
            QtWidgets.QMessageBox.warning(win, "Error", str(e))

    def _add_slab():
        if not engine.model.templates.slab_names():
            QtWidgets.QMessageBox.warning(
                win, "No slab templates",
                "Create a slab template first.")
            return
        dlg = AddSlabDialog(win)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        tpl = engine.model.templates.get_slab(dlg.tpl.currentText())
        try:
            corners = dlg.resolve_corners()
            s = Slab(dlg.name.text().strip(), tpl, corners,
                     dlg.nx.value(), dlg.ny.value())
            engine.model.add_slab(s)
        except Exception as e:
            QtWidgets.QMessageBox.warning(win, "Error", str(e))
        engine.refresh()
        _refresh_model_tree()
        rw.Render()

    def _delete_selected():
        sel = model_tree.selectedItems()
        if not sel:
            return
        kind, name = sel[0].data(0, QtCore.Qt.UserRole) or (None, None)
        if kind == "node":
            engine.model.remove_node(name)
        elif kind == "member":
            engine.model.remove_member(name)
        elif kind == "slab":
            engine.model.remove_slab(name)
        engine.refresh()
        _refresh_model_tree()
        rw.Render()

    def _add_restraint():
        if not engine.model.nodes:
            return
        node, ok = QtWidgets.QInputDialog.getItem(
            win, "Restraint", "Node:",
            list(engine.model.nodes.keys()), 0, False)
        if not ok:
            return
        engine.model.add_restraint(node, [True] * 6)
        _refresh_model_tree()

    def _add_point_load():
        if not engine.model.nodes:
            return
        node, ok = QtWidgets.QInputDialog.getItem(
            win, "Point load", "Node:",
            list(engine.model.nodes.keys()), 0, False)
        if not ok:
            return
        Fz, ok = QtWidgets.QInputDialog.getDouble(
            win, "Point load", "Fz (N):", -10000, -1e9, 1e9, 1)
        if not ok:
            return
        engine.model.add_point_load(
            node, (0.0, 0.0, Fz, 0.0, 0.0, 0.0))
        _refresh_model_tree()

    def _add_udl():
        if not engine.model.members:
            return
        mname, ok = QtWidgets.QInputDialog.getItem(
            win, "UDL", "Member:",
            list(engine.model.members.keys()), 0, False)
        if not ok:
            return
        w, ok = QtWidgets.QInputDialog.getDouble(
            win, "UDL", "w (N/m) downward:", -1000, -1e7, 1e7, 1)
        if not ok:
            return
        engine.model.add_member_udl(mname, "z", w)
        _refresh_model_tree()

    brow = QtWidgets.QHBoxLayout()
    for lbl, fn in [("Add Member", _add_member),
                    ("Edit Member", _edit_member),
                    ("Delete", _delete_selected)]:
        b = QtWidgets.QPushButton(lbl)
        b.clicked.connect(fn)
        brow.addWidget(b)
    mv.addLayout(brow)

    brow2 = QtWidgets.QHBoxLayout()
    for lbl, fn in [("Slab Tpl", _add_slab_tpl),
                    ("Add Slab", _add_slab)]:
        b = QtWidgets.QPushButton(lbl)
        b.clicked.connect(fn)
        brow2.addWidget(b)
    mv.addLayout(brow2)

    brow3 = QtWidgets.QHBoxLayout()
    for lbl, fn in [("Restraint", _add_restraint),
                    ("Point load", _add_point_load),
                    ("UDL", _add_udl)]:
        b = QtWidgets.QPushButton(lbl)
        b.clicked.connect(fn)
        brow3.addWidget(b)
    mv.addLayout(brow3)

    # ============================================================ ANALYZE TAB
    analyze_panel = QtWidgets.QWidget()
    av = QtWidgets.QVBoxLayout(analyze_panel)
    av.addWidget(QtWidgets.QLabel("<b>Analysis</b>"))

    def _run(mode):
        try:
            res = engine.run_analysis(mode)
            rw.Render()
            if not res.used_opensees:
                QtWidgets.QMessageBox.information(
                    win, "Analysis", res.message)
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                win, "Analysis failed", str(e))

    def _push_btn(label, fn):
        b = QtWidgets.QPushButton(label)
        b.clicked.connect(fn)
        return b

    av.addWidget(_push_btn("Run: concrete stress",
                           lambda: _run("concrete")))
    av.addWidget(_push_btn("Run: rebar stress",
                           lambda: _run("steel")))

    trans_cb = QtWidgets.QCheckBox("Translucent concrete")
    trans_cb.toggled.connect(scene.set_translucent)
    av.addWidget(trans_cb)
    av.addStretch(1)

    tabs.addTab(design, "Design")
    tabs.addTab(model_panel, "Model")
    tabs.addTab(analyze_panel, "Analyze")

    hl.addWidget(vtk_widget, 1)
    hl.addWidget(tabs)
    win.setCentralWidget(central)
    win.resize(1550, 920)
    win.show()

    # ---- Starter content
    engine.model.add_template(ComponentTemplate(
        "Col300", COLUMN, rectangle(0.3, 0.3),
        Concrete("C30", fc=30e6, Ec=30e9),
        Steel("Fe500", fy=500e6, Es=200e9),
        RebarLayout(0.04, 0.020, 0.010, 3, 3, 0.15)))
    engine.model.add_template(ComponentTemplate(
        "Beam300x500", BEAM, rectangle(0.3, 0.5),
        Concrete("C30", fc=30e6, Ec=30e9),
        Steel("Fe500", fy=500e6, Es=200e9),
        RebarLayout(0.04, 0.020, 0.010, 3, 2, 0.15)))
    engine.model.add_slab_template(SlabTemplate(
        "Slab200", 0.2, Concrete("C30", fc=30e6, Ec=30e9),
        Steel("Fe500", fy=500e6, Es=200e9),
        SlabRebar(0.025, 0.012, 0.15, 0.15)))

    n1 = engine.model.find_or_create_node(0.0, 0.0, 0.0, name_hint="N1")
    n2 = engine.model.find_or_create_node(0.0, 0.0, 3.0, name_hint="N2")
    n3 = engine.model.find_or_create_node(4.0, 0.0, 0.0, name_hint="N3")
    n4 = engine.model.find_or_create_node(4.0, 0.0, 3.0, name_hint="N4")
    engine.model.add_member(Member(
        "C1", engine.model.templates.get("Col300"), n1, n2))
    engine.model.add_member(Member(
        "C2", engine.model.templates.get("Col300"), n3, n4))
    engine.model.add_member(Member(
        "B1", engine.model.templates.get("Beam300x500"), n2, n4))
    engine.model.add_restraint(n1, [True] * 6)
    engine.model.add_restraint(n3, [True] * 6)

    _refresh_tpl_tree()
    _refresh_model_tree()
    engine.refresh()

    vtk_widget.Initialize()
    vtk_widget.Start()
    return app.exec()


if __name__ == "__main__":
    import sys
    sys.exit(run_gui())