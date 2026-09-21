"""VTK scene with translucency toggle and per-triangle gradient."""
from __future__ import annotations
import numpy as np

try:
    import vtk
    from vtk.util import numpy_support
    _HAS_VTK = True
except ImportError:
    _HAS_VTK = False


def has_vtk() -> bool: return _HAS_VTK


def _require_vtk():
    if not _HAS_VTK: raise ImportError("VTK required")


def build_polydata(vertices, triangles):
    _require_vtk()
    pd = vtk.vtkPolyData()
    pts = vtk.vtkPoints()
    pts.SetData(numpy_support.numpy_to_vtk(
        np.ascontiguousarray(vertices, dtype=np.float64), deep=True))
    pd.SetPoints(pts)
    tris = np.ascontiguousarray(triangles, dtype=np.int64)
    cells = vtk.vtkCellArray()
    if len(tris):
        flat = np.empty((len(tris), 4), dtype=np.int64)
        flat[:, 0] = 3; flat[:, 1:] = tris
        cells.SetCells(len(tris), numpy_support.numpy_to_vtkIdTypeArray(
            flat.ravel(), deep=True))
    pd.SetPolys(cells)
    return pd


def _fill_lut(lut, cmap):
    try:
        from matplotlib import cm
        try: cmap_obj = cm.get_cmap(cmap)
        except Exception: cmap_obj = cm.get_cmap("turbo")
        n = lut.GetNumberOfTableValues()
        for i in range(n):
            r, g, b, _ = cmap_obj(i / max(1, n-1))
            lut.SetTableValue(i, r, g, b, 1.0)
    except Exception:
        # fallback: turbo-ish 5-stop gradient
        stops = [(0.0, 0.19, 0.07, 0.23),
                 (0.25, 0.10, 0.60, 0.90),
                 (0.5, 0.20, 0.90, 0.40),
                 (0.75, 0.95, 0.75, 0.10),
                 (1.0, 0.95, 0.20, 0.10)]
        n = lut.GetNumberOfTableValues()
        for i in range(n):
            t = i / max(1, n-1)
            for k in range(len(stops)-1):
                t0, r0, g0, b0 = stops[k]
                t1, r1, g1, b1 = stops[k+1]
                if t0 <= t <= t1:
                    u = (t - t0) / (t1 - t0)
                    lut.SetTableValue(i,
                        r0 + u*(r1-r0), g0 + u*(g1-g0), b0 + u*(b1-b0), 1.0)
                    break


class Scene:
    def __init__(self):
        _require_vtk()
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.12, 0.14, 0.18)
        self.actors: dict = {}
        self.polydata: dict = {}
        self.raw_mesh: dict = {}
        self.base_colors: dict = {}
        self.window = None
        self.translucent = False

        self.scalar_bar = vtk.vtkScalarBarActor()
        self.scalar_bar.SetVisibility(False)
        self.renderer.AddActor2D(self.scalar_bar)

    def set_window(self, rw): self.window = rw

    def set_translucent(self, on: bool):
        self.translucent = bool(on)
        for name, actor in self.actors.items():
            if name.startswith("rebar:"):
                actor.GetProperty().SetOpacity(1.0)
            else:
                actor.GetProperty().SetOpacity(0.35 if self.translucent else 1.0)
        if self.window is not None:
            self.window.Render()

    def set_shape(self, name, verts, tris, color=(0.72, 0.72, 0.78),
                  opacity=None):
        pd = build_polydata(verts, tris)
        self.polydata[name] = pd
        self.raw_mesh[name] = (np.asarray(verts, dtype=float),
                               np.asarray(tris, dtype=int))
        self.base_colors[name] = color

        if name in self.actors:
            self.renderer.RemoveActor(self.actors[name])

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(pd)
        mapper.ScalarVisibilityOff()

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        if opacity is None:
            if self.translucent and not name.startswith("rebar:"):
                actor.GetProperty().SetOpacity(0.35)
            else:
                actor.GetProperty().SetOpacity(1.0)
        else:
            actor.GetProperty().SetOpacity(opacity)
        actor.GetProperty().SetInterpolationToFlat()
        self.renderer.AddActor(actor)
        self.actors[name] = actor

    def remove_shape(self, name):
        if name in self.actors:
            self.renderer.RemoveActor(self.actors[name]); del self.actors[name]
        self.polydata.pop(name, None); self.raw_mesh.pop(name, None)

    def clear(self):
        for n in list(self.actors.keys()): self.remove_shape(n)
        self.scalar_bar.SetVisibility(False)

    def color_by_cell_scalars(self, name, scalars,
                              vmin=None, vmax=None, cmap="turbo",
                              label="Stress (Pa)"):
        if name not in self.actors: return
        arr = np.asarray(scalars, dtype=float)
        if vmin is None: vmin = float(np.nanmin(arr))
        if vmax is None: vmax = float(np.nanmax(arr))
        if vmax == vmin: vmax = vmin + 1.0

        lut = vtk.vtkLookupTable()
        lut.SetNumberOfTableValues(256)
        lut.SetTableRange(vmin, vmax)
        _fill_lut(lut, cmap); lut.Build()

        mapper = self.actors[name].GetMapper()
        va = numpy_support.numpy_to_vtk(arr, deep=True, array_type=vtk.VTK_DOUBLE)
        va.SetName(label)
        pd = self.polydata[name]
        pd.GetCellData().SetScalars(va)
        mapper.SetScalarModeToUseCellData()
        mapper.SetLookupTable(lut)
        mapper.SetScalarRange(vmin, vmax)
        mapper.SetColorModeToMapScalars()
        mapper.ScalarVisibilityOn()
        self.actors[name].GetProperty().SetColor(1, 1, 1)

        self.scalar_bar.SetLookupTable(lut)
        self.scalar_bar.SetTitle(label)
        self.scalar_bar.SetNumberOfLabels(6)
        self.scalar_bar.SetVisibility(True)

    def clear_overlay(self, name):
        if name not in self.actors: return
        self.actors[name].GetMapper().ScalarVisibilityOff()
        c = self.base_colors.get(name, (0.72, 0.72, 0.78))
        self.actors[name].GetProperty().SetColor(*c)
        self.scalar_bar.SetVisibility(False)

    def fit_all(self):
        import math
        self.renderer.ResetCamera()
        b = self.renderer.ComputeVisiblePropBounds()
        if b[1] < b[0]:
            return
        cx = 0.5 * (b[0] + b[1])
        cy = 0.5 * (b[2] + b[3])
        cz = 0.5 * (b[4] + b[5])
        dx = b[1] - b[0]; dy = b[3] - b[2]; dz = b[5] - b[4]
        diag = math.sqrt(dx * dx + dy * dy + dz * dz)
        if diag < 1e-6:
            diag = 1.0
        cam = self.renderer.GetActiveCamera()
        cam.SetFocalPoint(cx, cy, cz)
        # Standard isometric-ish view: +X+Y+Z from the object, Z up on screen
        cam.SetPosition(cx + diag * 1.3, cy - diag * 1.3, cz + diag * 1.3)
        cam.SetViewUp(0.0, 0.0, 1.0)
        self.renderer.ResetCameraClippingRange()
        if self.window is not None:
            self.window.Render()
    def _set_camera(self, position, focal, up):
        cam = self.renderer.GetActiveCamera()
        cam.SetFocalPoint(*focal)
        cam.SetPosition(*position)
        cam.SetViewUp(*up)
        self.renderer.ResetCameraClippingRange()
        if self.window is not None:
            self.window.Render()

    def fit_all(self):
        import math
        b = self.renderer.ComputeVisiblePropBounds()
        if b[1] < b[0]:
            return
        cx = 0.5 * (b[0] + b[1])
        cy = 0.5 * (b[2] + b[3])
        cz = 0.5 * (b[4] + b[5])
        dx = b[1] - b[0]; dy = b[3] - b[2]; dz = b[5] - b[4]
        diag = math.sqrt(dx * dx + dy * dy + dz * dz) or 1.0
        self._set_camera((cx + diag * 1.3, cy - diag * 1.3, cz + diag * 1.3),
                         (cx, cy, cz), (0.0, 0.0, 1.0))

    def view_front(self):
        """Look along -Y: X is horizontal, Z is vertical."""
        import math
        b = self.renderer.ComputeVisiblePropBounds()
        if b[1] < b[0]:
            return
        cx = 0.5 * (b[0] + b[1]); cy = 0.5 * (b[2] + b[3]); cz = 0.5 * (b[4] + b[5])
        dz = b[5] - b[4] or 1.0
        self._set_camera((cx, cy - dz * 3.0, cz), (cx, cy, cz), (0.0, 0.0, 1.0))

    def view_top(self):
        """Look down from +Z."""
        b = self.renderer.ComputeVisiblePropBounds()
        if b[1] < b[0]:
            return
        cx = 0.5 * (b[0] + b[1]); cy = 0.5 * (b[2] + b[3]); cz = 0.5 * (b[4] + b[5])
        dz = b[5] - b[4] or 1.0
        self._set_camera((cx, cy, cz + dz * 3.0), (cx, cy, cz), (0.0, 1.0, 0.0))

    def view_iso(self):
        self.fit_all()