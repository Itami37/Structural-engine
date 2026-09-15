"""VTK scene + scalar overlay.

The Scene owns only a renderer + actor dict.  The Qt widget owns the render
window.  Raw vertex/triangle arrays are kept alongside each polydata so the
app can colour triangles individually (gradient along a member).
"""
from __future__ import annotations
import numpy as np

try:
    import vtk
    from vtk.util import numpy_support
    _HAS_VTK = True
except ImportError:
    _HAS_VTK = False


def has_vtk() -> bool:
    return _HAS_VTK


def _require_vtk():
    if not _HAS_VTK:
        raise ImportError("VTK is required for the viewer.")


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
        flat[:, 0] = 3
        flat[:, 1:] = tris
        cells.SetCells(len(tris), numpy_support.numpy_to_vtkIdTypeArray(
            flat.ravel(), deep=True))
    pd.SetPolys(cells)
    return pd


def _fill_lut(lut, cmap):
    try:
        from matplotlib import cm
        try:
            cmap_obj = cm.get_cmap(cmap)
        except Exception:
            cmap_obj = cm.get_cmap("coolwarm")
        n = lut.GetNumberOfTableValues()
        for i in range(n):
            t = i / max(1, n - 1)
            r, g, b, _ = cmap_obj(t)
            lut.SetTableValue(i, r, g, b, 1.0)
    except Exception:
        n = lut.GetNumberOfTableValues()
        for i in range(n):
            t = i / max(1, n - 1)
            if t < 0.5:
                r, g, b = 0.0, t * 2.0, 1.0 - t * 2.0
            else:
                r, g, b = (t - 0.5) * 2.0, 1.0 - (t - 0.5) * 2.0, 0.0
            lut.SetTableValue(i, r, g, b, 1.0)


class Scene:
    def __init__(self):
        _require_vtk()
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.12, 0.14, 0.18)
        self.actors: dict = {}
        self.polydata: dict = {}
        self.raw_mesh: dict = {}     # name -> (verts, tris) as numpy arrays
        self.window = None

        self.scalar_bar = vtk.vtkScalarBarActor()
        self.scalar_bar.SetVisibility(False)
        self.renderer.AddActor2D(self.scalar_bar)

    def set_window(self, rw):
        self.window = rw

    def set_shape(self, name, vertices, triangles,
                  color=(0.72, 0.72, 0.78), opacity=1.0):
        pd = build_polydata(vertices, triangles)
        self.polydata[name] = pd
        self.raw_mesh[name] = (np.asarray(vertices, dtype=float),
                               np.asarray(triangles, dtype=int))

        if name in self.actors:
            self.renderer.RemoveActor(self.actors[name])

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(pd)
        mapper.ScalarVisibilityOff()

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetOpacity(opacity)
        actor.GetProperty().SetInterpolationToFlat()
        self.renderer.AddActor(actor)
        self.actors[name] = actor

    def remove_shape(self, name):
        if name in self.actors:
            self.renderer.RemoveActor(self.actors[name])
            del self.actors[name]
        self.polydata.pop(name, None)
        self.raw_mesh.pop(name, None)

    def clear(self):
        for n in list(self.actors.keys()):
            self.remove_shape(n)
        self.scalar_bar.SetVisibility(False)

    def color_by_cell_scalars(self, name, scalars, vmin=None, vmax=None,
                              cmap="coolwarm", label="Stress (Pa)"):
        if name not in self.actors:
            return
        arr = np.asarray(scalars, dtype=float)
        if vmin is None:
            vmin = float(np.nanmin(arr))
        if vmax is None:
            vmax = float(np.nanmax(arr))
        if vmax == vmin:
            vmax = vmin + 1.0

        lut = vtk.vtkLookupTable()
        lut.SetNumberOfTableValues(256)
        lut.SetTableRange(vmin, vmax)
        _fill_lut(lut, cmap)
        lut.Build()

        mapper = self.actors[name].GetMapper()
        vtk_arr = numpy_support.numpy_to_vtk(arr, deep=True,
                                             array_type=vtk.VTK_DOUBLE)
        vtk_arr.SetName(label)
        pd = self.polydata[name]
        pd.GetCellData().SetScalars(vtk_arr)
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
        if name not in self.actors:
            return
        mapper = self.actors[name].GetMapper()
        mapper.ScalarVisibilityOff()
        self.actors[name].GetProperty().SetColor(0.72, 0.72, 0.78)
        self.scalar_bar.SetVisibility(False)

    def fit_all(self):
        self.renderer.ResetCamera()
        if self.window is not None:
            self.window.Render()