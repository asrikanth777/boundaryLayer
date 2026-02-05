from pathlib import Path
from paraview.simple import *
# import pandas as pd
import re
import os
import matplotlib.pyplot as plt 
import numpy as np

from paraview import servermanager
from vtk.util import numpy_support as ns
import numpy as np


paraview.simple._DisableFirstRenderCameraReset()

# function to just select the vts file in the pipeline
# maybe not most efficient but its a copy-paste from previous project
def getImport():
    sources = GetSources()
    object = []

    for key, src in list(sources.items()):
        object.append(src)

    return GroupDatasets(Input=object)

# gets vts file and does celldatatopointdata
flowfield = getImport()
cellDatatoPointData1 = CellDatatoPointData(Input=flowfield)
cellDatatoPointData1.CellDataArraytoprocess = ['H', 'M', 'T', 'p', 'rho', 'v']
renderView1 = GetActiveViewOrCreate('RenderView')
cellDatatoPointData1Display = Show(cellDatatoPointData1, renderView1)
Hide(flowfield, renderView1)
renderView1.Update()

# applys calculator filter on celldatatopointdata to get velocity vector from scalar quantities
calculator1 = Calculator(Input=cellDatatoPointData1)
calculator1.Function = 'v_X*iHat + v_Y*jHat'
calculator1Display = Show(calculator1, renderView1)
Hide(cellDatatoPointData1, renderView1)
renderView1.Update()

# compute derivatives to get gradient of velocity, most important is dv/dy
computeDerivatives1 = ComputeDerivatives(Input = calculator1)
computeDerivatives1.Vectors = ['POINTS', 'Result']
computeDerivatives1Display = Show(computeDerivatives1, renderView1)
Hide(calculator1, renderView1)
renderView1.Update()

cellDatatoPointData2 = CellDatatoPointData(Input=computeDerivatives1)
cellDatatoPointData2.CellDataArraytoprocess = ['ScalarGradient', 'VectorGradient']
renderView1 = GetActiveViewOrCreate('RenderView')
cellDatatoPointData2Display = Show(cellDatatoPointData2, renderView1)
Hide(computeDerivatives1, renderView1)
renderView1.Update()

# plots over line along stagnation point, this will be used to find inflection point
plotOverLine1 = PlotOverLine(Input=cellDatatoPointData2, Source='High Resolution Line Source')
plotOverLine1.Source.Point1 = [0.4, 0, 0]
plotOverLine1.Source.Point2 = [0.565, 0, 0]
plotOverLine1.Source.Resolution = 5000
plotOverLine1Display = Show(plotOverLine1, renderView1)

# --- make a Python View and inject script ---
pythonView1 = CreateView("PythonView")

SetActiveView(pythonView1)

# hide everything in this view except plotOverLine1
Hide(flowfield, pythonView1)
Hide(cellDatatoPointData1, pythonView1)
Hide(calculator1, pythonView1)
Hide(computeDerivatives1, pythonView1)
Hide(cellDatatoPointData2, pythonView1)

Show(plotOverLine1, pythonView1)
Render(pythonView1)

pythonView1.Script = r'''
from paraview import python_view
from paraview.numpy_support import vtk_to_numpy
import numpy as np

WINDOW_SIZE = 11
TAIL_FRAC = 0.10
VG_COMPONENT_INDEX = 4

def setup_data(view):
    view.EnableAllAttributeArrays()

def render(view, width, height):
    fig = python_view.matplotlib_figure(width, height)
    axT = fig.add_subplot(2, 1, 1)
    axG = fig.add_subplot(2, 1, 2)

    dobj = view.GetVisibleDataObjectForRendering(0)
    if dobj is None:
        axT.text(0.5, 0.5, "No visible data object", ha="center", va="center", transform=axT.transAxes)
        return python_view.figure_to_image(fig)

    pts_vtk = dobj.GetPoints()
    pd = dobj.GetPointData()

    if pts_vtk is None or pd is None:
        axT.text(0.5, 0.5, "Missing points or point data", ha="center", va="center", transform=axT.transAxes)
        return python_view.figure_to_image(fig)

    pts = vtk_to_numpy(pts_vtk.GetData())
    x = pts[:, 0]  # Points_0

    vectGrad = pd.GetArray("VectorGradient")
    T_arr = pd.GetArray("T")
    if vectGrad is None or T_arr is None:
        axT.text(0.5, 0.5, "Missing VectorGradient or T", ha="center", va="center", transform=axT.transAxes)
        return python_view.figure_to_image(fig)

    vg = vtk_to_numpy(vectGrad)
    dvdy = vg[:, VG_COMPONENT_INDEX]
    tempNP = vtk_to_numpy(T_arr)

    mask = np.isfinite(x) & np.isfinite(dvdy) & np.isfinite(tempNP)
    x = x[mask]; dvdy = dvdy[mask]; tempNP = tempNP[mask]
    if x.size < 5:
        axT.text(0.5, 0.5, "Too few valid points", ha="center", va="center", transform=axT.transAxes)
        return python_view.figure_to_image(fig)

    idx = np.argsort(x)
    x = x[idx]; dvdy = dvdy[idx]; tempNP = tempNP[idx]

    w = int(WINDOW_SIZE)
    if w < 3: w = 3
    if w % 2 == 0: w += 1

    if x.size <= w + 2:
        x_smooth = x.copy()
        dvdy_smooth = dvdy.copy()
    else:
        kernel = np.ones(w) / w
        x_smooth = np.convolve(x, kernel, mode="valid")
        dvdy_smooth = np.convolve(dvdy, kernel, mode="valid")

    final_grad = np.diff(dvdy) / np.diff(x)
    fg_smooth = np.diff(dvdy_smooth) / np.diff(x_smooth)

    x_d = x[1:]
    x_s_d = x_smooth[1:] if x_smooth.size > 1 else x_smooth
    temp_d = tempNP[1:]

    n = fg_smooth.size
    side = int((1.0 - TAIL_FRAC) * n)
    side = max(0, min(side, n-1))

    examine = x_s_d[side:]
    check = fg_smooth[side:]
    indx = int(np.argmin(np.abs(check)))

    inflect_x = float(examine[indx])
    inflect_y = float(check[indx])
    bl_thickness = float(examine[-1] - examine[indx])

    axT.set_title("Temperature Profile")
    axT.plot(x_d, temp_d)
    axT.set_xlabel("x_direction")
    axT.set_ylabel("Temperature")
    axT.minorticks_on()

    axG.set_title("Boundary Layer Capture")
    axG.plot(x_d, np.zeros_like(final_grad))
    axG.plot(x_s_d, fg_smooth)
    axG.plot(inflect_x, inflect_y, "ro")
    axG.set_xlabel("x_direction")
    axG.set_ylabel("dv2/dydx")
    axG.minorticks_on()


    axG.text(
        0.5, 0.95,
        f"Inflection Point: {inflect_x:.6g} m",
        ha="center",
        va="top",
        transform=axG.transAxes,
        fontsize=11
    )

    axG.text(
        0.5, 0.88,
        f"Boundary Layer Thickness: {bl_thickness*1000:.6g} mm",
        ha="center",
        va="top",
        transform=axG.transAxes,
        fontsize=11
    )

    
    fig.tight_layout()
    return python_view.figure_to_image(fig)
'''


