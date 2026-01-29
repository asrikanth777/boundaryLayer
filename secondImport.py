from pathlib import Path
from paraview.simple import *
import pandas as pd
import re
import os
import matplotlib.pyplot as plt 
import numpy as np


from paraview import servermanager
from vtk.util import numpy_support as ns
import numpy as np



# this stuff gets current working directory
base = Path.cwd()


paraview.simple._DisableFirstRenderCameraReset()

# function to just select the vts file in the pipeline
# maybe not most efficient but its a copy-paste from previous project
def getImport():
    sources = GetSources()
    object = []

    for key, src in list(sources.items()):
        object.append(src)

    return object[0]

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
cellDatatoPointData1Display = Show(cellDatatoPointData1, renderView1)
Hide(computeDerivatives1, renderView1)
renderView1.Update()

# plots over line along stagnation point, this will be used to find inflection point
plotOverLine1 = PlotOverLine(Input=cellDatatoPointData2, Source='High Resolution Line Source')
plotOverLine1.Source.Point1 = [0.5, 0, 0]
plotOverLine1.Source.Point2 = [0.565, 0, 0]
plotOverLine1.Source.Resolution = 5000
plotOverLine1Display = Show(plotOverLine1, renderView1)



# data = servermanager.Fetch(plotOverLine1)
# pd = data.GetPointData()
# print([pd.GetArrayName(i) for i in range(pd.GetNumberOfArrays())])


# pts_vtk = data.GetPoints()

# print("VTK points object:", pts_vtk)
# print("Number of points:", pts_vtk.GetNumberOfPoints())


# arr = pd.GetArray("VectorGradient")
# print("VectorGradient components:", arr.GetNumberOfComponents())
# print("VectorGradient tuples:", arr.GetNumberOfTuples())

# vg = ns.vtk_to_numpy(arr)
# print("numpy shape:", vg.shape)   # (N, ncomp)

# dvdy = vg[:,4]

# Fetch output of PlotOverLine

data = servermanager.Fetch(plotOverLine1)

# ---- coordinates (Points_0/1/2) ----
pts = ns.vtk_to_numpy(data.GetPoints().GetData())
x = pts[:, 0]
y = pts[:, 1]  # if you need it

# ---- point-data arrays ----
point_data = data.GetPointData()
print([point_data.GetArrayName(i) for i in range(point_data.GetNumberOfArrays())])

# ---- VectorGradient and dvdy component ----
arr = point_data.GetArray("VectorGradient")
vg = ns.vtk_to_numpy(arr)          # shape (N, 9)
dvdy = vg[:, 4]                    # VectorGradient_4

# ---- drop NaNs (keep alignment with x) ----
mask = np.isfinite(x) & np.isfinite(dvdy)   # isfinite handles inf too
x = x[mask]
dvdy = dvdy[mask]

# ---- sort by x (critical before any derivative) ----
idx = np.argsort(x)
x = x[idx]
dvdy = dvdy[idx]

print("After cleaning: N =", x.size, "x range =", (x.min(), x.max()))

final_grad = np.diff(dvdy)/np.diff(x)
x = np.delete(x, 0)

plt.figure()
plt.plot(x, final_grad)
plt.xlabel("x_direction")
plt.ylabel("dv2/dydx")
plt.show()



