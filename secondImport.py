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


"""
import numpy as np

def smooth_knn_poly(x, y, k=51, degree=2):
    """
#    Smooth y(x) on irregular x using k-nearest-neighbors local polynomial regression.
#    Returns y_smooth at the same x points (no resampling).
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # sort by x
    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]

    n = len(x)
    y_s = np.empty(n)

    # precompute neighbor indices by distance in x
    for i in range(n):
        # distances to all points (1D so cheap enough for ~5k)
        d = np.abs(x - x[i])
        nn = np.argpartition(d, kth=min(k, n-1))[:k]  # k nearest indices
        nn = nn[np.argsort(x[nn])]  # sort neighbors by x

        xw = x[nn] - x[i]           # center at x_i improves conditioning
        yw = y[nn]

        # weights: closer points matter more (avoid zero bandwidth)
        h = np.max(np.abs(xw))
        if h == 0:
            y_s[i] = y[i]
            continue
        w = np.exp(-0.5 * (xw / (0.35*h))**2)

        # design matrix for polynomial in (x - x_i)
        # columns: 1, x, x^2, ...
        A = np.vstack([xw**p for p in range(degree+1)]).T

        # weighted least squares
        W = np.sqrt(w)
        Aw = A * W[:, None]
        yw2 = yw * W
        coeff, *_ = np.linalg.lstsq(Aw, yw2, rcond=None)

        # value at x_i is polynomial at 0 => coeff[0]
        y_s[i] = coeff[0]

    return x, y_s  # returns sorted x and smoothed y

"""