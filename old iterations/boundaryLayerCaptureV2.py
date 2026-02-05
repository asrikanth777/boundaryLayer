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
cellDatatoPointData1Display = Show(cellDatatoPointData1, renderView1)
Hide(computeDerivatives1, renderView1)
renderView1.Update()

# plots over line along stagnation point, this will be used to find inflection point
plotOverLine1 = PlotOverLine(Input=cellDatatoPointData2, Source='High Resolution Line Source')
plotOverLine1.Source.Point1 = [0.4, 0, 0]
plotOverLine1.Source.Point2 = [0.565, 0, 0]
plotOverLine1.Source.Resolution = 5000
plotOverLine1Display = Show(plotOverLine1, renderView1)

# gets coordinate data from plot over line and converts to numpy arrays
#
#
data = servermanager.Fetch(plotOverLine1)
pts = ns.vtk_to_numpy(data.GetPoints().GetData())
x = pts[:, 0]


# gets point data along line and grabs dvdy numpy array
#
#
point_data = data.GetPointData()
print([point_data.GetArrayName(i) for i in range(point_data.GetNumberOfArrays())])
vectGrad = point_data.GetArray("VectorGradient")
vg = ns.vtk_to_numpy(vectGrad)          # shape (N, 9)
dvdy = vg[:, 4]                    # VectorGradient_4

temp = point_data.GetArray("T")
tempNP = ns.vtk_to_numpy(temp)

v = point_data.GetArray("v")
vNP = ns.vtk_to_numpy(v)
print(vNP.shape)
vU = vNP[:, 0]

# drops nan values
#
#
mask = np.isfinite(x) & np.isfinite(dvdy)   
x = x[mask]
dvdy = dvdy[mask]
idx = np.argsort(x)
x = x[idx]
dvdy = dvdy[idx]
tempNP = tempNP[idx]
vU = vU[idx]

# print("After cleaning: N =", x.size, "x range =", (x.min(), x.max()))


# smoothing function for prettier line
#
#
window_size = 11
x_smooth = np.convolve(x, np.ones(window_size)/window_size, mode='valid')
dvdy_smooth = np.convolve(dvdy, np.ones(window_size)/window_size, mode='valid')

# calculates dv2/dxdy
#
#
final_grad = np.diff(dvdy)/np.diff(x)
fg_smooth = np.diff(dvdy_smooth)/np.diff(x_smooth)

# for matching array sizes
#
#
x = np.delete(x, 0)
x_smooth = np.delete(x_smooth, 0)
tempNP = np.delete(tempNP, 0)
vU = np.delete(vU, 0)
dvdy = np.delete(dvdy, 0)

# extracts only later 1/3 of the data
# to get inflection point
#
#
size = final_grad.shape
#print(size)
size = int(size[0])
#print(type(size))
side = int(9/10*size)
#print(side)

examine = x_smooth[side:]
check = fg_smooth[side:]

# tries to find inflection point by smallest distance from zero
#
#
dist = [abs(val - 0) for val in check]
bound = min(dist)
indx = dist.index(bound)
# print("Where the inflection point is:")
# print(examine[indx])
bl_thickness = x[-1] - examine[indx]
# print("How thick the boundary layer is:")
# print(bl_thickness)


# plots it for visualization
#
#
location = f"Inflection Point: {examine[indx]}m"
bl_plot = f"Boundary Layer Thickness: {bl_thickness * 1000}mm"

plt.figure()
plt.title("Temperature Profile")
plt.plot(x, tempNP)
plt.xlabel("x_direction")
plt.ylabel("Temperature")

plt.figure()
plt.title("Axial Velocity Profile")
plt.plot(x, vU)
plt.xlabel("x_direction")
plt.ylabel("Axial Velocity")

plt.figure()
plt.title("dv/dy Profile")
plt.plot(x,dvdy)
plt.xlabel("x_direction")
plt.ylabel("dv/dy")


plt.figure()
# plt.plot(x, final_grad) this is the raw data plot
plt.plot(x, np.zeros_like(final_grad))
plt.plot(examine[indx], check[indx], 'ro') 
plt.title("Boundary Layer Capture")
plt.figtext(0.5, 0.5, location, ha="center", fontsize=11)
plt.figtext(0.5, 0.45, bl_plot, ha="center", fontsize=11)
plt.plot(x_smooth, fg_smooth)
plt.xlabel("x_direction")
plt.ylabel("dv2/dydx")
plt.show()

