from pathlib import Path
from paraview.simple import *
import re
import os
import matplotlib.pyplot as plt 
import numpy as np
from paraview import servermanager
from vtk.util import numpy_support as ns
import numpy as np


paraview.simple._DisableFirstRenderCameraReset()

def getImport():
    sources = GetSources()
    object = []

    for key, src in list(sources.items()):
        object.append(src)

    return GroupDatasets(Input=object)

H_index = 'H'
M_index = 'M'
T_index = 'T'
p_index = 'p'
rho_index = 'rho'
v_index = 'V'
v_x_index = 'V_X'
v_y_index = 'V_Y'
window_size = 5
window_2 = 15

def smooth_like_matlab(data, window):
    # 1. Create the kernel
    kernel = np.ones(window) / window
    
    # 2. Pad the edges with the first and last values
    # We pad by (window // 2) on each side to keep the 'same' length
    pad_size = window // 2
    padded_data = np.pad(data, pad_size, mode='edge')
    
    # 3. Convolve using 'valid' to remove the padding effects
    return np.convolve(padded_data, kernel, mode='valid')

flowfield = getImport()
cellDatatoPointData1 = CellDatatoPointData(Input=flowfield)
cellDatatoPointData1.CellDataArraytoprocess = [H_index, M_index, T_index, p_index, rho_index, v_index]
renderView1 = GetActiveViewOrCreate('RenderView')
cellDatatoPointData1Display = Show(cellDatatoPointData1, renderView1)
Hide(flowfield, renderView1)
renderView1.Update()

calculator1 = Calculator(Input=cellDatatoPointData1)
calculator1.Function = f"{v_x_index}*iHat + {v_y_index}*jHat"
calculator1Display = Show(calculator1, renderView1)
Hide(cellDatatoPointData1, renderView1)
renderView1.Update()

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

plotOverLine1 = PlotOverLine(Input=cellDatatoPointData2, Source='High Resolution Line Source')
plotOverLine1.Source.Point1 = [0.4, 0, 0]
plotOverLine1.Source.Point2 = [0.565, 0, 0]
plotOverLine1.Source.Resolution = 10000
plotOverLine1Display = Show(plotOverLine1, renderView1)

"""
THIS PART NEEDS TO BE CLEANED UP FOR SURE
"""


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

tempData = point_data.GetArray(T_index)
temp = ns.vtk_to_numpy(tempData)

v = point_data.GetArray(v_index)
vNP = ns.vtk_to_numpy(v)
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
temp = temp[idx]
vU = vU[idx]

# print("After cleaning: N =", x.size, "x range =", (x.min(), x.max()))


# smoothing function for prettier line
#
#

dv2dxdy = np.diff(dvdy)/np.diff(x)
# dv2dxdy = np.gradient(dvdy, x)
x_mid = 0.5*(x[:-1] + x[1:])


# for matching array sizes
#
#

# extracts only later 1/3 of the data
# to get inflection point
#
#
size = dv2dxdy.shape
#print(size)
size = int(size[0])
#print(type(size))
side = int(9/10*size)
#print(side)

examine = x_mid[side:]
check = dv2dxdy[side:]

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
print(x[-1])

plt.figure()
plt.title("Temperature Profile")
plt.plot(x, temp)
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
plt.plot(x, np.zeros_like(x))
plt.plot(examine[indx], check[indx], 'ro') 
plt.title("Boundary Layer Capture")
plt.figtext(0.5, 0.5, location, ha="center", fontsize=11)
plt.figtext(0.5, 0.45, bl_plot, ha="center", fontsize=11)
plt.plot(x_mid, dv2dxdy)
plt.xlabel("x_direction")
plt.ylabel("dv2/dydx")
plt.show()

