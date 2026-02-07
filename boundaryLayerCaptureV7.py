from paraview.simple import *
import pandas as pd
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

H_index = 'H'
M_index = 'M'
T_index = 'T'
p_index = 'p'
rho_index = 'rho'
v_index = 'V'
v_x_index = 'V_X'
v_y_index = 'V_Y'
window_size = 3
window_2 = 15



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
plotOverLine1.Source.Resolution = 5000
plotOverLine1Display = Show(plotOverLine1, renderView1)

##############################################################################33

data = servermanager.Fetch(plotOverLine1)
pts = ns.vtk_to_numpy(data.GetPoints().GetData())
Points_0 = pts[:, 0]


point_data = data.GetPointData()
vectGrad = point_data.GetArray("VectorGradient")
vg = ns.vtk_to_numpy(vectGrad)          # shape (N, 9)
VectorGradient_4 = vg[:, 4]                    # VectorGradient_4

tempData = point_data.GetArray(T_index)
temp = ns.vtk_to_numpy(tempData)

v = point_data.GetArray(v_index)
vNP = ns.vtk_to_numpy(v)
vU = vNP[:, 0]

import pandas as pd

df = pd.DataFrame({
    "Points_0": Points_0,
    "VectorGradient_4": VectorGradient_4,
    "T": temp,
    "u": vU
})

x = df["Points_0"]
y = df["VectorGradient_4"]

df["dv_dy_smooth"] = df["VectorGradient_4"].rolling(5, center=True, min_periods=1).mean()

y_s = np.array(df["dv_dy_smooth"])
x_s = np.array(x)

grad = np.diff(y_s) / np.diff(x_s)
grad = np.append(grad, grad[-1])
df["grad_raw"] = grad
x_f = x_s
