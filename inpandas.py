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

base = Path.cwd()

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

# spreadsheet view to export points on line as csv
CreateLayout('Layout #2')
SetActiveView(None)
spreadSheetView1 = CreateView('SpreadSheetView')
spreadSheetView1.ColumnToSort = ''
spreadSheetView1.BlockSize = 1024
plotOverLine1Display_2 = Show(plotOverLine1, spreadSheetView1)
layout2 = GetLayoutByName("Layout #2")
AssignViewToLayout(view=spreadSheetView1, layout=layout2, hint=0)
ExportView(str(base) + '/something.csv', view=spreadSheetView1)

# reads csv and drops nan columns
csvImport = pd.read_csv("something.csv")
csvForEdit1 = csvImport.dropna()
csvForEdit2 = csvForEdit1.reset_index(drop=True)
csvForEdit2.to_csv('droppednan.csv', index=False)

df = pd.read_csv("droppednan.csv")


x = df["Points_0"]
y = df["VectorGradient_4"]

df["du_dy_smooth"] = df["VectorGradient_4"].rolling(25, center=True, min_periods=1).mean()
df["x_smooth"] = df["Points_0"].rolling(25, center=True, min_periods=1).mean()


y_smooth = df['du_dy_smooth']
x_smooth = df['x_smooth']

x_np = np.array(x)
y_np = np.array(y)

x_smooth_np = np.array(x_smooth)
y_smooth_np = np.array(y_smooth)

final_grad = np.diff(y_np)/np.diff(x_np)

fg = np.diff(y_smooth_np)/np.diff(x_smooth_np)
x_np = np.delete(x_np, 0)
x_smooth_np1 = np.delete(x_smooth_np, 0)


plt.figure()
plt.plot(x_np, final_grad)
plt.plot(x_smooth_np1, fg)
plt.xlabel("x_direction")
plt.ylabel("dv2/dydx")
plt.show()
