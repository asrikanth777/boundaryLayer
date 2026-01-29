from pathlib import Path
from paraview.simple import *
import pandas as pd
import re
import os
import matplotlib.pyplot as plt 
import numpy as np
from scipy.signal import savgol_filter



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
plotOverLine1.Source.Point2 = [0.6, 0, 0]
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


x = csvForEdit2["Points_0"]
y = csvForEdit2["VectorGradient_4"]

# plt.figure()
# plt.plot(x, y)
# plt.xlabel("x_direction")
# plt.ylabel("dv/dy")
# plt.show()

x_np = np.array(x)
y_np = np.array(y)
final_grad = np.diff(y_np)/np.diff(x_np)
x_np = np.delete(x_np, 0)
# final_grad = np.gradient(y_np, x_np)

plt.figure()
plt.plot(x_np, final_grad)
plt.xlabel("x_direction")
plt.ylabel("dv2/dydx")
plt.show()




# plt.figure()
# plt.plot(x, y, label="dv/dy")
# plt.plot(x_np, final_grad, label="d/dx(dv/dy)")
# plt.xlabel("x_direction")
# plt.legend()
# plt.show()

