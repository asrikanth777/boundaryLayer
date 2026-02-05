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

base = Path.cwd()
print(base)
boundaryLayer = base / "boundaryLayer"
pythonScript = boundaryLayer / "pythonviewscript.py"


H_index = 'H'
M_index = 'M'
T_index = 'T'
p_index = 'p'
rho_index = 'rho'
v_index = 'V'
v_x_index = 'V_X'
v_y_index = 'V_Y'

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

pythonView1.Script = pythonScript.read_text(encoding="utf-8")
# put the view in a layout so it actually appears
layout = GetLayout()
if layout is None:
    layout = CreateLayout("Layout #1")

AssignViewToLayout(view=pythonView1, layout=layout, hint=0)

# make it active and show the data in THIS view
SetActiveView(pythonView1)
Show(plotOverLine1, pythonView1)

Render(pythonView1)
