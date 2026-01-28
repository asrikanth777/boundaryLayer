from pathlib import Path
from paraview.simple import *
import pandas as pd
import re
import os

base = Path.cwd()


paraview.simple._DisableFirstRenderCameraReset()

def getImport():
    sources = GetSources()
    object = []

    for key, src in list(sources.items()):
        object.append(src)

    # groupImports = GroupDatasets(Input=import_list)
    # return groupImports
    return object[0]

flowfield = getImport()
cellDatatoPointData1 = CellDatatoPointData(Input=flowfield)
cellDatatoPointData1.CellDataArraytoprocess = ['H', 'M', 'T', 'p', 'rho', 'v']

renderView1 = GetActiveViewOrCreate('RenderView')
cellDatatoPointData1Display = Show(cellDatatoPointData1, renderView1)

Hide(flowfield, renderView1)
renderView1.Update()

calculator1 = Calculator(Input=cellDatatoPointData1)
calculator1.Function = 'v_X*iHat + v_Y*jHat'
calculator1Display = Show(calculator1, renderView1)

Hide(cellDatatoPointData1, renderView1)
renderView1.Update()

computeDerivatives1 = ComputeDerivatives(Input = calculator1)
computeDerivatives1.Vectors = ['POINTS', 'Result']
computeDerivatives1Display = Show(computeDerivatives1, renderView1)

Hide(calculator1, renderView1)
renderView1.Update()

plotOverLine1 = PlotOverLine(Input=computeDerivatives1, Source='High Resolution Line Source')
plotOverLine1.Source.Point1 = [0.5, 0, 0]
plotOverLine1.Source.Point2 = [0.6, 0, 0]
plotOverLine1Display = Show(plotOverLine1, renderView1)

CreateLayout('Layout #2')
SetActiveView(None)
spreadSheetView1 = CreateView('SpreadSheetView')
spreadSheetView1.ColumnToSort = ''
spreadSheetView1.BlockSize = 1024
plotOverLine1Display_2 = Show(plotOverLine1, spreadSheetView1)
layout2 = GetLayoutByName("Layout #2")
AssignViewToLayout(view=spreadSheetView1, layout=layout2, hint=0)
ExportView(str(base) + '/something.csv', view=spreadSheetView1)

csvImport = pd.read_csv("something.csv")
csvForEdit = csvImport.dropna()
