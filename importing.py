from pathlib import Path
from paraview.simple import *
import re
import os
import pandas as pd
from paraview import servermanager
from vtk.util import numpy_support as ns


paraview.simple._DisableFirstRenderCameraReset()

def clear_pipeline():
    sources = GetSources()
    for key, src in list(sources.items()):
        Delete(src)

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
v_index = 'v'
v_x_index = 'v_X'
v_y_index = 'v_Y'
window_size = 3
window_2 = 15


base = Path.cwd()
name = input("folder with empty chamber")
folder = base / name
print(folder)

# collects all pvts and vts files 
pvts_files = sorted(folder.glob("*.pvts"))
vts_files  = sorted(folder.glob("*.vts"))

# Build a set of base stems from pvts
# this is for checking what its importing in later and for writing outputs for csv later
pvts_stems = [f.stem[-2:] for f in pvts_files]
pvts_stem1 = [f.stem for f in pvts_files]

# Filter vts_files: keep only those whose stem does NOT start with any pvts stem
clean_vts = []
for v in vts_files:
    if not any(v.stem.endswith(stem) for stem in pvts_stems):
        clean_vts.append(v)

# read this to make sure you are importing the right things
# if the above code is not there, it will import the pvts and the respective vts files, which arent needed
print("PVTS:", [f.name for f in pvts_files])
print("Filtered VTS:", [f.name for f in clean_vts])

# tracks name to use for outputting csvs
vts_stem1 = [f.stem for f in clean_vts]

# Make readers and import in pvts/vts files
pvts_readers = [XMLPartitionedStructuredGridReader(FileName=str(f)) for f in pvts_files]
vts_readers  = [XMLStructuredGridReader(FileName=str(f)) for f in clean_vts]

# renders them to use actions on them later
renderView1 = GetActiveViewOrCreate('RenderView')

empty_group = getImport()

cellDatatoPointData1 = CellDatatoPointData(Input=empty_group)
cellDatatoPointData1.CellDataArraytoprocess = [H_index, M_index, T_index, p_index, rho_index, v_index]
renderView1 = GetActiveViewOrCreate('RenderView')
cellDatatoPointData1Display = Show(cellDatatoPointData1, renderView1)
Hide(empty_group, renderView1)
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

data = servermanager.Fetch(plotOverLine1)
pts = ns.vtk_to_numpy(data.GetPoints().GetData())
Points_0 = pts[:, 0]


point_data = data.GetPointData()
print([point_data.GetArrayName(i) for i in range(point_data.GetNumberOfArrays())])

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

df = df.dropna()
df = df.reset_index(drop=True)
df.to_csv("empty_chamber.csv", index=False)
# clear_pipeline()




