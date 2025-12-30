from pathlib import Path
from paraview.simple import *
import re
import os

paraview.simple._DisableFirstRenderCameraReset()

def group_Imports():
    sources = GetSources()
    import_list = []

    for key, src in list(sources.items()):
        import_list.append(src)

    groupImports = GroupDatasets(Input=import_list)
    return groupImports

base = Path.cwd()
print(base)
name = input("name for folder with pvts and vts")
folder = base / name
print(folder)

# collects all pvts and vts files 
pvts_files = sorted(folder.glob("*.pvts"))
print("pvts files: ", [f.name for f in pvts_files])
vts_files  = sorted(folder.glob("*.vts"))
print("vts files: ", [f.name for f in vts_files])


pvts_num = [f.stem[-1:] for f in pvts_files]
pvts_stems = [f.stem for f in pvts_files]

import_vts = []
for files in vts_files:
    if int(files.stem[-1]) > 5 and not any(files.stem.endswith(stem) for stem in pvts_stems):
        import_vts.append(files)

print(f"import vts files: {import_vts}")


import_pvts = []
for files in pvts_files:
    if int(files.stem[-1]) > 5:
        import_pvts.append(files)

print(f"import pvts files: {import_pvts}")


import_stems = [f.stem for f in import_vts]

pvts_readers = [XMLPartitionedStructuredGridReader(FileName=str(f)) for f in pvts_files]
vts_readers  = [XMLStructuredGridReader(FileName=str(f)) for f in import_vts]

renderView1 = GetActiveViewOrCreate('RenderView')

grouped_imports = group_Imports()
groupDisplay = Show(grouped_imports, renderView1)
renderView1.Update()

# create a new 'Plot Over Line'
plotOverLine1 = PlotOverLine(registrationName='PlotOverLine1', Input=grouped_imports)
plotOverLine1.Point1 = [0.401885, 0.0, 0.0]
plotOverLine1.Point2 = [0.6, 0.0, 0.0]
plotOverLine1.UpdatePipeline()

# Create a new 'Line Chart View'
lineChartView1 = CreateView('XYChartView')

# show data in view
plotOverLine1Display_1 = Show(plotOverLine1, lineChartView1, 'XYChartRepresentation')
layout1 = GetLayoutByName("Layout #1")
AssignViewToLayout(view=lineChartView1, layout=layout1, hint=0)
renderView1.Update()
lineChartView1.Update()
plotOverLine1Display_1.SeriesVisibility = ['v_Magnitude']



# next steps
# ----1) make multiple plot over line of x-axis lines
# ----2) get tangential components
#           + first calculate tangential direction along wall
#           + curved wall so more complex calc needed
#           + then find Ut

# more to add but this is from chat, need to cross-validate.

# edge of the boundary layer as the inflection point of dv/dy velocity in the x direction
# refer to paper on more details about this
# need to trace the surface of the sample
