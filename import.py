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

