"""
Print what each block actually contains: bounds + array names/components.
Usage:  pvpython inspect_arrays.py ../results_sample/output_rk4_5kpa_50kw
"""
import sys
from pathlib import Path
from paraview.simple import *
from paraview import servermanager

folder = Path(sys.argv[1])

pvts_files = sorted(folder.glob("*.pvts"))
vts_files  = sorted(folder.glob("*.vts"))
pvts_stems = [f.stem[-2:] for f in pvts_files]
clean_vts  = [v for v in vts_files if not any(v.stem.endswith(s) for s in pvts_stems)]

for f in pvts_files + clean_vts:
    r = (XMLPartitionedStructuredGridReader if f.suffix == ".pvts"
         else XMLStructuredGridReader)(FileName=str(f))
    d = servermanager.Fetch(r)
    b = d.GetBounds()
    print(f"\n{f.name}")
    print(f"  bounds x: [{b[0]:.5f}, {b[1]:.5f}]  y: [{b[2]:.5f}, {b[3]:.5f}]  z: [{b[4]:.5f}, {b[5]:.5f}]")
    for label, fd in (("cell ", d.GetCellData()), ("point", d.GetPointData())):
        arrs = [f"{fd.GetArrayName(i)}({fd.GetArray(i).GetNumberOfComponents()})"
                for i in range(fd.GetNumberOfArrays())]
        print(f"  {label} arrays: {', '.join(arrs) if arrs else '(none)'}")
    Delete(r)