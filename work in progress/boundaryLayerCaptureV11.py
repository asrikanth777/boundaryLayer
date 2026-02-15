from paraview.simple import *
import pandas as pd
from pathlib import Path
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
v_index = 'v'
v_x_index = 'v_X'
v_y_index = 'v_Y'
window_size = 3
window_2 = 15

R_B = float(input("What is the radius of the sample body in meters?"))

base = Path.cwd()
name = input("folder with sample")
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

df = df.dropna()
df = df.reset_index(drop=True)

x = df["Points_0"]
y = df["VectorGradient_4"]
temperature = df["T"]
xVelocity = df["u"]


df["dv_dy_smooth"] = df["VectorGradient_4"].rolling(5, center=True, min_periods=1).mean()

y_s = np.array(df["dv_dy_smooth"])
x_s = np.array(x)

grad = np.diff(y_s) / np.diff(x_s)
grad = np.append(grad, grad[-1])
df["grad_raw"] = grad

df["grad_smooth"] = df["grad_raw"].rolling(5, center=True, min_periods=1).mean()
gs = df["grad_smooth"]

def find_inflection(x, grad_arr, tail_frac = 0.9):
    x = np.asarray(x, dtype=float)
    g = np.asarray(grad_arr, dtype=float)

    n = g.size
    side = int(tail_frac * n)

    x_tail = x[side:]
    g_tail = g[side:]

    # closest-to-zero (discrete)
    idx0 = np.argmin(np.abs(g_tail))
    location0 = x_tail[idx0]
    yloc0 = g_tail[idx0]
    thickness = x_tail[-1] - location0

    # largest positive value (discrete)
    max_idx = np.argmax(g_tail)
    max_val = g_tail[max_idx]
    max_loc = x_tail[max_idx]

    return location0, yloc0, thickness, max_val, max_loc


loc, ylc, bl, mv, ml = find_inflection(x_s, gs, tail_frac=0.9)


results = {
    "location": loc, 
    "Y-loc": ylc, 
    "bl_thickness": bl, 
    "max_val": mv, 
    "max_loc": ml 
}

resultsPrint = {
    "location": f"{loc:6f}m", 
    "Y-loc": f"{ylc:6f}m", 
    "bl_thickness": f"{bl*1000:6f}mm", 
    "max_val": f"{mv:6f}m" , 
    "max_loc": f"{ml:6f}m" 
}

x_zero = results["location"]
y_zero = results["Y-loc"]
x_max  = results["max_loc"]
y_max  = results["max_val"]

idx = np.argmin(np.abs(x_s - x_max))
x_mark = x_s[idx]
y_mark = y_s[idx]



print(resultsPrint)


df_empty = pd.read_csv("empty_chamber.csv")
xvel_empty = df_empty["u"]
x_empty = df_empty["Points_0"]

"""
TERMS LEFT TO CALCULATE FOR NONDIMENSIONAL VALUES
R_B =   radius of the body
x_e =   x location inflection point of dv/dy
        (where maximum of dv2/dxdy is)
delta = distance from x_e to sample edge
beta_e =  y-value of inflection point of dv/dy
U_t =   x-direction velocity right as it exits nozzle

# Need to run a second sim for these values
U_e = velocity when it deviates from freeflow simulation
U_s = U_t - U_e
"""

"""
NONDIMENSIONAL VALUE CALCULATIONS
T1 = delta/R_B
T2 = beta_e * R_B / U_t
T3 = d(beta_e)/dx * R_B**2 / U_t
T4 = U_e / U_t
T5 = U_e / U_s

"""
x_e = x_mark
beta_e = y_mark

delta = x_s[-1] - x_e
U_t = xVelocity[:3].mean()


T1 = delta/R_B
T2 = beta_e * R_B / U_t
T3 = mv * R_B**2 / U_t

nonDim = {
    "NDP1" : T1,
    "NDP2" : T2,
    "NDP3" : T3
}


###################### GRAPHS ######################

plt.figure()
plt.plot(x, temperature)
plt.xlabel("x")
plt.ylabel("temperature")
plt.grid(True, alpha=0.3)

plt.figure()
# plt.plot(x,xVelocity)
plt.plot(x_empty,xvel_empty)
plt.xlabel("x")
plt.ylabel("x-dir velocity")
plt.grid(True, alpha=0.3)



plt.figure()
plt.plot(x_s, y_s, label="dv/dy", lw=2, color="0.5")
plt.scatter(
    x_mark, y_mark,
    s=80, marker="x", color="k", zorder=10,
    label="dv/dy at derivative peak x"
)
plt.xlabel("x")
plt.ylabel("dv/dy")
plt.grid(True, alpha=0.3)
plt.legend()



plt.figure()
plt.plot(x_s, gs, label="gradient", lw=2)
plt.axhline(0.0, color="k", lw=1, alpha=0.5)
plt.scatter(
    x_zero, y_zero,
    s=70,
    marker="x",
    color="k",
    zorder=10,
    label="closest-to-zero"
)
plt.scatter(
    x_max, y_max,
    s=70,
    marker="o",
    facecolors="none",
    edgecolors="k",
    zorder=10,
    label="max positive"
)
plt.xlabel("x")
plt.ylabel("gradient")
plt.legend()
plt.grid(True, alpha=0.3)





plt.show()
