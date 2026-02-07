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

df["dv_dy_smooth"] = df["VectorGradient_4"].rolling(5, center=True, min_periods=1).mean()
# df["x_smooth"] = df["Points_0"].rolling(25, center=True, min_periods=1).mean()
print(df["VectorGradient_4"].head(3))
print(df["VectorGradient_4"].tail(3))


print(df["dv_dy_smooth"].head(3))
print(df["dv_dy_smooth"].tail(3))

print(x.tail(3))

y_s = np.array(df["dv_dy_smooth"])
x_s = np.array(x)

grad = np.diff(y_s) / np.diff(x_s)
grad = np.append(grad, grad[-1])
df["grad_raw"] = grad
print(grad.shape)
x_f = x_s

# --- 2) Precompute smoothed columns once ---
smooth_windows = [3,5,7,9,11,13,15,17,19,21,23,25]
for w in smooth_windows:
    df[f"grad_smooth_{w}"] = df["grad_raw"].rolling(w, center=True, min_periods=1).mean()

# --- 3) Inflection finder: use a grad array you give it (raw or already-smoothed) ---
def find_inflection_from_array(x, grad_arr, tail_frac=0.9):
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

# --- 4) Compute results for raw + selected windows (or all, if you want) ---
windows = [None,3,5,7,11,13,15,19]   # change to your full list if desired
results = {}

for w in windows:
    key = "raw" if w is None else f"w{w}"
    g_arr = df["grad_raw"].to_numpy() if w is None else df[f"grad_smooth_{w}"].to_numpy()

    loc, ylc, bl, mv, ml = find_inflection_from_array(x_f, g_arr, tail_frac=0.9)
    results[key] = {"location": loc, "Y-loc": ylc, "bl_thickness": bl, "max_val": mv, "max_loc": ml}

df_results = (
    pd.DataFrame(results).T
      .reset_index()
      .rename(columns={"index": "smoothing"})
)

print(df_results)

# --- 5) Build marker points automatically (no manual lists) ---
keys_to_plot = ["raw","w3","w5", "w7","w11", "w13","w15", "w19"]

x_zero = [results[k]["location"] for k in keys_to_plot]
y_zero = [results[b]["Y-loc"] for b in keys_to_plot]

x_max  = [results[k]["max_loc"] for k in keys_to_plot]
y_max  = [results[k]["max_val"] for k in keys_to_plot]


def snap_to_raw(x_raw, y_raw, x_target):
    idx = np.argmin(np.abs(x_raw - x_target))
    return x_raw[idx], y_raw[idx]


x_dvdy = []
y_dvdy = []

for k in keys_to_plot:
    xr, yr = snap_to_raw(x_f, y_s, results[k]["max_loc"])
    x_dvdy.append(xr)
    y_dvdy.append(yr)

# color per window (raw black, others use matplotlib cycle)
colors = {
    "raw": "k",
    "w3":  "C0",
    "w5":  "C1",
    "w7":  "C2",
    "w11": "C3",
    "w13": "C4",
    "w15": "C5",
    "w19": "C6",
}

plt.figure()

plt.plot(x_f, y_s, label="dv/dy", lw=2, color="0.5")

for k, xr, yr in zip(keys_to_plot, x_dvdy, y_dvdy):
    plt.scatter(
        xr, yr,
        s=80,
        marker="x",
        color=colors[k],
        zorder=10,
        label=k
    )

plt.xlabel("x")
plt.ylabel("dv/dy")
plt.grid(True, alpha=0.3)
plt.legend(ncols=2, fontsize=9, title="window")


# --- 6) Plot ---
plt.figure()

plt.plot(x_f, df["grad_raw"], label="raw", lw=1)
plt.plot(x_f, np.zeros_like(x_f), label="zero", lw=1)
plt.plot(x_f, df["grad_smooth_3"],  label="window = 3",  lw=2)
plt.plot(x_f, df["grad_smooth_5"],  label="window = 5",  lw=2)
plt.plot(x_f, df["grad_smooth_7"],  label="window = 7",  lw=2)
plt.plot(x_f, df["grad_smooth_9"],  label="window = 9",  lw=2)
plt.plot(x_f, df["grad_smooth_11"],  label="window = 11",  lw=2)
plt.plot(x_f, df["grad_smooth_13"],  label="window = 13",  lw=2)
plt.plot(x_f, df["grad_smooth_15"],  label="window = 15",  lw=2)
plt.plot(x_f, df["grad_smooth_19"], label="window = 19", lw=2)

# markers: closest-to-zero (on y=0)
plt.scatter(x_zero, y_zero, s=70, marker="x", color="k", zorder=10, label="closest-to-zero")

# markers: max positive on each curve
plt.scatter(x_max, y_max, s=70, marker="o", facecolors="none", edgecolors="k",
            zorder=10, label="max positive")

plt.xlabel("x")
plt.ylabel("gradient")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

