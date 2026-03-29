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

def clear_pipeline():
    sources = GetSources()
    for key, src in list(sources.items()):
        Delete(src)


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

R_B = float(input("What is the radius of the sample body in meters?\n"))

base = Path.cwd()

column_names = ['50kw', '100kw', '150kw', '200kw']
row_index = ['NDP1', 'NDP2', 'NDP3', 'NDP4', 'NDP5']   
summary_df = pd.DataFrame(index=row_index, columns=column_names, dtype=float)


folder_50kw_sample = input("folder with sample at 50kW\n")
folder_100kw_sample = input("folder with sample at 100kw\n")
folder_150kw_sample = input("folder with sample at 150kw\n")
folder_200kw_sample = input("folder with empty chamber at 200kw\n")
folder_list_1 = [folder_50kw_sample, folder_100kw_sample, folder_150kw_sample, folder_200kw_sample]
folders_list = []

for a in folder_list:
    folders_list.append(base/a)


for power_col, folder in zip(column_names, folders_list):
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



        return location0, yloc0, thickness


    loc, ylc, bl = find_inflection(x_s, gs, tail_frac=0.9)


    results = {
        "location": loc, 
        "Y-loc": ylc, 
        "bl_thickness": bl,   
    }

    resultsPrint = {
        "location": f"{loc:6f}m", 
        "Y-loc": f"{ylc:6f}", 
        "bl_thickness": f"{bl*1000:6f}mm", 
    }

    x_zero = results["location"]
    y_zero = results["Y-loc"]

    idx = np.argmin(np.abs(x_s - x_zero))
    x_mark = x_s[idx]
    y_mark = y_s[idx]

    idx2 = np.argmin(np.abs(x - loc))
    x_ue = x[idx2]
    y_ue = xVelocity[idx2]


    print(resultsPrint)

    x_e = x_mark
    beta_e = y_mark

    delta = x_s[-1] - x_e
    U_t = xVelocity[:3].mean()
    U_e = y_ue
    U_s = U_t - U_e


    T1 = delta/R_B
    T2 = beta_e * R_B / U_t
    T3 = y_mark * R_B**2 / U_t
    T4 = U_e/ U_t
    T5 = U_e / U_s


    # --- write into the SUMMARY dataframe by power ---
    summary_df.loc["NDP1", power_col] = T1
    summary_df.loc["NDP2", power_col] = T2
    summary_df.loc["NDP3", power_col] = T3
    summary_df.loc["NDP4", power_col] = T4
    summary_df.loc["NDP5", power_col] = T5

