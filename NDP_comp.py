from paraview.simple import *
import pandas as pd
from pathlib import Path
import numpy as np
from paraview import servermanager
from vtk.util import numpy_support as ns

paraview.simple._DisableFirstRenderCameraReset()

# ── helpers ──────────────────────────────────────────────────────────────────

def getImport():
    sources = GetSources()
    return GroupDatasets(Input=list(sources.values()))

def clear_pipeline():
    for src in list(GetSources().values()):
        Delete(src)

# ── constants ─────────────────────────────────────────────────────────────────

H_index   = 'H';  M_index = 'M';  T_index = 'T'
p_index   = 'p';  rho_index = 'rho';  v_index = 'v'
v_x_index = 'v_X';  v_y_index = 'v_Y'

POWER_LEVELS = ['50kw', '100kw', '150kw', '200kw']
ROW_INDEX    = ['NDP1', 'NDP2', 'NDP3', 'NDP4', 'NDP5']

# ── inputs ────────────────────────────────────────────────────────────────────

R_B          = float(input("Radius of sample body (meters): "))
base         = Path.cwd()
sample_root  = base / input("Top-level folder for SAMPLE runs (contains 50kw/, 100kw/, …): ").strip()
empty_root   = base / input("Top-level folder for EMPTY chamber runs (contains 50kw/, 100kw/, …): ").strip()

# Expected layout:
#   sample_root/
#       50kw/   *.pvts  *.vts
#       100kw/  ...
#   empty_root/
#       50kw/   ...
#       ...

# ── pipeline processing function ──────────────────────────────────────────────

def process_folder(folder: Path, R_B: float) -> dict:
    """
    Loads all pvts/vts files in *folder*, runs the ParaView pipeline,
    and returns a dict with NDP1-NDP5.
    Returns None if the folder doesn't exist or has no data files.
    """
    if not folder.exists():
        print(f"  [SKIP] folder not found: {folder}")
        return None

    pvts_files = sorted(folder.glob("*.pvts"))
    vts_files  = sorted(folder.glob("*.vts"))

    if not pvts_files and not vts_files:
        print(f"  [SKIP] no .pvts/.vts files in: {folder}")
        return None

    # Filter out vts files that are sub-files of a pvts (they share a 2-char suffix)
    pvts_stems = [f.stem[-2:] for f in pvts_files]
    clean_vts  = [v for v in vts_files
                  if not any(v.stem.endswith(s) for s in pvts_stems)]

    print(f"  PVTS:         {[f.name for f in pvts_files]}")
    print(f"  Filtered VTS: {[f.name for f in clean_vts]}")

    # ── build readers ──
    [XMLPartitionedStructuredGridReader(FileName=str(f)) for f in pvts_files]
    [XMLStructuredGridReader(FileName=str(f))            for f in clean_vts]

    renderView1 = GetActiveViewOrCreate('RenderView')

    # ── pipeline ──
    flowfield = getImport()

    cdpd1 = CellDatatoPointData(Input=flowfield)
    cdpd1.CellDataArraytoprocess = [H_index, M_index, T_index,
                                     p_index, rho_index, v_index]
    Show(cdpd1, renderView1);  Hide(flowfield, renderView1)
    renderView1.Update()

    calc1 = Calculator(Input=cdpd1)
    calc1.Function = f"{v_x_index}*iHat + {v_y_index}*jHat"
    Show(calc1, renderView1);  Hide(cdpd1, renderView1)
    renderView1.Update()

    deriv1 = ComputeDerivatives(Input=calc1)
    deriv1.Vectors = ['POINTS', 'Result']
    Show(deriv1, renderView1);  Hide(calc1, renderView1)
    renderView1.Update()

    # BUG FIX: was showing cdpd1 here instead of cdpd2
    cdpd2 = CellDatatoPointData(Input=deriv1)
    cdpd2.CellDataArraytoprocess = ['ScalarGradient', 'VectorGradient']
    Show(cdpd2, renderView1);  Hide(deriv1, renderView1)
    renderView1.Update()

    pol = PlotOverLine(Input=cdpd2, Source='High Resolution Line Source')
    pol.Source.Point1     = [0.4,   0, 0]
    pol.Source.Point2     = [0.565, 0, 0]
    pol.Source.Resolution = 5000
    Show(pol, renderView1)

    # ── extract data ──
    data       = servermanager.Fetch(pol)
    pts        = ns.vtk_to_numpy(data.GetPoints().GetData())
    Points_0   = pts[:, 0]
    pd_        = data.GetPointData()

    vg         = ns.vtk_to_numpy(pd_.GetArray("VectorGradient"))   # (N,9)
    VG4        = vg[:, 4]
    temp       = ns.vtk_to_numpy(pd_.GetArray(T_index))
    vNP        = ns.vtk_to_numpy(pd_.GetArray(v_index))
    xVelocity  = vNP[:, 0]

    df = pd.DataFrame({"Points_0": Points_0, "VG4": VG4,
                        "T": temp, "u": xVelocity}).dropna().reset_index(drop=True)

    x           = df["Points_0"].to_numpy()
    xVelocity   = df["u"].to_numpy()

    # ── smoothing & inflection ──
    df["dv_dy_smooth"] = df["VG4"].rolling(5, center=True, min_periods=1).mean()
    y_s  = df["dv_dy_smooth"].to_numpy()

    grad = np.diff(y_s) / np.diff(x)
    grad = np.append(grad, grad[-1])
    df["grad_smooth"] = pd.Series(grad).rolling(5, center=True, min_periods=1).mean().to_numpy()
    gs   = df["grad_smooth"].to_numpy()

    def find_inflection(x, grad_arr, tail_frac=0.9):
        n      = grad_arr.size
        side   = int(tail_frac * n)
        x_t, g_t = x[side:], grad_arr[side:]
        idx0   = np.argmin(np.abs(g_t))
        loc    = x_t[idx0]
        return loc, g_t[idx0], x_t[-1] - loc

    loc, ylc, bl = find_inflection(x, gs)

    idx   = np.argmin(np.abs(x - loc))
    x_e   = x[idx]
    beta_e = y_s[idx]

    idx2  = np.argmin(np.abs(x - loc))
    U_e   = xVelocity[idx2]
    U_t   = xVelocity[:3].mean()
    U_s   = U_t - U_e
    delta = x[-1] - x_e

    print(f"  location={loc:.6f}m  bl_thickness={bl*1000:.6f}mm")

    # ── clean up pipeline for next folder ──
    clear_pipeline()

    return {
        "NDP1": delta  / R_B,
        "NDP2": beta_e * R_B / U_t,
        "NDP3": ylc    * R_B**2 / U_t,
        "NDP4": U_e    / U_t,
        "NDP5": U_e    / U_s,
    }

# ── main loop ─────────────────────────────────────────────────────────────────

for label, root in [("sample", sample_root), ("empty", empty_root)]:
    summary_df = pd.DataFrame(index=ROW_INDEX, columns=POWER_LEVELS, dtype=float)

    for power_col in POWER_LEVELS:
        folder = root / power_col
        print(f"\n[{label.upper()} | {power_col}]  {folder}")
        result = process_folder(folder, R_B)

        if result:
            for ndp, val in result.items():
                summary_df.loc[ndp, power_col] = val

    out_path = base / f"summary_{label}.csv"
    summary_df.to_csv(out_path)
    print(f"\nSaved: {out_path}\n{summary_df}\n")