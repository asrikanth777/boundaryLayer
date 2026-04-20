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

R_B         = float(input("Radius of sample body (meters): "))
base        = Path.cwd()
sample_root = base / input("Top-level folder for SAMPLE runs (contains 50kw/, 100kw/, …): ").strip()

# ── pipeline processing function ──────────────────────────────────────────────

def process_folder(folder: Path, R_B: float) -> dict:
    if not folder.exists():
        print(f"  [SKIP] folder not found: {folder}")
        return None

    pvts_files = sorted(folder.glob("*.pvts"))
    vts_files  = sorted(folder.glob("*.vts"))

    if not pvts_files and not vts_files:
        print(f"  [SKIP] no .pvts/.vts files in: {folder}")
        return None

    pvts_stems = [f.stem[-2:] for f in pvts_files]
    clean_vts  = [v for v in vts_files
                  if not any(v.stem.endswith(s) for s in pvts_stems)]

    print(f"  PVTS:         {[f.name for f in pvts_files]}")
    print(f"  Filtered VTS: {[f.name for f in clean_vts]}")

    [XMLPartitionedStructuredGridReader(FileName=str(f)) for f in pvts_files]
    [XMLStructuredGridReader(FileName=str(f))            for f in clean_vts]

    renderView1 = GetActiveViewOrCreate('RenderView')

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

    cdpd2 = CellDatatoPointData(Input=deriv1)
    cdpd2.CellDataArraytoprocess = ['ScalarGradient', 'VectorGradient']
    Show(cdpd2, renderView1);  Hide(deriv1, renderView1)
    renderView1.Update()

    pol = PlotOverLine(Input=cdpd2, Source='High Resolution Line Source')
    pol.Source.Point1     = [0.4,   0, 0]
    pol.Source.Point2     = [0.565, 0, 0]
    pol.Source.Resolution = 5000
    Show(pol, renderView1)

    data       = servermanager.Fetch(pol)
    pts        = ns.vtk_to_numpy(data.GetPoints().GetData())
    Points_0   = pts[:, 0]
    pd_        = data.GetPointData()

    vg         = ns.vtk_to_numpy(pd_.GetArray("VectorGradient"))
    VG4        = vg[:, 4]
    temp       = ns.vtk_to_numpy(pd_.GetArray(T_index))
    vNP        = ns.vtk_to_numpy(pd_.GetArray(v_index))
    xVelocity  = vNP[:, 0]

    df = pd.DataFrame({"Points_0": Points_0, "VG4": VG4,
                        "T": temp, "u": xVelocity}).dropna().reset_index(drop=True)

    x         = df["Points_0"].to_numpy()
    xVelocity = df["u"].to_numpy()

    df["dv_dy_smooth"] = df["VG4"].rolling(5, center=True, min_periods=1).mean()
    y_s = df["dv_dy_smooth"].to_numpy()

    grad = np.diff(y_s) / np.diff(x)
    grad = np.append(grad, grad[-1])
    df["grad_smooth"] = pd.Series(grad).rolling(5, center=True, min_periods=1).mean().to_numpy()
    gs = df["grad_smooth"].to_numpy()

    def find_inflection(x, grad_arr, tail_frac=0.9):
        n    = grad_arr.size
        side = int(tail_frac * n)
        x_t, g_t = x[side:], grad_arr[side:]

        # F-method: largest positive value
        max_idx = np.argmax(g_t)
        mv      = g_t[max_idx]
        ml      = x_t[max_idx]
        bl1     = x_t[-1] - ml

        return mv, ml, bl1


    mv, ml, bl1 = find_inflection(x, gs)

    idx   = np.argmin(np.abs(x - ml))
    x_e   = x[idx]
    beta_e = y_s[idx]

    idx2  = np.argmin(np.abs(x - ml))
    U_e   = xVelocity[idx2]
    U_t   = xVelocity[:3].mean()
    U_s   = U_t - U_e
    delta = bl1

    return {
        "NDP1": delta  / R_B,
        "NDP2": beta_e * R_B / U_t,
        "NDP3": mv     * R_B**2 / U_t,
        "NDP4": U_e    / U_t,
        "NDP5": U_e    / U_s,
    }

# ── main loop ─────────────────────────────────────────────────────────────────

summary_df = pd.DataFrame(index=ROW_INDEX, columns=POWER_LEVELS, dtype=float)

for power_col in POWER_LEVELS:
    matches = [f for f in sample_root.iterdir() if f.is_dir() and f.name.endswith(power_col)]

    if not matches:
        print(f"  [SKIP] no folder ending in '{power_col}' found in {sample_root}")
        continue

    if len(matches) > 1:
        print(f"  [WARN] multiple folders match '{power_col}': {[f.name for f in matches]}, using {matches[0].name}")

    folder = matches[0]
    print(f"\n[{power_col}]  {folder}")
    result = process_folder(folder, R_B)

    if result:
        for ndp, val in result.items():
            summary_df.loc[ndp, power_col] = val

out_path = base / "summary.csv"
summary_df.to_csv(out_path)
print(f"\nSaved: {out_path}\n{summary_df}\n")
  