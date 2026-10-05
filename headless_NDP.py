"""
Stagnation-line NDP extraction for each power level.
Headless version. Run from a terminal:

    pvpython headless_NDP.py --radius 0.025 --root path/to/sample_runs
    (pvbatch works the same way)

Calculations are IDENTICAL to the original script. Changes:
  1. Loads ONLY .vts files (every piece); .pvts files are never read.
  2. The pipeline is deleted after each power level (no cross-contamination).
  3. Folder matching won't let '50kw' pick up the '150kw' folder.
  4. Array names match this output (V, V_X, V_Y) and PlotOverLine uses the
     ParaView 5.10+ API.
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from paraview.simple import *
from paraview import servermanager
from vtk.util import numpy_support as ns

paraview.simple._DisableFirstRenderCameraReset()

# ── constants ────────────────────────────────────────────────────────────────

H_index   = 'H';  M_index = 'M';  T_index = 'T'
p_index   = 'p';  rho_index = 'rho';  v_index = 'V'      # CHANGED: 'v' -> 'V'
v_x_index = 'V_X';  v_y_index = 'V_Y'                     # CHANGED: v_X/v_Y -> V_X/V_Y

POWER_LEVELS = ['50kw', '75kw', '100kw', '125kw', '150kw', '175kw', '200kw']
ROW_INDEX    = ['NDP1', 'NDP2', 'NDP3', 'NDP4', 'NDP5']


# ── pipeline processing function ─────────────────────────────────────────────

def process_folder(folder: Path, R_B: float) -> dict:
    if not folder.exists():
        print(f"  [SKIP] folder not found: {folder}")
        return None

    # CHANGED: only .vts files; .pvts wrappers are never opened
    vts_files = sorted(folder.glob("*.vts"))

    if not vts_files:
        print(f"  [SKIP] no .vts files in: {folder}")
        return None

    print(f"  VTS ({len(vts_files)}): {[f.name for f in vts_files]}")

    # CHANGED: track every proxy so the pipeline can be deleted afterwards
    created = []
    def track(proxy):
        created.append(proxy)
        return proxy

    try:
        readers = [track(XMLStructuredGridReader(FileName=str(f))) for f in vts_files]

        # CHANGED: group only THIS folder's readers (was: every source in the session)
        flowfield = track(GroupDatasets(Input=readers))

        cdpd1 = track(CellDatatoPointData(Input=flowfield))
        cdpd1.CellDataArraytoprocess = [H_index, M_index, T_index,
                                        p_index, rho_index, v_index]

        calc1 = track(Calculator(Input=cdpd1))
        calc1.Function = f"{v_x_index}*iHat + {v_y_index}*jHat"

        deriv1 = track(ComputeDerivatives(Input=calc1))
        deriv1.Vectors = ['POINTS', 'Result']

        cdpd2 = track(CellDatatoPointData(Input=deriv1))
        cdpd2.CellDataArraytoprocess = ['ScalarGradient', 'VectorGradient']

        # CHANGED: ParaView 5.10+ API (same line, same resolution)
        pol = track(PlotOverLine(Input=cdpd2))
        pol.Point1     = [0.4,   0, 0]
        pol.Point2     = [0.565, 0, 0]
        pol.Resolution = 5000

        data = servermanager.Fetch(pol)   # local copy; survives the Delete below

    finally:
        # CHANGED: delete downstream-first so the next power level starts clean
        for proxy in reversed(created):
            Delete(proxy)

    # ── everything below is the original calculation, untouched ──────────────

    pts        = ns.vtk_to_numpy(data.GetPoints().GetData())
    Points_0   = pts[:, 0]
    pd_        = data.GetPointData()

    # Diagnostic only (does not alter any values): warn if the line leaves the mesh
    mask_arr = pd_.GetArray("vtkValidPointMask")
    if mask_arr is not None:
        n_bad = int((ns.vtk_to_numpy(mask_arr) == 0).sum())
        if n_bad:
            print(f"  [WARN] {n_bad} of {mask_arr.GetNumberOfTuples()} line points "
                  f"are outside the mesh.")

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


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Stagnation-line NDP extraction")
    ap.add_argument("--radius", type=float, required=True, help="sample body radius [m]")
    ap.add_argument("--root", required=True,
                    help="top-level folder for SAMPLE runs (contains 50kw/, 100kw/, ...)")
    ap.add_argument("--out", default="summary.csv", help="output CSV path (default: summary.csv)")
    args = ap.parse_args()

    R_B         = args.radius
    sample_root = Path(args.root).resolve()
    if not sample_root.is_dir():
        raise FileNotFoundError(f"--root is not a folder: {sample_root}")

    summary_df = pd.DataFrame(index=ROW_INDEX, columns=POWER_LEVELS, dtype=float)

    for power_col in POWER_LEVELS:
        # (?<!\d) stops '50kw' from matching '150kw'
        pat = re.compile(rf'(?<!\d){re.escape(power_col)}$')
        matches = sorted(f for f in sample_root.iterdir() if f.is_dir() and pat.search(f.name))

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

    out_path = Path(args.out).resolve()
    summary_df.to_csv(out_path)
    print(f"\nSaved: {out_path}\n{summary_df}\n")


if __name__ == "__main__":
    main()