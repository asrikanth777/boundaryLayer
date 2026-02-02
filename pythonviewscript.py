from paraview import python_view
from paraview.numpy_support import vtk_to_numpy
import numpy as np


WINDOW_SIZE = 11
TAIL_FRAC = 0.10
VG_COMPONENT_INDEX = 4

def setup_data(view):
    view.EnableAllAttributeArrays()

def render(view, width, height):
    fig = python_view.matplotlib_figure(width, height)
    tempPLOT = fig.add_subplot(2, 2, 1)
    dv2dxdyPLOT = fig.add_subplot(2, 2, 4)
    dvdyPLOT = fig.add_subplot(2, 2, 3)
    uPLOT = fig.add_subplot(2, 2, 2)

    dobj = view.GetVisibleDataObjectForRendering(0)
    if dobj is None:
        tempPLOT.text(0.5, 0.5, "No visible data object", ha="center", va="center", transform=tempPLOT.transAxes)
        return python_view.figure_to_image(fig)

    pts_vtk = dobj.GetPoints()
    pd = dobj.GetPointData()

    if pts_vtk is None or pd is None:
        tempPLOT.text(0.5, 0.5, "Missing points or point data", ha="center", va="center", transform=tempPLOT.transAxes)
        return python_view.figure_to_image(fig)

    pts = vtk_to_numpy(pts_vtk.GetData())
    x = pts[:, 0]  # Points_0

    vectGrad = pd.GetArray("VectorGradient")
    T_arr = pd.GetArray("T")
    v_arr = pd.GetArray("v")
    if vectGrad is None or T_arr is None or v_arr is None:
        tempPLOT.text(0.5, 0.5, "Missing VectorGradient or T", ha="center", va="center", transform=tempPLOT.transAxes)
        return python_view.figure_to_image(fig)

    vg = vtk_to_numpy(vectGrad)
    dvdy = vg[:, VG_COMPONENT_INDEX]
    tempNP = vtk_to_numpy(T_arr)
    vNP = vtk_to_numpy(v_arr)
    vU = vNP[:, 0]

    mask = np.isfinite(x) & np.isfinite(dvdy) & np.isfinite(tempNP)
    x = x[mask]; dvdy = dvdy[mask]; tempNP = tempNP[mask]
    if x.size < 5:
        tempPLOT.text(0.5, 0.5, "Too few valid points", ha="center", va="center", transform=tempPLOT.transAxes)
        return python_view.figure_to_image(fig)

    idx = np.argsort(x)
    x = x[idx]; dvdy = dvdy[idx]; tempNP = tempNP[idx]; vU = vU[idx]

    w = int(WINDOW_SIZE)
    if w < 3: w = 3
    if w % 2 == 0: w += 1

    if x.size <= w + 2:
        x_smooth = x.copy()
        dvdy_smooth = dvdy.copy()
    else:
        kernel = np.ones(w) / w
        x_smooth = np.convolve(x, kernel, mode="valid")
        dvdy_smooth = np.convolve(dvdy, kernel, mode="valid")

    final_grad = np.diff(dvdy) / np.diff(x)
    fg_smooth = np.diff(dvdy_smooth) / np.diff(x_smooth)

    x_d = x[1:]
    dvdy_d = dvdy[1:]
    vU_d = vU[1:]
    x_s_d = x_smooth[1:] if x_smooth.size > 1 else x_smooth
    temp_d = tempNP[1:]

    n = fg_smooth.size
    side = int((1.0 - TAIL_FRAC) * n)
    side = max(0, min(side, n-1))

    examine = x_s_d[side:]
    check = fg_smooth[side:]
    indx = int(np.argmin(np.abs(check)))

    inflect_x = float(examine[indx])
    inflect_y = float(check[indx])
    bl_thickness = float(examine[-1] - examine[indx])

    tempPLOT.set_title("Temperature Profile")
    tempPLOT.plot(x_d, temp_d)
    tempPLOT.set_xlabel("x_direction")
    tempPLOT.set_ylabel("Temperature")
    tempPLOT.minorticks_on()

    dv2dxdyPLOT.set_title("Boundary Layer Capture")
    dv2dxdyPLOT.plot(x_d, np.zeros_like(final_grad))
    dv2dxdyPLOT.plot(x_s_d, fg_smooth)
    dv2dxdyPLOT.plot(inflect_x, inflect_y, "ro")
    dv2dxdyPLOT.set_xlabel("x_direction")
    dv2dxdyPLOT.set_ylabel("dv2/dydx")
    dv2dxdyPLOT.minorticks_on()

    dvdyPLOT.set_title("dv/dy Profile")
    dvdyPLOT.plot(x_d, dvdy_d)
    dvdyPLOT.set_xlabel("x_direction")
    dvdyPLOT.set_ylabel("dv/dy")
    dvdyPLOT.minorticks_on()

    uPLOT.set_title("Axial Velocity Profile")
    uPLOT.plot(x_d, vU_d)
    uPLOT.set_xlabel("x_direction")
    uPLOT.set_ylabel("dv/dy")
    uPLOT.minorticks_on()


    dv2dxdyPLOT.text(
        0.5, 0.95,
        f"Inflection Point: {inflect_x:.6g} m",
        ha="center",
        va="top",
        transform=dv2dxdyPLOT.transAxes,
        fontsize=11
    )

    dv2dxdyPLOT.text(
        0.5, 0.88,
        f"Boundary Layer Thickness: {bl_thickness*1000:.6g} mm",
        ha="center",
        va="top",
        transform=dv2dxdyPLOT.transAxes,
        fontsize=11
    )

    
    fig.tight_layout()
    return python_view.figure_to_image(fig)