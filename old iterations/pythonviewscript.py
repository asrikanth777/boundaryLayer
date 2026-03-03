from paraview import python_view
from paraview.numpy_support import vtk_to_numpy
import numpy as np

try:
    import pandas as pd
    print("pandas available:", pd.__version__)
except ImportError as e:
    print("pandas NOT available")



WINDOW_SIZE = 11
TAIL_FRAC = 0.9
VG_COMPONENT_INDEX = 4

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
    T_arr = pd.GetArray(T_index)
    v_arr = pd.GetArray(v_index)
    if vectGrad is None or T_arr is None or v_arr is None:
        tempPLOT.text(0.5, 0.5, "Missing VectorGradient or T", ha="center", va="center", transform=tempPLOT.transAxes)
        return python_view.figure_to_image(fig)

    vg = vtk_to_numpy(vectGrad)
    dvdy = vg[:, VG_COMPONENT_INDEX]
    temp = vtk_to_numpy(T_arr)
    vNP = vtk_to_numpy(v_arr)
    vU = vNP[:, 0]

    mask = np.isfinite(x) & np.isfinite(dvdy) & np.isfinite(temp)
    x = x[mask]; dvdy = dvdy[mask]; temp = temp[mask]; vU = vU[mask]
    if x.size < 5:
        tempPLOT.text(0.5, 0.5, "Too few valid points", ha="center", va="center", transform=tempPLOT.transAxes)
        return python_view.figure_to_image(fig)

    idx = np.argsort(x)
    x = x[idx]; dvdy = dvdy[idx]; temp = temp[idx]; vU = vU[idx]
    


    kernel1 = np.ones(window_size) / window_size
    kernel2 = np.ones(window_2) / window_2

    dvdy_smooth = np.convolve(dvdy, kernel1, mode='valid')
    final_grad = np.diff(dvdy) / np.diff(x)

    x = x[window_size-1:]
    fg_smooth = np.diff(dvdy_smooth) / np.diff(x)
    fg_s = np.convolve(fg_smooth, kernel2, mode='valid')

    x = np.delete(x, 0)
    temp = np.delete(temp, 0)
    vU = np.delete(vU, 0)
    dvdy = np.delete(dvdy, 0)

    size = final_grad.shape[0]
    side = int(TAIL_FRAC*size)
    examine = x[side:]
    check = fg_s[side:]
    dist = [abs(val-0) for val in check]
    bound = min(dist)
    indx = dist.index(bound)
    bl_thickness = examine[-1] - examine[indx]




    tempPLOT.set_title("Temperature Profile")
    tempPLOT.plot(x, temp[window_size-1:])
    tempPLOT.set_xlabel("x_direction")
    tempPLOT.set_ylabel("Temperature")
    tempPLOT.minorticks_on()

    dv2dxdyPLOT.set_title("Boundary Layer Capture")
    dv2dxdyPLOT.plot(x, np.zeros_like(x))
    dv2dxdyPLOT.plot(x[window_2-1:], fg_s)
    dv2dxdyPLOT.plot(examine[indx], check[indx], "ro")
    dv2dxdyPLOT.set_xlabel("x_direction")
    dv2dxdyPLOT.set_ylabel("dv2/dydx")
    dv2dxdyPLOT.minorticks_on()

    dvdyPLOT.set_title("dv/dy Profile")
    dvdyPLOT.plot(x, dvdy[window_size-1:])
    dvdyPLOT.set_xlabel("x_direction")
    dvdyPLOT.set_ylabel("dv/dy")
    dvdyPLOT.minorticks_on()

    uPLOT.set_title("Axial Velocity Profile")
    uPLOT.plot(x, vU[window_size-1:])
    uPLOT.set_xlabel("x_direction")
    uPLOT.set_ylabel("dv/dy")
    uPLOT.minorticks_on()


    dv2dxdyPLOT.text(
        0.5, 0.5,
        f"Inflection Point: {examine[indx]:.6g} m",
        ha="center",
        va="top",
        transform=dv2dxdyPLOT.transAxes,
        fontsize=11
    )

    dv2dxdyPLOT.text(
        0.5, 0.45,
        f"Boundary Layer Thickness: {bl_thickness*1000:.6g} mm",
        ha="center",
        va="top",
        transform=dv2dxdyPLOT.transAxes,
        fontsize=11
    )
    
    fig.tight_layout()
    return python_view.figure_to_image(fig)
