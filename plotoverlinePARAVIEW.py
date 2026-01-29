import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from scipy.signal import savgol_filter

df = pd.read_csv("droppednan.csv")


x = df["Points_0"]
y = df["VectorGradient_4"]

df['du_dy_smooth'] = df['VectorGradient_4'].rolling(window=25, center=True).mean()
df['x_smooth'] = df['Points_0'].rolling(window=25, center=True).mean()

y_smooth = df['du_dy_smooth']
x_smooth = df['x_smooth']

# plt.figure()
# plt.plot(x, y)
# plt.xlabel("x_direction")
# plt.ylabel("dv/dy")
# plt.show()

x_np = np.array(x)
y_np = np.array(y)

x_smooth_np = np.array(x_smooth)
y_smooth_np = np.array(y_smooth)

final_grad = np.diff(y_np)/np.diff(x_np)

fg = np.diff(y_smooth_np)/np.diff(x_smooth_np)
# final_grad_smooth = pd.Series(fg).rolling(
#     window=2, 
#     center=True, 
# ).mean().values

x_np = np.delete(x_np, 0)
x_smooth_np1 = np.delete(x_smooth_np, 0)


#x_smooth1 = np.delete(x_smooth, 0)
# final_grad = np.gradient(y_np, x_np)

plt.figure()
plt.plot(x_np, final_grad)
plt.plot(x_smooth_np1, fg)
plt.xlabel("x_direction")
plt.ylabel("dv2/dydx")
plt.show()

# plt.figure()
# plt.plot(x, y, label="dv/dy")
# plt.plot(x_np, final_grad, label="d/dx(dv/dy)")
# plt.xlabel("x_direction")
# plt.legend()
# plt.show()
