import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np

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

# create whatever smoothing windows you want
for w in [3,5,7,9,11,13,15,17,19,21,23,25]:
    df[f"grad_smooth_{w}"] = (
        df["grad_raw"]
        .rolling(w, center=True, min_periods=1)
        .mean()
    )


def find_inflection(x, grad, window=None, tail_frac=0.9):
    """
    x        : array-like (monotonic)
    grad     : array-like
    window   : int or None (rolling mean window)
    tail_frac: fraction of domain to search in (e.g. 0.9 = last 10%)
    """
    grad_s = grad

    if window is not None and window > 1:
        grad_s = (
            pd.Series(grad)
            .rolling(window, center=True, min_periods=1)
            .mean()
            .to_numpy()
        )

    n = grad_s.size
    side = int(tail_frac * n)

    x_tail = x[side:]
    g_tail = grad_s[side:]

    idx = np.argmin(np.abs(g_tail))   # closest to zero
    location = x_tail[idx]
    thickness = x_tail[-1] - location
    max_idx = np.argmax(g_tail)
    max_val = g_tail[max_idx]
    max_loc = x_tail[max_idx]


    return location, thickness, max_val, max_loc


windows = [None, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]

results = {}

for w in windows:
    key = "raw" if w is None else f"w{w}"
    loc, bl, mv, ml = find_inflection(x_f, grad, window=w, tail_frac=0.9)
    results[key] = {"location": loc, "bl_thickness": bl, "max_val": mv, "max_loc": ml}

df_results = (
    pd.DataFrame(results)
      .T                       # keys become rows
      .reset_index()
      .rename(columns={"index": "smoothing"})
)

print(df)

print(df_results)

x_pts = [
    results["raw"]["location"],
    results["w7"]["location"],
    results["w19"]["location"],
    results["raw"]["max_loc"],
    results["w7"]["max_loc"],
    results["w19"]["max_loc"],
]

# y locations (zero-crossing)
y_pts = [
    0.0, 
    0.0, 
    0.0,
    results["raw"]["max_val"],
    results["w7"]["max_val"],
    results["w19"]["max_val"],
]




plt.figure()

plt.plot(x_f, df["grad_raw"], label="raw", linewidth=1)
plt.plot(x_f, np.zeros_like(x_f), label="zero", linewidth=1)
plt.plot(x_f, df["grad_smooth_7"], label="window = 7", linewidth=2)
plt.plot(x_f, df["grad_smooth_19"], label="window = 19", linewidth=2)
plt.scatter(x_pts, y_pts,
            s=60,
            marker="x",
            color="k",
            zorder=10,
            label="inflection points")


plt.xlabel("x")
plt.ylabel("gradient")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


