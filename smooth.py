import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Create a sample DataFrame
data = {'Value': [10, 12, 11, 15, 14, 16, 18, 17, 19, 21, 20, 22]}
df = pd.DataFrame(data)

# 2. Define the window size
window_size = 3

# 3. Calculate the Simple Moving Average (SMA)
df['SMA'] = df['Value'].rolling(window=window_size).mean()

# 4. Print the result
print(df)

# You can also plot the results
# plt.figure(figsize=(10, 6))
# plt.plot(df['Value'], label='Original Data')
# plt.plot(df['SMA'], label=f'SMA (Window {window_size})')
# plt.legend()
# plt.title('Simple Moving Average Filter with Pandas')
# plt.show()
