import pandas as pd 

test_df = pd.read_csv("data07/file_index0_7.csv")
list_tracker = []
list_besttrack = []
current_file_name = ""

for i in range(len(test_df)):
    try:
        _, bt_path, bt_idx, nwp_path, nwp_id = test_df.values[i]
        cyclone_id_name = nwp_path.split("/")[2]
        file_name = nwp_path.split("/")[-1].split(".")[0].split("_")[0]
        year = file_name[:4]
        df_path = f"data/{year}/{cyclone_id_name}/tracker/{file_name}/tracker.csv"
        current_file_name = bt_path
        df = pd.read_csv(df_path)
        # if bt_path != current_file_name:
            
        list_tracker.append(df['NWP Maximum sustained wind speed'].values[nwp_id])
        list_besttrack.append(df['Best-track Maximum sustained wind speed'].values[nwp_id])
    except:
        pass
    

breakpoint()

import matplotlib.pyplot as plt
import numpy as np

# Generate random data for demonstration
# data = np.random.randn(1000)  # 1000 random values from a normal distribution

counts, bins = np.histogram(list_besttrack, bins=30)

# Normalize counts to sum to 100
counts = counts / counts.sum() * 100

# Plot histogram with normalized counts
plt.bar(bins[:-1], counts, width=(bins[1] - bins[0]), edgecolor='black', align='edge', color='skyblue')

# Adding titles and labels
plt.title('Distribution of M.S wind speed')
plt.xlabel('Value')
plt.ylabel('Percentage')

# Save the plot as a PNG file
plt.savefig("MSWS_distribution.png")
# Show the plot
plt.show()
quit()
from repo import model_utils
list_tracker = [i * 0.5 for i in list_tracker]
list_besttrack = [i * 0.5 for i in list_besttrack]

mae, mse, mape, rmse, r2, corr = model_utils.cal_acc(list_tracker, list_besttrack)

y_prd = list_tracker
y_true = list_besttrack

import matplotlib.pyplot as plt


# Plotting the lines
plt.figure(figsize=(10, 5))
plt.plot(y_prd , label='Tracker', linestyle='-', marker='o')
plt.plot(y_true, label='Besttrack', linestyle='-', marker='x')

# Adding titles and labels
plt.title('Comparison of Predicted and True Values')
plt.xlabel('Step')
plt.ylabel('M.S wind speed (m/s)')

# Adding a grid
plt.grid(True, linestyle=':')

# Adding a legend
plt.legend()

# Displaying the plot
plt.show()
plt.savefig("track.png")


breakpoint()