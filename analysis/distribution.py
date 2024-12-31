import numpy as np

folder_name = f"cropped_data_not_z/test"
arr = np.load(f"{folder_name}/data.npz")
y_arr = arr['groundtruth']

import matplotlib.pyplot as plt

# Assuming y_arr is your data array
# For example, y_arr could be arr['groundtruth'] or any other numerical dataset
plt.figure(figsize=(10, 6))

# Plotting the histogram
plt.hist(y_arr, bins=30, edgecolor='black')  # Adjust the `bins` parameter as needed
plt.title('Distribution of Ground Truth Values')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.savefig(f"{folder_name}/data_distribution.png")

