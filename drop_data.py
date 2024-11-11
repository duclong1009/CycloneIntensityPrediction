import numpy as np
import pandas as pd 
import os

modes = ["train", "valid", "test"]

for mode in modes:
    data = np.load(f"cropped_data/{mode}/data.npz")
    x_arr = data['x_arr']
    y_arr = data['groundtruth']
    
    # Find indices where y_arr is not equal to 0
    non_zero_indices = np.where(y_arr != 0)[0]
    
    # Select elements in x_arr where corresponding y_arr is not 0
    x_arr_non_zero = x_arr[non_zero_indices]
    y_arr_non_zero = y_arr[non_zero_indices]
    folder_dir = f"cropped_data_not_z"

    os.makedirs(f"{folder_dir}/{mode}", exist_ok=True)
    np.savez(f"{folder_dir}/{mode}/data.npz",x_arr = x_arr_non_zero, groundtruth=y_arr_non_zero)
    # Optional breakpoint for debugging
    