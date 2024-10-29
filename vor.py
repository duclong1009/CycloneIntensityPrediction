import numpy as np
import metpy.calc as mpcalc
import os

from tqdm import tqdm 
def calculate_wind_speed(u, v):
    """
    Calculate the wind speed from 2D arrays of u-wind and v-wind components.

    Parameters:
    u (2D array): U-wind component (east-west direction)
    v (2D array): V-wind component (north-south direction)

    Returns:
    2D array: Wind speed at each grid point
    """
    wind_speed = np.sqrt(u**2 + v**2)
    return wind_speed


from tqdm import tqdm

modes = ['train','valid','test']
for mode in modes:

    arr = np.load(f"cropped_data/{mode}/data.npz")

    list_v = []
    x_arr = arr['x_arr']
    y_arr = arr['groundtruth']

    for i in range(x_arr.shape[0]):
        x_i = x_arr[i]

        v10m = x_i[1]
        u10m = x_i[0]
        u = x_i[17:25]
        v = x_i[25:33]

        vs = calculate_wind_speed(u,v)
        vs_10m = calculate_wind_speed(u10m, v10m)
        v_combine = np.concatenate([np.expand_dims(vs_10m,0),vs],0)

        list_v.append(v_combine)
    v_all = np.stack(list_v,0)
    finall_arr = np.concatenate([x_arr, v_all],1)


    os.makedirs(f"cropped_data_vor/{mode}", exist_ok=True)
    np.savez(f"cropped_data_vor/{mode}/data.npz",x_arr = x_arr, groundtruth = y_arr)
