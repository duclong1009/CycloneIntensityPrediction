import numpy as np
import metpy.calc as mpcalc

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




arr = np.load("data07/cropped_data/train/data.npz")
list_v = []
i = 0
x_arr = arr['x_arr']
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

# a = arr['x_arr'][0]
breakpoint()