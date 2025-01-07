import numpy as np
list_arr = []
list_gr = []
for i in range(1,5,1):
    arr = np.load(f"cutted_data2/train/train_data{i}.npz")
    list_arr.append(arr['x_arr'])
    list_gr.append(arr['groundtruth'])
combined_arr = np.concatenate(list_arr,0)
combined_gr = np.concatenate(list_gr,0)
np.savez("cutted_data2/train/train_data", x_arr=combined_arr, groundtruth= combined_gr)

