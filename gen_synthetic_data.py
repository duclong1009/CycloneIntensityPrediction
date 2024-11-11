import numpy as np
from scipy.ndimage import rotate
from tqdm import tqdm
# Tải dữ liệu từ tệp npz
train_data = np.load("cropped_data/train/data.npz")
x_arr = train_data['x_arr']
y_arr = train_data['groundtruth']


# Lọc các chỉ số của phần tử mà y_arr khác 0
non_zero_indices = np.where(y_arr != 0)[0]

# Lấy các phần tử trong x_arr và y_arr tương ứng với các chỉ số khác 0
x_arr_non_zero = x_arr[non_zero_indices]
y_arr_non_zero = y_arr[non_zero_indices]

del x_arr, y_arr
print("Del x arr y arr")
# Đếm số lượng phần tử trong các khoảng giá trị của y_arr_non_zero
idx1 = np.where((y_arr_non_zero > 33) & (y_arr_non_zero < 48))[0]
idx2 = np.where((y_arr_non_zero > 47) & (y_arr_non_zero < 64))[0]
idx3 = np.where(y_arr_non_zero > 63)[0]

# Tạo các mẫu trùng lặp cho idx2 với góc xoay ngẫu nhiên
duplicated_samples_idx3 = []
list_y3 = []
for index in tqdm(idx3):
    angle = np.random.uniform(0, 360)  # Góc xoay ngẫu nhiên
    rotated_sample = rotate(x_arr_non_zero[index], angle=angle, axes=(1, 2), reshape=False)
    duplicated_samples_idx3.append(rotated_sample)
    list_y3.append(y_arr_non_zero[index])
y_arr3 = np.stack(list_y3,0)
arr_idx3 = np.stack(duplicated_samples_idx3,0)
del duplicated_samples_idx3, list_y3

x_syn1 = np.concatenate([x_arr_non_zero, arr_idx3],0)
y_syn1 = np.concatenate([y_arr_non_zero, y_arr3],0)
del x_arr_non_zero, arr_idx3, y_arr_non_zero, y_arr3

# Tạo hai bản sao cho mỗi mẫu trong idx3 với góc xoay ngẫu nhiên
print("Done idx3")
duplicated_samples_idx2 = []
list_y2 = []
for index in tqdm(idx2):
    for _ in range(2):  # Tạo hai mẫu
        angle = np.random.uniform(0, 360)  # Góc xoay ngẫu nhiên
        rotated_sample = rotate(x_syn1[index], angle=angle, axes=(1, 2), reshape=False)
        duplicated_samples_idx2.append(rotated_sample)
        list_y2.append(y_syn1[index])
y_arr2 = np.stack(list_y2,0)

arr_idx2 = np.stack(duplicated_samples_idx2,0)
 
del duplicated_samples_idx2, y_arr2
x_syn2 = np.concatenate([x_syn1, arr_idx2],0)
y_syn2 = np.concatenate([y_syn1, y_arr2],0)
del x_syn1, arr_idx2, y_syn1, y_arr2


np.savez("synthetic_data/train/data/npz",x_arr= x_syn2, groundtruth=y_syn2)
# Kiểm tra kết quả
print("Số lượng mẫu trùng lặp đã tạo cho idx2:", len(duplicated_samples_idx2))
print("Số lượng mẫu trùng lặp đã tạo cho idx3:", len(duplicated_samples_idx3))

