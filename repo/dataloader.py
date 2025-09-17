from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os 
from sklearn.preprocessing import MinMaxScaler
import pickle
import torch
class CycloneDataset(Dataset):
    def __init__(self,file_path ="file_index.csv", mode="train", args=None ,besttrack_scaler_path="output/scaler/besttrackscaler.pkl",nwp_scaler_path="output/scaler/nwpscaler.pkl", ):
        super().__init__()
        self.map_idx = pd.read_csv(file_path)
        self.besttrack_scaler_path = besttrack_scaler_path
        self.nwp_scaler_path = nwp_scaler_path
        
        self.besttrack_scaler = MinMaxScaler()
        self.nwp_scaler = MinMaxScaler()
        self.mode=  mode
        self.args = args

    def fit_data(self,arr,y):
        if self.mode == "train":
            arr_shape = arr.shape
            arr = arr.reshape((arr.shape[0], -1))
            y=  np.expand_dims(np.array(y),0).reshape((1,1))
            self.besttrack_scaler.partial_fit(y)
            self.nwp_scaler.partial_fit(arr)
            arr = self.nwp_scaler.transform(arr)
            y = self.besttrack_scaler.transform(y)
            arr = arr.reshape(arr_shape)
        else:
            arr_shape = arr.shape
            arr = arr.reshape((arr.shape[0], -1))
            y=  np.expand_dims(np.array(y),0).reshape((1,1))
            arr = self.nwp_scaler.transform(arr)
            y = self.besttrack_scaler.transform(y)
            arr = arr.reshape(arr_shape)
        return arr, y
    
    def convert_latlon_to_index(self, latitude,longitude):
        import math
        x = math.ceil((latitude + 20)/ 0.125)
        y = math.ceil((longitude - 80)/ 0.125)
        return x,y
        
    def crop_arr(self, arr, lat, lon, radius=80):

        # Ensure the input array is of the correct shape
        assert arr.shape == (58, 481, 481), "Input array must have shape (58, 481, 481)"
        
        # Create an empty array filled with zeros
        cropped_arr = np.zeros((58, radius * 2 + 1, radius * 2 + 1))
        
        # Calculate the start and end indices for the input array
        lat_start = max(lat - radius, 0)
        lat_end = min(lat + radius, arr.shape[1] - 1)
        lon_start = max(lon - radius, 0)
        lon_end = min(lon + radius, arr.shape[2] - 1)
        
        # Calculate the start and end indices for the cropped array
        crop_lat_start = max(0, radius - lat)
        crop_lat_end = crop_lat_start + (lat_end - lat_start + 1)
        crop_lon_start = max(0, radius - lon)
        crop_lon_end = crop_lon_start + (lon_end - lon_start + 1)
        
        # Copy the valid region from the input array to the cropped array
        cropped_arr[:, crop_lat_start:crop_lat_end, crop_lon_start:crop_lon_end] = arr[:, lat_start:lat_end+1, lon_start:lon_end+1]
        
        return cropped_arr
    
    def set_scaler(self, besttrack_scaler, nwp_scaler):
        self.besttrack_scaler = besttrack_scaler
        self.nwp_scaler = nwp_scaler
        
    def get_scaler(self,):
        from copy import deepcopy
        return deepcopy(self.besttrack_scaler), deepcopy(self.nwp_scaler)
    
    def __getitem__(self,idx):
        _,_, besttrack_path, besttrack_id, nwp_path, nwp_id = self.map_idx.values[idx]
        bt_df = pd.read_csv(besttrack_path)
    
        bt_lat, bt_lon =  bt_df['Latitude of the center'][besttrack_id], bt_df['Longitude of the center'][besttrack_id]
        converted_lat, converted_lon = self.convert_latlon_to_index(bt_lat, bt_lon)
        
        bt_wp = bt_df['Maximum sustained wind speed'][besttrack_id]
        
        nwp_arr = np.load(nwp_path)
        
        
        u10m = np.expand_dims(nwp_arr['u10m'][nwp_id],0) # 1,,481,481
        v10m = np.expand_dims(nwp_arr['v10m'][nwp_id],0)
        t2m = np.expand_dims(nwp_arr['t2m'][nwp_id],0)
        td2m = np.expand_dims(nwp_arr['td2m'][nwp_id],0)
        ps = np.expand_dims(nwp_arr['ps'][nwp_id],0)
        pmsl = np.expand_dims(nwp_arr['pmsl'][nwp_id],0)
        skt = np.expand_dims(nwp_arr['skt'][nwp_id],0)
        total_cloud = np.expand_dims(nwp_arr['total_cloud'][nwp_id],0)
        rain = np.expand_dims(nwp_arr['rain'][nwp_id],0)
        
        h = nwp_arr['h'][nwp_id]
        u = nwp_arr['u'][nwp_id]
        v = nwp_arr['v'][nwp_id]
        t = nwp_arr['t'][nwp_id]
        q = nwp_arr['q'][nwp_id]
        w = nwp_arr['w'][nwp_id]
        # Albedo = nwp_arr['Albedo']
        Z = np.expand_dims(nwp_arr['Z'],0)
        terrain = np.expand_dims(nwp_arr['terrain'],0)
       
        arr = np.concatenate([u10m, v10m, t2m, td2m, ps, pmsl, skt, total_cloud, rain, h, u, v, t, q, w, terrain], axis=0)
        arr, bt_wp = self.fit_data(arr,bt_wp)
        cropped_arr = self.crop_arr(arr, converted_lat,converted_lon, self.args.radius)
        return {"x": cropped_arr, "y": bt_wp, "loc": np.array([converted_lat, converted_lon])}
    

    def __len__(self):
        return len(self.map_idx)
        
class CycloneDataset2(Dataset):
    def __init__(self,data_dir ="cutted_data/train", mode="train", nwp_scaler=None, bt_scaler = None, args=None ,besttrack_scaler_path="output/scaler/besttrackscaler.pkl",nwp_scaler_path="output/scaler/nwpscaler.pkl", ):
        super().__init__()
        
        
        self.arr = np.load(data_dir)
        self.x_train, self.y_train, self.nwp_id = self.arr['x_arr'], self.arr['groundtruth'], self.arr['leading_time']
        
        self.besttrack_scaler_path = besttrack_scaler_path
        self.nwp_scaler_path = nwp_scaler_path
        
        self.nwp_scaler = nwp_scaler
        self.bt_scaler = bt_scaler
        
        self.mode=  mode
        self.args = args

    def fit_data(self,arr,y):
        #
        arr_shape = arr.shape
        reshaped_arr = arr.transpose((1,2,0))
        
        reshaped_arr = reshaped_arr.reshape((reshaped_arr.shape[0] * reshaped_arr.shape[1], -1))
        
        # self.bt_scaler.fit(y)
        reshaped_arr =  self.nwp_scaler.transform(reshaped_arr)
        reshaped_arr = reshaped_arr.reshape(arr_shape[1],arr_shape[2], arr_shape[0])
        reshaped_arr = reshaped_arr.transpose(2,0,1)
        # reshaped_y = y.reshape(y.shape[0],1)
        
        if self.args.transform_groundtruth:
            y=  np.expand_dims(np.array(y),0).reshape((1,1))
            y = self.bt_scaler.transform(y)
        # y = self.besttrack_scaler.transform(y)
        return reshaped_arr, y


    def __getitem__(self,idx):
        arr = self.x_train[idx]
        
        nwp_id = self.nwp_id[idx]
        if len(arr.shape) == 4:
            arr = arr[nwp_id]
        elif len(arr.shape) == 3:
            arr = arr

        bt_wp = self.y_train[idx]
        bt_wp = bt_wp * 0.5
        arr, bt_wp = self.fit_data(arr,bt_wp)
        # cropped_arr = self.crop_arr(arr, converted_lat,converted_lon, self.args.radius)
        return {"x": arr, "y": bt_wp}
    

    def __len__(self):
        return self.x_train.shape[0]
        
class VITDataset6(Dataset):
    """
    For training prompt6 the item format is: 
   
        
    """
    def __init__(self,data_dir ="cutted_data/train", mode="train", nwp_scaler=None, bt_scaler = None, args=None ,besttrack_scaler_path="output/scaler/besttrackscaler.pkl",nwp_scaler_path="output/scaler/nwpscaler.pkl", ):
        super().__init__()
        
        self.features = args.list_features        
            
        self.arr = np.load(data_dir)
        self.x_train, self.y_train, self.his = self.arr['x_arr'], self.arr['groundtruth'], self.arr['his']
        
        if self.features is not None:
            self.x_train = self.x_train[:,self.features, :,:]
            
        self.besttrack_scaler_path = besttrack_scaler_path
        self.nwp_scaler_path = nwp_scaler_path
        
        self.nwp_scaler = nwp_scaler
        self.bt_scaler = bt_scaler
        
        self.mode=  mode
        self.args = args
        self.image_size = args.image_size

    def fit_data(self,arr,y):
    
        arr_shape = arr.shape
        reshaped_arr = arr.transpose((1,2,0))
        
        reshaped_arr = reshaped_arr.reshape((reshaped_arr.shape[0] * reshaped_arr.shape[1], -1))
        
        # self.bt_scaler.fit(y)
        reshaped_arr =  self.nwp_scaler.transform(reshaped_arr)
        reshaped_arr = reshaped_arr.reshape(arr_shape[1],arr_shape[2], arr_shape[0])
        reshaped_arr = reshaped_arr.transpose(2,0,1)
        
        
        # reshaped_arr = reshaped_arr[:10,:,:]
        # reshaped_y = y.reshape(y.shape[0],1)
        
        if self.args.transform_groundtruth:
            y = np.expand_dims(np.array(y),0).reshape((1,1))
            y = self.bt_scaler.transform(y)
        # y = self.besttrack_scaler.transform(y)
        return reshaped_arr, y

    def __getitem__(self,idx):
        """
        x_train: [400,4,63,101,101] / [400,63,101,101]
        """

        arr = self.x_train[idx]

        ## Crop image to get data around the center, (51,51) is the center of the image
        if len(arr.shape) == 4:
            arr = arr[-1,:,51 - self.image_size // 2: 51 + self.image_size // 2, 51 - self.image_size // 2: 51 + self.image_size // 2]
        elif len(arr.shape) == 3:
            arr = arr[:, 51 - self.image_size // 2: 51 + self.image_size // 2, 51 - self.image_size // 2: 51 + self.image_size // 2]
    
        ## Get the groundtruth
        bt_wp = self.y_train[idx]
        bt_wp = bt_wp * 0.5
        
        ## Fit the data
        arr, bt_wp = self.fit_data(arr,bt_wp)
        
        ## Get the history data
        his = self.his[idx]
        arr = [arr, his]
        return {"x": arr, "y": bt_wp}

    def __len__(self):
        return self.x_train.shape[0]

class VITDataset6_2(Dataset):
    """
    For training prompt6 the item format is: 
        - arr: nwp data
    """
    def __init__(self,data_dir ="cutted_data/train", mode="train", nwp_scaler=None, bt_scaler = None, args=None ,besttrack_scaler_path="output/scaler/besttrackscaler.pkl",nwp_scaler_path="output/scaler/nwpscaler.pkl", ):
        super().__init__()
        
        self.features = args.list_features        
            
        self.arr = np.load(data_dir)

        self.x_train, self.y_train, self.his, self.nwp_id = self.arr['x_arr'], self.arr['groundtruth'], self.arr['his'], self.arr['leading_time']
        
        if self.features is not None:
            self.x_train = self.x_train[:,self.features, :,:]
            
        self.besttrack_scaler_path = besttrack_scaler_path
        self.nwp_scaler_path = nwp_scaler_path
        
        self.nwp_scaler = nwp_scaler
        self.bt_scaler = bt_scaler
        
        self.mode=  mode
        self.args = args
        self.image_size = args.image_size

    def fit_data(self,arr,y):
        # breakpoint()
        arr_shape = arr.shape
        if len(arr_shape) == 4:
            # print(arr_shape)
            reshaped_arr = arr.transpose((0,2,3,1))
            reshaped_arr = reshaped_arr.reshape((reshaped_arr.shape[0] * reshaped_arr.shape[1] * reshaped_arr.shape[2], -1))
            reshaped_arr =  self.nwp_scaler.transform(reshaped_arr)
            reshaped_arr = reshaped_arr.reshape(arr_shape[0],arr_shape[2], arr_shape[3], arr_shape[1])
            reshaped_arr = reshaped_arr.transpose(0,3,1,2)

        elif len(arr_shape) == 3:
            reshaped_arr = arr.transpose((1,2,0))
            
            reshaped_arr = reshaped_arr.reshape((reshaped_arr.shape[0] * reshaped_arr.shape[1], -1))
            
            # self.bt_scaler.fit(y)
            reshaped_arr =  self.nwp_scaler.transform(reshaped_arr)
            reshaped_arr = reshaped_arr.reshape(arr_shape[1],arr_shape[2], arr_shape[0])
            reshaped_arr = reshaped_arr.transpose(2,0,1)

        if self.args.transform_groundtruth:
            y = np.expand_dims(np.array(y),0).reshape((1,1))
            y = self.bt_scaler.transform(y)
        # y = self.besttrack_scaler.transform(y)
        return reshaped_arr, y

    def __getitem__(self,idx):
        """
        x_train: [400,4,63,101,101] / [400,63,101,101]
        """

        arr = self.x_train[idx]
        
        if len(arr.shape) == 4:
            arr = arr[:,:,:self.image_size,:self.image_size]
        elif len(arr.shape) == 3:
            arr = arr[:,:self.image_size,:self.image_size]
        

        bt_wp = self.y_train[idx]
        bt_wp = bt_wp * 0.5
        
        arr, bt_wp = self.fit_data(arr,bt_wp)
        his = self.his[idx]
        nwp_id = self.nwp_id[idx]
        arr = [arr, his, nwp_id]
        return {"x": arr, "y": bt_wp}

    def __len__(self):
        return self.x_train.shape[0]
        




class VITDataset6_4(Dataset):
    """
    For training prompt6 the item format is: 
        - arr: nwp data
    """
    def __init__(self,data_dir ="cutted_data/train", mode="train", nwp_scaler=None, bt_scaler = None, args=None ,besttrack_scaler_path="output/scaler/besttrackscaler.pkl",nwp_scaler_path="output/scaler/nwpscaler.pkl", ):
        super().__init__()
        
        self.features = args.list_features        
            
        self.arr = np.load(data_dir)

        self.x_train, self.y_train, self.his, self.nwp_id = self.arr['x_arr'], self.arr['groundtruth'], self.arr['his'], self.arr['leading_time']
        
        hres_data= np.load(f"{args.data_dir}/{mode}/hres_info.npz")
        self.hres_info = hres_data['hres_info']
        self.checked_list = hres_data['checked']
        
        if self.features is not None:
            self.x_train = self.x_train[:,self.features, :,:]
            
        self.besttrack_scaler_path = besttrack_scaler_path
        self.nwp_scaler_path = nwp_scaler_path
        
        self.nwp_scaler = nwp_scaler
        self.bt_scaler = bt_scaler
        
        self.mode=  mode
        self.args = args
        self.image_size = args.image_size

    def fit_data(self,arr,y):
        # breakpoint()
        
        arr_shape = arr.shape
        if len(arr_shape) == 4:
            # print(arr_shape)
            reshaped_arr = arr.transpose((0,2,3,1))
            reshaped_arr = reshaped_arr.reshape((reshaped_arr.shape[0] * reshaped_arr.shape[1] * reshaped_arr.shape[2], -1))
            reshaped_arr =  self.nwp_scaler.transform(reshaped_arr)
            reshaped_arr = reshaped_arr.reshape(arr_shape[0],arr_shape[2], arr_shape[3], arr_shape[1])
            reshaped_arr = reshaped_arr.transpose(0,3,1,2)

        elif len(arr_shape) == 3:
            reshaped_arr = arr.transpose((1,2,0))
            
            reshaped_arr = reshaped_arr.reshape((reshaped_arr.shape[0] * reshaped_arr.shape[1], -1))
            
            # self.bt_scaler.fit(y)
            reshaped_arr =  self.nwp_scaler.transform(reshaped_arr)
            reshaped_arr = reshaped_arr.reshape(arr_shape[1],arr_shape[2], arr_shape[0])
            reshaped_arr = reshaped_arr.transpose(2,0,1)

        if self.args.transform_groundtruth:
            y = np.expand_dims(np.array(y),0).reshape((1,1))
            y = self.bt_scaler.transform(y)
        # y = self.besttrack_scaler.transform(y)
        return reshaped_arr, y

    def __getitem__(self,idx):
        """
        x_train: [400,4,63,101,101] / [400,63,101,101]
        """
        checked = self.checked_list[idx]
        arr = self.x_train[idx]
        his = self.his[idx]
        nwp_id = self.nwp_id[idx]
        hres_info = self.hres_info
        
        arr = arr[nwp_id]
        
        if len(arr.shape) == 4:
            arr = arr[:,:,:self.image_size,:self.image_size]
        elif len(arr.shape) == 3:
            arr = arr[:,:self.image_size,:self.image_size]
        

        bt_wp = self.y_train[idx]
        bt_wp = bt_wp * 0.5
        
        arr, bt_wp = self.fit_data(arr,bt_wp)
        
        arr = [arr, his, nwp_id]
        return {"x": arr, "y": bt_wp}

    def __len__(self):
        return self.x_train.shape[0]
    
    
class VITDataset6_5(Dataset):
    def __init__(self, data_dir="cutted_data/train", mode="train", nwp_scaler=None, bt_scaler=None, args=None, 
                 besttrack_scaler_path="output/scaler/besttrackscaler.pkl", nwp_scaler_path="output/scaler/nwpscaler.pkl"):
        super().__init__()
        self.historical_data_length = args.historical_data_length
        self.features = args.list_features        
        self.arr = np.load(data_dir)

        self.x_train, self.y_train, self.his, self.nwp_id = self.arr['x_arr'], self.arr['groundtruth'], self.arr['his'], self.arr['leading_time']
        
        hres_data = np.load(f"{args.data_dir}/{mode}/hres_info.npz")
        self.hres_info = hres_data['hres_info']
        self.checked_list = hres_data['checked']
        
        # Filter indices where checked is 1
        valid_indices = np.where(self.checked_list == 1)[0]
        
        print(f"Data {mode}: No. sample{len(self.y_train)} No. valid sample {len(valid_indices)}")
        self.valid_indices = valid_indices
        self.x_train = self.x_train[valid_indices]
        self.y_train = self.y_train[valid_indices]
        self.his = self.his[valid_indices]
        self.nwp_id = self.nwp_id[valid_indices]
        self.checked_list = self.checked_list[valid_indices]  # Update checked_list to only valid samples
        self.check_data_values()
        if self.features is not None:
            self.x_train = self.x_train[:, self.features, :, :]

        self.besttrack_scaler_path = besttrack_scaler_path
        self.nwp_scaler_path = nwp_scaler_path
        
        self.nwp_scaler = nwp_scaler 
        self.bt_scaler = bt_scaler 
        
        self.mode = mode
        self.args = args
        self.image_size = args.image_size

        # Determine max sequence length for padding (assuming arr has variable sequence length)
        self.max_seq_len = max([nwp_id for x, nwp_id in zip(self.x_train, self.nwp_id)])
    
    def check_data_values(self):
        """Check for NaN, inf and print basic stats for all data arrays"""
        print("\nChecking data arrays for NaN/inf values and stats:")
        
        # Check x_train
        print("\nx_train:")
        print(f"Shape: {self.x_train.shape}")
        print(f"NaN count: {np.isnan(self.x_train).sum()}")
        print(f"Inf count: {np.isinf(self.x_train).sum()}")
        print(f"Min: {np.min(self.x_train)}, Max: {np.max(self.x_train)}")
        
        # Check y_train
        print("\ny_train:")
        print(f"Shape: {self.y_train.shape}")
        print(f"NaN count: {np.isnan(self.y_train).sum()}")
        print(f"Inf count: {np.isinf(self.y_train).sum()}")
        print(f"Min: {np.min(self.y_train)}, Max: {np.max(self.y_train)}")
        
        # Check his
        print("\nhis:")
        print(f"Shape: {self.his.shape}")
        print(f"NaN count: {np.isnan(self.his).sum()}")
        print(f"Inf count: {np.isinf(self.his).sum()}")
        print(f"Min: {np.min(self.his)}, Max: {np.max(self.his)}")
        
        # Check nwp_id
        print("\nnwp_id:")
        print(f"Shape: {self.nwp_id.shape}")
        print(f"NaN count: {np.isnan(self.nwp_id).sum()}")
        print(f"Inf count: {np.isinf(self.nwp_id).sum()}")
        print(f"Min: {np.min(self.nwp_id)}, Max: {np.max(self.nwp_id)}")
        
        # Check hres_info
        print("\nhres_info:")
        print(f"Shape: {self.hres_info.shape}")
        print(f"NaN count: {np.isnan(self.hres_info).sum()}")
        print(f"Inf count: {np.isinf(self.hres_info).sum()}")
        print(f"Min: {np.min(self.hres_info)}, Max: {np.max(self.hres_info)}")
        
        # Check checked_list
        print("\nchecked_list:")
        print(f"Shape: {self.checked_list.shape}")
        print(f"NaN count: {np.isnan(self.checked_list).sum()}")
        print(f"Inf count: {np.isinf(self.checked_list).sum()}")
        print(f"Min: {np.min(self.checked_list)}, Max: {np.max(self.checked_list)}")
        print(f"Unique values: {np.unique(self.checked_list, return_counts=True)}")

    def fit_data(self, arr, y):
        arr_shape = arr.shape
        if len(arr_shape) == 4:  # [seq_len, n_fts, height, width]
            # Transpose and reshape for scaling
            reshaped_arr = arr.transpose((0, 2, 3, 1))  # [seq_len, height, width, n_fts]
            reshaped_arr = reshaped_arr.reshape((-1, arr_shape[1]))  # Flatten spatial and feature dims
            reshaped_arr = self.nwp_scaler.transform(reshaped_arr)
            reshaped_arr = reshaped_arr.reshape(arr_shape[0], arr_shape[2], arr_shape[3], arr_shape[1])
            reshaped_arr = reshaped_arr.transpose(0, 3, 1, 2)  # Back to [seq_len, n_fts, height, width]

        elif len(arr_shape) == 3:  # [n_fts, height, width] or similar
            reshaped_arr = arr.transpose((1, 2, 0))
            reshaped_arr = reshaped_arr.reshape((-1, arr_shape[0]))
            reshaped_arr = self.nwp_scaler.transform(reshaped_arr)
            reshaped_arr = reshaped_arr.reshape(arr_shape[1], arr_shape[2], arr_shape[0])
            reshaped_arr = reshaped_arr.transpose(2, 0, 1)

        if self.args.transform_groundtruth:
            y = np.expand_dims(np.array(y), 0).reshape((1, 1))
            y = self.bt_scaler.transform(y)

        return reshaped_arr, y

    def __getitem__(self, idx):
        """
        Only processes samples where checked is 1 (already filtered in __init__)
        """
        arr = self.x_train[idx]
        his = self.his[idx]
        
        his_length = 64

        his = [0] * (his_length - self.historical_data_length) + his[- self.historical_data_length:].tolist()

        if len(his) < his_length:
            his = [0] * (his_length - len(his)) + list(his)
        elif len(his) > his_length:
            his = list(his[-his_length:])

        his = np.array(his)
        # his = np.array(his)
        

        nwp_id = self.nwp_id[idx]
        hres_info = self.hres_info[idx]

        # Subset by nwp_id and resize
        arr = arr[nwp_id]

        if len(arr.shape) == 4:  # [seq_len, n_fts, height, width]
            seq_len = arr.shape[0]  # Get actual sequence length
            arr = arr[:, :, :self.image_size, :self.image_size]
        elif len(arr.shape) == 3:  # [n_fts, height, width]
            seq_len = 1  # No sequence dimension, treat as single frame
            arr = arr[:, :self.image_size, :self.image_size]

        bt_wp = self.y_train[idx] * 0.5

        # Scale data
        arr, bt_wp = self.fit_data(arr, bt_wp)

        # # Pad arr to max_seq_len if it's a sequence
        # if len(arr.shape) == 4:  # Sequence data
        #     padding_size = self.max_seq_len - seq_len
        #     if padding_size > 0:
        #         padding = np.zeros((padding_size, arr.shape[1], arr.shape[2], arr.shape[3]))
        #         arr = np.concatenate([arr, padding], axis=0)


        # # Convert to torch tensor and return with length
        # # arr = torch.from_numpy(arr).float()
        # # his = torch.from_numpy(his).float() 
        # # nwp_id = torch.tensor(nwp_id).long()
        
        # # hres_info = torch.from_numpy(hres_info).float()
        
        
        return {"x": [arr, his, nwp_id, hres_info], "y": bt_wp, }

    def __len__(self):
        return len(self.valid_indices)  # Updated to reflect filtered size



    
class CycloneDataset3(Dataset):
    def __init__(self,data_dir ="cutted_data/train", mode="train", args=None, scaler=None ,besttrack_scaler_path="output/scaler/besttrackscaler.pkl",nwp_scaler_path="output/scaler/nwpscaler.pkl", ):
        super().__init__()
        
        
        self.arr = np.load(data_dir)
        self.x_train, self.y_train = self.arr['x_arr'], self.arr['groundtruth']
        
        self.besttrack_scaler_path = besttrack_scaler_path
        self.nwp_scaler_path = nwp_scaler_path
        
        self.mode=  mode
        self.args = args
        self.scaler = scaler

    # def fit_data(self,arr,y):
    #     arr_shape = arr.shape
        
    #     return arr, y

    def set_scaler(self, besttrack_scaler, nwp_scaler):
        self.besttrack_scaler = besttrack_scaler
        self.nwp_scaler = nwp_scaler
        
    def get_scaler(self,):
        from copy import deepcopy
        return deepcopy(self.besttrack_scaler), deepcopy(self.nwp_scaler)
    
    def __getitem__(self,idx):
        arr = self.x_train[idx][:,31,31]
        arr = arr.reshape(1,arr.shape[0])
        arr = self.scaler.transform(arr)
        arr = arr.squeeze()
        bt_wp = self.y_train[idx]
        
        # arr, bt_wp = self.fit_data(arr,bt_wp)
        # cropped_arr = self.crop_arr(arr, converted_lat,converted_lon, self.args.radius)
        return {"x": arr, "y": bt_wp}
    

    def __len__(self):
        return self.x_train.shape[0]
        
class VITDataset(Dataset):
    def __init__(self,data_dir ="cutted_data/train", mode="train", nwp_scaler=None, bt_scaler = None, args=None ,besttrack_scaler_path="output/scaler/besttrackscaler.pkl",nwp_scaler_path="output/scaler/nwpscaler.pkl", ):
        super().__init__()
        
        print(f"Initializingggg {mode} dataloader ....")
        self.features = args.list_features        
            
        self.arr = np.load(data_dir)
        self.x_train, self.y_train, self.nwp_id= self.arr['x_arr'], self.arr['groundtruth'], self.arr['leading_time']
        
        if self.features is not None:
            self.x_train = self.x_train[:,self.features, :,:]
            

        
        self.nwp_scaler = nwp_scaler
        self.bt_scaler = bt_scaler
        
        self.mode=  mode
        self.args = args
        self.image_size = args.image_size
        print(f"Initialized {mode} dataloader")

    def fit_data(self,arr,y):
    
        arr_shape = arr.shape
        reshaped_arr = arr.transpose((1,2,0))
        
        reshaped_arr = reshaped_arr.reshape((reshaped_arr.shape[0] * reshaped_arr.shape[1], -1))
        
        # self.bt_scaler.fit(y)
        reshaped_arr =  self.nwp_scaler.transform(reshaped_arr)
        reshaped_arr = reshaped_arr.reshape(arr_shape[1],arr_shape[2], arr_shape[0])
        reshaped_arr = reshaped_arr.transpose(2,0,1)
        
        
        # reshaped_arr = reshaped_arr[:10,:,:]
        # reshaped_y = y.reshape(y.shape[0],1)
        
        if self.args.transform_groundtruth:
            y=  np.expand_dims(np.array(y),0).reshape((1,1))
            y = self.bt_scaler.transform(y)
        # y = self.besttrack_scaler.transform(y)
        return reshaped_arr, y


    def __getitem__(self,idx):
        nwp_id = self.nwp_id[idx]

        arr = self.x_train[idx][nwp_id] #63,101,101

        if len(arr.shape) == 4:
            arr = arr[-1,:,:self.image_size,:self.image_size]
        elif len(arr.shape) == 3:
            arr = arr[:,:self.image_size,:self.image_size]
        
        bt_wp = self.y_train[idx]
        bt_wp = bt_wp * 0.5
        
        arr, bt_wp = self.fit_data(arr,bt_wp)
        
        return {"x": arr, "y": bt_wp}
    

    def __len__(self):
        return self.x_train.shape[0]
        
class VITDatasetSLW(Dataset):
    def __init__(self,data_dir ="cutted_data/train", mode="train", nwp_scaler=None, bt_scaler = None, args=None):
        super().__init__()
        
        self.features = args.list_features        
            
        self.arr = np.load(data_dir)

        # 1000, 4,63,61,61
        self.x_train, self.y_train, self.his = self.arr['x_arr'], self.arr['groundtruth'], self.arr['his']
        
        if self.features is not None:
            self.x_train = self.x_train[:,self.features, :,:]
            
        
        self.nwp_scaler = nwp_scaler
        self.bt_scaler = bt_scaler
        
        self.mode = mode
        self.args = args
        self.image_size = args.image_size

    def fit_data(self,arr,y):
        ## arr shape 4,63,100,100
        arr_shape = arr.shape
        reshaped_arr = arr.transpose((0,2,3,1)) # 4,100,100, 63
        
        reshaped_arr = reshaped_arr.reshape((reshaped_arr.shape[0] * reshaped_arr.shape[1] * reshaped_arr.shape[2], -1))
        
        # self.bt_scaler.fit(y)
        reshaped_arr =  self.nwp_scaler.transform(reshaped_arr)
        reshaped_arr = reshaped_arr.reshape(arr_shape[0],arr_shape[2],arr_shape[3], arr_shape[1]) # 4,100,100, 63
        reshaped_arr = reshaped_arr.transpose(0,3,1,2) # 4, 63, 100,100
        reshaped_arr = reshaped_arr.reshape(-1, reshaped_arr.shape[2], reshaped_arr.shape[3]) # 252,100,100
        
        if self.args.transform_groundtruth:
            y =  np.expand_dims(np.array(y),0).reshape((1,1))
            y = self.bt_scaler.transform(y)
        # y = self.besttrack_scaler.transform(y)
   
        return reshaped_arr, y


    def __getitem__(self,idx):
        arr = self.x_train[idx][:,:,:self.image_size,:self.image_size]
        
        bt_wp = self.y_train[idx]
        bt_wp = bt_wp * 0.5

        his = self.his[idx]
        arr, bt_wp = self.fit_data(arr,bt_wp)
        return {"x": arr, "y": bt_wp}
    
    def __len__(self):
        return self.x_train.shape[0]

class VITDatasetSLW6(Dataset):
    def __init__(self,data_dir ="cutted_data/train", mode="train", nwp_scaler=None, bt_scaler = None, args=None):
        super().__init__()
        
        self.features = args.list_features        
            
        self.arr = np.load(data_dir)

        # 1000, 4,63,61,61
        self.x_train, self.y_train, self.his = self.arr['x_arr'], self.arr['groundtruth'], self.arr['his']
        
        if self.features is not None:
            self.x_train = self.x_train[:,self.features, :,:]
            
        
        self.nwp_scaler = nwp_scaler
        self.bt_scaler = bt_scaler
        
        self.mode = mode
        self.args = args
        self.image_size = args.image_size

    def fit_data(self,arr,y):
        ## arr shape 4,63,100,100
        arr_shape = arr.shape
        reshaped_arr = arr.transpose((0,2,3,1)) # 4,100,100, 63
        
        reshaped_arr = reshaped_arr.reshape((reshaped_arr.shape[0] * reshaped_arr.shape[1] * reshaped_arr.shape[2], -1))
        
        # self.bt_scaler.fit(y)
        reshaped_arr =  self.nwp_scaler.transform(reshaped_arr)
        reshaped_arr = reshaped_arr.reshape(arr_shape[0],arr_shape[2],arr_shape[3], arr_shape[1]) # 4,100,100, 63
        reshaped_arr = reshaped_arr.transpose(0,3,1,2) # 4, 63, 100,100
        reshaped_arr = reshaped_arr.reshape(-1, reshaped_arr.shape[2], reshaped_arr.shape[3]) # 252,100,100
        
        if self.args.transform_groundtruth:
            y =  np.expand_dims(np.array(y),0).reshape((1,1))
            y = self.bt_scaler.transform(y)
        # y = self.besttrack_scaler.transform(y)
   
        return reshaped_arr, y

    def __getitem__(self,idx):
        arr = self.x_train[idx][:,:,:self.image_size,:self.image_size]
        
        bt_wp = self.y_train[idx]
        bt_wp = bt_wp * 0.5

        his = self.his[idx]
        arr, bt_wp = self.fit_data(arr,bt_wp)
        arr = [arr, his]
        return {"x": arr, "y": bt_wp}
    

    def __len__(self):
        return self.x_train.shape[0]
 
    def __init__(self,data_dir ="cutted_data/train", mode="train", nwp_scaler=None, bt_scaler = None, args=None ,besttrack_scaler_path="output/scaler/besttrackscaler.pkl",nwp_scaler_path="output/scaler/nwpscaler.pkl", ):
        super().__init__()
        
        self.features = args.list_features        
            
        self.arr = np.load(data_dir)
        self.x_train, self.y_train = self.arr['x_arr'], self.arr['groundtruth']
        
        if self.features is not None:
            self.x_train = self.x_train[:,self.features, :,:]
            
        self.besttrack_scaler_path = besttrack_scaler_path
        self.nwp_scaler_path = nwp_scaler_path
        
        self.nwp_scaler = nwp_scaler
        self.bt_scaler = bt_scaler
        
        self.mode=  mode
        self.args = args
        self.image_size = args.image_size
        self.cluster_index = args.cluster_index

    def fit_data(self,arr,y):
    
        arr_shape = arr.shape
        reshaped_arr = arr.transpose((1,2,0))
        
        reshaped_arr = reshaped_arr.reshape((reshaped_arr.shape[0] * reshaped_arr.shape[1], -1))
        
        # self.bt_scaler.fit(y)
        reshaped_arr =  self.nwp_scaler.transform(reshaped_arr)
        reshaped_arr = reshaped_arr.reshape(arr_shape[1],arr_shape[2], arr_shape[0])
        reshaped_arr = reshaped_arr.transpose(2,0,1)
        
        if self.args.transform_groundtruth:
            y=  np.expand_dims(np.array(y),0).reshape((1,1))
            y = self.bt_scaler.transform(y)
        # y = self.besttrack_scaler.transform(y)
        return reshaped_arr, y

    def __getitem__(self,idx):
        arr = self.x_train[idx][:,:self.image_size,:self.image_size]
        
        bt_wp = self.y_train[idx]
        bt_wp = bt_wp * 0.5
        
        arr, bt_wp = self.fit_data(arr,bt_wp)
        list_arr = []
        for index_list in self.cluster_index:
            list_arr.append(arr[index_list,:,:])
        # 
        return {"x": list_arr, "y": bt_wp}
    

    def __len__(self):
        return self.x_train.shape[0]
    
class VITDataset2(Dataset):
    def __init__(self,data_dir ="cutted_data/train", mode="train", nwp_scaler=None, bt_scaler = None, args=None):
        super().__init__()
        
        self.features = args.list_features        
            
        self.arr = np.load(data_dir)
        self.x_train, self.y_train, self.leading_time = self.arr['x_arr'], self.arr['groundtruth'], self.arr['leading_time']
        del self.arr
        if self.features is not None:
            self.x_train = self.x_train[:,self.features, :,:]
            
        
        self.nwp_scaler = nwp_scaler
        self.bt_scaler = bt_scaler
        
        self.args = args
        self.image_size = args.image_size

    def fit_data(self,arr,y):
    
        arr_shape = arr.shape
        reshaped_arr = arr.transpose((1,2,0))
        
        reshaped_arr = reshaped_arr.reshape((reshaped_arr.shape[0] * reshaped_arr.shape[1], -1))
        
        # self.bt_scaler.fit(y)
        reshaped_arr =  self.nwp_scaler.transform(reshaped_arr)
        reshaped_arr = reshaped_arr.reshape(arr_shape[1],arr_shape[2], arr_shape[0])
        reshaped_arr = reshaped_arr.transpose(2,0,1)
        
        
        # reshaped_arr = reshaped_arr[:10,:,:]
        # reshaped_y = y.reshape(y.shape[0],1)
        
        if self.args.transform_groundtruth:
            y=  np.expand_dims(np.array(y),0).reshape((1,1))
            y = self.bt_scaler.transform(y)
        # y = self.besttrack_scaler.transform(y)
        return reshaped_arr, y

    def __getitem__(self,idx):  
        arr = self.x_train[idx][:,:self.image_size,:self.image_size]
        leading_time = self.leading_time[idx]
        bt_wp = self.y_train[idx]
        bt_wp = bt_wp * 0.5
        
        arr, bt_wp = self.fit_data(arr,bt_wp)
        
        return {"x": arr, "y": bt_wp,'leading_time': leading_time}

    

    def __len__(self):
        return self.x_train.shape[0]
   
class VITDataset6_6(Dataset):
    def __init__(self, data_dir="cutted_data/train", mode="train", nwp_scaler=None, bt_scaler=None, args=None, 
                 besttrack_scaler_path="output/scaler/besttrackscaler.pkl", nwp_scaler_path="output/scaler/nwpscaler.pkl"):
        super().__init__()
        self.historical_data_length = args.historical_data_length
        self.historical_nwp_length = args.historical_nwp_length
        self.features = args.list_features        
        self.arr = np.load(data_dir)

        self.x_train, self.y_train, self.his, self.nwp_id = self.arr['x_arr'], self.arr['groundtruth'], self.arr['his'], self.arr['leading_time']
        
        hres_data = np.load(f"{args.data_dir}/{mode}/hres_info.npz")
        self.hres_info = hres_data['hres_info']
        self.checked_list = hres_data['checked']
        
        # Filter indices where checked is 1
        valid_indices = np.where(self.checked_list == 1)[0]
        
        print(f"Data {mode}: No. sample{len(self.y_train)} No. valid sample {len(valid_indices)}")
        self.valid_indices = valid_indices
        self.x_train = self.x_train[valid_indices]
        self.y_train = self.y_train[valid_indices]
        self.his = self.his[valid_indices]
        self.nwp_id = self.nwp_id[valid_indices]
        self.checked_list = self.checked_list[valid_indices]  # Update checked_list to only valid samples
        self.check_data_values()
        if self.features is not None:
            self.x_train = self.x_train[:, self.features, :, :]

        self.besttrack_scaler_path = besttrack_scaler_path
        self.nwp_scaler_path = nwp_scaler_path
        
        self.nwp_scaler = nwp_scaler 
        self.bt_scaler = bt_scaler 
        
        self.mode = mode
        self.args = args
        self.image_size = args.image_size

        # Determine max sequence length for padding (assuming arr has variable sequence length)
        self.max_seq_len = max([nwp_id for x, nwp_id in zip(self.x_train, self.nwp_id)])
    
    def check_data_values(self):
        """Check for NaN, inf and print basic stats for all data arrays"""
        print("\nChecking data arrays for NaN/inf values and stats:")
        
        # Check x_train
        print("\nx_train:")
        print(f"Shape: {self.x_train.shape}")
        print(f"NaN count: {np.isnan(self.x_train).sum()}")
        print(f"Inf count: {np.isinf(self.x_train).sum()}")
        print(f"Min: {np.min(self.x_train)}, Max: {np.max(self.x_train)}")
        
        # Check y_train
        print("\ny_train:")
        print(f"Shape: {self.y_train.shape}")
        print(f"NaN count: {np.isnan(self.y_train).sum()}")
        print(f"Inf count: {np.isinf(self.y_train).sum()}")
        print(f"Min: {np.min(self.y_train)}, Max: {np.max(self.y_train)}")
        
        # Check his
        print("\nhis:")
        print(f"Shape: {self.his.shape}")
        print(f"NaN count: {np.isnan(self.his).sum()}")
        print(f"Inf count: {np.isinf(self.his).sum()}")
        print(f"Min: {np.min(self.his)}, Max: {np.max(self.his)}")
        
        # Check nwp_id
        print("\nnwp_id:")
        print(f"Shape: {self.nwp_id.shape}")
        print(f"NaN count: {np.isnan(self.nwp_id).sum()}")
        print(f"Inf count: {np.isinf(self.nwp_id).sum()}")
        print(f"Min: {np.min(self.nwp_id)}, Max: {np.max(self.nwp_id)}")
        
        # Check hres_info
        print("\nhres_info:")
        print(f"Shape: {self.hres_info.shape}")
        print(f"NaN count: {np.isnan(self.hres_info).sum()}")
        print(f"Inf count: {np.isinf(self.hres_info).sum()}")
        print(f"Min: {np.min(self.hres_info)}, Max: {np.max(self.hres_info)}")
        
        # Check checked_list
        print("\nchecked_list:")
        print(f"Shape: {self.checked_list.shape}")
        print(f"NaN count: {np.isnan(self.checked_list).sum()}")
        print(f"Inf count: {np.isinf(self.checked_list).sum()}")
        print(f"Min: {np.min(self.checked_list)}, Max: {np.max(self.checked_list)}")
        print(f"Unique values: {np.unique(self.checked_list, return_counts=True)}")

    def fit_data(self, arr, y):
        arr_shape = arr.shape
        if len(arr_shape) == 4:  # [seq_len, n_fts, height, width]
            # Transpose and reshape for scaling
            reshaped_arr = arr.transpose((0, 2, 3, 1))  # [seq_len, height, width, n_fts]
            reshaped_arr = reshaped_arr.reshape((-1, arr_shape[1]))  # Flatten spatial and feature dims
            reshaped_arr = self.nwp_scaler.transform(reshaped_arr)
            reshaped_arr = reshaped_arr.reshape(arr_shape[0], arr_shape[2], arr_shape[3], arr_shape[1])
            reshaped_arr = reshaped_arr.transpose(0, 3, 1, 2)  # Back to [seq_len, n_fts, height, width]

        elif len(arr_shape) == 3:  # [n_fts, height, width] or similar
            reshaped_arr = arr.transpose((1, 2, 0))
            reshaped_arr = reshaped_arr.reshape((-1, arr_shape[0]))
            reshaped_arr = self.nwp_scaler.transform(reshaped_arr)
            reshaped_arr = reshaped_arr.reshape(arr_shape[1], arr_shape[2], arr_shape[0])
            reshaped_arr = reshaped_arr.transpose(2, 0, 1)

        if self.args.transform_groundtruth:
            y = np.expand_dims(np.array(y), 0).reshape((1, 1))
            y = self.bt_scaler.transform(y)

        return reshaped_arr, y

    def __getitem__(self, idx):
        """
        Only processes samples where checked is 1 (already filtered in __init__)
        """
        arr = self.x_train[idx]
        his = self.his[idx]
        
        his_length = 64

        his = [0] * (his_length - self.historical_data_length) + his[- self.historical_data_length:].tolist()

        if len(his) < his_length:
            his = [0] * (his_length - len(his)) + list(his)
        elif len(his) > his_length:
            his = list(his[-his_length:])
            
        his = np.array(his)
        # his = np.array(his)
        

        nwp_id = self.nwp_id[idx]
        hres_info = self.hres_info[idx]

        # Subset by nwp_id and resize
        arr = arr[nwp_id - self.historical_nwp_length + 1 :nwp_id+1]

        if len(arr.shape) == 4:  # [seq_len, n_fts, height, width]
            seq_len = arr.shape[0]  # Get actual sequence length
            arr = arr[:, :, :self.image_size, :self.image_size]
        elif len(arr.shape) == 3:  # [n_fts, height, width]
            seq_len = 1  # No sequence dimension, treat as single frame
            arr = arr[:, :self.image_size, :self.image_size]

        bt_wp = self.y_train[idx] * 0.5

        # Scale data
        arr, bt_wp = self.fit_data(arr, bt_wp)
        
        return {"x": [arr, his, nwp_id, hres_info], "y": bt_wp, }

    def __len__(self):
        return len(self.valid_indices)  # Updated to reflect filtered size