from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os 
from sklearn.preprocessing import MinMaxScaler
import pickle

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
        



from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os 
from sklearn.preprocessing import MinMaxScaler
import pickle

class CycloneDataset2(Dataset):
    def __init__(self,data_dir ="cutted_data/train", mode="train", nwp_scaler=None, bt_scaler = None, args=None ,besttrack_scaler_path="output/scaler/besttrackscaler.pkl",nwp_scaler_path="output/scaler/nwpscaler.pkl", ):
        super().__init__()
        
        
        self.arr = np.load(data_dir)
        self.x_train, self.y_train = self.arr['x_arr'], self.arr['groundtruth']
        
        self.besttrack_scaler_path = besttrack_scaler_path
        self.nwp_scaler_path = nwp_scaler_path
        
        self.nwp_scaler = nwp_scaler
        self.bt_scaler = bt_scaler
        
        self.mode=  mode
        self.args = args

    def fit_data(self,arr,y):
        # breakpoint()
    
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
        bt_wp = self.y_train[idx]
        bt_wp = bt_wp * 0.5
        arr, bt_wp = self.fit_data(arr,bt_wp)
        # cropped_arr = self.crop_arr(arr, converted_lat,converted_lon, self.args.radius)
        return {"x": arr, "y": bt_wp}
    

    def __len__(self):
        return self.x_train.shape[0]
        


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
        
        
        self.arr = np.load(data_dir)
        self.x_train, self.y_train = self.arr['x_arr'], self.arr['groundtruth']
        
        self.besttrack_scaler_path = besttrack_scaler_path
        self.nwp_scaler_path = nwp_scaler_path
        
        self.nwp_scaler = nwp_scaler
        self.bt_scaler = bt_scaler
        
        self.mode=  mode
        self.args = args

    def fit_data(self,arr,y):
    
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
        arr = self.x_train[idx][:,:100,:100]
        bt_wp = self.y_train[idx]
        bt_wp = bt_wp * 0.5
        arr, bt_wp = self.fit_data(arr,bt_wp)
        # cropped_arr = self.crop_arr(arr, converted_lat,converted_lon, self.args.radius)
        return {"x": arr, "y": bt_wp}
    

    def __len__(self):
        return self.x_train.shape[0]
        