from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os 
from sklearn.preprocessing import MinMaxScaler
import pickle



class InferenceDataset(Dataset):
    def __init__(self, nwp_scaler=None, bt_scaler = None, args=None):
        super().__init__()
        
        self.features = args.list_features        
        self.data_file = pd.read_csv("map_index4inference.csv")
        
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

        if len(arr.shape) == 4:
            arr = arr[-1,:,:self.image_size,:self.image_size]
        elif len(arr.shape) == 3:
            arr = arr[:,:self.image_size,:self.image_size]
        

        bt_wp = self.y_train[idx]
        bt_wp = bt_wp * 0.5
        
        arr, bt_wp = self.fit_data(arr,bt_wp)
        his = self.his[idx]
        arr = [arr, his]
        return {"x": arr, "y": bt_wp}

    def __len__(self):
        return self.x_train.shape[0]