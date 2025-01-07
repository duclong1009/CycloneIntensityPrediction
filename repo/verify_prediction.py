import argparse
import model_utils
import model
import dataloader
from torch.utils.data import DataLoader
import torch 
import torch.nn as nn 
import wandb
import os
from datetime import datetime
import pandas as pd 
import numpy as np
import tqdm 
import json

list_key = ['h.npz.npy', 'landsea_mask.npz.npy',
       'lat.npz.npy', 'lev.npz.npy', 'lev_soil.npz.npy', 'lon.npz.npy',
       'pmsl.npz.npy', 'ps.npz.npy', 'q.npz.npy', 'rain.npz.npy',
       't.npz.npy', 't2m.npz.npy', 'td2m.npz.npy',
       'terrain.npz.npy', 'time.npz.npy', 'total_cloud.npz.npy',
       'u.npz.npy', 'u10m.npz.npy', 'v.npz.npy', 'v10m.npz.npy',
       'w.npz.npy']


def get_history( bt_df, besttrack_id):
        bt_his = []
        bt_his = bt_df['Maximum sustained wind speed'].values[:besttrack_id]
        return bt_his


def convert_latlon_to_index(latitude,longitude):
        import math
        x = math.ceil((latitude + 20)/ 0.125)
        y = math.ceil((longitude - 80)/ 0.125)
        return x,y

def crop_arr(arr, lat, lon, radius=25):

        # Create an empty array filled with zeros
        cropped_arr = np.zeros((arr.shape[0], radius * 2 + 1, radius * 2 + 1))
        
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

def fit_data(arr, nwp_scaler):
    
        arr_shape = arr.shape
        reshaped_arr = arr.transpose((1,2,0))
        
        reshaped_arr = reshaped_arr.reshape((reshaped_arr.shape[0] * reshaped_arr.shape[1], -1))
        
        # self.bt_scaler.fit(y)
        reshaped_arr =  nwp_scaler.transform(reshaped_arr)
        reshaped_arr = reshaped_arr.reshape(arr_shape[1],arr_shape[2], arr_shape[0])
        reshaped_arr = reshaped_arr.transpose(2,0,1)
        
        

        

        # y = self.besttrack_scaler.transform(y)
        return reshaped_arr


def get_option():
    parser = argparse.ArgumentParser()
    
    ## CNN config
    parser.add_argument("--output_channels",type=int, default=128)
    parser.add_argument("--kernel_size", type=int,  default=3)
    parser.add_argument("--padding",type=int, default=0)
    parser.add_argument("--stride",type=int, default=1)
    ##  VIT config
    parser.add_argument("--patch_size",type=int, default=10)
    parser.add_argument("--dim",type=int, default=1024)
    parser.add_argument("--heads",type=int, default=16)
    
    parser.add_argument("--model_type", type=str, default= "simple_cnn")
    parser.add_argument("--backbone_name",type=str, default='resnet18')
    parser.add_argument("--radius",type=int, default=30)
    # parser.add_argument("--backbone_name", ty)
    ## Config 
    parser.add_argument("--seed",type=int, default=42)
    parser.add_argument("--epochs",type=int, default=10)
    parser.add_argument("--data_dir",type=str, default="cutted_data2")
    parser.add_argument("--transform_groundtruth",action="store_true", default=False)
    
    parser.add_argument(
        '--list_features', 
        metavar='N', 
        type=int, 
        nargs='+', 
        help='a list of integers'
    )
    ### early stopping
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--checkpoint_dir",type=str, default='checkpoint')
    parser.add_argument("--delta", type=float, default= 1e-4)
    
    ### Dataloader
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers",type=int, default=0)
    # parser
    
    ### optimizer
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--l2_coef",type=float, default= 1e-3)
    
    ### Wandb
    parser.add_argument("--group_name",type=str, default='test_group')
    parser.add_argument("--_use_wandb",action="store_true", default=False)
    parser.add_argument("--debug",action="store_true", default=False)
    # par
    parser.add_argument("--loss_func",type=str, default='mse', choices=['weighted_mse','mse'])
    
    ## scheduler
    parser.add_argument("--_use_scheduler_lr",action="store_true", default=False)
    parser.add_argument("--scheduler_type",type=str, default="steplr")
    ### promt setting
    parser.add_argument("--prompt_dims",type=int, default=128)
    parser.add_argument("--image_size",type=int, default=100)
    ###
    parser.add_argument("--use_position_embedding", action="store_true", default=False)

    parser.add_argument("--use_cls_for_region", action="store_true", default=False)
    parser.add_argument("--combining_layer_type",type=int, default=0)
    # training 
    parser.add_argument("--body_model_name", type=str, default="vit")
    parser.add_argument("--freeze", action="store_true", default=False)
    
    parser.add_argument("--prompt_length", type=int, default=50)
    
    # parser.add_argument("--input_channels",type=int, default=58)
    args = parser.parse_args()
    return args 


if __name__ == "__main__":
    
    args = get_option()
    
    try:
        config = vars(args)
    except IOError as msg:
        args.error(str(msg))

    model_utils.seed_everything(args.seed)

    
    # scaler = model_utils.get_scaler()
    ### Init wandb
    
    nwp_scaler, bt_scaler, n_fts = model_utils.get_scaler2(args)

    if args.model_type == "prompt_vit0":
        import orca_model
        cnn_embedder = orca_model.CNNEmbedder(input_channels=n_fts[0], output_dim=768, kernel_size=10)
        
        prediction_head = orca_model.PredictionHead(dim = 768)
        
        train_model = orca_model.Prompt_Tuning_Model0(cnn_embedder, args.body_model_name, prediction_head,args)
        
        args.name = (f"{args.model_type}-pse_{args.use_position_embedding}-SLr_{args._use_scheduler_lr}_{args.scheduler_type}-loss_func_{args.loss_func}-{args.backbone_name}__{args.seed}_{args.batch_size}-lr_{args.lr}-tf_gr_{args.transform_groundtruth}-ps_{args.patch_size}-dim_{args.dim}-head_{args.heads}")
        train_dataset = dataloader.VITDataset(data_dir= f"{args.data_dir}/train/data.npz",mode="train", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        valid_dataset = dataloader.VITDataset(data_dir= f"{args.data_dir}/valid/data.npz", mode="valid", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        test_dataset = dataloader.VITDataset(data_dir= f"{args.data_dir}/test/data.npz", mode="test", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
    
    
    elif args.model_type == "prompt_vit1":
        import orca_model
        cnn_embedder = orca_model.CNNEmbedder(input_channels=n_fts[0], output_dim=768, kernel_size=10)
        
        prediction_head = orca_model.PredictionHead(dim = 768 + args.prompt_dims)
        
        train_model = orca_model.Prompt_Tuning_Model1(cnn_embedder, args.body_model_name, prediction_head,args)
        
        args.name = (f"{args.model_type}-freee_{args.freeze}-pse_{args.use_position_embedding}-SLr_{args._use_scheduler_lr}_{args.scheduler_type}-loss_func_{args.loss_func}-{args.backbone_name}__{args.seed}_{args.batch_size}-lr_{args.lr}-tf_gr_{args.transform_groundtruth}-ps_{args.patch_size}-dim_{args.dim}-head_{args.heads}")
        train_dataset = dataloader.VITDataset(data_dir= f"{args.data_dir}/train/data.npz",mode="train", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        valid_dataset = dataloader.VITDataset(data_dir= f"{args.data_dir}/valid/data.npz", mode="valid", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        test_dataset = dataloader.VITDataset(data_dir= f"{args.data_dir}/test/data.npz", mode="test", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
    
    elif args.model_type == "prompt_vit2":
        import orca_model
        cnn_embedder = orca_model.CNNEmbedder(input_channels=n_fts[0], output_dim=768, kernel_size=10)
        
        prediction_head = orca_model.PredictionHead(dim = 768 + args.prompt_dims)
        
        train_model = orca_model.Prompt_Tuning_Model2(cnn_embedder, args.body_model_name, prediction_head,args)
        
        args.name = (f"{args.model_type}-freee_{args.freeze}-pse_{args.use_position_embedding}-SLr_{args._use_scheduler_lr}_{args.scheduler_type}-loss_func_{args.loss_func}-{args.backbone_name}__{args.seed}_{args.batch_size}-lr_{args.lr}-tf_gr_{args.transform_groundtruth}-ps_{args.patch_size}-dim_{args.dim}-head_{args.heads}")
        train_dataset = dataloader.VITDataset(data_dir= f"{args.data_dir}/train/data.npz",mode="train", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        valid_dataset = dataloader.VITDataset(data_dir= f"{args.data_dir}/valid/data.npz", mode="valid", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        test_dataset = dataloader.VITDataset(data_dir= f"{args.data_dir}/test/data.npz", mode="test", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        
    elif args.model_type == "prompt_vit3":
        import orca_model
        cnn_embedder = orca_model.CNNEmbedder(input_channels=n_fts[0], output_dim=768 - args.prompt_dims, kernel_size=10)
        
        prediction_head = orca_model.PredictionHead()
        
        train_model = orca_model.Prompt_Tuning_Model3(cnn_embedder, args.body_model_name, prediction_head,args)
        
        args.name = (f"{args.model_type}-freee_{args.freeze}-pse_{args.use_position_embedding}-SLr_{args._use_scheduler_lr}_{args.scheduler_type}-loss_func_{args.loss_func}-{args.backbone_name}__{args.seed}_{args.batch_size}-lr_{args.lr}-tf_gr_{args.transform_groundtruth}-ps_{args.patch_size}-dim_{args.dim}-head_{args.heads}")
        train_dataset = dataloader.VITDataset(data_dir= f"{args.data_dir}/train/data.npz",mode="train", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        valid_dataset = dataloader.VITDataset(data_dir= f"{args.data_dir}/valid/data.npz", mode="valid", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        test_dataset = dataloader.VITDataset(data_dir= f"{args.data_dir}/test/data.npz", mode="test", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        
    elif args.model_type == "prompt_vit4":
        import orca_model
        cnn_embedder = orca_model.CNNEmbedder(input_channels=58, output_dim=768 - args.prompt_dims, kernel_size=10)
        
        prediction_head = orca_model.PredictionHead()
        
        train_model = orca_model.Prompt_Tuning_Model4(cnn_embedder, "vit", prediction_head,args.prompt_dims)
        
        args.name = (f"{args.model_type}-pse_{args.use_position_embedding}-SLr_{args._use_scheduler_lr}_{args.scheduler_type}-loss_func_{args.loss_func}-{args.backbone_name}__{args.seed}_{args.batch_size}-lr_{args.lr}-tf_gr_{args.transform_groundtruth}-ps_{args.patch_size}-dim_{args.dim}-head_{args.heads}")
        train_dataset = dataloader.VITDataset(data_dir= f"{args.data_dir}/train/data.npz",mode="train", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        valid_dataset = dataloader.VITDataset(data_dir= f"{args.data_dir}/valid/data.npz", mode="valid", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        test_dataset = dataloader.VITDataset(data_dir= f"{args.data_dir}/test/data.npz", mode="test", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
   
    elif args.model_type == "prompt_vit5":
        import orca_model
        cnn_embedder = orca_model.CNNEmbedder(input_channels=58, output_dim=768 - args.prompt_dims, kernel_size=10)
        
        prediction_head = orca_model.PredictionHead(dim = 768 + args.prompt_dims)
        
        train_model = orca_model.Prompt_Tuning_Model5(cnn_embedder, "vit", prediction_head,args.prompt_dims)
        
        args.name = f"{args.model_type}-pse_{args.use_position_embedding}-SLr_{args._use_scheduler_lr}_{args.scheduler_type}-loss_func_{args.loss_func}-{args.backbone_name}__{args.seed}_{args.batch_size}-lr_{args.lr}-tf_gr_{args.transform_groundtruth}-ps_{args.patch_size}-dim_{args.dim}-head_{args.heads}"
        train_dataset = dataloader.VITDataset(data_dir= f"{args.data_dir}/train/data.npz",mode="train", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        valid_dataset = dataloader.VITDataset(data_dir= f"{args.data_dir}/valid/data.npz", mode="valid", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        test_dataset = dataloader.VITDataset(data_dir= f"{args.data_dir}/test/data.npz", mode="test", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
    
    elif args.model_type == "prompt_vit3_leading_time":
        import orca_model
        cnn_embedder = orca_model.CNNEmbedder(input_channels=n_fts[0], output_dim=768 - args.prompt_dims, kernel_size=10)
        
        prediction_head = orca_model.PredictionHead2(prompt_dim = args.prompt_dims)
        
        train_model = orca_model.Prompt_Tuning_Model_Leading_t(cnn_embedder, args.body_model_name, prediction_head,args)
        
        args.name = (f"{args.model_type}-freee_{args.freeze}-pse_{args.use_position_embedding}-SLr_{args._use_scheduler_lr}_{args.scheduler_type}-loss_func_{args.loss_func}-{args.backbone_name}__{args.seed}_{args.batch_size}-lr_{args.lr}-tf_gr_{args.transform_groundtruth}-ps_{args.patch_size}-dim_{args.dim}-head_{args.heads}")
        train_dataset = dataloader.VITDataset2(data_dir= f"{args.data_dir}/train/data.npz",mode="train", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        valid_dataset = dataloader.VITDataset2(data_dir= f"{args.data_dir}/valid/data.npz", mode="valid", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        test_dataset = dataloader.VITDataset2(data_dir= f"{args.data_dir}/test/data.npz", mode="test", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        
        
    elif args.model_type == "prompt_vit7":
        import orca_model
        cnn_embedder = orca_model.CNNEmbedder(input_channels=n_fts[0], output_dim=768, kernel_size=10)
        
        prediction_head = orca_model.PredictionHead(n_patchs = 100 + args.prompt_length)
        
        train_model = orca_model.Prompt_Tuning_Model7(cnn_embedder, args.body_model_name, prediction_head,args)
        
        args.name = (f"{args.model_type}-prl_{args.prompt_length}-freee_{args.freeze}-pse_{args.use_position_embedding}-SLr_{args._use_scheduler_lr}_{args.scheduler_type}-loss_func_{args.loss_func}-{args.backbone_name}__{args.seed}_{args.batch_size}-lr_{args.lr}-tf_gr_{args.transform_groundtruth}-ps_{args.patch_size}-dim_{args.dim}-head_{args.heads}")
        train_dataset = dataloader.VITDataset(data_dir= f"{args.data_dir}/train/data.npz",mode="train", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        valid_dataset = dataloader.VITDataset(data_dir= f"{args.data_dir}/valid/data.npz", mode="valid", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        test_dataset = dataloader.VITDataset(data_dir= f"{args.data_dir}/test/data.npz", mode="test", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
    
    elif args.model_type == "prompt_vit6":
        import orca_model
        cnn_embedder = orca_model.CNNEmbedder(input_channels=n_fts[0], output_dim=768 - args.prompt_dims, kernel_size=10)
        
        prediction_head = orca_model.PredictionHead(n_patchs=101)
        
        train_model = orca_model.Prompt_Tuning_Model6(cnn_embedder, args.body_model_name, prediction_head,args)

        args.name = (f"{args.model_type}-pse_{args.use_position_embedding}-SLr_{args._use_scheduler_lr}_{args.scheduler_type}-loss_func_{args.loss_func}-{args.backbone_name}__{args.seed}_{args.batch_size}-lr_{args.lr}-tf_gr_{args.transform_groundtruth}-ps_{args.patch_size}-dim_{args.dim}-head_{args.heads}")

        # args.name = (f"{args.model_type}-prl_{args.prompt_length}-freee_{args.freeze}-pse_{args.use_position_embedding}-SLr_{args._use_scheduler_lr}_{args.scheduler_type}-loss_func_{args.loss_func}-{args.backbone_name}__{args.seed}_{args.batch_size}-lr_{args.lr}-tf_gr_{args.transform_groundtruth}-ps_{args.patch_size}-dim_{args.dim}-head_{args.heads}")
        train_dataset = dataloader.VITDataset6(data_dir= f"{args.data_dir}/train/data.npz",mode="train", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        valid_dataset = dataloader.VITDataset6(data_dir= f"{args.data_dir}/valid/data.npz", mode="valid", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        test_dataset = dataloader.VITDataset6(data_dir= f"{args.data_dir}/test/data.npz", mode="test", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
    
    elif args.model_type == 'cnn':
        args.name = (f"{args.model_type}-Freeze_{args.freeze}-SLr_{args._use_scheduler_lr}_{args.scheduler_type}-loss_func_{args.loss_func}-{args.backbone_name}__{args.seed}_{args.batch_size}-lr_{args.lr}-tf_gr_{args.transform_groundtruth}")
        train_model = model.FeatureExtractorModel(num_input_channels=n_fts[0], output_dim=1, backbone_name=args.backbone_name)
        train_dataset = dataloader.CycloneDataset2(data_dir= f"{args.data_dir}/train/data.npz",mode="train", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        valid_dataset = dataloader.CycloneDataset2(data_dir= f"{args.data_dir}/valid/data.npz", mode="valid", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        test_dataset = dataloader.CycloneDataset2(data_dir= f"{args.data_dir}/test/data.npz", mode="test", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
    
    elif args.model_type == "prompt_vit6_use_historicaldata":
        import orca_model
        cnn_embedder = orca_model.CNNEmbedder(input_channels=n_fts[0] * n_fts[1], output_dim=768 - args.prompt_dims, kernel_size=10)
        
        prediction_head = orca_model.PredictionHead(n_patchs=101)
        
        train_model = orca_model.Prompt_Tuning_Model6(cnn_embedder, args.body_model_name, prediction_head,args)
        
        args.name = (f"{args.model_type}-prl_{args.prompt_length}-freee_{args.freeze}-pse_{args.use_position_embedding}-SLr_{args._use_scheduler_lr}_{args.scheduler_type}-loss_func_{args.loss_func}-{args.backbone_name}__{args.seed}_{args.batch_size}-lr_{args.lr}-tf_gr_{args.transform_groundtruth}-ps_{args.patch_size}-dim_{args.dim}-head_{args.heads}")
        train_dataset = dataloader.VITDatasetSLW6(data_dir= f"{args.data_dir}/train/data.npz",mode="train", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        valid_dataset = dataloader.VITDatasetSLW6(data_dir= f"{args.data_dir}/valid/data.npz", mode="valid", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        test_dataset = dataloader.VITDatasetSLW6(data_dir= f"{args.data_dir}/test/data.npz", mode="test", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)

    elif args.model_type == "prompt_vit3_use_historicaldata":
        import orca_model
        cnn_embedder = orca_model.CNNEmbedder(input_channels=n_fts[0], output_dim=768 - args.prompt_dims, kernel_size=10)
        
        prediction_head = orca_model.PredictionHead()
        
        train_model = orca_model.Prompt_Tuning_Model3(cnn_embedder, args.body_model_name, prediction_head,args)
        
        args.name = (f"{args.model_type}-freee_{args.freeze}-pse_{args.use_position_embedding}-SLr_{args._use_scheduler_lr}_{args.scheduler_type}-loss_func_{args.loss_func}-{args.backbone_name}__{args.seed}_{args.batch_size}-lr_{args.lr}-tf_gr_{args.transform_groundtruth}-ps_{args.patch_size}-dim_{args.dim}-head_{args.heads}")
    
        # train_dataset = dataloader.VITDatasetSLW(data_dir= f"{args.data_dir}/train/data.npz",mode="train", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        # valid_dataset = dataloader.VITDatasetSLW(data_dir= f"{args.data_dir}/valid/data.npz", mode="valid", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        # test_dataset = dataloader.VITDatasetSLW(data_dir= f"{args.data_dir}/test/data.npz", mode="test", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        
    
    # args.name = "test"

    if args._use_wandb:
        wandb.login(key='ab2505638ca8fabd9114e88f3449ddb51e15a942')
        wandb.init(
            entity="aiotlab",
            project="Cyclone intensity prediction3",
            group= "Inference",
            name=f"{args.name}",
            config=config,
        )

    # print(f"Number input channel {input_shape}")
    print("Model", args.model_type)

    #### Model trainning 
    
    # device = torch.device("cuda:0")
    if args.debug:
        print("Using CPU")
        device = torch.device("cpu")
    else:
        print("Using GPU")
        device = torch.device("cuda:0")
    
    #### Model testing 
    print("Model loading")
    model_utils.load_model(train_model, f"output/{args.group_name}/checkpoint/stdgi_{args.name}.pt")
    
    
    # test_dataset.set_scaler(besttrack_scaler,nwp_scaler)

    index_df = pd.read_csv("2023_2024.csv")
    save_name = "prompt6_data4_2324"

    date_format = "%Y-%m-%d %H:%M:%S"
    date_str = "2014-01-31 00:00:00"
    # Create a datetime object from the string
    dt = datetime.strptime(date_str, date_format) 

    test_list = [0,1,2,3,4]
    list_gr = [[], [], [], [], [] ] 
    list_prd = [[], [], [], [], [] ]
    list_infor = []
    train_model.eval()
    with torch.no_grad():
        for idx in tqdm.tqdm(range(len(index_df))):
            list_his = []
            _, tc_id, tc, date, nwp_path, time_str, besttrack_path, besttrack_id = index_df.values[idx]
            # nwp_path = "unzip_data/2014/1402_KAJIKI/arr/2014020100_4HD.nc"
            # time_str = ""
            # besttrack_path = "single-tc-bessttrack2/2014/1402.csv"
            # besttrack_id = 12
            # breakpoint()
            nwp_path = nwp_path.split(".")[0]
            bt_df = pd.read_csv(f"/home/user01/aiotlab/longnd/cyclone_prediction/raw_data/{besttrack_path}")
            raw_bt_his = get_history(bt_df, besttrack_id).tolist()

            for test_idx in test_list:
                bt_his = raw_bt_his + list_his
                bt_his = [0] * (64 - len(bt_his)) + bt_his
                current_bt_idx = test_idx + besttrack_id
                if current_bt_idx < len(bt_df):

                    bt_lat, bt_lon = bt_df['Latitude of the center'][current_bt_idx], bt_df['Longitude of the center'][current_bt_idx]
                
                    converted_lat, converted_lon = convert_latlon_to_index(bt_lat, bt_lon)


                    list_arr = []
                    try:
                        for key in list_key:
                            
                            arr = np.load(f"/home/user01/aiotlab/longnd/cyclone_prediction/raw_data/{nwp_path}/{key}")
                            if len(arr.shape) == 2:
                                arr_ = np.expand_dims(arr,0)
                            elif len(arr.shape) == 3:
                                arr_ = np.expand_dims(arr[test_idx],0)
                            elif len(arr.shape) == 4:
                                arr_ = arr[test_idx]
                            else:
                                pass
                            list_arr.append(arr_)
                        stacked_arr = np.concatenate(list_arr,0)        

                        # Crop the array around the cyclone's location
                        cropped_arr = crop_arr(stacked_arr, converted_lat, converted_lon, 50)
                        cropped_arr = cropped_arr[:,:100,:100]
                        if cropped_arr.shape == (63, 100, 100):
                            cropped_arr = fit_data(cropped_arr, nwp_scaler)
                            cropped_arr = torch.tensor(cropped_arr,).to(device).unsqueeze(0).float()
                            bt_his = torch.tensor(bt_his).unsqueeze(0).float()
                            
                            # breakpoint
                            x = [cropped_arr, bt_his]
                            output = train_model(x)
                            out = output.detach().numpy()[0][0]
                            list_his.append(out)
                            bt_wp = bt_df['Maximum sustained wind speed'][current_bt_idx] / 2
                            print(out, bt_wp)
                            list_gr[test_idx].append(bt_wp)
                            list_prd[test_idx].append(out)
                            list_infor.append([tc_id, tc, date, time_str,test_idx, bt_wp, out])
                            
                    except:
                        pass

    
    import json

    json_file_path = f"list_infor_{save_name}.json"
    columns = ["tc_id", "tc", "date", "time_str", "test_idx", "bt_wp", "out"]
    pd.DataFrame(list_infor, columns=columns).to_json(json_file_path, orient="records", indent=4)
    
    json_file_path = f"list_gr_{save_name}.json"
    with open(json_file_path, "w") as json_file:
        json.dump(list_gr, json_file)

    json_file_path = f"list_prd_{save_name}.json"
    with open(json_file_path, "w") as json_file:
        json.dump(list_prd, json_file)
    


    # print("--------Testing-------")
    # data = [[pred, gt] for pred, gt in zip(list_prd, list_gr)]
    # table = wandb.Table(data=data, columns=["Prediction", "Ground Truth"])
    # wandb.log({"predictions_vs_groundtruths_table": table})

    # # Optionally, create a plot to visualize the predictions vs ground truths
    # import matplotlib.pyplot as plt

    # plt.figure(figsize=(20, 5))
    # plt.plot(list_prd, label='Predictions', marker='o')
    # plt.plot(list_grt, label='Ground Truths', marker='x')
    # plt.xlabel('Sample Index')
    # plt.ylabel('Value')
    # plt.title('Predictions vs Ground Truths')
    # plt.legend()
    # plt.grid(True)

    # # Log the plot to W&B
    # wandb.log({"predictions_vs_groundtruths_plot": wandb.Image(plt)})
    # plt.close()

    # # Finish the W&B run
    # wandb.finish()
