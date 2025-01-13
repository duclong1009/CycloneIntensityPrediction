import argparse
import model_utils
import model
import dataloader
from torch.utils.data import DataLoader
import torch 
import torch.nn as nn 
import wandb
import os
import baseline

import orca_model
        
def get_option():
    parser = argparse.ArgumentParser()
    
    ## data
    
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
    ###
    parser.add_argument("--use_position_embedding", action="store_true", default=False)

    parser.add_argument("--use_cls_for_region", action="store_true", default=False)
    parser.add_argument("--combining_layer_type",type=int, default=0)
    # training 
    parser.add_argument("--body_model_name", type=str, default="vit")
    parser.add_argument("--freeze", action="store_true", default=False)
    
    parser.add_argument("--prompt_length", type=int, default=50)
    parser.add_argument("--image_size", type=int, default=100, help="image size for the input image")
    
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
    
    nwp_scaler, bt_scaler, n_fts = model_utils.fit_scalers_in_batches(args)

    if args.model_type == "prompt_vit0":
        cnn_embedder = orca_model.CNNEmbedder(input_channels=n_fts[0], output_dim=768, kernel_size=10)
        
        prediction_head = orca_model.PredictionHead(dim = 768)
        
        train_model = orca_model.Prompt_Tuning_Model0(cnn_embedder, args.body_model_name, prediction_head,args)
        
        args.name = (f"{args.model_type}-pse_{args.use_position_embedding}-loss_func_{args.loss_func}-{args.body_model_name}__{args.seed}_{args.batch_size}-lr_{args.lr}-tf_gr_{args.transform_groundtruth}-ps_{args.patch_size}-dim_{args.dim}-head_{args.heads}")
        train_dataset = dataloader.VITDataset(data_dir= f"{args.data_dir}/train/data.npz",mode="train", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        valid_dataset = dataloader.VITDataset(data_dir= f"{args.data_dir}/valid/data.npz", mode="valid", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        test_dataset = dataloader.VITDataset(data_dir= f"{args.data_dir}/test/data.npz", mode="test", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
    
    
    elif args.model_type == "prompt_vit1":
        cnn_embedder = orca_model.CNNEmbedder(input_channels=n_fts[0], output_dim=768, kernel_size=10)
        
        prediction_head = orca_model.PredictionHead(dim = 768 + args.prompt_dims)
        
        train_model = orca_model.Prompt_Tuning_Model1(cnn_embedder, args.body_model_name, prediction_head,args)
        
        args.name = (f"{args.model_type}-freee_{args.freeze}-pse_{args.use_position_embedding}-loss_func_{args.loss_func}-{args.body_model_name}__{args.seed}_{args.batch_size}-lr_{args.lr}-tf_gr_{args.transform_groundtruth}-ps_{args.patch_size}-dim_{args.dim}-head_{args.heads}")
        train_dataset = dataloader.VITDataset(data_dir= f"{args.data_dir}/train/data.npz",mode="train", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        valid_dataset = dataloader.VITDataset(data_dir= f"{args.data_dir}/valid/data.npz", mode="valid", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        test_dataset = dataloader.VITDataset(data_dir= f"{args.data_dir}/test/data.npz", mode="test", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
    
    elif args.model_type == "prompt_vit2":
        cnn_embedder = orca_model.CNNEmbedder(input_channels=n_fts[0], output_dim=768, kernel_size=10)
        
        prediction_head = orca_model.PredictionHead(dim = 768 + args.prompt_dims)
        
        train_model = orca_model.Prompt_Tuning_Model2(cnn_embedder, args.body_model_name, prediction_head,args)
        
        args.name = (f"{args.model_type}-freee_{args.freeze}-pse_{args.use_position_embedding}-loss_func_{args.loss_func}-{args.body_model_name}__{args.seed}_{args.batch_size}-lr_{args.lr}-tf_gr_{args.transform_groundtruth}-ps_{args.patch_size}-dim_{args.dim}-head_{args.heads}")
        train_dataset = dataloader.VITDataset(data_dir= f"{args.data_dir}/train/data.npz",mode="train", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        valid_dataset = dataloader.VITDataset(data_dir= f"{args.data_dir}/valid/data.npz", mode="valid", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        test_dataset = dataloader.VITDataset(data_dir= f"{args.data_dir}/test/data.npz", mode="test", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        
    elif args.model_type == "prompt_vit3":
        cnn_embedder = orca_model.CNNEmbedder(input_channels=n_fts[0], output_dim=768 - args.prompt_dims, kernel_size=10)
        
        prediction_head = orca_model.PredictionHead(n_patchs=(args.image_size // 10) ** 2 +1)
        
        train_model = orca_model.Prompt_Tuning_Model3(cnn_embedder, args.body_model_name, prediction_head,args)
        
        args.name = (f"{args.model_type}-freee_{args.freeze}-pse_{args.use_position_embedding}-loss_func_{args.loss_func}-{args.body_model_name}__{args.seed}_{args.batch_size}-lr_{args.lr}-tf_gr_{args.transform_groundtruth}-ps_{args.patch_size}-dim_{args.dim}-head_{args.heads}")
        train_dataset = dataloader.VITDataset(data_dir= f"{args.data_dir}/train/data.npz",mode="train", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        valid_dataset = dataloader.VITDataset(data_dir= f"{args.data_dir}/valid/data.npz", mode="valid", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        test_dataset = dataloader.VITDataset(data_dir= f"{args.data_dir}/test/data.npz", mode="test", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        
    elif args.model_type == "prompt_vit4":
        cnn_embedder = orca_model.CNNEmbedder(input_channels=58, output_dim=768 - args.prompt_dims, kernel_size=10)
        
        prediction_head = orca_model.PredictionHead()
        
        train_model = orca_model.Prompt_Tuning_Model4(cnn_embedder, "vit", prediction_head,args.prompt_dims)
        
        args.name = (f"{args.model_type}-pse_{args.use_position_embedding}-loss_func_{args.loss_func}-{args.body_model_name}__{args.seed}_{args.batch_size}-lr_{args.lr}-tf_gr_{args.transform_groundtruth}-ps_{args.patch_size}-dim_{args.dim}-head_{args.heads}")
        train_dataset = dataloader.VITDataset(data_dir= f"{args.data_dir}/train/data.npz",mode="train", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        valid_dataset = dataloader.VITDataset(data_dir= f"{args.data_dir}/valid/data.npz", mode="valid", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        test_dataset = dataloader.VITDataset(data_dir= f"{args.data_dir}/test/data.npz", mode="test", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
   
    elif args.model_type == "prompt_vit5":
        cnn_embedder = orca_model.CNNEmbedder(input_channels=58, output_dim=768 - args.prompt_dims, kernel_size=10)
        
        prediction_head = orca_model.PredictionHead(dim = 768 + args.prompt_dims)
        
        train_model = orca_model.Prompt_Tuning_Model5(cnn_embedder, "vit", prediction_head,args.prompt_dims)
        
        args.name = f"{args.model_type}-pse_{args.use_position_embedding}-loss_func_{args.loss_func}-{args.body_model_name}__{args.seed}_{args.batch_size}-lr_{args.lr}-tf_gr_{args.transform_groundtruth}-ps_{args.patch_size}-dim_{args.dim}-head_{args.heads}"
        train_dataset = dataloader.VITDataset(data_dir= f"{args.data_dir}/train/data.npz",mode="train", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        valid_dataset = dataloader.VITDataset(data_dir= f"{args.data_dir}/valid/data.npz", mode="valid", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        test_dataset = dataloader.VITDataset(data_dir= f"{args.data_dir}/test/data.npz", mode="test", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
    
    elif args.model_type == "prompt_vit3_leading_time":
        cnn_embedder = orca_model.CNNEmbedder(input_channels=n_fts[0], output_dim=768 - args.prompt_dims, kernel_size=10)
        
        prediction_head = orca_model.PredictionHead2(prompt_dim = args.prompt_dims)
        
        train_model = orca_model.Prompt_Tuning_Model_Leading_t(cnn_embedder, args.body_model_name, prediction_head,args)
        
        args.name = (f"{args.model_type}-freee_{args.freeze}-pse_{args.use_position_embedding}-loss_func_{args.loss_func}-{args.body_model_name}__{args.seed}_{args.batch_size}-lr_{args.lr}-tf_gr_{args.transform_groundtruth}-ps_{args.patch_size}-dim_{args.dim}-head_{args.heads}")
        train_dataset = dataloader.VITDataset2(data_dir= f"{args.data_dir}/train/data.npz",mode="train", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        valid_dataset = dataloader.VITDataset2(data_dir= f"{args.data_dir}/valid/data.npz", mode="valid", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        test_dataset = dataloader.VITDataset2(data_dir= f"{args.data_dir}/test/data.npz", mode="test", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        
        
    elif args.model_type == "prompt_vit7":
        cnn_embedder = orca_model.CNNEmbedder(input_channels=n_fts[0], output_dim=768, kernel_size=10)
        
        prediction_head = orca_model.PredictionHead(n_patchs = 100 + args.prompt_length)
        
        train_model = orca_model.Prompt_Tuning_Model7(cnn_embedder, args.body_model_name, prediction_head,args)
        
        args.name = (f"{args.model_type}-prl_{args.prompt_length}-freee_{args.freeze}-pse_{args.use_position_embedding}-loss_func_{args.loss_func}-{args.body_model_name}__{args.seed}_{args.batch_size}-lr_{args.lr}-tf_gr_{args.transform_groundtruth}-ps_{args.patch_size}-dim_{args.dim}-head_{args.heads}")
        train_dataset = dataloader.VITDataset(data_dir= f"{args.data_dir}/train/data.npz",mode="train", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        valid_dataset = dataloader.VITDataset(data_dir= f"{args.data_dir}/valid/data.npz", mode="valid", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        test_dataset = dataloader.VITDataset(data_dir= f"{args.data_dir}/test/data.npz", mode="test", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
    
    elif args.model_type == "prompt_vit6":
        cnn_embedder = orca_model.CNNEmbedder(input_channels=n_fts[0], output_dim=768 - args.prompt_dims, kernel_size=10)
        
        prediction_head = orca_model.PredictionHead(n_patchs=(args.image_size // 10) ** 2 +1)
        
        train_model = orca_model.Prompt_Tuning_Model6(cnn_embedder, args.body_model_name, prediction_head,args)
        
        args.name = (f"{args.model_type}-prl_{args.prompt_length}-freee_{args.freeze}-pse_{args.use_position_embedding}-loss_func_{args.loss_func}-{args.body_model_name}__{args.seed}_{args.batch_size}-lr_{args.lr}-tf_gr_{args.transform_groundtruth}-ps_{args.patch_size}-dim_{args.dim}-head_{args.heads}")
        train_dataset = dataloader.VITDataset6(data_dir= f"{args.data_dir}/train/data.npz",mode="train", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        valid_dataset = dataloader.VITDataset6(data_dir= f"{args.data_dir}/valid/data.npz", mode="valid", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        test_dataset = dataloader.VITDataset6(data_dir= f"{args.data_dir}/test/data.npz", mode="test", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
    
    elif args.model_type == 'cnn':
        args.name = (f"{args.model_type}-Freeze_{args.freeze}-loss_func_{args.loss_func}-{args.body_model_name}__{args.seed}_{args.batch_size}-lr_{args.lr}-tf_gr_{args.transform_groundtruth}")
        train_model = model.FeatureExtractorModel(num_input_channels=n_fts[0], output_dim=1, backbone_name=args.backbone_name)
        train_dataset = dataloader.CycloneDataset2(data_dir= f"{args.data_dir}/train/data.npz",mode="train", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        valid_dataset = dataloader.CycloneDataset2(data_dir= f"{args.data_dir}/valid/data.npz", mode="valid", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        test_dataset = dataloader.CycloneDataset2(data_dir= f"{args.data_dir}/test/data.npz", mode="test", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
    
    elif args.model_type == "prompt_vit6_use_historicaldata":
        cnn_embedder = orca_model.CNNEmbedder(input_channels=n_fts[0] * n_fts[1], output_dim=768 - args.prompt_dims, kernel_size=10)
        
        prediction_head = orca_model.PredictionHead(n_patchs=101)
        
        train_model = orca_model.Prompt_Tuning_Model6(cnn_embedder, args.body_model_name, prediction_head,args)
        
        args.name = (f"{args.model_type}-prl_{args.prompt_length}-freee_{args.freeze}-pse_{args.use_position_embedding}-loss_func_{args.loss_func}-{args.body_model_name}__{args.seed}_{args.batch_size}-lr_{args.lr}-tf_gr_{args.transform_groundtruth}-ps_{args.patch_size}-dim_{args.dim}-head_{args.heads}")
        train_dataset = dataloader.VITDatasetSLW6(data_dir= f"{args.data_dir}/train/data.npz",mode="train", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        valid_dataset = dataloader.VITDatasetSLW6(data_dir= f"{args.data_dir}/valid/data.npz", mode="valid", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        test_dataset = dataloader.VITDatasetSLW6(data_dir= f"{args.data_dir}/test/data.npz", mode="test", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)

    elif args.model_type == "prompt_vit3_use_historicaldata":
        cnn_embedder = orca_model.CNNEmbedder(input_channels=n_fts[0], output_dim=768 - args.prompt_dims, kernel_size=10)
        
        prediction_head = orca_model.PredictionHead()
        
        train_model = orca_model.Prompt_Tuning_Model3(cnn_embedder, args.body_model_name, prediction_head,args)
        
        args.name = (f"{args.model_type}-freee_{args.freeze}-pse_{args.use_position_embedding}-loss_func_{args.loss_func}-{args.body_model_name}__{args.seed}_{args.batch_size}-lr_{args.lr}-tf_gr_{args.transform_groundtruth}-ps_{args.patch_size}-dim_{args.dim}-head_{args.heads}")
        train_dataset = dataloader.VITDatasetSLW(data_dir= f"{args.data_dir}/train/data.npz",mode="train", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        valid_dataset = dataloader.VITDatasetSLW(data_dir= f"{args.data_dir}/valid/data.npz", mode="valid", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        test_dataset = dataloader.VITDatasetSLW(data_dir= f"{args.data_dir}/test/data.npz", mode="test", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
    
    elif args.model_type == "prompt_vit6_2":
        print("No. fts", n_fts[0])
        cnn_embedder = orca_model.CNNEmbedder(input_channels=n_fts[0], output_dim=768 - args.prompt_dims, kernel_size=10)
        
        prediction_head = orca_model.PredictionHead(n_patchs=(args.image_size // 10) ** 2 +1)
        
        train_model = orca_model.Prompt_Tuning_Model6_Progressive(cnn_embedder, args.body_model_name, prediction_head,args)
        
        args.name = (f"{args.model_type}-prl_{args.prompt_length}-freee_{args.freeze}-pse_{args.use_position_embedding}-loss_func_{args.loss_func}-{args.body_model_name}__{args.seed}_{args.batch_size}-lr_{args.lr}-tf_gr_{args.transform_groundtruth}-ps_{args.patch_size}-dim_{args.dim}-head_{args.heads}")
        train_dataset = dataloader.VITDataset6_2(data_dir= f"{args.data_dir}/train/data.npz",mode="train", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        valid_dataset = dataloader.VITDataset6_2(data_dir= f"{args.data_dir}/valid/data.npz", mode="valid", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        test_dataset = dataloader.VITDataset6_2(data_dir= f"{args.data_dir}/test/data.npz", mode="test", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)

    
    elif args.model_type == "prompt_vit6_3":
        print("No. fts", n_fts[0])
        cnn_embedder = orca_model.CNNEmbedder(input_channels=n_fts[0], output_dim=768 - args.prompt_dims, kernel_size=10)
        
        prediction_head = orca_model.PredictionHead(n_patchs= (args.image_size // 10) ** 2 +1)
        
        train_model = orca_model.Prompt_Tuning_Model6_Progressive2(cnn_embedder, args.body_model_name, prediction_head,args)
        
        args.name = (f"{args.model_type}-prl_{args.prompt_length}-freee_{args.freeze}-pse_{args.use_position_embedding}-loss_func_{args.loss_func}-{args.body_model_name}__{args.seed}_{args.batch_size}-lr_{args.lr}-tf_gr_{args.transform_groundtruth}-ps_{args.patch_size}-dim_{args.dim}-head_{args.heads}")
        train_dataset = dataloader.VITDataset6_2(data_dir= f"{args.data_dir}/train/data.npz",mode="train", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        valid_dataset = dataloader.VITDataset6_2(data_dir= f"{args.data_dir}/valid/data.npz", mode="valid", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        test_dataset = dataloader.VITDataset6_2(data_dir= f"{args.data_dir}/test/data.npz", mode="test", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)

    elif args.model_type == "convlstm1":
        # cnn_embedder = orca_model.CNNEmbedder(input_channels=n_fts[0], output_dim=768 - args.prompt_dims, kernel_size=10)
        
        prediction_head = orca_model.PredictionHead(n_patchs=100)
        
        train_model = baseline.ConvLSTM2(input_channels=n_fts[0], output_dim=768, kernel_size=5, num_layers=2, prediction_head=prediction_head, args=args)
                
        args.name = (f"{args.model_type}-prl_{args.prompt_length}-freee_{args.freeze}-pse_{args.use_position_embedding}-loss_func_{args.loss_func}-{args.body_model_name}__{args.seed}_{args.batch_size}-lr_{args.lr}-tf_gr_{args.transform_groundtruth}-ps_{args.patch_size}-dim_{args.dim}-head_{args.heads}")
        train_dataset = dataloader.VITDataset6_2(data_dir= f"{args.data_dir}/train/data.npz",mode="train", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        valid_dataset = dataloader.VITDataset6_2(data_dir= f"{args.data_dir}/valid/data.npz", mode="valid", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        test_dataset = dataloader.VITDataset6_2(data_dir= f"{args.data_dir}/test/data.npz", mode="test", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)

    
    # args.name = "test"

    if args._use_wandb:
        wandb.login(key='ab2505638ca8fabd9114e88f3449ddb51e15a942')
        wandb.init(
            entity="aiotlab",
            project="Cyclone intensity prediction3",
            group=args.group_name,
            name=f"{args.name}",
            config=config,
        )

    # print(f"Number input channel {input_shape}")
    print("Model", args.model_type)

    ### Data loading and Data preprocess
    if not os.path.exists(f"output/{args.group_name}/checkpoint/"):
        print(f"Make dir output/{args.group_name}/checkpoint/ ...")
        os.makedirs(f"output/{args.group_name}/checkpoint/")

    ## Initialize early stopping
    early_stopping = model_utils.EarlyStopping(
        patience=args.patience,
        verbose=True,
        delta=args.delta,
        path=f"output/{args.group_name}/checkpoint/stdgi_{args.name}.pt",
    )

    #### Model initialization
    ### dataset
    ### loss & optimizer
    if args.loss_func == "mse":
        loss_func = nn.MSELoss()
    elif args.loss_func == "weighted_mse":
        import loss
        loss_func = loss.WeightedMSELoss()
    else:
        raise("Not correct loss function!")
    trainable_params = filter(lambda p: p.requires_grad, train_model.parameters())
    optimizer = torch.optim.Adam(
        trainable_params, lr=args.lr, weight_decay=args.l2_coef
    )
    #### Model trainning 
    
    
    # device = torch.device("cuda:0")
    # if torch.cuda.is_availabel():

    if args.debug:
        print("Using CPU")
        device = torch.device("cpu")
    else:
        print("Using GPU")
        device = torch.device("cuda:0")
    
    # dataset = dataloader
    list_train_loss, list_valid_loss = model_utils.train_func(train_model, train_dataset, valid_dataset, early_stopping, loss_func, optimizer, args, device)
    
    ### 
    #### Model testing 
    model_utils.load_model(train_model, f"output/{args.group_name}/checkpoint/stdgi_{args.name}.pt")
    
    if args._use_wandb:
        wandb.run.summary["beet_training_loss"] = early_stopping.best_score

    # besttrack_scaler, nwp_scaler = train_dataset.get_scaler() 
            
    # test_dataset.set_scaler(besttrack_scaler,nwp_scaler)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print("--------Testing-------")
    list_prd, list_grt, epoch_loss, mae, mse, mape, rmse, r2, corr_ = model_utils.test_func(train_model, test_dataloader, loss_func, args, bt_scaler,device=device)
    if args._use_wandb:
        wandb.log({"mae":mae,
                   "mse":mse,
                   "mape":mape,
                   "rmse":rmse,
                   "r2":r2,
                   "corr":corr_})
    print(f"MSE: {mse} MAE:{mae} MAPE:{mape} RMSE:{rmse} R2:{r2} Corr:{corr_}")
    data = [[pred, gt] for pred, gt in zip(list_prd, list_grt)]
    table = wandb.Table(data=data, columns=["Prediction", "Ground Truth"])
    wandb.log({"predictions_vs_groundtruths_table": table})

    # Optionally, create a plot to visualize the predictions vs ground truths
    import matplotlib.pyplot as plt

    plt.figure(figsize=(20, 5))
    plt.plot(list_prd, label='Predictions', marker='o')
    plt.plot(list_grt, label='Ground Truths', marker='x')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.title('Predictions vs Ground Truths')
    plt.legend()
    plt.grid(True)

    # Log the plot to W&B
    wandb.log({"predictions_vs_groundtruths_plot": wandb.Image(plt)})
    plt.close()

    # Finish the W&B run
    wandb.finish()
