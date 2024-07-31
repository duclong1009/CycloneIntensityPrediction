import argparse
import model_utils
import model
import dataloader
from torch.utils.data import DataLoader
import torch 
import torch.nn as nn 
import wandb
import os
def get_option():
    parser = argparse.ArgumentParser()
    
    ## CNN config
    parser.add_argument("--output_channels",type=int, default=128)
    parser.add_argument("--kernel_size", type=int,  default=3)
    parser.add_argument("--padding",type=int, default=0)
    parser.add_argument("--stride",type=int, default=1)
    
    parser.add_argument("--model_type",  choices=['simple_cnn','cnn','c_attention_cnn','g_attention_cnn'], default= "simple_cnn")
    parser.add_argument("--backbone_name",type=str, default='resnet18')
    parser.add_argument("--radius",type=int, default=30)
    # parser.add_argument("--backbone_name", ty)
    ## Config 
    parser.add_argument("--seed",type=int, default=42)
    parser.add_argument("--epochs",type=int, default=10)
    ### early stopping
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--checkpoint_dir",type=str, default='checkpoint')
    parser.add_argument("--delta", type=float, default= 0.01)
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
    # training 
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
    
    print("Model", args.model_type)
    if  args.model_type == 'simple_cnn':
        args.name = (f"{args.model_type}__{args.seed}_{args.batch_size}-lr_{args.lr}")
        train_model = model.SimpleCNN(input_channels= 58, output_channels=args.output_channels,args=args,)
    elif args.model_type == 'cnn':
        args.name = (f"{args.model_type}__{args.backbone_name}__{args.seed}_{args.batch_size}-lr_{args.lr}")
        train_model = model.FeatureExtractorModel(num_input_channels=58, output_dim=1, backbone_name=args.backbone_name)
    elif args.model_type == 'c_attention_cnn':
        train_model = model.Channel_SelfAttentionCNN(num_input_channels=58, output_dim=1, backbone_name=args.backbone_name)
        args.name = (f"{args.model_type}__{args.backbone_name}__{args.seed}_{args.batch_size}-lr_{args.lr}")
    elif args.model_type == 'g_attention_cnn':
        train_model = model.Grid_SelfAttentionCNN(num_input_channels=58, output_dim=1, backbone_name=args.backbone_name)
        args.name = (f"{args.model_type}__{args.backbone_name}__{args.seed}_{args.batch_size}-lr_{args.lr}")
    # args.name = "test"
    if args._use_wandb:
        wandb.init(
            entity="aiotlab",
            project="Cyclone intensity prediction",
            group=args.group_name,
            name=f"{args.name}",
            config=config,
        )
    
    ### Data loading and Data preprocess
    session_name = f"test_model"
    
    if not os.path.exists(f"output/{args.group_name}/checkpoint/"):
        print(f"Make dir output/{args.group_name}/checkpoint/ ...")
        os.makedirs(f"output/{args.group_name}/checkpoint/")

    ## Initialize early stopping
    early_stopping = model_utils.EarlyStopping(
        patience=args.patience,
        verbose=True,
        delta=args.delta,
        path=f"output/{args.group_name}/checkpoint/{args.name}.pt",
    )
    
    #### Model initialization
    # print("Use backbone cnn", args._use_backbone_cnn)
    
    
    ### dataset
    train_dataset = dataloader.CycloneDataset(file_path= "train_index.csv",mode="train", args=args)
    valid_dataset = dataloader.CycloneDataset(file_path= "valid_index.csv", mode="valid", args=args)
    test_dataset = dataloader.CycloneDataset(file_path= "test_index.csv", mode="test", args=args)
    
    
    train_dataloader=  DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_dataloader=  DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_dataloader=  DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    # breakpoint()
    ### loss & optimizer
    mse_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(
        train_model.parameters(), lr=args.lr, weight_decay=args.l2_coef
    )
    #### Model trainning 
    
    
    # dataset = dataloader
    list_train_loss, list_valid_loss = model_utils.train_func(train_model, train_dataloader, valid_dataloader, early_stopping, mse_loss, optimizer, args, torch.device("cuda:0"))
    
    
    ### 
    #### Model testing 
    model_utils.load_model(train_model, f"output/{args.group_name}/checkpoint/stdgi_{args.name}.pt")
    
    if args._use_wandb:
        wandb.run.summary["beet_training_loss"] = early_stopping.best_score
        
    besttrack_scaler, nwp_scaler = train_dataloader.get_scaler() 
            
    test_dataloader.set_scaler(besttrack_scaler,nwp_scaler)

    print("--------Testing-------")
    list_prd, list_grt, epoch_loss, mae, mse, mape, rmse, r2, corr_ = model_utils.test_func(train_model, test_dataloader, mse_loss, args, device=torch.device("cuda:0"))
    if args._use_wandb:
        wandb.log({"mae":mae,
                   "mse":mse,
                   "mape":mape,
                   "rmse":rmse,
                   "r2":r2,
                   "corr":corr_})
    print(f"MSE: {mse} MAE:{mae} MAPE:{mape} RMSE:{rmse} R2:{r2} Corr:{corr_}")
    