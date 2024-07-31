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
    
    parser.add_argument("--model_type",  choices=['regression','regression2','c_attention_cnn','g_attention_cnn'], default= "simple_cnn")
    parser.add_argument("--backbone_name",type=str, default='resnet18')
    parser.add_argument("--radius",type=int, default=30)
    # parser.add_argument("--backbone_name", ty)
    ## Config 
    parser.add_argument("--seed",type=int, default=42)
    parser.add_argument("--epochs",type=int, default=10)
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
    if  args.model_type == 'regression':
        args.name = (f"{args.model_type}__{args.seed}_{args.batch_size}-lr_{args.lr}")
        train_model = model.SimpleTwoLayerNN(58)
    elif  args.model_type == 'regression2':
        args.name = (f"{args.model_type}__{args.seed}_{args.batch_size}-lr_{args.lr}")
        train_model = model.DeepRegressionNN(58)

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
    
    ### dataloading
    scaler = model_utils.get_scaler()
    ### dataset
    train_dataset = dataloader.CycloneDataset3(data_dir= "cutted_data/train/train_data.npz",mode="train", args=args,scaler=scaler)
    valid_dataset = dataloader.CycloneDataset3(data_dir= "cutted_data/valid/train_data.npz", mode="valid", args=args,scaler=scaler)
    test_dataset = dataloader.CycloneDataset3(data_dir= "cutted_data/test/train_data.npz", mode="test", args=args,scaler=scaler)
    
    
    
    
    # breakpoint()
    ### loss & optimizer
    mse_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(
        train_model.parameters(), lr=args.lr, weight_decay=args.l2_coef
    )
    #### Model trainning 
    
    
    # dataset = dataloader
    list_train_loss, list_valid_loss = model_utils.train_func(train_model, train_dataset, valid_dataset, early_stopping, mse_loss, optimizer, args, torch.device("cuda:0"))
    
    
    ### 
    #### Model testing 
    model_utils.load_model(train_model, f"output/{args.group_name}/checkpoint/{args.name}.pt")
    
    if args._use_wandb:
        wandb.run.summary["beet_training_loss"] = early_stopping.best_score

    # besttrack_scaler, nwp_scaler = train_dataset.get_scaler() 
            
    # test_dataset.set_scaler(besttrack_scaler,nwp_scaler)
    test_dataloader=  DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print("--------Testing-------")
    list_prd, list_grt, epoch_loss, mae, mse, mape, rmse, r2, corr_ = model_utils.test_func(train_model, test_dataloader, mse_loss, args, None,device=torch.device("cuda:0"))
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