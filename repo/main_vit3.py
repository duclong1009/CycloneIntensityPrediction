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
    ##  VIT config
    parser.add_argument("--patch_size",type=int, default=10)
    parser.add_argument("--dim",type=int, default=1024)
    parser.add_argument("--heads",type=int, default=16)
    
    parser.add_argument("--model_type",  choices=['simple_vit3'], default= "simple_vit3")
    parser.add_argument("--backbone_name",type=str, default='resnet18')
    parser.add_argument("--radius",type=int, default=30)
    # parser.add_argument("--backbone_name", ty)
    ## Config 
    parser.add_argument("--seed",type=int, default=42)
    parser.add_argument("--epochs",type=int, default=10)
    parser.add_argument("--data_dir",type=str, default="cutted_data2")
    parser.add_argument("--transform_groundtruth",action="store_true", default=False)
    
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
    
    parser.add_argument("--loss_func",type=str, default='mse', choices=['weighted_mse','mse'])
    ###
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
    
    nwp_scaler, bt_scaler, input_channels = model_utils.get_scaler2(args)
    print(f"Number input channel {input_channels}")
    print("Model", args.model_type)
    

    
    
    import simple_vit
    train_model = simple_vit.SimpleViT2(image_size = 100,
                                        patch_size = args.patch_size,
                                        num_classes = 1,
                                        channels=58,
                                        dim = args.dim,
                                        depth = 6,
                                        heads = args.heads,
                                        mlp_dim = 2048)
    
    args.name = (f"{args.model_type}-loss_func_{args.loss_func}-{args.backbone_name}__{args.seed}_{args.batch_size}-lr_{args.lr}-tf_gr_{args.transform_groundtruth}-ps_{args.patch_size}-dim_{args.dim}-head_{args.heads}")
    train_dataset = dataloader.VITDataset(data_dir= f"{args.data_dir}/train/data.npz",mode="train", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
    valid_dataset = dataloader.VITDataset(data_dir= f"{args.data_dir}/valid/data.npz", mode="valid", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
    test_dataset = dataloader.VITDataset(data_dir= f"{args.data_dir}/test/data.npz", mode="test", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        
    # args.name = "test"
    if args._use_wandb:
        wandb.login(key='ab2505638ca8fabd9114e88f3449ddb51e15a942')
        wandb.init(
            entity="aiotlab",
            project="Cyclone intensity prediction",
            group=args.group_name,
            name=f"{args.name}",
            config=config,
        )
    
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
    optimizer = torch.optim.Adam(
        train_model.parameters(), lr=args.lr, weight_decay=args.l2_coef
    )
    #### Model trainning 
    
    
    # dataset = dataloader
    list_train_loss, list_valid_loss = model_utils.train_func(train_model, train_dataset, valid_dataset, early_stopping, loss_func, optimizer, args, torch.device("cuda:0"))
    
    
    ### 
    #### Model testing 
    model_utils.load_model(train_model, f"output/{args.group_name}/checkpoint/stdgi_{args.name}.pt")
    
    if args._use_wandb:
        wandb.run.summary["beet_training_loss"] = early_stopping.best_score

    # besttrack_scaler, nwp_scaler = train_dataset.get_scaler() 
            
    # test_dataset.set_scaler(besttrack_scaler,nwp_scaler)
    test_dataloader=  DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print("--------Testing-------")
    list_prd, list_grt, epoch_loss, mae, mse, mape, rmse, r2, corr_ = model_utils.test_func(train_model, test_dataloader, loss_func, args, bt_scaler,device=torch.device("cuda:0"))
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