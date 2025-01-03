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

    n_fts = [63, 1]
    if args.model_type == "prompt_vit6":
        import orca_model
        cnn_embedder = orca_model.CNNEmbedder(input_channels=n_fts[0], output_dim=768 - args.prompt_dims, kernel_size=10)
        
        prediction_head = orca_model.PredictionHead(n_patchs=101)
        
        train_model = orca_model.Prompt_Tuning_Model6(cnn_embedder, args.body_model_name, prediction_head,args)
        
        args.name = (f"{args.model_type}-prl_{args.prompt_length}-freee_{args.freeze}-pse_{args.use_position_embedding}-SLr_{args._use_scheduler_lr}_{args.scheduler_type}-loss_func_{args.loss_func}-{args.backbone_name}__{args.seed}_{args.batch_size}-lr_{args.lr}-tf_gr_{args.transform_groundtruth}-ps_{args.patch_size}-dim_{args.dim}-head_{args.heads}")

    elif args.model_type == "prompt_vit6_2":

        import orca_model
        cnn_embedder = orca_model.CNNEmbedder(input_channels=n_fts[0], output_dim=768 - args.prompt_dims, kernel_size=10)
        
        prediction_head = orca_model.PredictionHead(n_patchs=101)
        
        train_model = orca_model.Prompt_Tuning_Model6_Progressive(cnn_embedder, args.body_model_name, prediction_head,args)
        
        args.name = (f"{args.model_type}-prl_{args.prompt_length}-freee_{args.freeze}-pse_{args.use_position_embedding}-SLr_{args._use_scheduler_lr}_{args.scheduler_type}-loss_func_{args.loss_func}-{args.backbone_name}__{args.seed}_{args.batch_size}-lr_{args.lr}-tf_gr_{args.transform_groundtruth}-ps_{args.patch_size}-dim_{args.dim}-head_{args.heads}")

    # print(f"Number input channel {input_shape}")
    print("Model", args.model_type)

    arr = torch.rand((32,6,63,101,101))
    his = torch.rand((32,64))
    nwp_id = torch.randint(low=0,high=5, size=(32,1))
    x = [arr,his, nwp_id]
    train_model(x)
    
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
    if args.debug:
        print("Using CPU")
        device = torch.device("cpu")
    else:
        print("Using GPU")
        device = torch.device("cuda:0")
    