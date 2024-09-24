import argparse
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
    args = parser.parse_args()
    return args 

import repo.orca_model as orca_model

args = get_option()

prediction_head = orca_model.PredictionHead()

if args.use_cls_for_region:
    if args.combining_layer_type == 1:
        prediction_head = orca_model.PredictionHead(dim = 768 + args.prompt_dims, n_patchs=100)
    elif args.combining_layer_type == 2:
        prediction_head = orca_model.PredictionHead(dim = 768 + args.prompt_dims, n_patchs=100)
    else:
        raise("")
else:
    prediction_head = orca_model.PredictionHead(dim = 768 + args.prompt_dims)

train_model = orca_model.Region_Attention(58, "vit", prediction_head, args)
import torch

tensor = torch.rand((10,58,100,100))

x = train_model(tensor)
