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
    parser.add_argument("--freeze",action='store_true', default=False)
    # 
    args = parser.parse_args()
    return args 

if __name__ == "__main__":
    args = get_option()
    
    try:
        config = vars(args)
    except IOError as msg:
        args.error(str(msg))

    # 
    import orca_model
    model_utils.seed_everything(args.seed)

    nwp_scaler, bt_scaler, input_channels = model_utils.get_scaler2(args)
    print(f"Number input channel {input_channels}")
    print("Model", args.model_type)

    
    if  args.model_type == 'simple_cnn':
        args.name = (f"{args.model_type}-Freeze_{args.freeze}-SLr_{args._use_scheduler_lr}_{args.scheduler_type}-loss_func_{args.loss_func}-{args.seed}_{args.batch_size}-lr_{args.lr}-tf_gr_{args.transform_groundtruth}")
        train_model = model.SimpleCNN(input_channels= input_channels, output_channels=args.output_channels,args=args,)
        train_dataset = dataloader.CycloneDataset2(data_dir= f"{args.data_dir}/train/data.npz",mode="train", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        valid_dataset = dataloader.CycloneDataset2(data_dir= f"{args.data_dir}/valid/data.npz", mode="valid", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        test_dataset = dataloader.CycloneDataset2(data_dir= f"{args.data_dir}/test/data.npz", mode="test", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        
    elif args.model_type == 'cnn':
        args.name = (f"{args.model_type}-Freeze_{args.freeze}-SLr_{args._use_scheduler_lr}_{args.scheduler_type}-loss_func_{args.loss_func}-{args.backbone_name}__{args.seed}_{args.batch_size}-lr_{args.lr}-tf_gr_{args.transform_groundtruth}")
        train_model = model.FeatureExtractorModel(num_input_channels=input_channels, output_dim=1, backbone_name=args.backbone_name)
        train_dataset = dataloader.CycloneDataset2(data_dir= f"{args.data_dir}/train/data.npz",mode="train", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        valid_dataset = dataloader.CycloneDataset2(data_dir= f"{args.data_dir}/valid/data.npz", mode="valid", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        test_dataset = dataloader.CycloneDataset2(data_dir= f"{args.data_dir}/test/data.npz", mode="test", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
    elif args.model_type == 'c_attention_cnn':
        train_model = model.Channel_SelfAttentionCNN(num_input_channels=input_channels, output_dim=1, backbone_name=args.backbone_name)
        args.name = (f"{args.model_type}-Freeze_{args.freeze}-SLr_{args._use_scheduler_lr}_{args.scheduler_type}-loss_func_{args.loss_func}-{args.backbone_name}__{args.seed}_{args.batch_size}-lr_{args.lr}-tf_gr_{args.transform_groundtruth}")
        train_dataset = dataloader.CycloneDataset2(data_dir= f"{args.data_dir}/train/data.npz",mode="train", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        valid_dataset = dataloader.CycloneDataset2(data_dir= f"{args.data_dir}/valid/data.npz", mode="valid", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        test_dataset = dataloader.CycloneDataset2(data_dir= f"{args.data_dir}/test/data.npz", mode="test", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
    
    elif args.model_type == 'g_attention_cnn':
        train_model = model.Grid_SelfAttentionCNN(num_input_channels=input_channels, output_dim=1, backbone_name=args.backbone_name)
        args.name = (f"{args.model_type}-Freeze_{args.freeze}-SLr_{args._use_scheduler_lr}_{args.scheduler_type}-loss_func_{args.loss_func}-{args.backbone_name}__{args.seed}_{args.batch_size}-lr_{args.lr}-tf_gr_{args.transform_groundtruth}")
        train_dataset = dataloader.CycloneDataset2(data_dir= f"{args.data_dir}/train/data.npz",mode="train", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        valid_dataset = dataloader.CycloneDataset2(data_dir= f"{args.data_dir}/valid/data.npz", mode="valid", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        test_dataset = dataloader.CycloneDataset2(data_dir= f"{args.data_dir}/test/data.npz", mode="test", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        
    elif args.model_type == "simple_vit":
        import simple_vit
        train_model = simple_vit.SimpleViT(image_size = 100,
                                            patch_size = args.patch_size,
                                            num_classes = 1,
                                            channels=58,
                                            dim = args.dim,
                                            depth = 6,
                                            heads = args.heads,
                                            mlp_dim = 2048)
        
        args.name = (f"{args.model_type}-Freeze_{args.freeze}-SLr_{args._use_scheduler_lr}_{args.scheduler_type}-loss_func_{args.loss_func}-{args.backbone_name}__{args.seed}_{args.batch_size}-lr_{args.lr}-tf_gr_{args.transform_groundtruth}-ps_{args.patch_size}-dim_{args.dim}-head_{args.heads}")
        train_dataset = dataloader.VITDataset(data_dir= f"{args.data_dir}/train/data.npz",mode="train", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        valid_dataset = dataloader.VITDataset(data_dir= f"{args.data_dir}/valid/data.npz", mode="valid", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        test_dataset = dataloader.VITDataset(data_dir= f"{args.data_dir}/test/data.npz", mode="test", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
    elif args.model_type == "simple_vit2":
        import simple_vit
        train_model = simple_vit.SimpleViT2(image_size = 100,
                                            patch_size = args.patch_size,
                                            num_classes = 1,
                                            channels=58,
                                            dim = args.dim,
                                            depth = 6,
                                            heads = args.heads,
                                            mlp_dim = 2048)
        
        args.name = (f"{args.model_type}-Freeze_{args.freeze}-SLr_{args._use_scheduler_lr}_{args.scheduler_type}-loss_func_{args.loss_func}-{args.backbone_name}__{args.seed}_{args.batch_size}-lr_{args.lr}-tf_gr_{args.transform_groundtruth}-ps_{args.patch_size}-dim_{args.dim}-head_{args.heads}")
        train_dataset = dataloader.VITDataset(data_dir= f"{args.data_dir}/train/data.npz",mode="train", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        valid_dataset = dataloader.VITDataset(data_dir= f"{args.data_dir}/valid/data.npz", mode="valid", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        test_dataset = dataloader.VITDataset(data_dir= f"{args.data_dir}/test/data.npz", mode="test", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
    
    elif args.model_type == "simple_vit3":
        import simple_vit
        train_model = simple_vit.SimpleViT3(image_size = 100,
                                            patch_size = args.patch_size,
                                            num_classes = 1,
                                            channels=58,
                                            dim = args.dim,
                                            depth = 6,
                                            heads = args.heads,
                                            mlp_dim = 2048)
        
        args.name = (f"{args.model_type}-Freeze_{args.freeze}-SLr_{args._use_scheduler_lr}_{args.scheduler_type}-loss_func_{args.loss_func}-{args.backbone_name}__{args.seed}_{args.batch_size}-lr_{args.lr}-tf_gr_{args.transform_groundtruth}-ps_{args.patch_size}-dim_{args.dim}-head_{args.heads}")
        train_dataset = dataloader.VITDataset(data_dir= f"{args.data_dir}/train/data.npz",mode="train", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        valid_dataset = dataloader.VITDataset(data_dir= f"{args.data_dir}/valid/data.npz", mode="valid", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        test_dataset = dataloader.VITDataset(data_dir= f"{args.data_dir}/test/data.npz", mode="test", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
    
    elif args.model_type == "orca_based_e2e":
        import orca_model
        cnn_embedder = orca_model.CNNEmbedder(input_channels=58, output_dim=768, kernel_size=10)
        prediction_head = orca_model.PredictionHead()
        
        train_model = orca_model.CrossTuningModel(cnn_embedder, "vit", prediction_head)
        args.name = (f"{args.model_type}-Freeze_{args.freeze}-SLr_{args._use_scheduler_lr}_{args.scheduler_type}-loss_func_{args.loss_func}-{args.backbone_name}__{args.seed}_{args.batch_size}-lr_{args.lr}-tf_gr_{args.transform_groundtruth}-ps_{args.patch_size}-dim_{args.dim}-head_{args.heads}")
        train_dataset = dataloader.VITDataset(data_dir= f"{args.data_dir}/train/data.npz",mode="train", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        valid_dataset = dataloader.VITDataset(data_dir= f"{args.data_dir}/valid/data.npz", mode="valid", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        test_dataset = dataloader.VITDataset(data_dir= f"{args.data_dir}/test/data.npz", mode="test", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
    
    elif args.model_type == "prompt_vit0":
        import orca_model
        cnn_embedder = orca_model.CNNEmbedder(input_channels=input_channels, output_dim=768, kernel_size=10)
        
        prediction_head = orca_model.PredictionHead(dim = 768)
        
        train_model = orca_model.Prompt_Tuning_Model0(cnn_embedder, args.body_model_name, prediction_head,args)
        
        args.name = (f"{args.model_type}-pse_{args.use_position_embedding}-SLr_{args._use_scheduler_lr}_{args.scheduler_type}-loss_func_{args.loss_func}-{args.backbone_name}__{args.seed}_{args.batch_size}-lr_{args.lr}-tf_gr_{args.transform_groundtruth}-ps_{args.patch_size}-dim_{args.dim}-head_{args.heads}")
        train_dataset = dataloader.VITDataset(data_dir= f"{args.data_dir}/train/data.npz",mode="train", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        valid_dataset = dataloader.VITDataset(data_dir= f"{args.data_dir}/valid/data.npz", mode="valid", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        test_dataset = dataloader.VITDataset(data_dir= f"{args.data_dir}/test/data.npz", mode="test", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
    
    
    elif args.model_type == "prompt_vit1":
        import orca_model
        cnn_embedder = orca_model.CNNEmbedder(input_channels=input_channels, output_dim=768, kernel_size=10)
        
        prediction_head = orca_model.PredictionHead(dim = 768 + args.prompt_dims)
        
        train_model = orca_model.Prompt_Tuning_Model1(cnn_embedder, args.body_model_name, prediction_head,args)
        
        args.name = (f"{args.model_type}-freee_{args.freeze}-pse_{args.use_position_embedding}-SLr_{args._use_scheduler_lr}_{args.scheduler_type}-loss_func_{args.loss_func}-{args.backbone_name}__{args.seed}_{args.batch_size}-lr_{args.lr}-tf_gr_{args.transform_groundtruth}-ps_{args.patch_size}-dim_{args.dim}-head_{args.heads}")
        train_dataset = dataloader.VITDataset(data_dir= f"{args.data_dir}/train/data.npz",mode="train", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        valid_dataset = dataloader.VITDataset(data_dir= f"{args.data_dir}/valid/data.npz", mode="valid", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        test_dataset = dataloader.VITDataset(data_dir= f"{args.data_dir}/test/data.npz", mode="test", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
    
    elif args.model_type == "prompt_vit2":
        import orca_model
        cnn_embedder = orca_model.CNNEmbedder(input_channels=input_channels, output_dim=768, kernel_size=10)
        
        prediction_head = orca_model.PredictionHead(dim = 768 + args.prompt_dims)
        
        train_model = orca_model.Prompt_Tuning_Model2(cnn_embedder, args.body_model_name, prediction_head,args)
        
        args.name = (f"{args.model_type}-freee_{args.freeze}-pse_{args.use_position_embedding}-SLr_{args._use_scheduler_lr}_{args.scheduler_type}-loss_func_{args.loss_func}-{args.backbone_name}__{args.seed}_{args.batch_size}-lr_{args.lr}-tf_gr_{args.transform_groundtruth}-ps_{args.patch_size}-dim_{args.dim}-head_{args.heads}")
        train_dataset = dataloader.VITDataset(data_dir= f"{args.data_dir}/train/data.npz",mode="train", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        valid_dataset = dataloader.VITDataset(data_dir= f"{args.data_dir}/valid/data.npz", mode="valid", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        test_dataset = dataloader.VITDataset(data_dir= f"{args.data_dir}/test/data.npz", mode="test", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        
    elif args.model_type == "prompt_vit3":
        import orca_model
        cnn_embedder = orca_model.CNNEmbedder(input_channels=input_channels, output_dim=768 - args.prompt_dims, kernel_size=10)
        
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
    
    elif args.model_type == "individual_prompt_vit1":
        import orca_model

        prediction_head = orca_model.PredictionHead(dim = 768 + args.prompt_dims)
        
        train_model = orca_model.Individual_Embeder_Tuning_Model1(58, "vit", prediction_head,args.prompt_dims)
        
        args.name = f"{args.model_type}-SLr_{args._use_scheduler_lr}_{args.scheduler_type}-loss_func_{args.loss_func}-{args.backbone_name}__{args.seed}_{args.batch_size}-lr_{args.lr}-tf_gr_{args.transform_groundtruth}-ps_{args.patch_size}-dim_{args.dim}-head_{args.heads}"
        train_dataset = dataloader.VITDataset(data_dir= f"{args.data_dir}/train/data.npz",mode="train", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        valid_dataset = dataloader.VITDataset(data_dir= f"{args.data_dir}/valid/data.npz", mode="valid", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        test_dataset = dataloader.VITDataset(data_dir= f"{args.data_dir}/test/data.npz", mode="test", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
    
    
    elif args.model_type == "individual_prompt_vit2":
        import orca_model

        prediction_head = orca_model.PredictionHead(dim = 768 + args.prompt_dims)
        
        train_model = orca_model.Individual_Embeder_Tuning_Model2(58, "vit", prediction_head,args.prompt_dims)
        
        args.name = f"{args.model_type}-SLr_{args._use_scheduler_lr}_{args.scheduler_type}-loss_func_{args.loss_func}-{args.backbone_name}__{args.seed}_{args.batch_size}-lr_{args.lr}-tf_gr_{args.transform_groundtruth}-ps_{args.patch_size}-dim_{args.dim}-head_{args.heads}"
        valid_dataset = dataloader.VITDataset(data_dir= f"{args.data_dir}/valid/data.npz", mode="valid", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        test_dataset = dataloader.VITDataset(data_dir= f"{args.data_dir}/test/data.npz", mode="test", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
    
    elif args.model_type == "individual_prompt_vit3":
        import orca_model

        prediction_head = orca_model.PredictionHead()
        
        train_model = orca_model.Individual_Embeder_Tuning_Model3(58, "vit", prediction_head,args)
        
        args.name = f"{args.model_type}-SLr_{args._use_scheduler_lr}_{args.scheduler_type}-loss_func_{args.loss_func}-{args.backbone_name}__{args.seed}_{args.batch_size}-lr_{args.lr}-tf_gr_{args.transform_groundtruth}-ps_{args.patch_size}-dim_{args.dim}-head_{args.heads}"
        train_dataset = dataloader.VITDataset(data_dir= f"{args.data_dir}/train/data.npz",mode="train", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        valid_dataset = dataloader.VITDataset(data_dir= f"{args.data_dir}/valid/data.npz", mode="valid", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        test_dataset = dataloader.VITDataset(data_dir= f"{args.data_dir}/test/data.npz", mode="test", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
    
    
    elif args.model_type == "region_attention":
        import orca_model
        if args.use_cls_for_region:
            if args.combining_layer_type == 1:
                prediction_head = orca_model.PredictionHead(dim = 768 + args.prompt_dims, n_patchs=101)
            elif args.combining_layer_type == 2:
                prediction_head = orca_model.PredictionHead(dim = 768 + args.prompt_dims, n_patchs=100)
            else:
                raise("")
        else:
            prediction_head = orca_model.PredictionHead(dim = 768 + args.prompt_dims)
        
        train_model = orca_model.Region_Attention(58, "vit", prediction_head, args)
        
        args.name = f"{args.model_type}-RCls_{args.use_cls_for_region}-CT_{args.combining_layer_type}-SLr_{args._use_scheduler_lr}_{args.scheduler_type}-loss_func_{args.loss_func}-{args.backbone_name}__{args.seed}_{args.batch_size}-lr_{args.lr}-tf_gr_{args.transform_groundtruth}-ps_{args.patch_size}-dim_{args.dim}-head_{args.heads}"
        train_dataset = dataloader.VITDataset(data_dir= f"{args.data_dir}/train/data.npz",mode="train", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        valid_dataset = dataloader.VITDataset(data_dir= f"{args.data_dir}/valid/data.npz", mode="valid", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        test_dataset = dataloader.VITDataset(data_dir= f"{args.data_dir}/test/data.npz", mode="test", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
    
    

    elif args.model_type == "metnet_model":
        import metnet_model
        train_model = metnet_model.MetNet_Tuning_Model()
        
        args.name = f"{args.model_type}-SLr_{args._use_scheduler_lr}_{args.scheduler_type}-loss_func_{args.loss_func}-{args.backbone_name}__{args.seed}_{args.batch_size}-lr_{args.lr}-tf_gr_{args.transform_groundtruth}-ps_{args.patch_size}-dim_{args.dim}-head_{args.heads}"
        train_dataset = dataloader.VITDataset(data_dir= f"{args.data_dir}/train/data.npz",mode="train", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        valid_dataset = dataloader.VITDataset(data_dir= f"{args.data_dir}/valid/data.npz", mode="valid", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        test_dataset = dataloader.VITDataset(data_dir= f"{args.data_dir}/test/data.npz", mode="test", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
    
    elif args.model_type == "metnet_model2":
        import metnet_model
        train_model = metnet_model.MetNet_Tuning_Model2()
        
        args.name = f"{args.model_type}-SLr_{args._use_scheduler_lr}_{args.scheduler_type}-loss_func_{args.loss_func}-{args.backbone_name}__{args.seed}_{args.batch_size}-lr_{args.lr}-tf_gr_{args.transform_groundtruth}-ps_{args.patch_size}-dim_{args.dim}-head_{args.heads}"
        train_dataset = dataloader.VITDataset(data_dir= f"{args.data_dir}/train/data.npz",mode="train", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        valid_dataset = dataloader.VITDataset(data_dir= f"{args.data_dir}/valid/data.npz", mode="valid", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        test_dataset = dataloader.VITDataset(data_dir= f"{args.data_dir}/test/data.npz", mode="test", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
    

    args.name = (f"{args.model_type}-pse_{args.use_position_embedding}-SLr_{args._use_scheduler_lr}_{args.scheduler_type}-loss_func_{args.loss_func}-{args.backbone_name}__{args.seed}_{args.batch_size}-lr_{args.lr}-tf_gr_{args.transform_groundtruth}-ps_{args.patch_size}-dim_{args.dim}-head_{args.heads}")
    train_dataset = dataloader.VITDataset(data_dir= f"{args.data_dir}/train/data.npz",mode="train", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
    valid_dataset = dataloader.VITDataset(data_dir= f"{args.data_dir}/valid/data.npz", mode="valid", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
    test_dataset = dataloader.VITDataset(data_dir= f"{args.data_dir}/test/data.npz", mode="test", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
        

    model_utils.seed_everything(args.seed)
    # scaler = model_utils.get_scaler()
    ### Init wandb
    
    # nwp_scaler, bt_scaler, input_channels = model_utils.get_scaler2(args)
    # print(f"Number input channel {input_channels}")
    # print("Model", args.model_type)

    if args._use_wandb:
        wandb.login(key='ab2505638ca8fabd9114e88f3449ddb51e15a942')
        wandb.init(
            entity="aiotlab",
            project="Cyclone intensity prediction",
            group=f"test_only-{args.group_name}",
            name=f"{args.name}",
            config=config,
        )
    device = torch.device("cpu")
    train_model = train_model.to(device)
    model_path = "output/prompt_vit3_seed79_rm_0/checkpoint/stdgi_prompt_vit3-pse_False-SLr_False_steplr-loss_func_mse-resnet18__79_32-lr_0.0005-tf_gr_False-ps_10-dim_1024-head_16.pt"
    model_utils.load_model(train_model,model_path)
    sample1 = test_dataset[0]
    x_train, y_grt = model_utils.to_float(torch.tensor(sample1['x']).unsqueeze(0), device), model_utils.to_float(torch.tensor(sample1['y']).unsqueeze(0),device)
    y_prd = train_model(x_train)

    
    
    test_dataloader=  DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    if args.loss_func == "mse":
        loss_func = nn.MSELoss()
    elif args.loss_func == "weighted_mse":
        import loss
        loss_func = loss.WeightedMSELoss()
    else:
        raise("Not correct loss function!")

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
    