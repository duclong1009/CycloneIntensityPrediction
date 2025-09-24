import numpy as np
import torch
import torch.nn as nn
from repo.orca_model import PatchEmbedding3D_2, PredictionHead, Prompt_Tuning_Model6_6



class Args:
        def __init__(self):
            self.prompt_dims = 256  # Prompt dimension size
            self.use_position_embedding = False

            self.historical_nwp_length = 5  # Number of historical NWP timesteps

            self.patch_size_t = 1  # Temporal patch size
            self.patch_size_h = 10  # Height patch size 
            self.patch_size_w = 10  # Width patch size
            self.use_position_embedding = True
            self.max_lead_time = 72
            self.start_lead_time = 0
            self.freeze = False
            self.image_size = 50
            self.data_dir = "/mnt/disk1/aiotlab/longnd/data/tc_data/basedyear_data_3days/data2"
            self.historical_data_length = 20

args = Args()

arr = torch.rand((32,5, 63,50,50))

patch_size = (1, 10, 10)

patch_size = (args.patch_size_t, args.patch_size_h, args.patch_size_w)
embedder = PatchEmbedding3D_2( expected_output_dim=768 - 128, patch_size=patch_size, n_timestep=args.historical_nwp_length, args=args)
breakpoint()
prediction_head = PredictionHead(n_patchs= 63 +2)
train_model = Prompt_Tuning_Model6_6(embedder, 'vit', prediction_head, args)

output = train_model(arr)
