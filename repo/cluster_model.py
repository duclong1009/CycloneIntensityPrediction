import torch
import torch.nn as nn
import orca_model  # Assuming this module contains the necessary classes

class MultiHeadModel(nn.Module):
    def __init__(self, model_name ,prediction_head, args):
        super(MultiHeadModel, self).__init__()
        self.model_name = model_name
        n_clusters = len(args.cluster_index)
        self.cluster_index = args.cluster_index
        self.prediction_head = prediction_head
        
        # Use nn.ModuleList to store the heads
        self.list_embeder = nn.ModuleList()
        
        for i in range(n_clusters):
            if model_name == "prompt_vit1":
                cnn_embedder = orca_model.CNNEmbedder(
                    input_channels=len(self.cluster_index[i]), 
                    output_dim=768, 
                    kernel_size=10
                )
                train_model = orca_model.Prompt_Tuning_Model1_Embeder(
                    cnn_embedder, 
                    "vit",
                    args.prompt_dims
                )
                
            elif model_name == "prompt_vit2":
                cnn_embedder = orca_model.CNNEmbedder(
                    input_channels=len(self.cluster_index[i]), 
                    output_dim=768, 
                    kernel_size=10
                )
                train_model = orca_model.Prompt_Tuning_Model2_Embeder(
                    cnn_embedder, 
                    "vit",
                    args.prompt_dims
                )
                
            elif model_name == "prompt_vit3":
                cnn_embedder = orca_model.CNNEmbedder(
                    input_channels=len(self.cluster_index[i]), 
                    output_dim=768 - args.prompt_dims, 
                    kernel_size=10
                )
                train_model = orca_model.Prompt_Tuning_Model3_Embeder(
                    cnn_embedder, 
                    "vit",
                    args.prompt_dims
                )
                
            else:
                raise("")
            self.list_embeder.append(train_model)

    def forward(self, x):
        list_embedding_output = []
        for i, cluster_x in enumerate(x):
            embedding_output = self.list_embeder[i](cluster_x)
            list_embedding_output.append(embedding_output)
        final_embedding_output = torch.concat(list_embedding_output,-1) 
        return self.prediction_head(final_embedding_output) # Add return to output the results
    
class MultiHead_MultiOutput_Model(nn.Module):
    def __init__(self, model_name, args):
        super(MultiHead_MultiOutput_Model, self).__init__()
        self.model_name = model_name
        n_clusters = len(args.cluster_index)
        self.cluster_index = args.cluster_index
        
        # Use nn.ModuleList to store the heads
        self.list_embeder = nn.ModuleList()
        self.list_head = nn.ModuleList()
        
        
        if model_name == "prompt_vit1":
            self.prediction_head = orca_model.PredictionHead(dim = (768 + args.prompt_dims) * len(args.cluster_index),n_patchs=100)
            for i in range(n_clusters):
                head_i = orca_model.PredictionHead(dim = 768 + args.prompt_dims,n_patchs=100)
                cnn_embedder = orca_model.CNNEmbedder(
                    input_channels=len(self.cluster_index[i]), 
                    output_dim=768, 
                    kernel_size=10
                )
                train_model = orca_model.Prompt_Tuning_Model1_Embeder(
                    cnn_embedder, 
                    "vit",
                    args.prompt_dims
                )
                self.list_embeder.append(train_model)
                self.list_head.append(head_i)
        elif model_name == "prompt_vit2":
            self.prediction_head = orca_model.PredictionHead(dim = (768 + args.prompt_dims) * len(args.cluster_index),n_patchs=100)
            for i in range(n_clusters):
                head_i = orca_model.PredictionHead(dim = 768 + args.prompt_dims,n_patchs=100)
                cnn_embedder = orca_model.CNNEmbedder(
                    input_channels=len(self.cluster_index[i]), 
                    output_dim=768, 
                    kernel_size=10
                )
                train_model = orca_model.Prompt_Tuning_Model2_Embeder(
                    cnn_embedder, 
                    "vit",
                    args.prompt_dims
                )
                self.list_embeder.append(train_model)
                self.list_head.append(head_i)
                
        elif model_name == "prompt_vit3":
            self.prediction_head = orca_model.PredictionHead(dim = 768 * len(args.cluster_index),n_patchs=100)
            for i in range(n_clusters):
                head_i = orca_model.PredictionHead(dim = 768,n_patchs=100)
                cnn_embedder = orca_model.CNNEmbedder(
                    input_channels=len(self.cluster_index[i]), 
                    output_dim=768 - args.prompt_dims, 
                    kernel_size=10
                )
                train_model = orca_model.Prompt_Tuning_Model3_Embeder(
                    cnn_embedder, 
                    "vit",
                    args.prompt_dims
                )
                self.list_embeder.append(train_model)
                self.list_head.append(head_i)
                
        else:
            raise("")
            self.list_embeder.append(train_model)

    def forward_train(self, x):
        list_embedding_output = []
        list_output = []
        for i, cluster_x in enumerate(x):
            embedding_output = self.list_embeder[i](cluster_x)
            output_i = self.list_head[i](embedding_output)
            
            list_embedding_output.append(embedding_output)
            list_output.append(output_i)
            
        final_embedding_output = torch.concat(list_embedding_output,-1) 
        return self.prediction_head(final_embedding_output), list_output  # Add return to output the results
    
    def forward(self, x):
        list_embedding_output = []
        for i, cluster_x in enumerate(x):
            embedding_output = self.list_embeder[i](cluster_x)            
            list_embedding_output.append(embedding_output)
            
        final_embedding_output = torch.concat(list_embedding_output,-1) 
        return self.prediction_head(final_embedding_output)  # Add return to output the results
    
    def forward_stage1(self, x):
        list_embedding_output = []
        list_output = []
        for i, cluster_x in enumerate(x):
            embedding_output = self.list_embeder[i](cluster_x)
            output_i = self.list_head[i](embedding_output)
            
            list_embedding_output.append(embedding_output)
            list_output.append(output_i)
        return  list_output  # Add return to output the results
    
    