import torch
import torch.nn as nn
import orca_model  # Assuming this module contains the necessary classes

class MultiHeadModel(nn.Module):
    def __init__(self, prediction_head, args):
        super(MultiHeadModel, self).__init__()
        n_clusters = len(args.cluster_index)
        self.cluster_index = args.cluster_index
        self.prediction_head = prediction_head
        
        # Use nn.ModuleList to store the heads
        self.list_head = nn.ModuleList()
        
        for i in range(n_clusters):
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
            self.list_head.append(train_model)

    def forward(self, x):
        list_embedding_output = []
        for i, cluster_x in enumerate(x):
            embedding_output = self.list_head[i](cluster_x)
            list_embedding_output.append(embedding_output)
        final_embedding_output = torch.concat(list_embedding_output,-1) 
        return self.prediction_head(final_embedding_output) # Add return to output the results
    


x1 = torch.rand((10,3,100,100))
x2 = torch.rand((10,3,100,100))
x3 = torch.rand((10,3,100,100))
x4 = torch.rand((10,49,100,100))