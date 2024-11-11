import torch

# Define the Data Tensor (random values for illustration, shape 32x5x100)
data_tensor = torch.randint(1, 100, (10, 5, 100))

# Define the Index Tensor (indices for selection, shape 32x1)
index_tensor = torch.randint(0, 5, (10, 1))  # each value is an index from 0 to 4

# Use indexing to select the rows based on index_tensor
selected_elements = data_tensor[torch.arange(data_tensor.shape[0]).unsqueeze(1), index_tensor]
from repo.orca_model import Prompt_Tuning_Model_Leading_t, CNNEmbedder, PredictionHead2

x = torch.rand((10,63,101,101))
leading_time = torch.randint(0, 5, (10, 1))
prompt_dims = 128
cnn_embedder = CNNEmbedder(63,768 - prompt_dims, 10)
prediction_head = PredictionHead2()
args = {}
# args.prompt_dims = prompt_dims
# args.use_position_embedding = False
train_model = Prompt_Tuning_Model_Leading_t(cnn_embedder,'vit',prediction_head, args)

a = train_model((x, leading_time))
# Display the result
selected_elements
