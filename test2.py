import torch
from transformers import PatchTSTConfig, PatchTSTModel
from huggingface_hub import hf_hub_download

# Download the dataset
file = hf_hub_download(
    repo_id="hf-internal-testing/etth1-hourly-batch", filename="train-batch.pt", repo_type="dataset"
)
batch = torch.load(file)

# Modify the model configuration for your dataset shape (32, 32, 3)
config = PatchTSTConfig.from_pretrained("namctin/patchtst_etth1_pretrain")
config.context_length = 48  # Set sequence length to 32
config.num_input_channels = 3  # Set number of input features to 3

# Initialize the model with the modified configuration
model = PatchTSTModel(config)

# Prepare your dataset (assuming it has shape [32, 32, 3])
# For demonstration, we'll slice the original batch to match (32, 32, 3)
# In practice, replace this with your actual dataset
your_past_values = batch["past_values"][:, :48, :3]  # Shape: [32, 32, 3]
your_future_values = batch["future_values"][:, :48, :3]  # Shape: [32, 32, 3]

# Run the model
outputs = model(
    past_values=your_past_values,
    future_values=your_future_values,
)

# Access the last hidden state
last_hidden_state = outputs.last_hidden_state

# Optional: Print shapes to verify
print(f"Input past_values shape: {your_past_values.shape}")
print(f"Output last_hidden_state shape: {last_hidden_state.shape}")

breakpoint()