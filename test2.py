
from transformers import AutoImageProcessor, ViTModel


from transformers import ViTModel, ViTConfig

# Load the architecture from the configuration
config = ViTConfig()  # Use default configuration or modify as needed
model = ViTModel(config)  # This initializes the model with random weights
model2 = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
# model = ViTModel._from_config(ViTModel.config_class.from_pretrained("google/vit-base-patch16-224-in21k"))
