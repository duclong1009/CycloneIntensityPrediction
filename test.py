import torch
import torch.nn as nn
import torchvision
class UpsampleNet(nn.Module):
    def __init__(self):
        super(UpsampleNet, self).__init__()
        
        # First upsampling layer
        self.upsample1 = nn.ConvTranspose2d(
            in_channels=58,
            out_channels=58,
            kernel_size=4,  # Kernel size
            stride=2,       # Stride
            padding=1,      # Padding
            output_padding=0  # Upsample to around 200x200
        )
        
        # Second upsampling layer to fine-tune size to 201x201
        self.upsample2 = nn.ConvTranspose2d(
            in_channels=58,
            out_channels=58,
            kernel_size=3,  # Smaller kernel size for fine-tuning
            stride=2,       # Stride of 1 to add minimal increase
            padding=1,      # Padding
            output_padding=1  # Adjust to achieve exactly 201x201
        )
        self.center_crop = torchvision.transforms.CenterCrop(size=(256,256))
        
    def forward(self, x):
        x = self.upsample1(x)
        
        x = self.upsample2(x)
        x = self.center_crop(x)
        return x

# Create an instance of the UpsampleNet
upsample_net = UpsampleNet()

# Example input tensor with shape (1, 58, 100, 100)
input_tensor = torch.rand(1, 58, 100, 100)  # Add batch dimension

# Forward pass through the two-layer upsampling network
output_tensor = upsample_net(input_tensor)

# Check the shape of the output tensor
print("Output shape:", output_tensor.shape)