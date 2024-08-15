import torch
import torch.nn as nn
import torchvision
from metnet import MetNet, MetNet2
import copy

class UpsampleNet(nn.Module):
    def __init__(self):
        super(UpsampleNet, self).__init__()
        
        # First upsampling layer
        self.upsample1 = nn.ConvTranspose2d(
            in_channels=58,
            out_channels=100,
            kernel_size=4,  # Kernel size
            stride=2,       # Stride
            padding=1,      # Padding
            output_padding=0  # Upsample to around 200x200
        )
        
        # Second upsampling layer to fine-tune size to 201x201
        self.upsample2 = nn.ConvTranspose2d(
            in_channels=100,
            out_channels=144,
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

class MetNet_Tuning_Model(nn.Module):
    def __init__(self,):
        super(MetNet_Tuning_Model,self).__init__()
        
        self.upsample_module = UpsampleNet()
        
        metnet_model = MetNet().from_pretrained("openclimatefix/metnet")

        self.metnet_encoder = copy.deepcopy(metnet_model.image_encoder)
        
        self.prediction_head = nn.Sequential(nn.Linear(1048576, 512),
                                             nn.GELU(),
                                             nn.Linear(512, 128),
                                             nn.GELU(),
                                             nn.Linear(128, 1))
        
    def forward(self, x):
        x = self.upsample_module(x)
        x = x.unsqueeze(1)
        encoder_output = self.metnet_encoder(x)
        encoder_output = encoder_output.squeeze(1)
        encoder_output = encoder_output.flatten(1)
        x = self.prediction_head(encoder_output)
        return x

# model = MetNet_Tuning_Model()

# arr = torch.rand((32,58,100,100))
# out = model(arr)
