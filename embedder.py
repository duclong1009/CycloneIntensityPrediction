import torch
import torch.nn as nn

class CNNEmbedder(nn.Module):
    def __init__(self, input_channels, output_dim, kernel_size=3):
        super(CNNEmbedder, self).__init__()
        self.conv = nn.Conv2d(in_channels=input_channels, 
                              out_channels=output_dim, 
                              kernel_size=(kernel_size, kernel_size), 
                              stride=(kernel_size, kernel_size))
        # self.sequence_length = sequence_length
        self.output_dim = output_dim

    def forward(self, x):
        # Apply convolution
        
        x = self.conv(x)
        
        # Flatten the non-channel dimensions
        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, channels, -1)
        
        # Transpose to get the shape (batch_size, sequence_length, embedding_dim)
        x = x.permute(0, 2, 1)
    
        return x

# Example usage
kernel_size = 10
input_tensor = torch.randn(1, 58, 100, 100)
embedder = CNNEmbedder(input_channels=58, output_dim=768, kernel_size=kernel_size)
output_tensor = embedder(input_tensor)

print(output_tensor.shape)  # Expected output: torch.Size([1, 50, 768])