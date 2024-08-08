import torch
import torch.nn as nn


class CNNEmbedder(nn.Module):
    def __init__(self, input_channels, output_dim, sequence_length):
        super(CNNEmbedder, self).__init__()
        self.conv = nn.Conv2d(in_channels=input_channels, 
                              out_channels=output_dim, 
                              kernel_size=(3, 3), 
                              stride=(3, 3))
        self.sequence_length = sequence_length
        self.output_dim = output_dim

    def forward(self, x):
        # Apply convolution
        
        x = self.conv(x)
        
        # Flatten the non-channel dimensions
        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, channels, -1)
        
        # Transpose to get the shape (batch_size, sequence_length, embedding_dim)
        x = x.permute(0, 2, 1)
        
        # Pad/truncate to the desired sequence length if necessary
        # if x.size(1) > self.sequence_length:
        #     x = x[:, :self.sequence_length, :]
        # elif x.size(1) < self.sequence_length:
        #     padding = torch.zeros(batch_size, self.sequence_length - x.size(1), self.output_dim, device=x.device)
        #     x = torch.cat((x, padding), dim=1)

        return x


# # Example usage
# input_tensor = torch.randn(1, 58, 100, 100)
# embedder = CNNEmbedder(input_channels=58, output_dim=768, sequence_length=50)
# output_tensor = embedder(input_tensor)

# print(output_tensor.shape)  # Expected output: torch.Size([1, 50, 768])