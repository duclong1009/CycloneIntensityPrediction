import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torch.nn.functional as F
class SimpleCNN(nn.Module):
    def __init__(self, input_channels, output_channels,args):
        super(SimpleCNN, self).__init__()
        
        self.CNN_extract = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            
        )
        
        self.linear1 = nn.Sequential(nn.Linear(40000,128),
            nn.ReLU(),
            nn.Linear(128, 1))
    
    def forward(self,x, loc=None):
        # breakpoint()
        cnn_embedding = self.CNN_extract(x)
        # breakpoint()
        # cnn_embedding = torch.flatten(cnn_embedding, start_dim=1, end_dim=-1)
        output = self.linear1(cnn_embedding)
        
        return output
    
    
# class CNN.
class FeatureExtractorModel(nn.Module):
    def __init__(self, num_input_channels=58, output_dim=1, backbone_name='resnet18'):
        super(FeatureExtractorModel, self).__init__()
        # Load pre-trained ResNet18 model
        if backbone_name == "resnet18":
            self.backbone = models.resnet18(pretrained=True)
        
            # Modify the first convolutional layer to accept the desired number of input channels
            self.backbone.conv1 = nn.Conv2d(num_input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            num_ftrs = 512
            self.backbone.fc = nn.Linear(num_ftrs, output_dim)
        elif backbone_name == "vgg16":
            self.backbone = models.vgg16(pretrained=True)
            # Modify the first convolutional layer to accept the desired number of input channels
            self.backbone.features[0] = nn.Conv2d(num_input_channels, 64, kernel_size=3, stride=1, padding=1)
            num_ftrs = 4096
            
            self.backbone.classifier[6] = nn.Linear(num_ftrs, output_dim)
        else:
            raise("Not correct backbone")
        
        # Freeze the parameters of the backbone
        # for param in self.backbone.parameters():
        #     param.requires_grad = False
        
        # Replace the final fully connected layer with a regression layer
        # num_ftrs = self.backbone.fc.in_features
        # self.backbone.fc = nn.Linear(num_ftrs, output_dim)

    def forward(self, x):
        return self.backbone(x)


import torch
import torch.nn as nn
import torchvision.models as models

# Channel Attention Module
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

# Feature Extractor Model with Attention
class Attention_CNN(nn.Module):
    def __init__(self, num_input_channels=58, output_dim=1, backbone_name='resnet18'):
        super(Attention_CNN, self).__init__()
        self.backbone_name = backbone_name
        
        if backbone_name == "resnet18":
            self.backbone = models.resnet18(pretrained=True)
            self.backbone.conv1 = nn.Conv2d(num_input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            num_ftrs = 512
            self.backbone.fc = nn.Linear(num_ftrs, output_dim)
        elif backbone_name == "vgg16":
            self.backbone = models.vgg16(pretrained=True)
            self.backbone.features[0] = nn.Conv2d(num_input_channels, 64, kernel_size=3, stride=1, padding=1)
            num_ftrs = 512
            self.backbone.classifier[6] = nn.Linear(num_ftrs, output_dim)
        else:
            raise ValueError("Not correct backbone")
        
        self.channel_attention = ChannelAttention(num_input_channels)

    def forward(self, x):
        x = self.channel_attention(x) * x
        if self.backbone_name == "resnet18":
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)
            x = self.backbone.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.backbone.fc(x)
        elif self.backbone_name == "vgg16":
            x = self.backbone.features(x)
            x = self.backbone.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.backbone.classifier(x)
        return x

    def forward_with_attention(self, x):
        
        
        # Pass through the backbone model
        return self.forward(x)



import torch
import torch.nn as nn
import torchvision.models as models

class ChannelSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelSelfAttention, self).__init__()
        self.query_fc = nn.Linear(3721, 3721)
        self.key_fc = nn.Linear(3721, 3721)
        self.value_fc = nn.Linear(3721, 3721)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # Flatten the spatial dimensions
        x_flat = x.view(batch_size, channels, -1)  # B x C x (H*W)
        
        # Generate query, key, and value tensors
        query = self.query_fc(x_flat)  # B x C x (H*W) 
        key = self.key_fc(x_flat)      # B x C x (H*W) 
        value = self.value_fc(x_flat)  # B x C x (H*W) 

        # Compute attention scores
        attention = torch.bmm(query, key.permute(0, 2, 1))  # B x C X C
        attention = F.softmax(attention, dim=-1)  # Apply softmax to get attention weights
        
        # Apply attention to the value tensor
        out = torch.bmm(attention, value)  # B x C X (H*W)
        out = out.view(batch_size, channels, height, width)  # Reshape back to original dimensions
        
        # Apply the learned scaling parameter
        out = self.gamma * out + x
        
        return out

class Channel_SelfAttentionCNN(nn.Module):
    def __init__(self, num_input_channels=58, output_dim=1, backbone_name='resnet18'):
        super(Channel_SelfAttentionCNN, self).__init__()
        self.backbone_name = backbone_name
        # Load pre-trained ResNet18 model
        if backbone_name == "resnet18":
            self.backbone = models.resnet18(pretrained=True)
            # Modify the first convolutional layer to accept the desired number of input channels
            self.backbone.conv1 = nn.Conv2d(num_input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            num_ftrs = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(512,1)  # Remove the final fully connected layer
        elif backbone_name == "vgg16":
            self.backbone = models.vgg16(pretrained=True)
            # Modify the first convolutional layer to accept the desired number of input channels
            self.backbone.features[0] = nn.Conv2d(num_input_channels, 64, kernel_size=3, stride=1, padding=1)
            num_ftrs = self.backbone.classifier[6].in_features
            self.backbone.classifier[6] = nn.Linear(512,1)  # Remove the final fully connected layer
        else:
            raise("Not correct backbone")
        
        # Freeze the parameters of the backbone if needed
        # for param in self.backbone.parameters():
        #     param.requires_grad = False

        # Channel self-attention
        self.channel_attention = ChannelSelfAttention(in_channels=58)
        
        # New fully connected layer for regression
        self.fc = nn.Linear(num_ftrs + 64, output_dim)  # Concatenate original features with attention features

    def forward(self, x):
        attention_out = self.channel_attention(x)
        if self.backbone_name == "resnet18":
            x = self.backbone.conv1(attention_out)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)
            x = self.backbone.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.backbone.fc(x)
        elif self.backbone_name == "vgg16":
            x = self.backbone.features(attention_out)
            x = self.backbone.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.backbone.classifier(x)
        return x
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class GridSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(GridSelfAttention, self).__init__()
        self.query_fc = nn.Linear(in_channels, in_channels)
        self.key_fc = nn.Linear(in_channels, in_channels)
        self.value_fc = nn.Linear(in_channels, in_channels)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # Flatten the spatial dimensions
        x_flat = x.view(batch_size, channels, -1)  # B x C x (H*W)
        
        # Generate query, key, and value tensors
        query = self.query_fc(x_flat.permute(0, 2, 1))  # B x (H*W) x C
        key = self.key_fc(x_flat.permute(0, 2, 1))      # B x (H*W) x C
        value = self.value_fc(x_flat.permute(0, 2, 1))  # B x (H*W) x C
        
        # key = key.permute(0, 2, 1)      # B x C x (H*W)
        # value = value.permute(0, 2, 1)  # B x C x (H*W)
        
        # Compute attention scores
        attention = torch.bmm(query, key.permute(0, 2, 1))  # B x (H*W) x (H*W)
        attention = F.softmax(attention, dim=-1)  # Apply softmax to get attention weights
        # Apply attention to the value tensor
        out = torch.bmm(attention, value).permute(0, 2, 1)  # B x C x (H*W)
        out = out.view(batch_size, channels, height, width)  # Reshape back to original dimensions
        
        # Apply the learned scaling parameter
        out = self.gamma * out + x
        
        return out
class Grid_SelfAttentionCNN(nn.Module):
    def __init__(self, num_input_channels=58, output_dim=1, backbone_name='resnet18'):
        super(Grid_SelfAttentionCNN, self).__init__()
        self.backbone_name = backbone_name
        # Load pre-trained ResNet18 model
        if backbone_name == "resnet18":
            self.backbone = models.resnet18(pretrained=True)
            # Modify the first convolutional layer to accept the desired number of input channels
            self.backbone.conv1 = nn.Conv2d(num_input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            num_ftrs = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(512,1)  # Remove the final fully connected layer
        elif backbone_name == "vgg16":
            self.backbone = models.vgg16(pretrained=True)
            # Modify the first convolutional layer to accept the desired number of input channels
            self.backbone.features[0] = nn.Conv2d(num_input_channels, 64, kernel_size=3, stride=1, padding=1)
            num_ftrs = self.backbone.classifier[6].in_features
            self.backbone.classifier[6] = nn.Linear(512,1)  # Remove the final fully connected layer
        else:
            raise("Not correct backbone")
        
        # Freeze the parameters of the backbone if needed
        # for param in self.backbone.parameters():
        #     param.requires_grad = False

        # Channel self-attention
        self.channel_attention = GridSelfAttention(in_channels=58)
        
        # New fully connected layer for regression
        self.fc = nn.Linear(num_ftrs + 64, output_dim)  # Concatenate original features with attention features
        self.relu = nn.ReLU()
    def forward(self, x):
        attention_out = self.channel_attention(x)
        if self.backbone_name == "resnet18":
            x = self.backbone.conv1(attention_out)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)
            x = self.backbone.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.backbone.fc(x)
        elif self.backbone_name == "vgg16":
            x = self.backbone.features(attention_out)
            x = self.backbone.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.backbone.classifier(x)
        return self.relu(x)

# Define the neural network class
class SimpleTwoLayerNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleTwoLayerNN, self).__init__()
        # Define the first linear layer
        self.fc1 = nn.Linear(input_size, 64)
        # Define the second linear layer
        self.fc2 = nn.Linear(64, 32)
        # Define the output layer (assuming a regression task)
        self.output = nn.Linear(32, 1)

    def forward(self, x):
        # Forward pass through the first layer with ReLU activation
        x = torch.relu(self.fc1(x))
        # Forward pass through the second layer with ReLU activation
        x = torch.relu(self.fc2(x))
        # Forward pass through the output layer
        x = self.output(x)
        return x
    
class DeepRegressionNN(nn.Module):
    def __init__(self, input_size):
        super(DeepRegressionNN, self).__init__()
        # Define the first linear layer
        self.fc1 = nn.Linear(input_size, 128)
        # Define the second linear layer
        self.fc2 = nn.Linear(128, 64)
        # Define the third linear layer
        self.fc3 = nn.Linear(64, 32)
        # Define the fourth linear layer
        self.fc4 = nn.Linear(32, 16)
        # Define the output layer
        self.output = nn.Linear(16, 1)
    
    def forward(self, x):
        # Forward pass through the first layer with ReLU activation
        x = torch.relu(self.fc1(x))
        # Forward pass through the second layer with ReLU activation
        x = torch.relu(self.fc2(x))
        # Forward pass through the third layer with ReLU activation
        x = torch.relu(self.fc3(x))
        # Forward pass through the fourth layer with ReLU activation
        x = torch.relu(self.fc4(x))
        # Forward pass through the output layer
        x = self.output(x)
        return x

        
if __name__ == "__main__":
    
    x = torch.randn(32, 58, 64, 64)  # Example input tensor with shape (batch_size, num_input_channels, height, width)
    model = FeatureExtractorModel(num_input_channels=58, output_dim=1, backbone_name='resnet18')
    output = model(x)
    print(output.shape)  # Should print torch.Size([32, 1])
    