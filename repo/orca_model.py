import torch
import torch.nn as nn
from transformers import AutoImageProcessor, ViTModel
import copy
from transformers import ViTModel, ViTConfig
import numpy as np

class CNNEmbedder(nn.Module):
    def __init__(self, input_channels, output_dim, kernel_size=3):
        super(CNNEmbedder, self).__init__()
        self.conv = nn.Conv2d(in_channels=input_channels, 
                              out_channels=output_dim, 
                              kernel_size=(kernel_size, kernel_size), 
                              stride=(kernel_size, kernel_size))
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

class CrossTuningModel(nn.Module):
    def __init__(self,cnn_embed, body_model_name="vit", prediction_head=None):
        super(CrossTuningModel, self).__init__()
        if body_model_name == 'vit':
            model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
            
            self.body_model =  copy.deepcopy(model.encoder)
        else:
            raise ValueError("Not correct body model name")
        self.layernorm = nn.LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        
        self.cnn_embed = cnn_embed
        self.prediction_head = prediction_head
    
    def forward(self,x):
        embedding_x = self.cnn_embed(x)
        body_output=  self.body_model(embedding_x)
        body_output = body_output.last_hidden_state
        body_output = self.layernorm(body_output)
        return self.prediction_head(body_output)

class PredictionHead(nn.Module):
    def __init__(self,dim=768, n_patchs=100):
        super(PredictionHead, self).__init__()
        
        self.linear_head1 = nn.Linear(dim * n_patchs, 512)
        self.linear_head2 = nn.Linear(512, 128)
        self.linear_head3 = nn.Linear(128, 1)
        self.gelu = nn.GELU()
    def forward(self,x):
        ### adding promt token at the end of body model
        x = x.reshape(x.shape[0], -1)
        x = self.gelu(self.linear_head1(x))
        x = self.gelu(self.linear_head2(x))
        return self.linear_head3(x)



class Prompt_Tuning_Model0(nn.Module):
    def __init__(self,cnn_embed, body_model_name="vit", prediction_head=None, args=None ):
        super(Prompt_Tuning_Model0,self).__init__()
        prompt_dim = args.prompt_dims

        # self.use_position_embedding = args.use_position_embedding

        if body_model_name == 'vit':
            model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
            self.body_model =  copy.deepcopy(model.encoder)
        elif body_model_name == 'scratch_vit':
            config = ViTConfig()  # Use default configuration or modify as needed   
            model = ViTModel(config)
            self.body_model =  copy.deepcopy(model.encoder)

        else:
            raise ValueError("Not correct body model name")

        self.layernorm = nn.LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        
        
        self.cnn_embed = cnn_embed
        self.prediction_head = prediction_head
        self.prompt_token = nn.Parameter(torch.randn(1, prompt_dim)) 
        self.use_position_embedding = args.use_position_embedding
        if self.use_position_embedding:
            emb_size = 768
            # self.cls_token = nn.Parameter(torch.randn(1,100, emb_size))
            self.positions = nn.Parameter(torch.randn(100, emb_size))

    def forward(self,x):
        ### adding promt token at the end of body model

        embedding_x = self.cnn_embed(x)
        if self.use_position_embedding:
            embedding_x += self.positions
            
        body_output=  self.body_model(embedding_x)
        body_output = body_output.last_hidden_state
        
        ### output shape [batch, n_patchs, 768 + 128 ]
        body_output = self.layernorm(body_output)
        return self.prediction_head(body_output)
    
    
class Prompt_Tuning_Model1(nn.Module):
    def __init__(self,cnn_embed, body_model_name="vit", prediction_head=None, args=None ):
        super(Prompt_Tuning_Model1,self).__init__()
        prompt_dim = args.prompt_dims

        # self.use_position_embedding = args.use_position_embedding

        if body_model_name == 'vit':
            model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
            self.body_model =  copy.deepcopy(model.encoder)
        elif body_model_name == 'scratch_vit':
            config = ViTConfig()  # Use default configuration or modify as needed   
            model = ViTModel(config)
            self.body_model =  copy.deepcopy(model.encoder)

        else:
            raise ValueError("Not correct body model name")

        if args.freeze:
            for param in self.body_model.parameters():
                param.requires_grad = False
                
        self.layernorm = nn.LayerNorm((768+ prompt_dim,), eps=1e-12, elementwise_affine=True)
        
        
        self.cnn_embed = cnn_embed
        self.prediction_head = prediction_head
        self.prompt_token = nn.Parameter(torch.randn(1, prompt_dim)) 
        self.use_position_embedding = args.use_position_embedding
        if self.use_position_embedding:
            emb_size = 768
            # self.cls_token = nn.Parameter(torch.randn(1,100, emb_size))
            self.positions = nn.Parameter(torch.randn(100, emb_size))

    def forward(self,x):
        ### adding promt token at the end of body model
        batch_size = x.shape[0]
        prompt_token_expanded = self.prompt_token.expand(batch_size, -1)  # Expand prompt token to batch 
        
        embedding_x = self.cnn_embed(x)
        if self.use_position_embedding:
            embedding_x += self.positions
            
        body_output=  self.body_model(embedding_x)
        body_output = body_output.last_hidden_state
        
        body_output = torch.cat([prompt_token_expanded.unsqueeze(1).repeat(1, body_output.size(1), 1), body_output], dim=-1)
        ### output shape [batch, n_patchs, 768 + 128 ]
        body_output = self.layernorm(body_output)
        return self.prediction_head(body_output)
        
class Prompt_Tuning_Model1_Embeder(nn.Module):
    def __init__(self,cnn_embed, body_model_name="vit", prompt_dim=128):
        super(Prompt_Tuning_Model1_Embeder,self).__init__()
        
        if body_model_name == 'vit':
            model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
            
            self.body_model =  copy.deepcopy(model.encoder)
        else:
            raise ValueError("Not correct body model name")
        self.layernorm = nn.LayerNorm((768+ prompt_dim,), eps=1e-12, elementwise_affine=True)
        
        self.cnn_embed = cnn_embed
        self.prompt_token = nn.Parameter(torch.randn(1, prompt_dim)) 
        
    def forward(self,x):
        ### adding promt token at the end of body model
        batch_size = x.shape[0]
        prompt_token_expanded = self.prompt_token.expand(batch_size, -1)  # Expand prompt token to batch 
        
        embedding_x = self.cnn_embed(x)
        body_output=  self.body_model(embedding_x)
        body_output = body_output.last_hidden_state
        
        body_output = torch.cat([prompt_token_expanded.unsqueeze(1).repeat(1, body_output.size(1), 1), body_output], dim=-1)
        ### output shape [batch, n_patchs, 768 + 128 ]
        body_output = self.layernorm(body_output)
        
        return body_output
        
class Prompt_Tuning_Model2(nn.Module):
    def __init__(self,cnn_embed, body_model_name="vit", prediction_head=None, args=None):
        super(Prompt_Tuning_Model2,self).__init__()
        
        prompt_dim = args.prompt_dims

        if body_model_name == 'vit':
            model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
            self.body_model =  copy.deepcopy(model.encoder)

        elif body_model_name == 'scratch_vit':
            config = ViTConfig()  # Use default configuration or modify as needed   
            model = ViTModel(config)
            self.body_model =  copy.deepcopy(model.encoder)

        else:
            raise ValueError("Not correct body model name")

        if args.freeze:
            for param in self.body_model.parameters():
                param.requires_grad = False
                
        self.layernorm = nn.LayerNorm((768 + prompt_dim,), eps=1e-12, elementwise_affine=True)
        

        self.cnn_embed = cnn_embed
        self.prediction_head = prediction_head
        self.prompt_token = nn.Parameter(torch.randn(1,100, prompt_dim)) 
        

        self.use_position_embedding = args.use_position_embedding

        if self.use_position_embedding:
            emb_size = 768
            # self.cls_token = nn.Parameter(torch.randn(1,100, emb_size))
            self.positions = nn.Parameter(torch.randn(100, emb_size))


    def forward(self,x):
        ### adding promt token at the end of body model
        # 
        batch_size = x.shape[0]
        # 
        # prompt_token_expanded = self.prompt_token.expand(batch_size, -1)  # Expand prompt token to batch 
        prompt_token_expanded = self.prompt_token.repeat(batch_size, 1,1)
        embedding_x = self.cnn_embed(x)
        if self.use_position_embedding:
            embedding_x += self.positions

        body_output=  self.body_model(embedding_x)
        body_output = body_output.last_hidden_state
        
        body_output = torch.cat([prompt_token_expanded, body_output], dim=-1)
        
        body_output = self.layernorm(body_output)
        return self.prediction_head(body_output)
    
class Prompt_Tuning_Model2_Embeder(nn.Module):
    def __init__(self,cnn_embed, body_model_name="vit",prompt_dim=128):
        super(Prompt_Tuning_Model2_Embeder,self).__init__()
        
        if body_model_name == 'vit':
            model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
            
            self.body_model =  copy.deepcopy(model.encoder)
        else:
            raise ValueError("Not correct body model name")
        self.layernorm = nn.LayerNorm((768 + prompt_dim,), eps=1e-12, elementwise_affine=True)
        
        self.cnn_embed = cnn_embed
        self.prompt_token = nn.Parameter(torch.randn(1,100, prompt_dim)) 
        
    def forward(self,x):
        ### adding promt token at the end of body model
        batch_size = x.shape[0]
        # 
        # prompt_token_expanded = self.prompt_token.expand(batch_size, -1)  # Expand prompt token to batch 
        prompt_token_expanded = self.prompt_token.repeat(batch_size, 1,1)
        embedding_x = self.cnn_embed(x)
        body_output=  self.body_model(embedding_x)
        body_output = body_output.last_hidden_state
        
        body_output = torch.cat([prompt_token_expanded, body_output], dim=-1)
        
        body_output = self.layernorm(body_output)
        return body_output
    

class Prompt_Tuning_Model3(nn.Module):
    def __init__(self,cnn_embed, body_model_name="vit", prediction_head=None, args=None):
        super(Prompt_Tuning_Model3,self).__init__()
        
        prompt_dim = args.prompt_dims
        if body_model_name == 'vit':
            model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
            self.body_model =  copy.deepcopy(model.encoder)

        elif body_model_name == 'scratch_vit':
            config = ViTConfig()  # Use default configuration or modify as needed   
            model = ViTModel(config)
            self.body_model =  copy.deepcopy(model.encoder)

        else:
            raise ValueError("Not correct body model name")

        if args.freeze:
            for param in self.body_model.parameters():
                param.requires_grad = False
                
        self.layernorm = nn.LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        
        self.cnn_embed = cnn_embed
        self.prediction_head = prediction_head
        self.prompt_token = nn.Parameter(torch.randn(1, prompt_dim)) 
        
        self.use_position_embedding = args.use_position_embedding
        if self.use_position_embedding:
            emb_size = 768
            # self.cls_token = nn.Parameter(torch.randn(1,100, emb_size))
            self.positions = nn.Parameter(torch.randn(100, emb_size))

    def forward(self,x):
        ### adding promt token at the begin of body model
        # 
        batch_size = x.shape[0]
        prompt_token_expanded = self.prompt_token.expand(batch_size, -1)  # Expand prompt token to batch 
        
        embedding_x = self.cnn_embed(x) # 100 640
        # 
        ### add promt token
        embedding_x = torch.cat([embedding_x, prompt_token_expanded.unsqueeze(1).repeat(1,embedding_x.shape[1],1)], dim=-1)
        
        if self.use_position_embedding:
            embedding_x += self.positions
            
        body_output=  self.body_model(embedding_x)
        body_output = body_output.last_hidden_state
        
        # body_output = torch.cat([prompt_token_expanded.unsqueeze(1).repeat(1, body_output.size(1), 1), body_output], dim=-1)
        
        body_output = self.layernorm(body_output)
        # body_output = 
        return self.prediction_head(body_output)
    
class Prompt_Tuning_Model3_Embeder(nn.Module):
    def __init__(self,cnn_embed, body_model_name="vit", prompt_dim=128):
        super(Prompt_Tuning_Model3_Embeder,self).__init__()
        
        if body_model_name == 'vit':
            model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
            
            self.body_model =  copy.deepcopy(model.encoder)
        else:
            raise ValueError("Not correct body model name")
        self.layernorm = nn.LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        
        self.cnn_embed = cnn_embed
        self.prompt_token = nn.Parameter(torch.randn(1, prompt_dim)) 
        
    def forward(self,x):
        ### adding promt token at the begin of body model
        batch_size = x.shape[0]
        prompt_token_expanded = self.prompt_token.expand(batch_size, -1)  # Expand prompt token to batch 
        
        embedding_x = self.cnn_embed(x)
        # 
        ### add promt token
        embedding_x = torch.cat([embedding_x, prompt_token_expanded.unsqueeze(1).repeat(1,embedding_x.shape[1],1)], dim=-1)

        
        body_output=  self.body_model(embedding_x)
        body_output = body_output.last_hidden_state
        
        # body_output = torch.cat([prompt_token_expanded.unsqueeze(1).repeat(1, body_output.size(1), 1), body_output], dim=-1)
        
        body_output = self.layernorm(body_output)
        # body_output = 
        return body_output


class Prompt_Tuning_Model4(nn.Module):
    def __init__(self,cnn_embed, body_model_name="vit", prediction_head=None, prompt_dim=128):
        super(Prompt_Tuning_Model4,self).__init__()
        
        if body_model_name == 'vit':
            model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
            
            self.body_model =  copy.deepcopy(model.encoder)
        else:
            raise ValueError("Not correct body model name")

        if args.freeze:
            for param in self.body_model.parameters():
                param.requires_grad = False
                
        self.layernorm = nn.LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        
        self.cnn_embed = cnn_embed
        self.prediction_head = prediction_head
        self.prompt_token = nn.Parameter(torch.randn(1,100, prompt_dim)) 
        
    def forward(self,x):
        ### adding promt token at the begin of body model
        batch_size = x.shape[0]
        # prompt_token_expanded = self.prompt_token.expand(batch_size, -1)  # Expand prompt token to batch 
        prompt_token_expanded = self.prompt_token.repeat(batch_size, 1,1)
        
        embedding_x = self.cnn_embed(x)
        
        ### add promt token
        embedding_x = torch.cat([embedding_x, prompt_token_expanded], dim=-1)
        
        
        body_output=  self.body_model(embedding_x)
        body_output = body_output.last_hidden_state
        
        # body_output = torch.cat([prompt_token_expanded.unsqueeze(1).repeat(1, body_output.size(1), 1), body_output], dim=-1)
        
        body_output = self.layernorm(body_output)
        # body_output = 
        return self.prediction_head(body_output)
    
    
class Prompt_Tuning_Model5(nn.Module):
    def __init__(self,cnn_embed, body_model_name="vit", prediction_head=None, prompt_dim=128):
        super(Prompt_Tuning_Model5,self).__init__()
        
        if body_model_name == 'vit':
            model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
            
            self.body_model =  copy.deepcopy(model.encoder)
        else:
            raise ValueError("Not correct body model name")

        if args.freeze:
            for param in self.body_model.parameters():
                param.requires_grad = False
                
        self.layernorm = nn.LayerNorm((768 + prompt_dim,), eps=1e-12, elementwise_affine=True)
        
        self.cnn_embed = cnn_embed
        self.prediction_head = prediction_head
        self.prompt_token1 = nn.Parameter(torch.randn(1, prompt_dim)) 
        self.prompt_token2 = nn.Parameter(torch.randn(1, prompt_dim)) 
        
    def forward(self,x):
        ### adding promt token at the begin of body model
        batch_size = x.shape[0]
        prompt_token_expanded = self.prompt_token1.expand(batch_size, -1)  # Expand prompt token to batch 
        
        embedding_x = self.cnn_embed(x)
        # 
        ### add promt token
        embedding_x = torch.cat([embedding_x, prompt_token_expanded.unsqueeze(1).repeat(1,embedding_x.shape[1],1)], dim=-1)
        
        
        body_output=  self.body_model(embedding_x)
        body_output = body_output.last_hidden_state
        
        # body_output = torch.cat([prompt_token_expanded.unsqueeze(1).repeat(1, body_output.size(1), 1), body_output], dim=-1)
        prompt_token_expanded2 = self.prompt_token2.expand(batch_size, -1)  # Expand prompt token to batch 
        body_output = torch.cat([body_output, prompt_token_expanded2.unsqueeze(1).repeat(1,embedding_x.shape[1],1)], dim=-1)
        
        body_output = self.layernorm(body_output)
        # body_output = 
        return self.prediction_head(body_output)


class Individual_Embeder_Tuning_Model1(nn.Module):
    def __init__(self, input_channels=58, body_model_name="vit", prediction_head=None, prompt_dim=128):
        super(Individual_Embeder_Tuning_Model1,self).__init__()
        
        self.list_embeder = nn.ModuleList()
        self.input_channels = input_channels
        for i in range(self.input_channels):
            self.list_embeder.append(CNNEmbedder(input_channels=1, output_dim=14, kernel_size=10))
        self.project_layer = nn.Linear(14 * self.input_channels, 768)
        
        if body_model_name == 'vit':
            model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
            
            self.body_model =  copy.deepcopy(model.encoder)
        else:
            raise ValueError("Not correct body model name")
        self.layernorm = nn.LayerNorm((768 + prompt_dim,), eps=1e-12, elementwise_affine=True)
        
        self.prediction_head = prediction_head
        self.prompt_token = nn.Parameter(torch.randn(1, prompt_dim)) 
        
    def cnn_embed(self,x):
        list_output = []
        for i in range(self.input_channels):
            x_i = x[:, i,:,:].unsqueeze(1)
            y_i = self.list_embeder[i](x_i)
            list_output.append(y_i)
        output_vec = torch.concat(list_output,-1)
        return self.project_layer(output_vec)
    
        # 
        
    def forward(self,x):
        ### adding promt token at the end of body model
        batch_size = x.shape[0]
        prompt_token_expanded = self.prompt_token.expand(batch_size, -1)  # Expand prompt token to batch 

        embedding_x = self.cnn_embed(x)
        
        body_output=  self.body_model(embedding_x)
        body_output = body_output.last_hidden_state
        
        body_output = torch.cat([prompt_token_expanded.unsqueeze(1).repeat(1, body_output.size(1), 1), body_output], dim=-1)
        ### output shape [batch, n_patchs, 768 + 128 ]
        body_output = self.layernorm(body_output)
        
        return self.prediction_head(body_output)
    
    
class Individual_Embeder_Tuning_Model2(nn.Module):
    def __init__(self,input_channels=58, body_model_name="vit", prediction_head=None, prompt_dim=128):
        super(Individual_Embeder_Tuning_Model2,self).__init__()
        
        self.list_embeder = nn.ModuleList()
        self.input_channels = input_channels
        for i in range(self.input_channels):
            self.list_embeder.append(CNNEmbedder(input_channels=1, output_dim=14, kernel_size=10))
        self.project_layer = nn.Linear(14 * self.input_channels, 768)
        
        if body_model_name == 'vit':
            model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
            
            self.body_model =  copy.deepcopy(model.encoder)
        else:
            raise ValueError("Not correct body model name")
        self.layernorm = nn.LayerNorm((768 + prompt_dim,), eps=1e-12, elementwise_affine=True)
        

        self.prediction_head = prediction_head
        self.prompt_token = nn.Parameter(torch.randn(1,100, prompt_dim)) 
    
    def cnn_embed(self,x):
        list_output = []
        for i in range(self.input_channels):
            x_i = x[:, i,:,:].unsqueeze(1)
            y_i = self.list_embeder[i](x_i)
            list_output.append(y_i)
        output_vec = torch.concat(list_output,-1)
        return self.project_layer(output_vec)
    
       
    def forward(self,x):
        ### adding promt token at the end of body model
        batch_size = x.shape[0]
        # 
        # prompt_token_expanded = self.prompt_token.expand(batch_size, -1)  # Expand prompt token to batch 
        prompt_token_expanded = self.prompt_token.repeat(batch_size, 1,1)
        embedding_x = self.cnn_embed(x)
        body_output=  self.body_model(embedding_x)
        body_output = body_output.last_hidden_state
        
        body_output = torch.cat([prompt_token_expanded, body_output], dim=-1)
        
        body_output = self.layernorm(body_output)
        return self.prediction_head(body_output)
    
    
class Individual_Embeder_Tuning_Model3(nn.Module):
    def __init__(self,input_channels=58, body_model_name="vit", prediction_head=None,args=None):
        super(Individual_Embeder_Tuning_Model3,self).__init__()
        self.prediction_head = prediction_head
        self.list_embeder = nn.ModuleList()
        self.input_channels = input_channels
        for i in range(self.input_channels):
            self.list_embeder.append(CNNEmbedder(input_channels=1, output_dim=14, kernel_size=10))
        self.project_layer = nn.Linear(14 * self.input_channels, 768 - args.prompt_dims)
        
        if body_model_name == 'vit':
            model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
            
            self.body_model =  copy.deepcopy(model.encoder)
        else:
            raise ValueError("Not correct body model name")
        self.layernorm = nn.LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        
        self.prompt_token = nn.Parameter(torch.randn(1, args.prompt_dims)) 
    
    def cnn_embed(self,x):
        list_output = []
        for i in range(self.input_channels):
            x_i = x[:, i,:,:].unsqueeze(1)
            y_i = self.list_embeder[i](x_i)
            list_output.append(y_i)
        output_vec = torch.concat(list_output,-1)
        return self.project_layer(output_vec)
       
    def forward(self,x):
        ### adding promt token at the begin of body model
        batch_size = x.shape[0]
        prompt_token_expanded = self.prompt_token.expand(batch_size, -1)  # Expand prompt token to batch 
        embedding_x = self.cnn_embed(x)
        # 
        ### add promt token
        embedding_x = torch.cat([embedding_x, prompt_token_expanded.unsqueeze(1).repeat(1,embedding_x.shape[1],1)], dim=-1)

        
        body_output=  self.body_model(embedding_x)
        body_output = body_output.last_hidden_state
        
        # body_output = torch.cat([prompt_token_expanded.unsqueeze(1).repeat(1, body_output.size(1), 1), body_output], dim=-1)
        
        body_output = self.layernorm(body_output)
        # body_output = 
        return self.prediction_head(body_output)


class Region_Attention(nn.Module):
    def __init__(self, input_channels=58, body_model_name="vit", prediction_head=None, args=None):
        super(Region_Attention,self).__init__()
        
        prompt_dim = args.prompt_dims
        self.combining_layer_type = args.combining_layer_type

        self.list_embeder = nn.ModuleList()
        self.input_channels = input_channels
        output_dim = 14
        self.use_cls_for_region = args.use_cls_for_region
        for i in range(self.input_channels):
            self.list_embeder.append(CNNEmbedder(input_channels=1, output_dim=output_dim, kernel_size=10))
        if self.use_cls_for_region:
            self.cls_token = nn.Parameter(torch.randn(self.input_channels, output_dim))
            
        if self.use_cls_for_region and self.combining_layer_type ==2: 
            self.project_layer = nn.Linear(output_dim, 768)

        elif self.use_cls_for_region and self.combining_layer_type ==1: 
            self.project_layer = nn.Linear(output_dim * (self.input_channels+1), 768)

        else:
            self.project_layer = nn.Linear(output_dim * self.input_channels, 768)
        # self

        if body_model_name == 'vit':
            model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
            
            self.body_model =  copy.deepcopy(model.encoder)
        else:
            raise ValueError("Not correct body model name")
        self.layernorm = nn.LayerNorm((768 + prompt_dim,), eps=1e-12, elementwise_affine=True)
        
        self.prediction_head = prediction_head
        self.prompt_token = nn.Parameter(torch.randn(1, prompt_dim)) 

        encoder_layer = nn.TransformerEncoderLayer(d_model=output_dim, nhead=2,batch_first=True)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

    def cnn_embed(self,x):
        list_output = []
        for i in range(self.input_channels):
            x_i = x[:, i,:,:].unsqueeze(1)
            y_i = self.list_embeder[i](x_i)
            list_output.append(y_i)
        return list_output
    
        
    def forward(self,x):
        ### adding promt token at the end of body model
        batch_size = x.shape[0]
        prompt_token_expanded = self.prompt_token.expand(batch_size, -1)  # Expand prompt token to batch 
        
        list_output = self.cnn_embed(x)
        
        stacked_embed = torch.stack(list_output,2)
        if self.use_cls_for_region:
            self.cls_token = nn.Parameter(self.cls_token.unsqueeze(0).repeat(batch_size,1,1).unsqueeze(1))
            stacked_embed = torch.concat([stacked_embed,self.cls_token],1)

            
        input_size = stacked_embed.shape

        stacked_embed = stacked_embed.reshape(-1, input_size[2], input_size[3])
        stacked_embed = self.transformer_encoder(stacked_embed)
        stacked_embed = stacked_embed.reshape(input_size)
        if self.use_cls_for_region and self.combining_layer_type == 2:
            stacked_embed = stacked_embed[:, :,0,:]
        else:
            stacked_embed = stacked_embed.reshape(input_size[0], input_size[1], -1)

        embedding_x = self.project_layer(stacked_embed)
        body_output=  self.body_model(embedding_x)
        body_output = body_output.last_hidden_state
        
        body_output = torch.cat([prompt_token_expanded.unsqueeze(1).repeat(1, body_output.size(1), 1), body_output], dim=-1)
        ### output shape [batch, n_patchs, 768 + 128 ]
        body_output = self.layernorm(body_output)
        
        return self.prediction_head(body_output)


# class Prompt_Tuning_Model_Leading_t(nn.Module):
#     def __init__(self,cnn_embed, body_model_name="vit", prediction_head=None, args=None):
#         super(Prompt_Tuning_Model_Leading_t, self).__init__()
#         n_leading_times = 6
#         prompt_dim = args.prompt_dims
#         # prompt_dim = 128
#         if body_model_name == 'vit':
#             model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
#             self.body_model =  copy.deepcopy(model.encoder)

#         elif body_model_name == 'scratch_vit':
#             config = ViTConfig()  # Use default configuration or modify as needed   
#             model = ViTModel(config)
#             self.body_model =  copy.deepcopy(model.encoder)

#         else:
#             raise ValueError("Not correct body model name")
            
#         if args.freeze:
#             for param in self.body_model.parameters():
#                 param.requires_grad = False

#         self.layernorm = nn.LayerNorm((768,), eps=1e-12, elementwise_affine=True)
#         # 
#         self.cnn_embed = cnn_embed
#         self.prediction_head = prediction_head
#         self.prompt_token = nn.Parameter(torch.randn(1, prompt_dim)) 
#         self.leading_tokens = nn.Parameter(torch.randn(n_leading_times, prompt_dim))
#         # self.use_position_embedding = False
#         self.use_position_embedding = args.use_position_embedding
#         if self.use_position_embedding:
#             emb_size = 768
#             self.positions = nn.Parameter(torch.randn(100, emb_size))

    # def forward(self,x):
    #     ### adding promt token at the begin of body model

    #     x, leading_time = x
    #     batch_size = x.shape[0]

    #     leading_tokens_expanded = self.leading_tokens.unsqueeze(0).expand(batch_size, -1,-1)
    #     leading_time = leading_time.int().unsqueeze(-1)
    #     selected_leading_tokens = leading_tokens_expanded[torch.arange(leading_tokens_expanded.shape[0]).unsqueeze(1), leading_time]
        
    #     prompt_token_expanded = self.prompt_token.expand(batch_size, -1)  # Expand prompt token to batch 
        
    #     embedding_x = self.cnn_embed(x) # 100 640

    #     ### add promt token
    #     embedding_x = torch.cat([embedding_x, prompt_token_expanded.unsqueeze(1).repeat(1,embedding_x.shape[1],1)], dim=-1)
        
    #     if self.use_position_embedding:
    #         embedding_x += self.positions
            
    #     body_output=  self.body_model(embedding_x)
    #     body_output = body_output.last_hidden_state
    #     # body_output = torch.cat([prompt_token_expanded.unsqueeze(1).repeat(1, body_output.size(1), 1), body_output], dim=-1)
        
    #     body_output = self.layernorm(body_output)
    #     # body_output = 
    #     return self.prediction_head(body_output, selected_leading_tokens)

class PredictionHead2(nn.Module):
    def __init__(self,dim=768, n_patchs=100, prompt_dim = 128):
        super(PredictionHead2, self).__init__()
        
        self.linear_head1 = nn.Linear(dim * n_patchs+ prompt_dim, 512)
        self.linear_head2 = nn.Linear(512, 128)
        self.linear_head3 = nn.Linear(128, 1)
        self.gelu = nn.GELU()
    def forward(self,x, prompt_token):
        ### adding promt token at the end of body model
        prompt_token = prompt_token.squeeze(1)
        x = x.reshape(x.shape[0], -1)
        x = torch.concat([x,prompt_token], -1)
        x = self.gelu(self.linear_head1(x))
        x = self.gelu(self.linear_head2(x))
        return self.linear_head3(x)
    

# class Prompt_Tuning_Model7(nn.Module):
#     def __init__(self,cnn_embed, body_model_name="vit", prediction_head=None, args=None):
#         super(Prompt_Tuning_Model7,self).__init__()
#         """_summary_

#         Raises:
#             ValueError: Concatenate prompt tokens to embedded vecotors 
#         """
#         prompt_dim = 768
        
#         if body_model_name == 'vit':
#             model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
#             self.body_model =  copy.deepcopy(model.encoder)

#         elif body_model_name == 'scratch_vit':
#             config = ViTConfig()  # Use default configuration or modify as needed   
#             model = ViTModel(config)
#             self.body_model =  copy.deepcopy(model.encoder)

#         else:
#             raise ValueError("Not correct body model name")

#         if args.freeze:
#             for param in self.body_model.parameters():
#                 param.requires_grad = False
                
#         self.layernorm = nn.LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        
#         self.cnn_embed = cnn_embed
#         self.prediction_head = prediction_head
#         self.prompt_token = nn.Parameter(torch.randn(args.prompt_length, prompt_dim)) 
        
#         self.use_position_embedding = args.use_position_embedding
        
#         if self.use_position_embedding:
#             emb_size = 768
#             # self.cls_token = nn.Parameter(torch.randn(1,100, emb_size))
#             self.positions = nn.Parameter(torch.randn(100, emb_size))

#     def forward(self,x):
#         ### adding promt token at the begin of body model
#         batch_size = x.shape[0]
#         prompt_token_expanded = self.prompt_token.unsqueeze(0).expand(batch_size,-1, -1)  # Expand prompt token to batch 
        
#         embedding_x = self.cnn_embed(x) # 100 640
#         # 
#         ### add promt token
#         embedding_x = torch.cat([embedding_x, prompt_token_expanded], dim=1)
#         if self.use_position_embedding:
#             embedding_x += self.positions
            
#         body_output=  self.body_model(embedding_x)
#         body_output = body_output.last_hidden_state
        
#         # body_output = torch.cat([prompt_token_expanded.unsqueeze(1).repeat(1, body_output.size(1), 1), body_output], dim=-1)

#         body_output = self.layernorm(body_output)
#         # body_output = 
#         return self.prediction_head(body_output)

# class Prompt_Tuning_Model6(nn.Module):
#     def __init__(self,cnn_embed, body_model_name="vit", prediction_head=None, args=None):
#         super(Prompt_Tuning_Model6,self).__init__()
        
#         prompt_dim = args.prompt_dims
#         if body_model_name == 'vit':
#             model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
#             self.body_model =  copy.deepcopy(model.encoder)

#         elif body_model_name == 'scratch_vit':
#             config = ViTConfig()  # Use default configuration or modify as needed   
#             model = ViTModel(config)
#             self.body_model =  copy.deepcopy(model.encoder)

#         else:
#             raise ValueError("Not correct body model name")
        
#         self.layernorm = nn.LayerNorm((768,), eps=1e-12, elementwise_affine=True)

#         self.cnn_embed = cnn_embed
#         self.linear = nn.Linear(64, 768)
#         self.prediction_head = prediction_head
#         self.prompt_token = nn.Parameter(torch.randn(1, prompt_dim)) 
        
#         self.use_position_embedding = args.use_position_embedding

#         if self.use_position_embedding:
#             emb_size = 768
#             # self.cls_token = nn.Parameter(torch.randn(1,100, emb_size))
#             self.positions = nn.Parameter(torch.randn(100, emb_size))

#     def forward(self,x):
#         ### adding promt token at the begin of body model
#         # 
#         batch_size = x[0].shape[0]
#         prompt_token_expanded = self.prompt_token.expand(batch_size, -1)  # Expand prompt token to batch 
        
#         embedding_x = self.cnn_embed(x[0]) # 100 640
#         # 
#         ### add promt token
#         embedding_x = torch.cat([embedding_x, prompt_token_expanded.unsqueeze(1).repeat(1,embedding_x.shape[1],1)], dim=-1)
        
#         if self.use_position_embedding:
#             embedding_x += self.positions
            
#         body_output=  self.body_model(embedding_x)
#         body_output = body_output.last_hidden_state
#         his = self.linear(x[1]) #768
        
#         body_output = torch.cat([body_output, his[:, None, :]], 1)
#         # body_output = torch.cat([prompt_token_expanded.unsqueeze(1).repeat(1, body_output.size(1), 1), body_output], dim=-1)
        
#         body_output = self.layernorm(body_output)
#         # body_output = 
#         return self.prediction_head(body_output)

import copy 
class Prompt_Tuning_Model6_Progressive(nn.Module):
    def __init__(self,cnn_embed, body_model_name="vit", prediction_head=None, args=None):
        super(Prompt_Tuning_Model6_Progressive,self).__init__()
        
        prompt_dim = args.prompt_dims
        if body_model_name == 'vit':
            model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
            self.body_model =  copy.deepcopy(model.encoder)

        elif body_model_name == 'scratch_vit':
            config = ViTConfig()  # Use default configuration or modify as needed   
            model = ViTModel(config)
            self.body_model =  copy.deepcopy(model.encoder)

        else:
            raise ValueError("Not correct body model name")
        
        if args.freeze:
            for param in self.body_model.parameters():
                param.requires_grad = False
        
        self.image_size = args.image_size
        self.kernel_size = 10
        
        self.layernorm = nn.LayerNorm((768,), eps=1e-12, elementwise_affine=True)

        self.cnn_embed = cnn_embed
        self.linear = nn.Linear(64, 768)
        self.prediction_head = prediction_head
        self.prompt_token = nn.Parameter(torch.randn(1, (self.image_size // self.kernel_size) ** 2, prompt_dim)) 
        
        self.use_position_embedding = args.use_position_embedding
        
        if self.use_position_embedding:
            emb_size = 768
            self.positions = nn.Parameter(torch.randn(100, emb_size))


    def forward(self,x):
        ### adding promt token at the begin of body model
        
        """
        format for x: [nwp_data, his, nwp_id]
        nwp_data.shape [32,5, 63,100,100]]

        """
        
        batch_size = x[0].shape[0]
        nwp_data = x[0]
        his = x[1]
        nwp_id = x[2]

        
        # prompt_token_expanded = self.prompt_token.expand(batch_size, )  # Expand prompt token to batch 

        list_output = []
        for sample_id in range(batch_size):
            extra_his = None
            list_extra_his = []
            sample_nwp_id = nwp_id[sample_id]
            sample_nwp_data = nwp_data[sample_id] ### 6,63,101,101
            his_i = his[sample_id].unsqueeze(0)
            current_his = his_i
            # print(sample_nwp_id)
            for lead_time in range(int(sample_nwp_id)+1):
                # print(sample_nwp_data[lead_time,:,:,:].unsqueeze(0).shape, self.cnn_embed)
                embedding_x = self.cnn_embed(sample_nwp_data[lead_time,:,:,:].unsqueeze(0)) ### 1, 100,x
                embedding_x = torch.cat([embedding_x, self.prompt_token], dim=-1) ### 1, 100, 768
                if self.use_position_embedding:
                    embedding_x += self.positions
                body_output=  self.body_model(embedding_x)
                body_output = body_output.last_hidden_state
                
                his_embed = self.linear(current_his) #768

                body_output = torch.cat([body_output, his_embed[:, None, :]], 1)
                body_output = self.layernorm(body_output)
                prediction_lead_tine = self.prediction_head(body_output)
             

                current_his = torch.concat([current_his, prediction_lead_tine], -1)[:,-64:]
            
            list_output.append(prediction_lead_tine)
        output = torch.concat(list_output,0)
        return output

class Prompt_Tuning_Model6_Progressive2(nn.Module):
    def __init__(self,cnn_embed, body_model_name="vit", prediction_head=None, args=None):
        super(Prompt_Tuning_Model6_Progressive2,self).__init__()
        
        prompt_dim = args.prompt_dims
        if body_model_name == 'vit':
            model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
            self.body_model =  copy.deepcopy(model.encoder)

        elif body_model_name == 'scratch_vit':
            config = ViTConfig()  # Use default configuration or modify as needed   
            model = ViTModel(config)
            self.body_model =  copy.deepcopy(model.encoder)

        else:
            raise ValueError("Not correct body model name")
        
        if args.freeze:
            for param in self.body_model.parameters():
                param.requires_grad = False
                
        self.layernorm = nn.LayerNorm((768,), eps=1e-12, elementwise_affine=True)

        self.cnn_embed = cnn_embed
        self.linear = nn.Linear(64, 768)
        self.prediction_head = prediction_head
        self.prompt_token = nn.Parameter(torch.randn(1, prompt_dim)) 
        
        self.use_position_embedding = args.use_position_embedding
        self.image_size = args.image_size
        self.kernel_size = 10
        
        
        if self.use_position_embedding:
            emb_size = 768
            self.positions = nn.Parameter(torch.randn((self.image_size // self.kernel_size) ** 2, emb_size))


    def forward(self,x):
        ### adding promt token at the begin of body model
        
        """
        format for x: [nwp_data, his, nwp_id]
        nwp_data.shape [32,5, 63,100,100]]

        """
        
        batch_size = x[0].shape[0]
        nwp_data = x[0]
        his = x[1]
        nwp_id = x[2]

        
        # prompt_token_expanded = self.prompt_token.expand(batch_size, )  # Expand prompt token to batch 

        list_output = []
        expaned_prompt_token = self.prompt_token.unsqueeze(1)
        
        expaned_prompt_token = expaned_prompt_token.repeat(1, (self.image_size // self.kernel_size) ** 2,1)
        for sample_id in range(batch_size):
            sample_nwp_id = nwp_id[sample_id]
            sample_nwp_data = nwp_data[sample_id] ### 6,63,101,101
            his_i = his[sample_id].unsqueeze(0)
            current_his = his_i
            # print(sample_nwp_id)
            
            for lead_time in range(int(sample_nwp_id)+1):
                embedding_x = self.cnn_embed(sample_nwp_data[lead_time,:,:,:].unsqueeze(0)) ### 1, 100,x

                embedding_x = torch.cat([embedding_x, expaned_prompt_token], dim=-1) ### 1, 100, 768 , 
                if self.use_position_embedding:
                    embedding_x += self.positions
                body_output=  self.body_model(embedding_x)
                body_output = body_output.last_hidden_state
                
                his_embed = self.linear(current_his) #768

                body_output = torch.cat([body_output, his_embed[:, None, :]], 1)
                body_output = self.layernorm(body_output)
                prediction_lead_tine = self.prediction_head(body_output)
             

                current_his = torch.concat([current_his, prediction_lead_tine], -1)[:,-64:]
            
            list_output.append(prediction_lead_tine)
        output = torch.concat(list_output,0)
        return output
    
    
    
import torch
import torch.nn as nn
import copy
from transformers import ViTModel, ViTConfig

class Prompt_Tuning_Model6_4(nn.Module):
    def __init__(self, cnn_embed, body_model_name="vit", prediction_head=None, args=None):
        super(Prompt_Tuning_Model6_4, self).__init__()
        
        self.use_position_embedding = args.use_position_embedding
        self.image_size = args.image_size
        self.kernel_size = 10
        
        prompt_dim = args.prompt_dims
        if body_model_name == 'vit':
            model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
            self.body_model = copy.deepcopy(model.encoder)

        elif body_model_name == 'scratch_vit':
            config = ViTConfig()  # Use default configuration or modify as needed   
            model = ViTModel(config)
            self.body_model = copy.deepcopy(model.encoder)

        else:
            raise ValueError("Not correct body model name")
        
        if args.freeze:
            for param in self.body_model.parameters():
                param.requires_grad = False
                
        self.layernorm = nn.LayerNorm((768,), eps=1e-12, elementwise_affine=True)

        self.cnn_embed = cnn_embed
        self.linear = nn.Linear(64, 768)
        self.prediction_head = prediction_head
        self.prompt_token = nn.Parameter(torch.randn(1, prompt_dim)) 
        
        self.max_lead_time = args.max_lead_time
        self.start_lead_time = args.start_lead_time
        
        if self.use_position_embedding:
            emb_size = 768
            self.positions = nn.Parameter(torch.randn((self.image_size // self.kernel_size) ** 2, emb_size))
        self.delta_t = nn.Parameter(torch.rand((self.max_lead_time - self.start_lead_time + 1), 768))
        self.n_patches = (self.image_size // self.kernel_size) ** 2
        
        # Replace GRU with Transformer for hres_info
        self.hres_real_len = 4  # Real data length for hres_info
        self.hres_padded_len = 10  # Padded length for hres_info
        self.transform_hres = nn.Linear(4,768)
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=768,  # Input dimension for hres_info (matches your 4 features)
                nhead=2,    # Number of attention heads
                dim_feedforward=768,  # Feedforward dimension
                dropout=0.1,
                batch_first=True
            ),
            num_layers=2  # Number of Transformer layers
        )
        
        # Update layernorm to account for Transformer output (assuming output dim is still 768 for consistency)
        # self.layernorm = nn.LayerNorm((768 + 768,), eps=1e-12, elementwise_affine=True)  # Adjusted for Transformer output

    def add_delta_t(self, arr, lead_time):
        list_lead = []
        for lt in lead_time:
            lt = lt - self.start_lead_time
            corress_prompt = self.delta_t[int(lt)]
            list_lead.append(corress_prompt)
        add_prompt = torch.stack(list_lead, 0)
        add_prompt = add_prompt.unsqueeze(1).repeat(1, self.n_patches, 1)
        return arr + add_prompt

    # def get_mask
    def forward(self, x):
        """
        format for x: [nwp_data, his, nwp_id, hres_info]
        nwp_data.shape [batch_size, seq_len, 63, 100, 100]
        hres_info.shape [batch_size, hres_padded_len, 4] (e.g., [32, 10, 4])
        """
        
        batch_size = x[0].shape[0]
        nwp_data = x[0]  # [batch_size, seq_len, n_fts, height, width]
        his = x[1]
        nwp_id = x[2]



        # Process nwp_data as before
        expanded_prompt_token = self.prompt_token.unsqueeze(1)
        expanded_prompt_token = expanded_prompt_token.repeat(batch_size, self.n_patches, 1)
        
        embedding_x = self.cnn_embed(nwp_data)
        embedding_x = torch.cat([embedding_x, expanded_prompt_token], dim=-1)
        
        embedding_x = self.add_delta_t(embedding_x, nwp_id)
        
        if self.use_position_embedding:
            embedding_x += self.positions
            
        body_output = self.body_model(embedding_x)
        body_output = body_output.last_hidden_state
        body_output = body_output.view(batch_size, -1)
        
        his_embed = self.linear(his)
        
        # Concatenate Transformer output with other features
        body_output = torch.cat([body_output, his_embed], dim=-1)
        
        # body_output = self.layernorm(body_output)
        prediction_lead_time = self.prediction_head(body_output)
        
        return prediction_lead_time


class Prompt_Tuning_Model6_5(nn.Module):
    def __init__(self, cnn_embed, body_model_name="vit", prediction_head=None, args=None):
        super(Prompt_Tuning_Model6_5, self).__init__()
        
        self.args = args
        self.use_position_embedding = args.use_position_embedding
        self.image_size = args.image_size
        self.kernel_size = 10
        
        prompt_dim = args.prompt_dims
        if body_model_name == 'vit':
            model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
            self.body_model = copy.deepcopy(model.encoder)

        elif body_model_name == 'scratch_vit':
            config = ViTConfig()  # Use default configuration or modify as needed   
            model = ViTModel(config)
            self.body_model = copy.deepcopy(model.encoder)

        else:
            raise ValueError("Not correct body model name")
        
        if args.freeze:
            for param in self.body_model.parameters():
                param.requires_grad = False
                
        self.layernorm = nn.LayerNorm((768,), eps=1e-12, elementwise_affine=True)

        self.cnn_embed = cnn_embed
        self.linear = nn.Linear(64, 768)
        self.prediction_head = prediction_head
        self.prompt_type = args.prompt_type
        
        self.max_lead_time = args.max_lead_time
        self.start_lead_time = args.start_lead_time
        
        self.delta_t = nn.Parameter(torch.rand((self.max_lead_time - self.start_lead_time + 1), 768))
        self.n_patches = (self.image_size // self.kernel_size) ** 2
        
        # Replace Transformer with GRU for hres_info
        self.hres_real_len = 4  # Real data length for hres_info
        self.hres_padded_len = 10  # Padded length for hres_info
        self.transform_hres = nn.Linear(4, 768)  # Transform hres_info features to 768
        
        # Use GRU instead of Transformer
        self.gru = nn.GRU(input_size=768, hidden_size=768, num_layers=2, batch_first=True, dropout=0.1)
        
        ## Init prompt
        if self.prompt_type ==0:
            self.prompt_token = nn.Parameter(torch.randn(1, prompt_dim)) 
        elif self.prompt_type == 1:
            self.prompt_token = nn.Parameter(torch.randn(self.n_patches, prompt_dim))
        
        if self.use_position_embedding:
            emb_size = 768
            self.positions = nn.Parameter(torch.randn(1, self.n_patches, emb_size))
        
        
    def add_prompt(self,embeddings):
        """
        embeddings shape: [batch size, n_patches, hidden_dim]
        
        """
        batch_size = embeddings.shape[0]
        if self.prompt_type == 0:
            expanded_prompt_token = self.prompt_token.unsqueeze(0)
            expanded_prompt_token = expanded_prompt_token.repeat(batch_size, self.n_patches, 1)
        
        elif self.prompt_type == 1:
            expanded_prompt_token = self.prompt_token.unsqueeze(0)
            expanded_prompt_token = expanded_prompt_token.repeat(batch_size, 1, 1)
        # print(f"Shape : {embeddings.shape}, {expanded_prompt_token.shape}")
        return torch.cat([embeddings, expanded_prompt_token], dim=-1)
    
    def add_delta_t(self, arr, lead_time):
        list_lead = []
        for lt in lead_time:
            lt = lt - self.start_lead_time
            corress_prompt = self.delta_t[int(lt)]
            list_lead.append(corress_prompt)
        add_prompt = torch.stack(list_lead, 0)
        add_prompt = add_prompt.unsqueeze(1).repeat(1, self.n_patches, 1)
        return arr + add_prompt

    def get_hres_mask(self, hres_info, nwp_id):
        device = next(self.parameters()).device
        hres_padding_mask = torch.zeros((hres_info.shape[0], hres_info.shape[1]))
        for i, lead_time in enumerate(nwp_id):
            hres_padding_mask[i, :int(lead_time)] = 1
        return hres_padding_mask.to(device)

    def get_hres_embedding(self, embeddings, nwp_id, hres_padding_mask):
        list_embeddings = []
        batch_size = embeddings.shape[0]
        for i in range(batch_size):
            # Find the last non-padded position (where mask is 1)
            last_valid_idx = int(nwp_id[i]) - 1  # Convert to 0-based indexing
            if last_valid_idx < 0:
                last_valid_idx = 0  # Handle edge case
            list_embeddings.append(embeddings[i, last_valid_idx, :])
        return torch.stack(list_embeddings, 0)

    def forward(self, x):
        """
        format for x: [nwp_data, his, nwp_id, hres_info]
        nwp_data.shape [batch_size, seq_len, 63, 100, 100]
        hres_info.shape [batch_size, hres_padded_len, 4] (e.g., [32, 10, 4])
        nwp_id: tensor of shape [batch_size] indicating the valid length for each sample
        """
        
        batch_size = x[0].shape[0]
        nwp_data = x[0]  # [batch_size, seq_len, n_fts, height, width]
        his = x[1]
        nwp_id = x[2]  # [batch_size], integer values indicating valid sequence length
        hres_info = x[3]  # [batch_size, hres_padded_len, 4]  

        # Transform hres_info to embedding
        hres_embedding = self.transform_hres(hres_info)  # [batch_size, hres_padded_len, 768]
        
        # Process hres_info through GRU
        hres_output, _ = self.gru(hres_embedding)  # [batch_size, hres_padded_len, 768]
        
        # Get padding mask
        hres_padding_mask = self.get_hres_mask(hres_info, nwp_id)
        
        # Extract the last valid output for each sequence using the mask and nwp_id
        
        hres_last = self.get_hres_embedding(hres_output, nwp_id, hres_padding_mask)

        # Process nwp_data 
        embedding_x = self.cnn_embed(nwp_data)
        embedding_x = self.add_prompt(embedding_x)
        
        if self.args.delta_t_position == 0:
            embedding_x = self.add_delta_t(embedding_x, nwp_id)
        
        if self.use_position_embedding:
            embedding_x += self.positions
            
        body_output = self.body_model(embedding_x)
        body_output = body_output.last_hidden_state
        
        if self.args.delta_t_position == 1:
            body_output = self.add_delta_t(body_output, nwp_id)
            
        body_output = body_output.view(batch_size, -1)
        
        his_embed = self.linear(his)
        
        # Concatenate GRU output with other features
        body_output = torch.cat([body_output, his_embed, hres_last], dim=-1)
        
        # Apply layer normalization
        # body_output = self.layernorm(body_output) 
        
        prediction_lead_time = self.prediction_head(body_output)
        return prediction_lead_time
    

class ChannelSELayer(nn.Module):
    """
    Re-implementation of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*

    """

    def __init__(self, num_channels, reduction_ratio=2):
        """

        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """

        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output tensor
        """
        batch_size, num_channels, H, W = input_tensor.size()
        # Average along each channel
        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        return output_tensor

class SEResNet(nn.Module):
    def __init__(self, in_channels, out_channels, reduction_ratio=2):
        super(SEResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.se = ChannelSELayer(out_channels, reduction_ratio)

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out = self.se(out)  # Apply SE block here

        out += identity  # Residual connection (optional)
        out = self.relu(out)

        return out
    
class Prompt_Tuning_Model6_52(Prompt_Tuning_Model6_5):
    def __init__(self, cnn_embed, body_model_name="vit", prediction_head=None, args=None):
        super(Prompt_Tuning_Model6_52, self).__init__(cnn_embed, body_model_name, prediction_head, args)
        self.channel_atten = ChannelSELayer(args.n_fts)
        
    
    def forward(self, x):
        batch_size = x[0].shape[0]
        nwp_data = x[0]  # [batch_size, seq_len, n_fts, height, width]
        his = x[1]
        nwp_id = x[2]  # [batch_size], integer values indicating valid sequence length
        hres_info = x[3]  # [batch_size, hres_padded_len, 4]  

        

        # Transform hres_info to embedding
        hres_embedding = self.transform_hres(hres_info)  # [batch_size, hres_padded_len, 768]
        
        # Process hres_info through GRU
        hres_output, _ = self.gru(hres_embedding)  # [batch_size, hres_padded_len, 768]
        
        # Get padding mask
        hres_padding_mask = self.get_hres_mask(hres_info, nwp_id)
        
        # Extract the last valid output for each sequence using the mask and nwp_id
        
        hres_last = self.get_hres_embedding(hres_output, nwp_id, hres_padding_mask)
        
        
        ## Do the channel attention in this place  for the nwp_data to highlight the importance of each feature
        
        
        nwp_data = self.channel_atten(nwp_data)
        
        # Process nwp_data 
        embedding_x = self.cnn_embed(nwp_data)
        embedding_x = self.add_prompt(embedding_x)
        
        if self.args.delta_t_position == 0:
            embedding_x = self.add_delta_t(embedding_x, nwp_id)
        
        if self.use_position_embedding:
            embedding_x += self.positions
            
        body_output = self.body_model(embedding_x)
        body_output = body_output.last_hidden_state
        
        if self.args.delta_t_position == 1:
            body_output = self.add_delta_t(body_output, nwp_id)
            
        body_output = body_output.view(batch_size, -1)
        
        his_embed = self.linear(his)
        
        # Concatenate GRU output with other features
        body_output = torch.cat([body_output, his_embed, hres_last], dim=-1)
        
        # Apply layer normalization
        # body_output = self.layernorm(body_output) 
        
        prediction_lead_time = self.prediction_head(body_output)
        return prediction_lead_time
    

class Prompt_Tuning_Model6_6(nn.Module):
    def __init__(self, cnn_embed, body_model_name="vit", args=None):
        super(Prompt_Tuning_Model6_6, self).__init__()
        
        prompt_dim = args.prompt_dims
        if body_model_name == 'vit':
            model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
            self.body_model = copy.deepcopy(model.encoder)

        elif body_model_name == 'scratch_vit':
            config = ViTConfig()  # Use default configuration or modify as needed   
            model = ViTModel(config)
            self.body_model = copy.deepcopy(model.encoder)

        else:
            raise ValueError("Not correct body model name")
        
        if args.freeze:
            for param in self.body_model.parameters():
                param.requires_grad = False
                
        self.layernorm = nn.LayerNorm((768,), eps=1e-12, elementwise_affine=True)

        self.cnn_embed = cnn_embed
        self.linear = nn.Linear(64, 768)
        self.prompt_token = nn.Parameter(torch.randn(1, prompt_dim)) 
        
        self.use_position_embedding = args.use_position_embedding
        self.image_size = args.image_size
        self.kernel_size = 10
        
        self.num_patches = (self.image_size // self.kernel_size) ** 2
        self.max_lead_time = args.max_lead_time
        self.delta_t = nn.Parameter(torch.rand((self.max_lead_time + 1), 768))
        
        if self.use_position_embedding:
            emb_size = 768
            self.positions = nn.Parameter(torch.randn(self.num_patches, emb_size))
            
        # Add GRU layer for temporal information
        self.gru = nn.GRU(input_size=768,  # Match the target embedding size
                         hidden_size=768,  # Same as input for consistency
                         batch_first=True)

        # Define list of FC layers to map from 768 * 100 to 768
        
        self.fc_layers = nn.Sequential(
            nn.Linear(768 * self.num_patches, 1024),  # First FC layer: increase to 1024 for processing
            nn.ReLU(),                   # Activation
            nn.LayerNorm(1024),        # Normalization
            nn.Linear(1024, 768),        # Final FC layer: reduce back to 768
            nn.ReLU()                    # Activation
        )

        # # Embedding for nwp_id (match the 768 dimension)
        # self.nwp_embedding = nn.Embedding(num_embeddings=100, embedding_dim=768)  # Match final output dim

        self.prediction_head = nn.Sequential(
            nn.Linear(768 * 2, 768),  # First FC layer: increase to 1024 for processing
            nn.ReLU(),                   # Activation
            nn.LayerNorm(768),        # Normalization
            nn.Linear(768, 1),        # Final FC layer: reduce back to 768
        )
        
    def add_time_embedding(self, embeddings):
        batch_size, n_ts, _, _ = embeddings.shape
        # print(embeddings.shape, self.delta_t.shape)
        expanded_leadtime = self.delta_t.unsqueeze(0).unsqueeze(2)
        
        expanded_leadtime = expanded_leadtime.repeat(batch_size, 1, self.num_patches, 1)
        
        return embeddings + expanded_leadtime
    
        
        
    def do_spatial_exactor(self, x):
        """
        x.shape: [batch_size, n_ts, n_fts, h, w]
        """
        batch_size, n_ts, n_fts, h, w = x.shape
        
        reshaped_x = x.view(batch_size * n_ts, n_fts, h, w)
        embedding_x = self.cnn_embed(reshaped_x)  # batch_size * n_ts, n_patches, embedding
        expanded_prompt_token = self.prompt_token.repeat(batch_size * n_ts, self.num_patches, 1)
        
        embedding_x = torch.cat([embedding_x, expanded_prompt_token], dim=-1)  # Batch_size * n_ts, n_patches, embedding
        
        if self.use_position_embedding:
            embedding_x += self.positions
            
        body_output = self.body_model(embedding_x)
        body_output = body_output.last_hidden_state
        body_output = body_output.view(batch_size, n_ts,self.num_patches, -1) 
        
        return body_output
        
        return embedding_x.view(batch_size, n_ts, self.num_patches, -1) ## Batch_size, n_ts, 768

    # def add_delta_t(self, embeddings, ):
    #     list_output = []
    #     batch_size,  n_ts, n_patches, _ = embeddings.shape
    #     for lt in range(n_ts):
    #         corress_prompt = self.delta_t[int(lt)]
            
        
    #     list_lead = []
        
    #     for lt in lead_time:
    #         lt = lt - self.start_lead_time
    #         corress_prompt = self.delta_t[int(lt)]
    #         list_lead.append(corress_prompt)
    #     add_prompt = torch.stack(list_lead, 0)
    #     add_prompt = add_prompt.unsqueeze(1).repeat(1, self.n_patches, 1)
    #     return embeddings + add_prompt
    
    def forward(self, x):
        """
        format for x: [nwp_data, n_step, his, nwp_id]
        nwp_data.shape [batch_size, n_ts, n_fts, h, w]
        """
        
        batch_size = x[0].shape[0]
        nwp_data = x[0]  # [batch_size, n_ts, n_fts, h, w]
        his = x[1]
        nwp_id = x[2]   # This should be a tensor of indices or IDs

        batch_size, n_ts, n_fts, h, w = nwp_data.shape
        # Do spatial embedding
        spatial_embeddings = self.do_spatial_exactor(nwp_data)  # [batch_size * n_ts, n_patches, embedding_dim]

        tem_spatial_embeddings = self.add_time_embedding(spatial_embeddings)
        
        tem_spatial_embeddings = tem_spatial_embeddings.view(batch_size, n_ts, -1)
        
        embeddings = self.fc_layers(tem_spatial_embeddings) ### batch, n_ts, 768
        
        
        gru_output, _ = self.gru(embeddings) ### 
        
        
        gru_embedding = gru_output[np.arange(batch_size), nwp_id.long(), :] # batch_size, 768
        sp_embeddings = embeddings[np.arange(batch_size), nwp_id.long(), :]
        
        emb = gru_embedding + sp_embeddings
        his_embed = self.linear(his)
        
        # Concatenate Transformer output with other features
        final_embedding = torch.cat([emb, his_embed], dim=-1) ## batch , 768, 2 
        
        
        return self.prediction_head(final_embedding)
        
        

        
        
        # Reshape spatial embedding for FC layers: [batch_size * n_ts, n_patches * embedding_dim]
    



