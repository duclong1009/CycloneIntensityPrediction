import torch
import torch.nn as nn
from transformers import AutoImageProcessor, ViTModel
import copy
from transformers import ViTModel, ViTConfig

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


class Prompt_Tuning_Model_Leading_t(nn.Module):
    def __init__(self,cnn_embed, body_model_name="vit", prediction_head=None, args=None):
        super(Prompt_Tuning_Model_Leading_t, self).__init__()
        n_leading_times = 6
        prompt_dim = args.prompt_dims
        # prompt_dim = 128
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
        # 
        self.cnn_embed = cnn_embed
        self.prediction_head = prediction_head
        self.prompt_token = nn.Parameter(torch.randn(1, prompt_dim)) 
        self.leading_tokens = nn.Parameter(torch.randn(n_leading_times, prompt_dim))
        # self.use_position_embedding = False
        self.use_position_embedding = args.use_position_embedding
        if self.use_position_embedding:
            emb_size = 768
            self.positions = nn.Parameter(torch.randn(100, emb_size))

    def forward(self,x):
        ### adding promt token at the begin of body model

        x, leading_time = x
        batch_size = x.shape[0]

        leading_tokens_expanded = self.leading_tokens.unsqueeze(0).expand(batch_size, -1,-1)
        leading_time = leading_time.int().unsqueeze(-1)
        selected_leading_tokens = leading_tokens_expanded[torch.arange(leading_tokens_expanded.shape[0]).unsqueeze(1), leading_time]
        
        prompt_token_expanded = self.prompt_token.expand(batch_size, -1)  # Expand prompt token to batch 
        
        embedding_x = self.cnn_embed(x) # 100 640

        ### add promt token
        embedding_x = torch.cat([embedding_x, prompt_token_expanded.unsqueeze(1).repeat(1,embedding_x.shape[1],1)], dim=-1)
        
        if self.use_position_embedding:
            embedding_x += self.positions
            
        body_output=  self.body_model(embedding_x)
        body_output = body_output.last_hidden_state
        # body_output = torch.cat([prompt_token_expanded.unsqueeze(1).repeat(1, body_output.size(1), 1), body_output], dim=-1)
        
        body_output = self.layernorm(body_output)
        # body_output = 
        return self.prediction_head(body_output, selected_leading_tokens)

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
    

class Prompt_Tuning_Model7(nn.Module):
    def __init__(self,cnn_embed, body_model_name="vit", prediction_head=None, args=None):
        super(Prompt_Tuning_Model7,self).__init__()
        """_summary_

        Raises:
            ValueError: Concatenate prompt tokens to embedded vecotors 
        """
        prompt_dim = 768
        
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
        self.prompt_token = nn.Parameter(torch.randn(args.prompt_length, prompt_dim)) 
        
        self.use_position_embedding = args.use_position_embedding
        
        if self.use_position_embedding:
            emb_size = 768
            # self.cls_token = nn.Parameter(torch.randn(1,100, emb_size))
            self.positions = nn.Parameter(torch.randn(100, emb_size))

    def forward(self,x):
        ### adding promt token at the begin of body model
        batch_size = x.shape[0]
        prompt_token_expanded = self.prompt_token.unsqueeze(0).expand(batch_size,-1, -1)  # Expand prompt token to batch 
        
        embedding_x = self.cnn_embed(x) # 100 640
        # 
        ### add promt token
        embedding_x = torch.cat([embedding_x, prompt_token_expanded], dim=1)
        if self.use_position_embedding:
            embedding_x += self.positions
            
        body_output=  self.body_model(embedding_x)
        body_output = body_output.last_hidden_state
        
        # body_output = torch.cat([prompt_token_expanded.unsqueeze(1).repeat(1, body_output.size(1), 1), body_output], dim=-1)

        body_output = self.layernorm(body_output)
        # body_output = 
        return self.prediction_head(body_output)

class Prompt_Tuning_Model6(nn.Module):
    def __init__(self,cnn_embed, body_model_name="vit", prediction_head=None, args=None):
        super(Prompt_Tuning_Model6,self).__init__()
        
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
        
        self.layernorm = nn.LayerNorm((768,), eps=1e-12, elementwise_affine=True)

        self.cnn_embed = cnn_embed
        self.linear = nn.Linear(64, 768)
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
        batch_size = x[0].shape[0]
        prompt_token_expanded = self.prompt_token.expand(batch_size, -1)  # Expand prompt token to batch 
        
        embedding_x = self.cnn_embed(x[0]) # 100 640
        # 
        ### add promt token
        embedding_x = torch.cat([embedding_x, prompt_token_expanded.unsqueeze(1).repeat(1,embedding_x.shape[1],1)], dim=-1)
        
        if self.use_position_embedding:
            embedding_x += self.positions
            
        body_output=  self.body_model(embedding_x)
        body_output = body_output.last_hidden_state
        his = self.linear(x[1]) #768
        
        body_output = torch.cat([body_output, his[:, None, :]], 1)
        # body_output = torch.cat([prompt_token_expanded.unsqueeze(1).repeat(1, body_output.size(1), 1), body_output], dim=-1)
        
        body_output = self.layernorm(body_output)
        # body_output = 
        return self.prediction_head(body_output)

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