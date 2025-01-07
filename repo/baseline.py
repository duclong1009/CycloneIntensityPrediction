import torch
import torch.nn as nn
import torch.nn.functional as F

# Co history
class CNN1(nn.Module):
    def __init__(self, input_channels, output_dim, kernel_size=20, prediction_head=None, args=None):
        super(CNN1,self).__init__()

        self.conv1 = nn.Conv2d(input_channels, input_channels * 2, kernel_size)
        self.conv2 = nn.Conv2d(input_channels*2, output_dim, kernel_size)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(output_dim * 10 * 10, 12000) # *11 neu height, width = 101
        self.fc2 = nn.Linear(12000, 1000)
        self.fc3 = nn.Linear(1000, 768)
        
        self.layernorm = nn.LayerNorm((768,), eps=1e-12, elementwise_affine=True)

        self.linear = nn.Linear(64, 768)
        self.prediction_head = prediction_head
        
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
                x = self.pool(F.relu(self.conv1(sample_nwp_data[lead_time,:,:,:].unsqueeze(0)))) # 1, 63*2, 40, 40 (1, 63*2, 41, 41)
                x = self.pool(F.relu(self.conv2(x))) # 1, 768, 10, 10 (1, 768, 11, 11)
                x = torch.flatten(x, 1) # 1, 768*10*10 (1, 768*11*11)
                x = F.relu(self.fc1(x)) # 1, 12000
                x = F.relu(self.fc2(x)) # 1, 1000
                x = self.fc3(x) # 1, 768
                
                his_embed = self.linear(current_his) #768

                body_output = torch.cat([x, his_embed[:, None, :]], 1)
                body_output = self.layernorm(body_output)
                prediction_lead_tine = self.prediction_head(body_output)
             
                current_his = torch.concat([current_his, prediction_lead_tine], -1)[:,-64:]
            
            list_output.append(prediction_lead_tine)
        output = torch.concat(list_output,0)
        return output
    
# Khong co history
class CNN2(nn.Module):
    def __init__(self, input_channels, output_dim, kernel_size=20, prediction_head=None, args=None):
        super(CNN1,self).__init__()

        self.conv1 = nn.Conv2d(input_channels, input_channels * 2, kernel_size)
        self.conv2 = nn.Conv2d(input_channels*2, output_dim, kernel_size)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(output_dim * 10 * 10, 12000) # *11 neu height, width = 101
        self.fc2 = nn.Linear(12000, 1000)
        self.fc3 = nn.Linear(1000, 768)
        
        self.layernorm = nn.LayerNorm((768,), eps=1e-12, elementwise_affine=True)

        self.linear = nn.Linear(64, 768)
        self.prediction_head = prediction_head
        
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
        nwp_id = x[2]

        
        # prompt_token_expanded = self.prompt_token.expand(batch_size, )  # Expand prompt token to batch 

        list_output = []
        for sample_id in range(batch_size):
            extra_his = None
            list_extra_his = []
            sample_nwp_id = nwp_id[sample_id]
            sample_nwp_data = nwp_data[sample_id] ### 6,63,101,101
            # print(sample_nwp_id)
            for lead_time in range(int(sample_nwp_id)+1):
                # print(sample_nwp_data[lead_time,:,:,:].unsqueeze(0).shape, self.cnn_embed)
                x = self.pool(F.relu(self.conv1(sample_nwp_data[lead_time,:,:,:].unsqueeze(0)))) # 1, 63*2, 40, 40 (1, 63*2, 41, 41)
                x = self.pool(F.relu(self.conv2(x))) # 1, 768, 10, 10 (1, 768, 11, 11)
                x = torch.flatten(x, 1) # 1, 768*10*10 (1, 768*11*11)
                x = F.relu(self.fc1(x)) # 1, 12000
                x = F.relu(self.fc2(x)) # 1, 1000
                x = self.fc3(x) # 1, 768
                
                body_output = self.layernorm(body_output)
                prediction_lead_tine = self.prediction_head(body_output)
                         
            list_output.append(prediction_lead_tine)
        output = torch.concat(list_output,0)
        return output
    
