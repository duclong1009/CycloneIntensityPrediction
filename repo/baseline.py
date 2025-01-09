import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



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
    
    
    
class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):

    """

    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.

    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor
        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :], cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])
            
        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
    
class CNN1(nn.Module):
    def __init__(self, input_channels, output_dim, kernel_size=20, prediction_head=None, args=None):
        super(CNN1,self).__init__()

        self.conv1 = nn.Conv2d(input_channels, input_channels * 2, kernel_size)
        self.conv2 = nn.Conv2d(input_channels*2, output_dim, kernel_size)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(output_dim * 10 * 10, 12000) # *11 neu height, width = 101
        self.fc2 = nn.Linear(12000, 1000)
        self.fc3 = nn.Linear(1000, 768)
        
        self.layernorm = nn.LayerNorm((768*2,), eps=1e-12, elementwise_affine=True)

        self.linear = nn.Linear(64, 768)
        self.prediction_head = prediction_head
        
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

                his_embed = self.linear(current_his) # 1, 768

                body_output = torch.cat([x, his_embed], 1)
                body_output = self.layernorm(body_output)
                prediction_lead_tine = self.prediction_head(body_output)
                
                current_his = torch.concat([current_his, prediction_lead_tine], -1)[:,-64:]
            
            list_output.append(prediction_lead_tine)
        output = torch.concat(list_output,0)
        return output
    
# Khong co history
class CNN2(nn.Module):
    def __init__(self, input_channels, output_dim, kernel_size=20, prediction_head=None, args=None):
        super(CNN2,self).__init__()

        self.conv1 = nn.Conv2d(input_channels, input_channels * 2, kernel_size)
        self.conv2 = nn.Conv2d(input_channels*2, output_dim, kernel_size)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(output_dim * 10 * 10, 12000) # *11 neu height, width = 101
        self.fc2 = nn.Linear(12000, 1000)
        self.fc3 = nn.Linear(1000, 768)
        
        self.layernorm = nn.LayerNorm((768,), eps=1e-12, elementwise_affine=True)

        self.linear = nn.Linear(64, 768)
        self.prediction_head = prediction_head

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

                body_output = self.layernorm(x)
                prediction_lead_tine = self.prediction_head(body_output)
                
            list_output.append(prediction_lead_tine)
        output = torch.concat(list_output,0)
        return output
    
# Co history
class ConvLSTM1(nn.Module):
    def __init__(self, input_channels, output_dim, kernel_size=20, num_layers=1, prediction_head=None, args=None):
        super(ConvLSTM1,self).__init__()

        self.convLSTM1 = ConvLSTM(input_channels, input_channels, (kernel_size, kernel_size), num_layers, True, True, False)
        self.conv1 = nn.Conv2d(input_channels, output_dim, 10, 10)
        self.layernorm = nn.LayerNorm((768,), eps=1e-12, elementwise_affine=True)

        self.linear = nn.Linear(64, 768)
        self.prediction_head = prediction_head

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
                # print(sample_nwp_data[:lead_time+1,:,:,:].unsqueeze(0).shape)
                _, x = self.convLSTM1(sample_nwp_data[:lead_time+1,:,:,:].unsqueeze(0))
                x = x[0][0] # 1, 63, 100, 100
                x = F.relu(self.conv1(x)) # 1, 768, 10, 10
                x = torch.flatten(x, 2) # 1, 768, 100
                his_embed = self.linear(current_his) # 1, 768

                body_output = torch.cat([x, his_embed.unsqueeze(-1)], 2)
                body_output = body_output.permute(0, 2, 1) # 1, 101, 768
                body_output = self.layernorm(body_output)
                prediction_lead_tine = self.prediction_head(body_output)
                
                current_his = torch.concat([current_his, prediction_lead_tine], -1)[:,-64:]
            
            list_output.append(prediction_lead_tine)
        output = torch.concat(list_output,0)
        return output
    
# Khong co history
class ConvLSTM2(nn.Module):
    def __init__(self, input_channels, output_dim, kernel_size=20, num_layers=1, prediction_head=None, args=None):
        super(ConvLSTM2,self).__init__()

        self.convLSTM1 = ConvLSTM(input_channels, input_channels, (kernel_size, kernel_size), num_layers, True, True, False)
        self.conv1 = nn.Conv2d(input_channels, output_dim, 10, 10)
        self.layernorm = nn.LayerNorm((768,), eps=1e-12, elementwise_affine=True)

        self.linear = nn.Linear(64, 768)
        self.prediction_head = prediction_head

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
            _, x = self.convLSTM1(sample_nwp_data[:int(sample_nwp_id)+1,:,:,:].unsqueeze(0))
            x = x[0][0] # 1, 63, 100, 100
            x = F.relu(self.conv1(x)) # 1, 768, 10, 10
            x = torch.flatten(x, 2) # 1, 768, 100

            body_output = x.permute(0, 2, 1) # 1, 100, 768
            body_output = self.layernorm(body_output)
            prediction_lead_tine = self.prediction_head(body_output)
                        
            list_output.append(prediction_lead_tine)
        output = torch.concat(list_output,0)
        return output