
import torch
import torch.nn as nn
from ret_benchmark.modeling import registry
import torch.nn.functional as F


@registry.BACKBONES.register('RNN')
class DecoderRNN(nn.Module):

    def __init__(self, cfg):
        super(DecoderRNN, self).__init__()
        
        self.RNN_input_size = 256
        self.h_RNN_layers = 3   # RNN hidden layers
        self.h_RNN = 1024                 # RNN hidden nodes
        self.drop_p = 0.5
    
        self.LSTM = nn.LSTM(
                input_size=self.RNN_input_size,
                hidden_size=self.h_RNN,        
                num_layers=self.h_RNN_layers,
                batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            bidirectional = True)
    
        self.fc1 = nn.Linear(self.h_RNN, 1024)
        self.fc2 = nn.Linear(512, 512)
        

    def forward(self, x):
        
        # change shape of x to (batch, time_step, input_size)
        self.LSTM.flatten_parameters()
        RNN_out, (h_n, h_c) = self.LSTM(x.squeeze(), None)
        """ h_n shape (n_layers, batch, hidden_size), h_c shape (n_layers, batch, hidden_size) """ 
        """ None represents zero initial hidden state. RNN_out has shape=(batch, time_step, output_size) """

        # FC layers
        x = self.fc1(RNN_out[:, -1, :])   # choose RNN_out at the last time step
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc2(x)

        return x
    
    def load_imagenet_param(self, model_path):
        
        # TODO: currently not skipping 
        # 1) the last layer of classification in case of pretrained wts
        # 2) first layer as input size has changed
        param_dict = torch.load(model_path)
        for i in param_dict:
            if 'last_linear' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])
            
    def load_classification_param(self, model_path):

        # this is to load the weight trained using classification problem
        param_dict = torch.load(model_path)['model']
        print(param_dict.keys())
        for i in param_dict:
            if 'head' in i: # no need to intialize the last layer
                continue
            k_m = i[len('backbone')+1:]
            print(k_m)
            self.state_dict()[k_m].copy_(param_dict[i])