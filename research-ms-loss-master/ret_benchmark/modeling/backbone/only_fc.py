import torch
import torch.nn as nn
from ret_benchmark.modeling import registry


@registry.BACKBONES.register('only_fc')
class Only_FC(nn.Module):

    def __init__(self, cfg):
        super(Only_FC, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(cfg.INPUT.DIM1*cfg.INPUT.DIM2, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.features(x)
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