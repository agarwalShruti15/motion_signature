import torch
import torch.nn as nn
from ret_benchmark.modeling import registry


@registry.BACKBONES.register('alexnet')
class AlexNet(nn.Module):

    def __init__(self, cfg):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=5),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 1024),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
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