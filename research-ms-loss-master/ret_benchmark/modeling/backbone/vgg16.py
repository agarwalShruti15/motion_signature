import torch
import torch.nn as nn

from ret_benchmark.modeling import registry

cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 1
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
    

@registry.BACKBONES.register('vgg16_bn')
class VGG(nn.Module):

    def __init__(self):
        super(VGG, self).__init__()
        self.features = make_layers(cfgs['D'], batch_norm=True)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 2048),
            nn.ReLU(True),
            nn.Dropout(),
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
        param_dict = torch.load(model_path)
        for i in param_dict:
            if 'head' in i: # no need to intialize the last layer
                continue
            self.state_dict()[i].copy_(param_dict[i])