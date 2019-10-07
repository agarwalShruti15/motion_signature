
import torchvision.transforms as T
from ret_benchmark.data.registry import TRANSFORM
import numpy as np


@TRANSFORM.register('standard_scalar')
class StandardScalar():
    
    def __init__(self, cfg, is_train):

        self.mean = np.load(cfg.INPUT.MEAN)
        self.std = np.load(cfg.INPUT.STD)   
        
    def __new__(self):
        
        normalize_transform = T.Lambda(lambda x: self.normalize(x))
        transform = T.Compose([T.ToTensor(),
            normalize_transform,
        ])
        return transform
    
    def normalize(self, input):
        
        return (input - self.mean)/self.std