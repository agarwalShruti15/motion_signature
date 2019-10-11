
import torchvision.transforms as T
from ret_benchmark.data.registry import TRANSFORM
import numpy as np


@TRANSFORM.register('standard_scalar')
class StandardScalar(object):
    def __init__(self, cfg, **kargs):
        self.mean = np.reshape(np.load(cfg.INPUT.MEAN), (1, -1))
        self.std = np.reshape(np.load(cfg.INPUT.STD), (1, -1))
        
    def __call__(self, input):
        
        # unit length the input
        norm1 = input/ np.linalg.norm(input, axis=1, keepdims=True)
        #after = norm1*255 # just to make it image scale
        
        after = (norm1 - self.mean)/self.std
        
        """with open('tmp.txt', 'a') as f:
            
            f.write('---------------------------------------------------------------------------------\n')
            for i in range(after.shape[1]):
                f.write(f'{input[0, i]} --> ({self.mean[0, i]}, {self.std[0, i]}) --> {after[0, i]} \n')"""
        
        return after