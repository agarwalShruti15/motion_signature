
import torchvision.transforms as T
from ret_benchmark.data.registry import TRANSFORM
import numpy as np


@TRANSFORM.register('standard_scalar')
class StandardScalar(object):
    def __init__(self, cfg, **kargs):
        self.mean = np.reshape(np.load(cfg.INPUT.MEAN), (1, -1))
        self.std = np.reshape(np.load(cfg.INPUT.STD), (1, -1))
        
    def __call__(self, input):
        return (input - self.mean)/self.std