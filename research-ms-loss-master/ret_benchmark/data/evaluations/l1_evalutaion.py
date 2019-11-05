# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

import numpy as np
from ret_benchmark.data.registry import EVAL

@EVAL.register('l1_loss')
class L1_Loss(object):      
    
    def eval(self, **kargs):
        
        feats = kargs['feats']
        labels = kargs['labels']
        
        return -np.mean(np.abs(feats - labels))
