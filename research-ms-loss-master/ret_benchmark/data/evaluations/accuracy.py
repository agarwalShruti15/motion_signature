# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

import numpy as np
from ret_benchmark.data.registry import EVAL

@EVAL.register('accuracy')
class Accuracy(object):      
    
    def eval(self, **kargs):
        
        feats = kargs['feats']
        labels = kargs['labels']
        
        return np.sum(np.argmax(feats, axis=1) == labels)/len(feats)
