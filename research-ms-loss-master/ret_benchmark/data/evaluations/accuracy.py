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
        
        softmax_feat = np.exp(feats)/np.sum(np.exp(feats), axis=1, keepdims=True)
        
        r_idx = np.random.choice(range(len(feats)), 20, replace=False)
        print(np.argmax(softmax_feat, axis=1)[r_idx], labels[r_idx])
        
        return np.sum(np.argmax(softmax_feat, axis=1) == labels)/len(feats)
