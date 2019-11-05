# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.
from ret_benchmark.data.registry import EVAL
from .ret_metric import RetMetric
from .accuracy import Accuracy
from .l1_evalutaion import L1_Loss

def build_evaluation(cfg):
    
    # set the transform
    eval_name = cfg.INPUT.EVAL
    assert eval_name in EVAL, \
        f'sampler name {eval_name} is not registered in registry'
    return EVAL[eval_name]()
