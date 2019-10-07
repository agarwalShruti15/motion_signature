# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.
from ret_benchmark.data.registry import TRANSFORM
from .standard_scalar import StandardScalar

def build_transforms(cfg, is_train=True):
    
    # set the transform
    trans_name = cfg.DATA.TRANSFORM
    assert trans_name in TRANSFORM, \
        f'transform name {trans_name} is not registered in registry'
    return TRANSFORM[trans_name](cfg, is_train)
