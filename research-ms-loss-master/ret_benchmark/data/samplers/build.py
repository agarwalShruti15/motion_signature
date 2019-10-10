# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.
from ret_benchmark.data.registry import SAMPLER
from .random_identity_sampler import RandomIdentitySampler

def build_sampler(cfg, dataset):
    
    # set the transform
    sampler_name = cfg.INPUT.SAMPLER
    assert sampler_name in SAMPLER, \
        f'sampler name {sampler_name} is not registered in registry'
    return SAMPLER[sampler_name](cfg, dataset)
