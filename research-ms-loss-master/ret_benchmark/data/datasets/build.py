# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.
from ret_benchmark.data.registry import DATA_LOADER
from .fabnet import FabNetDataLoader
from .aus190_lbl import AUS190_LBLDataLoader
from .aus_190aus import AUS2_190ausDataLoader
from .fabnet_190aus import FabNet2_190ausDataLoader

def build_data_loader(cfg, is_train, transforms):
    
    # set the transform
    loader_name = cfg.INPUT.DATA_LOADER
    assert loader_name in DATA_LOADER, \
        f'transform name {loader_name} is not registered in registry'
    return DATA_LOADER[loader_name](cfg, is_train, transforms)
