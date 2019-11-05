# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

from ret_benchmark.modeling.registry import HEADS

from .linear_norm import LinearNorm
from .linear_only import LinearOnly


def build_head(cfg):
    assert cfg.MODEL.HEAD.NAME in HEADS, f"head {cfg.MODEL.HEAD.NAME} is not defined"
    return HEADS[cfg.MODEL.HEAD.NAME](cfg, in_channels=cfg.MODEL.HEAD.IN_CHANNEL)
