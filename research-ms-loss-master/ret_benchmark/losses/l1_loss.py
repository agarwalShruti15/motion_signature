# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

from torch import nn
from ret_benchmark.losses.registry import LOSS


@LOSS.register('l1_loss')
class L1Loss(nn.L1Loss):
    def __init__(self, cfg):
        super(L1Loss, self).__init__()