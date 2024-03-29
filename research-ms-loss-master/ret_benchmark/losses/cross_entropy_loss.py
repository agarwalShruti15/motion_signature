# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

from torch import nn
from ret_benchmark.losses.registry import LOSS


@LOSS.register('ce_loss')
class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, cfg):
        super(CrossEntropyLoss, self).__init__()
        
    def forward(self, feats, in_labels):
        labels = in_labels.squeeze()
        return super().forward(feats, labels)