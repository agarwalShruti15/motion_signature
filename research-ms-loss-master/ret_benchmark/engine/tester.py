# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

import datetime
import time

import numpy as np
import torch

from ret_benchmark.data.evaluations import RetMetric
from ret_benchmark.utils.feat_extractor import feat_extractor
from ret_benchmark.utils.metric_logger import MetricLogger
from ret_benchmark.data.evaluations.build import build_evaluation


def do_test(
        cfg,
        model,
        val_loader,
        device,
        arguments,
        logger
):
    logger.info("Start training")

    model.eval()
    logger.info('Validation')
    if cfg.INPUT.PERFORM_METRIC:
        feats, labels = feat_extractor(model, val_loader, logger=logger)
    else:
        feats = list()
        labels = list()
        for i, batch in enumerate(val_loader):

            feats.append(np.reshape(batch[0].cpu().numpy(), (batch[0].shape[0], -1)))
            labels.append(batch[1].numpy())
            del batch
        feats = np.vstack(feats)
        labels = np.concatenate(labels, axis=0)
    
    metric = build_evaluation(cfg)
    recall_curr = metric.eval(feats=feats, labels=labels, k=1)
        
    logger.info(f"Total instance: {len(feats) :06d} | Recall: {recall_curr} ")
