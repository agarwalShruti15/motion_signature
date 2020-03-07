# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

import torch
import numpy as np


def feat_extractor(model, data_loader, logger=None):
    model.eval()
    feats = list()
    labels = list()

    for i, batch in enumerate(data_loader):
        imgs = batch[0].cuda()
        lbls = batch[1].numpy()
        
        with torch.no_grad():
            out = model(imgs).data.cpu().numpy()
            feats.append(out)
            labels.append(lbls)

        del out
        del imgs
        del lbls
        del batch
        
        if logger is not None and (i + 1) % 1000 == 0:
            logger.debug(f'Extract Features: [{i + 1}/{len(data_loader)}]')
            break
    feats = np.vstack(feats)
    #print(feats[0, :10], np.concatenate(labels, axis=0))
    #print(feats[0, :5], np.concatenate(labels, axis=0)[0, :5])
    return feats, np.concatenate(labels, axis=0)