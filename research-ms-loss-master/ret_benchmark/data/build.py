# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

from torch.utils.data import DataLoader

from .collate_batch import collate_fn
from .datasets.build import build_data_loader
from .samplers.build import build_sampler
from .transforms.build import build_transforms

def build_data(cfg, is_train=True):
    
    transforms = build_transforms(cfg, is_train=is_train)
    dataset_loader = build_data_loader(cfg, is_train=is_train, transforms=transforms)
    sampler = build_sampler(cfg, dataset=dataset_loader)
    
    if is_train:
        data_loader = DataLoader(dataset_loader,
                                 collate_fn=collate_fn,
                                 batch_sampler=sampler,
                                 num_workers=cfg.DATA.NUM_WORKERS,
                                 pin_memory=True
                                 )
    else:
        data_loader = DataLoader(dataset_loader,
                                 collate_fn=collate_fn,
                                 shuffle=False,
                                 batch_size=cfg.DATA.TEST_BATCHSIZE,
                                 num_workers=cfg.DATA.NUM_WORKERS
                                 )
    return data_loader
