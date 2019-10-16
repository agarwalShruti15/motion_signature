# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

import os
import re
from collections import defaultdict

from torch.utils.data import Dataset
from ret_benchmark.data.registry import DATA_LOADER
import numpy as np

def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    import matplotlib.pyplot as plt
    from mpl_toolkits import axes_grid1

    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)

@DATA_LOADER.register('FabNet')
class FabNetDataLoader(Dataset):
    """
    Basic Dataset read image path from img_source
    img_source: list of img_path and label
    """

    def __init__(self, cfg, is_train, transforms):
        if is_train:
            self.img_source = cfg.DATA.TRAIN_IMG_SOURCE
        else:
            self.img_source = cfg.DATA.TEST_IMG_SOURCE
        
        self.T = cfg.INPUT.FRAME_LENGTH
        self.min_instances = cfg.DATA.NUM_INSTANCES
        self.transforms = transforms
        self.root = os.path.dirname(self.img_source)
        assert os.path.exists(self.img_source), f"{img_source} NOT found."

        self.label_list = list()
        self.path_list = list()
        self._load_data()
        self.label_index_dict = self._build_label_index_dict()

    def __len__(self):
        return len(self.label_list)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"| Dataset Info |datasize: {self.__len__()}|num_labels: {len(set(self.label_list))}|"

    def _load_data(self):
        with open(self.img_source, 'r') as f:
            for line in f:
                _path, _label = re.split(r",| ", line.strip())
                self.path_list.append(_path)
                self.label_list.append(int(_label))
                
        uniq_lbls, unq_cnt = np.unique(self.label_list, return_counts=True)
        uniq_lbls = uniq_lbls[unq_cnt>=self.min_instances].copy()
        self.path_list = [self.path_list[i] for i in range(len(self.path_list)) if self.label_list[i] in uniq_lbls]
        self.label_list = [self.label_list[i] for i in range(len(self.label_list)) if self.label_list[i] in uniq_lbls]

    def _build_label_index_dict(self):
        
        index_dict = defaultdict(list)
        for i, label in enumerate(self.label_list):
            index_dict[label].append(i)
        print(len(index_dict.keys()))
        return index_dict

    def __getitem__(self, index):
        path = self.path_list[index]
        img_path = os.path.join(self.root, path)
        label = self.label_list[index]
        
        path = self.path_list[index]
        img_path = os.path.join(self.root, path)
        img = np.load(img_path)        
        
        # if there is audio feature, pick only FabNet
        if len(img.shape)>2:
            img = img[:, :, 0].copy()
        
        # pick a random frame sequence of T length
        r_idx = np.random.choice(np.arange(len(img)-self.T+1), 1)[0]
        img = img[r_idx:r_idx+self.T, :].copy()
        
        #img = img[:self.T, :].copy()
                
        if self.transforms is not None:
            img = self.transforms(img)
        
        return img.float(), label
