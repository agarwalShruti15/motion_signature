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
from itertools import combinations
import torch

@DATA_LOADER.register('aus_190aus')
class AUS2_190ausDataLoader(Dataset):
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
        self.feat_pair = list(combinations(range(20), 2))
        self.feat_pair = np.array([[j[0], j[1]] for j in self.feat_pair])

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
            
        # get the first 20 features to the y label and rest to aus embeddings
        y_lbl = img[:, :20]
        aus = img[:, :20]
        
        # pick a sequence where the correlation is not nan
        out_X = []; out_Y = []
        i = 0
        while len(out_X)<1:
            # pick a random frame sequence of T length
            r_idx = np.random.choice(np.arange(len(aus)-self.T+1), 1)[0]
            corr = np.corrcoef(y_lbl[r_idx:r_idx+self.T, :][:, self.feat_pair[:, 0]].T, 
                               y_lbl[r_idx:r_idx+self.T, :][:, self.feat_pair[:, 1]].T)[0:len(self.feat_pair), 
                                                                          len(self.feat_pair):].diagonal()
            if np.sum(np.isnan(corr))>0 and i<len(aus):
                i = i+1
                continue
            else:
                out_X = aus[r_idx:r_idx+self.T, :].copy()
                out_X_norm = (out_X-np.mean(out_X, axis=0, keepdims=True))
                out_X_norm = out_X_norm/np.linalg.norm(out_X_norm, axis=0, keepdims=True) #np.std(out_X_norm, axis=0, keepdims=True)
                out_X_norm[np.isinf(out_X_norm)] = out_X[np.isinf(out_X_norm)] # replace the values by 
                
                out_Y = corr.ravel().copy()
                out_Y[np.isnan(corr)] = 0
                                    
        tensor = torch.from_numpy(out_X)
        return tensor.float(), np.reshape(out_Y, (1, -1)).astype(np.float32)