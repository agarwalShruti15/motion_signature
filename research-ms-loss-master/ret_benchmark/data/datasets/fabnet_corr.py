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

@DATA_LOADER.register('FabNet_Corr')
class FabNetCorrDataLoader(Dataset):
    """
    Basic Dataset read image path from img_source
    img_source: list of img_path and label
    """

    def __init__(self, cfg, is_train, transforms):
        
        self.is_train = is_train
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
        self.feat_pair = list(combinations(range(256), 2))
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
        if self.is_train:            
            uniq_lbls, unq_cnt = np.unique(self.label_list, return_counts=True)
            uniq_lbls = uniq_lbls[unq_cnt>=self.min_instances].copy()
            self.path_list = [self.path_list[i] for i in range(len(self.path_list)) if self.label_list[i] in uniq_lbls]
            self.label_list = [self.label_list[i] for i in range(len(self.label_list)) if self.label_list[i] in uniq_lbls]

    def _build_label_index_dict(self):
        
        index_dict = defaultdict(list)
        for i, label in enumerate(self.label_list):
            index_dict[label].append(i)
        return index_dict

    def __getitem__(self, index):
        path = self.path_list[index]
        img_path = os.path.join(self.root, path)
        label = self.label_list[index]
        
        path = self.path_list[index]
        img_path = os.path.join(self.root, path)
        
        if os.path.exists(img_path):
            img = np.load(img_path)
        else:
            p, f = os.path.split(img_path)
            img = np.load(os.path.join(p, '_' + f))
                    
        # if there is audio feature, pick only FabNet
        if len(img.shape)>2:
            img = img[:, :, 0].copy()
        
        out_X = []
        i = 0
        while len(out_X)<1:
            # pick a random frame sequence of T length
            r_idx = np.random.choice(np.arange(len(img)-self.T+1), 1)[0]
            X = img[r_idx:r_idx+self.T, :].copy()
            
            X = X / np.linalg.norm(X, axis=1, keepdims=True)
            X = X - np.mean(X, axis=0, keepdims=True)
            X = X / np.linalg.norm(X, axis=0, keepdims=True) # normalize
            corr = np.sum(X[:, self.feat_pair[:, 0]] * X[:, self.feat_pair[:, 1]], axis=0)
        
            if np.sum((np.isnan(corr)) | (np.isinf(corr)))>0 and i < len(img):
                i = i+1
                continue
            else:
                if np.sum((np.isnan(corr)) | (np.isinf(corr)))>0:
                    #assert False, f'{img_path} {np.sum((np.isnan(corr)) | (np.isinf(corr)))}'
                    corr[(np.isnan(corr)) | (np.isinf(corr))] = 0
                
                out_X = np.reshape(corr, (128, 255)).copy()
        
        assert np.sum(np.isnan(out_X))<1, 'Nan values in correlation computation, please handle it'
        if self.transforms is not None:
            out_X = self.transforms(out_X)
            
        X = []; corr = []; img = [];
            
        return out_X.float(), np.int64(label)
