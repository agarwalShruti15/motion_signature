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
import matplotlib.pyplot as plt

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

    def _build_label_index_dict(self):
        index_dict = defaultdict(list)
        for i, label in enumerate(self.label_list):
            index_dict[label].append(i)
        return index_dict

    def __getitem__(self, index):
        path = self.path_list[index]
        img_path = os.path.join(self.root, path)
        label = self.label_list[index]
        
        
        """
        label = np.random.choice([0, 1], 1)[0]
        index = np.argwhere(np.array(self.label_list) == label).ravel()[0]
        
        # data of zero and data of one
        id0 = np.argwhere(np.array(self.label_list) == 0).ravel()[0]
        img0 = np.load(os.path.join(self.root, self.path_list[id0]))
        id1 = np.argwhere(np.array(self.label_list) == 1).ravel()[0]
        img1 = np.load(os.path.join(self.root, self.path_list[id1]))
        
        # display the data
        full_im_orig = np.hstack([img0[:self.T, :, 0].T, img1[:self.T, :, 0].T])
        print(full_im_orig.shape)
        
        add_colorbar(plt.imshow(full_im_orig))
        plt.savefig('original.png')
        plt.close()
        
        norm1 = full_im_orig/ np.linalg.norm(full_im_orig, axis=0, keepdims=True)
        add_colorbar(plt.imshow(norm1))
        plt.savefig('unit_scale.png')
        plt.close()
        
        norm2 = self.transforms(norm1.T).T
        add_colorbar(plt.imshow(norm2))
        plt.savefig('standar_scale.png')
        plt.close()
        
        with open('tmp.txt', 'w') as f:
            
            for i in range(full_im_orig.shape[0]):
                
                for j in range(int(full_im_orig.shape[1]/2)):
                    f.write('%2.4f '%full_im_orig[i, j])
                f.write('*****')
                for j in range(int(full_im_orig.shape[1]/2)):
                    f.write('%2.4f '%full_im_orig[i, j+int(full_im_orig.shape[1]/2)])
                    
                f.write('\n')
                for j in range(int(full_im_orig.shape[1]/2)):
                    f.write('%2.4f '%norm1[i, j])
                f.write('*****')
                for j in range(int(full_im_orig.shape[1]/2)):
                    f.write('%2.4f '%norm1[i, j+int(full_im_orig.shape[1]/2)])                
                
                f.write('\n')
                for j in range(int(full_im_orig.shape[1]/2)):
                    f.write('%2.4f '%norm2[i, j])
                f.write('*****')
                for j in range(int(full_im_orig.shape[1]/2)):
                    f.write('%2.4f '%norm2[i, j+int(full_im_orig.shape[1]/2)])
                    
                f.write('\n---------------------------------------------------------------------\n')
                
        
        raise ValueError
        """
        
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
