import numpy as np
import matplotlib.pyplot as plt
import os
import utils as u
import pandas as pd
import seaborn as sns
import multiprocessing
from joblib import Parallel, delayed
import natsort
import time
import re

from sklearn.metrics import roc_curve, auc
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree
from collections import Counter

# build the repo
# collect the files with which repo needs to build    
# construct a class with these file names to read the embeddings
# I need the embeddings, each embedding should have a label and a name, filename
# for each label get N embeddings to keep in the repo

class Repo2(object):
    
    """class to get the repo per label"""
    def __init__(self, bs_fldr, frames, step, pool_func, N_per_id):
        super(Repo2, self).__init__()
        self.emb = None # to store the embeddings
        self.label = None # integer labels
        self.name2label_dict = {} # integer dict to string dict        
        self.bs_fldr = bs_fldr
        self.frames = frames
        self.step = step
        self.pool_func = pool_func
        self.N_per_id = N_per_id
        self.pca = None
        self.pcaEmb = None
        self.knnrepo = None
        
    def read_feat_file(self, f, n1, n2):
        
        if not os.path.exists(os.path.join(self.bs_fldr, f)):
            path, file = os.path.split(f)
            if file[0] == '_':
                f = os.path.join(path, file[1:])
            else:
                f = os.path.join(path, '_' + file)
        try:
            feat = np.load(os.path.join(self.bs_fldr, f))    
        except Exception as e:
            print(os.path.join(self.bs_fldr, f))
            return None

        if len(feat.shape)>2:
            feat = feat[:,:,0].copy()
        if len(feat) < (self.frames+ self.step):
            return None
        
        # pool the features according to pool_params
        # for vgg pool all 100 frames, whereas for resnet3D pool only 84 frames. 
        # for then step to another window to pool. The pool function is given in the params
        col_feat = u.im2col_sliding_strided(feat, (self.frames, feat.shape[1]), stepsize=self.step).T
        tmp = self.pool_func(np.reshape(col_feat, (col_feat.shape[0], self.frames, feat.shape[1])), axis=1)

        n1 = int(n1*len(tmp))
        n2 = int(n2*len(tmp))
        tmp = tmp[n1:n2+1, :].copy()
        
        feat = []
        col_feat = []
        
        return tmp
        
    def remove_ids(self, id_list):
        
        int_lbl = [v for i, (k, v) in enumerate(self.name2label_dict.items()) if k in id_list]
        rm_idx = np.array([f in int_lbl for f in self.label])
        
        self.label = self.label[~rm_idx].copy()
        self.emb = self.emb[~rm_idx, :].copy()
        
        self.name2label_dict = {k:v for i, (k, v) in enumerate(self.name2label_dict.items()) if k not in id_list }
        
    def pop_ids(self, id_list):
        
        int_lbl = [v for i, (k, v) in enumerate(self.name2label_dict.items()) if k in id_list]
        rm_idx = np.array([f in int_lbl for f in self.label])
        
        out_emb = self.emb[rm_idx, :].copy()
        self.label = self.label[~rm_idx].copy()
        self.emb = self.emb[~rm_idx, :].copy()
        
        self.name2label_dict = {k:v for i, (k, v) in enumerate(self.name2label_dict.items()) if k not in id_list }        
        
        return out_emb
    
    def add_emb_ids(self, all_id_feat, id_lbl):
        
        bas_label = np.max(list(self.name2label_dict.values())) + 1
        
        self.label = np.concatenate((self.label, np.zeros((len(all_id_feat), ), dtype=np.int32)+bas_label))
        self.emb = np.concatenate((self.emb, all_id_feat), axis=0)
        assert id_lbl not in list(self.name2label_dict.keys())
        self.name2label_dict[id_lbl] = bas_label
        
    # n1, n2 gives the part of the video to use
    def add_ids(self, in_file_dict, n1, n2):
        
        all_keys = list(in_file_dict.keys()) # list of all identities
        all_keys.sort()
        
        if len(list(self.name2label_dict.values()))<1:
            bas_label = 0
        else:
            bas_label = np.max(list(self.name2label_dict.values())) + 1
            
        for k in range(len(all_keys)):
                        
            id_feat = {}
            for f in in_file_dict[all_keys[k]]:
                
                tmp = self.read_feat_file(f, n1, n2)
                if tmp is not None:
                    id_feat[f] = tmp.copy()
                    tmp = []
    
            all_id_feat = np.vstack(list(id_feat.values()))
            id_feat = []
            
            if self.N_per_id > 0 and self.N_per_id < len(all_id_feat):
                np.random.seed(seed=0)
                idx = np.random.choice(range(len(all_id_feat)), self.N_per_id, replace=False)
                all_id_feat = all_id_feat[idx, :].copy()
                
            # add the labels and embeddings
            if self.label is None:
                self.label = np.zeros((len(all_id_feat), ), dtype=np.int32)+bas_label+k
                self.emb = all_id_feat.copy()
            else:
                self.label = np.concatenate((self.label, np.zeros((len(all_id_feat), ), dtype=np.int32)+bas_label+k))
                self.emb = np.concatenate((self.emb, all_id_feat), axis=0)
                
            all_id_feat = []
            assert all_keys[k] not in list(self.name2label_dict.keys())
            self.name2label_dict[all_keys[k]] = bas_label+k            
                    
        self.pca = None
        self.pcaEmb = None
        self.knnrepo = None
        
        
    def build_repo_noKDD(self):
        
        # build k-Nearest Neghbor
        start = time.time()
        
        print(f'Number of labels {len(self.name2label_dict.values())}')                
        X = self.emb - np.mean(self.emb, axis=1, keepdims=True)
        X = X / np.linalg.norm(X, axis=1, keepdims=True) # normalize
        self.emb = X.copy()
        X = []
        
        print(f'Build repo time: {time.time()-start:0.3f} Size: {self.emb.shape}')
    
    # cosine distance
    def get_one_cosine_dist(self, in_test_file, n1, n2, ids):
        
        X = self.read_feat_file(in_test_file, n1, n2)
        
        if X is None or len(X)<1:
            out_df = pd.DataFrame({'predLabel':[], 'fileName':[], 'actualLabel':[], 'dist':[]})
            return
        
        if self.pca is not None:
            X = self.pca.transform(X)

        X = X - np.mean(X, axis=1, keepdims=True)        
        X = X / np.linalg.norm(X, axis=1, keepdims=True) # normalize
        sim_mat = np.matmul(self.emb, X.T)
        
        out_dist_mat = np.zeros((len(X), len(self.name2label_dict.keys())))
        c = 0
        for c, (k, i) in enumerate(self.name2label_dict.items()):
            if len(sim_mat[self.label==i, :])>0:
                out_dist_mat[:, c]=np.max(sim_mat[self.label==i, :], axis=0)
            else:
                out_dist_mat[:, c]=-np.nan
            c = c+1
        
        out_df = pd.DataFrame(data=np.array([f'{in_test_file}_{i}' for i in range(len(out_dist_mat))]), columns=['fileName'])
        out_df['sim'] = np.array(out_dist_mat).tolist()
        out_df['actualLabel'] = ids
        X = []
        return out_df
        
    def dist_using_dict(self, in_test_dict, n1, n2, parallel=False, dist='kdd'):
        
        # for this test file return the dataframe with true label, pred label, file id
        # I'll store the index of the data frame as file name and the index of the feature
        # collect all the parallel struct
        full_struct = []
        for ids in in_test_dict.keys():
            
            # ensure this key should be in the repo
            #assert np.any(ids==self.label_dict), f'test label should be in the repo dict {ids}'
            
            files = in_test_dict[ids]
            for f in files:
                full_struct.append((f, n1, n2, ids))
            
        if parallel:            
            num_cores = multiprocessing.cpu_count()
            df_list = Parallel(n_jobs=num_cores, verbose=1)(delayed(self.get_one_cosine_dist)(*full_struct[c]) for c in range(len(full_struct)))
            out_df = pd.concat(df_list, ignore_index=True, sort=False)
            df_list = []
        else:
            out_df = {}
            for i in range(len(full_struct)):
                print(full_struct[i])
                out_df[i] = self.get_one_cosine_dist(*full_struct[i])
                
            out_df = pd.concat(list(out_df.values()), ignore_index=True, sort=False)
            
        return out_df
