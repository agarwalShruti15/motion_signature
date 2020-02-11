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

def im2col_sliding_strided(A, BSZ, stepsize=1):
    # Parameters
    m,n = A.shape
    s0, s1 = A.strides
    nrows = m-BSZ[0]+1
    ncols = n-BSZ[1]+1
    shp = BSZ[0],BSZ[1],nrows,ncols
    strd = s0,s1,s0,s1

    out_view = np.lib.stride_tricks.as_strided(A, shape=shp, strides=strd)
    return out_view.reshape(BSZ[0]*BSZ[1],-1)[:,::stepsize]

def load_file_names(bs_fldr, in_fldr, join_bsfldr=False):
    if join_bsfldr:
        return [os.path.join(bs_fldr, in_fldr, f) for f in os.listdir(os.path.join(bs_fldr, in_fldr)) if f.endswith('.npy')]
    else:
        return [os.path.join(in_fldr, f) for f in os.listdir(os.path.join(bs_fldr, in_fldr)) if f.endswith('.npy')]


GG_train_files = ['__talking_against_wall', 
                  '__outside_talking_still_laughing', 
                  '__talking_angry_couch',
                  '__podium_speech_happy',
                  '__kitchen_still']
GG_real_test_files = ['__outside_talking_pan_laughing',
                      '__kitchen_pan']
GG_fake_test_files = ['__talking_against_wall', 
                      '__outside_talking_still_laughing', 
                      '__talking_angry_couch',
                      '__podium_speech_happy', 
                      '__outside_talking_pan_laughing',
                      '__kitchen_pan']


# get all the identity specific train and test files for all available dataset
def train_test_all_ids(bs_fldr):
    
    # REPO files are all the real files
    train_dict = u.load_dict_file('/data/home/shruti/voxceleb/motion_signature/data/utils/leaders_100_train.txt', join_bsfldr=False, inbsfldr=bs_fldr) # leaders repo file
    for k in ['bo_imposter', 'bs_imposter', 'ew_imposter', 'dt_imposter', 'hc_imposter', 'jb_imposter']:   # imposter videos
        train_dict[k] = load_file_names(bs_fldr, k, join_bsfldr=False)
    for k in range(1000):
        cur_name = '{0:03d}'.format(k)
        train_dict['FF_'+cur_name] = [os.path.join('FF_orig', cur_name + '.npy')]
        
    
    # TEST files, two dict for real and fake
    test_dict = {}; test_dict['real'] = {}; test_dict['fake'] = {}; 
    #REAL
    test_dict['real'] = u.load_dict_file('/data/home/shruti/voxceleb/motion_signature/data/utils/leaders_100_test.txt', join_bsfldr=False, inbsfldr=bs_fldr)
    for k in range(1000):
        cur_name = '{0:03d}'.format(k)
        test_dict['real']['FF_'+cur_name] = [os.path.join('FF_orig', cur_name + '.npy')]
    
    #FAKE:leaders
    fake_name = ['bo_faceswap', 'bs_faceswap', 'ew_faceswap', 'dt_faceswap', 'hc_faceswap', 'jb_faceswap']
    lbl = ['bo', 'bs', 'ew', 'dt', 'hc', 'jb']
    for i in range(len(lbl)):
        test_dict['fake'][lbl[i]] = load_file_names(bs_fldr, fake_name[i], join_bsfldr=False)
    
    #FAKE: FaceForensics
    for ff_fldr in ['FF_Deepfakes', 'FF_FaceSwap']:            
        ff_files = load_file_names(bs_fldr, ff_fldr, join_bsfldr=False)
        for k in range(1000):
            cur_name = '{0:03d}'.format(k)
            df = [f for f in ff_files if cur_name + '.npy' in f]
            assert len(df)==1, f'{cur_name} {df} {ff_files}'
            
            if 'FF_'+cur_name not in list(test_dict['fake'].keys()):
                test_dict['fake']['FF_'+cur_name] = df
            else:
                test_dict['fake']['FF_'+cur_name] = test_dict['fake']['FF_'+cur_name] + df
                
    # google dataset original videos divide the original in train and test
    # identify the fake indentites 
    GG_orig_allfiles = load_file_names(bs_fldr, 'GG_orig', join_bsfldr=False)
    GG_fake_allfiles = load_file_names(bs_fldr, 'GG_fake', join_bsfldr=False)
    
    id_lbl = np.unique([f.split('/')[-1][:2] for f in GG_orig_allfiles])  # get the unique ids
    for k in id_lbl:
        
        #real Train
        orig_files = [f for f in GG_orig_allfiles if f.split('/')[-1][:2]==k and 
                      np.any([x in f for x in GG_train_files])] # current file label
        orig_files.sort()
        train_dict['GG_'+k] = orig_files.copy()
        
        #real Test
        orig_test_files = [f for f in GG_orig_allfiles if f.split('/')[-1][:2]==k and np.any([x in f for x in GG_real_test_files])] # current file label
        test_dict['real']['GG_'+k] = orig_test_files.copy()
        
        #fake
        fake_files = [f for f in GG_fake_allfiles if (f'_{k}_' in f.split('/')[-1][2:]) and np.any([x in f for x in GG_fake_test_files])] # current file label
        fake_files.sort()
        test_dict['fake']['GG_'+k] = fake_files.copy()
        
    # steve bushemi and jennifer lawrence
    train_dict['steve_b'] = load_file_names(bs_fldr, 'steve_b', join_bsfldr=False)
    train_dict['jen_l'] = load_file_names(bs_fldr, 'jennifer_l', join_bsfldr=False)
    # remove pW7TbJJMVak.mp4 file which is ground-truth from 
    train_dict['jen_l'] = [f for f in train_dict['jen_l'] if 'pW7TbJJMVak_0' not in f]
    
    # steve faceswap
    test_dict['fake']['steve_b'] = load_file_names(bs_fldr, 'steve_faceswap', join_bsfldr=False)
    test_dict['real']['jen_l'] = ['jennifer_l/pW7TbJJMVak_0.npy']
        
    return train_dict, test_dict



def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    from mpl_toolkits import axes_grid1
 
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)

class Repo(object):
    """class to get the repo per label"""
    def __init__(self, bs_fldr, frames, step, pool_func, N_per_id):
        super(Repo, self).__init__()
        self.emb = [] # to store the embeddings
        self.label = [] # integer labels
        self.label_dict = [] # integer dict to string dict
        self.bs_fldr = bs_fldr
        self.frames = frames
        self.step = step
        self.pool_func = pool_func
        self.N_per_id = N_per_id
        self.pca = None
        self.pcaEmb = None
        self.knnrepo = None
        self.dist_dict = {'cosine': self.get_one_cosine_dist, 'kdd':self.get_one_kdd_dist}
        
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
        col_feat = im2col_sliding_strided(feat, (self.frames, feat.shape[1]), stepsize=self.step).T
        tmp = self.pool_func(np.reshape(col_feat, (col_feat.shape[0], self.frames, feat.shape[1])), axis=1)

        n1 = int(n1*len(tmp))
        n2 = int(n2*len(tmp))
        tmp = tmp[n1:n2+1, :].copy()
        
        feat = []
        col_feat = []
        
        return tmp
        
        
    # n1, n2 gives the part of the video to use
    def add_ids(self, in_file_dict, n1, n2):
        
        all_keys = list(in_file_dict.keys()) # list of all identities
        all_keys.sort()
        
        bas_label = len(self.label_dict)
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
            self.label = self.label + [np.zeros((len(all_id_feat), ), dtype=np.int32)+bas_label+k]
            self.label_dict = self.label_dict + [all_keys[k]]
            self.emb = self.emb + [all_id_feat]
            all_id_feat = []
            
        # need to build the repo again
        self.pca = None
        self.pcaEmb = None
        self.knnrepo = None
            
    def build_repo(self, N_pca_comp):
        
        self.label_dict = np.array(self.label_dict)
        print(f'Number of labels {len(self.label_dict)}')        
        
        # perform PCA first, store the pca embeddings, store the pca, store the KDD
        X = np.vstack(self.emb)
        X = X - np.mean(X, axis=1, keepdims=True)        
        X = X / np.linalg.norm(X, axis=1, keepdims=True) # normalize
        self.label = np.concatenate(self.label, axis=0)
        if N_pca_comp < 0:
            # take all the comp
            N_pca_comp = X.shape[1]
        
        self.pca = PCA(n_components=N_pca_comp, copy=True, whiten=False, svd_solver='auto', 
                       tol=0.0, iterated_power='auto', random_state=0).fit(X)
        print('pca components: {}, explained variance {}'.format(N_pca_comp, np.sum(self.pca.explained_variance_ratio_)))
        self.pcaEmb = self.pca.transform(X)
        X = []
        self.emb = []
        
        # build k-Nearest Neghbor
        start = time.time()
        self.knnrepo = KDTree(self.pcaEmb, leaf_size=100)
        print('Build repo time: {0:0.3f}'.format(time.time()-start))
        
        
    def build_repo_noPCA(self):
        
        self.label_dict = np.array(self.label_dict)
        print(f'Number of labels {len(self.label_dict)}')        
        
        # perform PCA first, store the pca embeddings, store the pca, store the KDD
        X = np.vstack(self.emb)
        X = X - np.mean(X, axis=1, keepdims=True)        
        X = X / np.linalg.norm(X, axis=1, keepdims=True) # normalize
        self.label = np.concatenate(self.label, axis=0)
        self.knnrepo = KDTree(X, leaf_size=100)
        X = []
        self.emb = []
        
        # build k-Nearest Neghbor
        start = time.time()
        
        print('Build repo time: {0:0.3f}'.format(time.time()-start))    
        
    def build_repo_noKDD(self):
        
        # build k-Nearest Neghbor
        start = time.time()
        
        self.label_dict = np.array(self.label_dict)
        print(f'Number of labels {len(self.label_dict)}')        
        
        # perform PCA first, store the pca embeddings, store the pca, store the KDD
        X = np.vstack(self.emb)
        
        X = X - np.mean(X, axis=1, keepdims=True)
        X = X / np.linalg.norm(X, axis=1, keepdims=True) # normalize
        self.label = np.concatenate(self.label, axis=0)
        self.emb = X.copy()
        X = []
        
        print('Build repo time: {0:0.3f}'.format(time.time()-start))        
        
    def get_one_kdd_dist(self, in_test_file, n1, n2, ids):
        
        X = self.read_feat_file(in_test_file, n1, n2)
        if self.pca is not None:
            X = self.pca.transform(X)
        
        X = X - np.mean(X, axis=1, keepdims=True)        
        X = X / np.linalg.norm(X, axis=1, keepdims=True) # normalize
        dist, pred_id = self.knnrepo.query(X, k=1)
        pred_lbl = self.label_dict[self.label[pred_id]].copy()
        file_list = [f'{in_test_file}_{i}' for i in range(len(pred_id))]
        out_df = pd.DataFrame(data=pred_lbl, columns=['predLabel'])
        out_df['fileName'] = file_list.copy()
        out_df['actualLabel'] = ids
        out_df['dist'] = dist
        X = []; pred_id = []; pred_lbl = []; file_list  = []
        return out_df
    
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
        cur_max_dist = np.max(sim_mat, axis=0)
        cur_max_id = np.argmax(sim_mat, axis=0)
        file_list = [f'{in_test_file}_{i}' for i in range(len(cur_max_id))]
        out_df = pd.DataFrame(data=self.label_dict[self.label[cur_max_id]], columns=['predLabel'])
        out_df['fileName'] = file_list.copy()
        out_df['actualLabel'] = ids
        out_df['dist'] = 1-cur_max_dist
        X = []; X_pca = []; cur_max_id = []; cur_max_dist = []; file_list  = []
        return out_df    
        
    def dist_using_dict(self, in_test_dict, n1, n2, parallel=False, dist='kdd'):
        
        # for this test file return the dataframe with true label, pred label, file id
        # I'll store the index of the data frame as file name and the index of the feature
        # collect all the parallel struct
        full_struct = []
        for ids in in_test_dict.keys():
            
            # ensure this key should be in the repo
            assert np.any(ids==self.label_dict), f'test label should be in the repo dict {ids}'
            
            files = in_test_dict[ids]
            for f in files:
                full_struct.append((f, n1, n2, ids))
            
        # distance function
        func = self.dist_dict[dist]
        
        if parallel:            
            num_cores = multiprocessing.cpu_count()
            df_list = Parallel(n_jobs=num_cores, verbose=1)(delayed(func)(*full_struct[c]) for c in range(len(full_struct)))
            out_df = pd.concat(df_list, ignore_index=True, sort=False)
            df_list = []
        else:
            out_df = {}
            for i in range(len(full_struct)):
                print(full_struct[i])
                out_df[i] = func(*full_struct[i])
                
            out_df = pd.concat(list(out_df.values()), ignore_index=True, sort=False)
            
        return out_df

def get_stat(in_lbl, in_dist):
    
    if len(in_lbl) == 1:
        return np.array([in_lbl, in_dist])
    
    maj_id = Counter(in_lbl).most_common()[0][0]
    maj_dist = np.mean(in_dist[in_lbl==maj_id])
    return np.array([maj_id, maj_dist])
    

class Repo_maj_pool(object):
    """class to get the repo per label"""
    def __init__(self, bs_fldr, frames, step, pool_func, N_per_id):
        super(Repo_maj_pool, self).__init__()
        self.emb = [] # to store the embeddings
        self.label = [] # integer labels
        self.label_dict = [] # integer dict to string dict
        self.bs_fldr = bs_fldr
        self.frames = frames
        self.step = step
        self.pool_func = pool_func
        self.N_per_id = N_per_id
        self.pca = None
        self.pcaEmb = None
        self.knnrepo = None
        self.dist_dict = {'cosine': self.get_one_cosine_dist}
        
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
         
        n1 = int(n1*len(feat))
        n2 = int(n2*len(feat))
        feat = feat[n1:n2+1, :].copy()
                
        return feat
        
        
    # n1, n2 gives the part of the video to use
    def add_ids(self, in_file_dict, n1, n2):
        
        all_keys = list(in_file_dict.keys()) # list of all identities
        all_keys.sort()
        
        bas_label = len(self.label_dict)
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
                idx = []
                
            # add the labels and embeddings
            self.label = self.label + [np.zeros((len(all_id_feat), ), dtype=np.int32)+bas_label+k]
            self.label_dict = self.label_dict + [all_keys[k]]
            self.emb = self.emb + [all_id_feat]
            all_id_feat = []
            
        # need to build the repo again
        self.pca = None
        self.pcaEmb = None
        self.knnrepo = None
    
    def build_repo(self, N_pca_comp):
        
        # build k-Nearest Neghbor
        start = time.time()
        
        self.label_dict = np.array(self.label_dict)
        print(f'Number of labels {len(self.label_dict)}')        
        
        # perform PCA first, store the pca embeddings, store the pca, store the KDD
        X = np.vstack(self.emb)
        
        X = X - np.mean(X, axis=1, keepdims=True)
        X = X / np.linalg.norm(X, axis=1, keepdims=True) # normalize        
        if N_pca_comp > 0:
            self.pca = PCA(n_components=N_pca_comp, copy=True, whiten=False, svd_solver='auto', 
                           tol=0.0, iterated_power='auto', random_state=0).fit(X)
            print('pca components: {}, explained variance {}'.format(N_pca_comp, np.sum(self.pca.explained_variance_ratio_)))
            X = self.pca.transform(X)        
        
        self.label = np.concatenate(self.label, axis=0)
        self.emb = X.copy()
        X = []
        
        print('Build repo time: {0:0.3f}'.format(time.time()-start))
    
    # cosine distance
    def get_one_cosine_dist(self, in_test_file, n1, n2, ids):
        
        X = self.read_feat_file(in_test_file, n1, n2)
        
        if X is None or len(X) < (self.frames + self.step):
            X = []
            print(in_test_file)
            out_df = pd.DataFrame({'predLabel':[], 'fileName':[], 'actualLabel':[], 'dist':[]})
            return out_df
        
        X = X - np.mean(X, axis=1, keepdims=True)
        X = X / np.linalg.norm(X, axis=1, keepdims=True) # normalize        
        if self.pca is not None:
            X = self.pca.transform(X)

        sim_mat = np.matmul(self.emb, X.T)
        
        cur_max_dist = np.max(sim_mat, axis=0)[:, np.newaxis]
        cur_max_id = np.argmax(sim_mat, axis=0)
        pred_lbl = self.label[cur_max_id][:, np.newaxis]
        
        # get the majority class for the pool params
        pool_lbl = im2col_sliding_strided(pred_lbl, (self.frames, 1), stepsize=self.step).T
        pool_dist = im2col_sliding_strided(cur_max_dist, (self.frames, 1), stepsize=self.step).T
        
        info = np.array([get_stat(pool_lbl[i, :], pool_dist[i, :]) for i in range(len(pool_dist))])

        file_list = [f'{in_test_file}_{i}' for i in range(len(info))]
        out_df = pd.DataFrame(data=self.label_dict[info[:, 0].astype(np.int32)], columns=['predLabel'])
        out_df['fileName'] = file_list.copy()
        out_df['actualLabel'] = ids
        out_df['dist'] = 1-info[:, 1]
        X = []; X_pca = []; cur_max_id = []; cur_max_dist = []; file_list  = [];
        pred_lbl = []; pool_lbl = []; pool_dist = []; info = []
        return out_df    
        
    def dist_using_dict(self, in_test_dict, n1, n2, parallel=False, dist='cosine'):
        
        # for this test file return the dataframe with true label, pred label, file id
        # I'll store the index of the data frame as file name and the index of the feature
        # collect all the parallel struct
        full_struct = []
        for ids in in_test_dict.keys():
            
            # ensure this key should be in the repo
            assert np.any(ids==self.label_dict), f'test label should be in the repo dict {ids}'
            
            files = in_test_dict[ids]
            for f in files:
                full_struct.append((f, n1, n2, ids))
            
        # distance function
        func = self.dist_dict[dist]
        
        if parallel:            
            num_cores = multiprocessing.cpu_count()
            df_list = Parallel(n_jobs=num_cores, verbose=1)(delayed(func)(*full_struct[c]) for c in range(len(full_struct)))
            out_df = pd.concat(df_list, ignore_index=True, sort=False)
            df_list = []
        else:
            out_df = {}
            for i in range(len(full_struct)):
                print(full_struct[i])
                out_df[i] = func(*full_struct[i])
                
            out_df = pd.concat(list(out_df.values()), ignore_index=True, sort=False)
            
        return out_df
