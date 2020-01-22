""" combine the transcript file with the landmark, au csv file

"""

import numpy as np
import os
import argparse
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from itertools import combinations
import re
import pickle
import multiprocessing
from joblib import Parallel, delayed


def save_obj(obj, fldr, name ):
    os.makedirs(fldr, exist_ok=True)
    with open(os.path.join(fldr, name + '.pkl'), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(fldr, name ):
    if not os.path.exists(os.path.join(fldr, name + '.pkl')):
        return None
    with open(os.path.join(fldr, name + '.pkl'), 'rb') as f:
        return pickle.load(f)

def get_second_col_df(in_feat, cor_feat):
    # normalize for correlation
    feat_df = in_feat - np.mean(in_feat, axis=0, keepdims=True)
    feat_df = feat_df / np.linalg.norm(feat_df, axis=0, keepdims=True)
    
    return feat_df[:, cor_feat[:, 0]], feat_df[:, cor_feat[:, 1]]


def get_pca_train_data_ldrs(bs_fldr, train_file, T, N=100000):
    
    # collect paths
    path_files = []
    with open(os.path.join(bs_fldr, train_file), 'r') as f:
        for line in f:
            _path, _label = re.split(r",| ", line.strip())
            path_files.extend([_path])
    if N>0:
        path_files = np.random.choice(path_files, N, replace=False)
        
    out_data = {}
    for p in range(len(path_files)):
        
        cur_data = np.load(os.path.join(bs_fldr, path_files[p]))
        out_data[p] = cur_data.copy()
    
    return np.vstack(list(out_data.values()))

def get_pca_train_data(bs_fldr, train_file, T, N=100000):
    
    # collect paths
    path_files = []
    with open(os.path.join(bs_fldr, train_file), 'r') as f:
        for line in f:
            _path, _label = re.split(r",| ", line.strip())
            path_files.extend([_path])
    if N>0:
        path_files = np.random.choice(path_files, N, replace=False)
        
    out_data = {}
    cor_feat = None
    for p in range(len(path_files)):
        
        cur_data = np.load(os.path.join(bs_fldr, path_files[p]))
        idx = np.random.choice(range(len(cur_data)-T+1), 1)[0]
        cur_data = cur_data[idx:idx+T, :].copy()
        if cor_feat is None:
            all_feat_comb = list(combinations(range(cur_data.shape[1]), 2))
            cor_feat = np.array([[j[0], j[1]] for j in all_feat_comb])
        
        f, s = get_second_col_df(cur_data, cor_feat)
        cur_cor = np.sum(f * s, axis=0, keepdims=True)
        if np.sum(np.isnan(cur_cor)) > 0:
            continue
        out_data[p] = cur_cor.copy()
    
    return np.vstack(list(out_data.values()))

def one_file_pca(infile, outfile, pca_model):
    
    cur_feat = np.load(infile)
    cur_feat = cur_feat[np.sum(np.isnan(cur_feat), axis=1)<1, :].copy()
    pca_feat = pca_model['pca_model'].transform(pca_model['pca_scalar'].transform(cur_feat))
    np.save(outfile, pca_feat.astype(np.float32))

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--ptf', type=str, help='file with relative path to npy files for pcs training')
    parser.add_argument('--ncomp', type=np.int32, help='number of components for PCA')
    parser.add_argument('--T', type=np.int32, help='number of frames to compute corr')
    parser.add_argument('--njobs', type=np.int32, help='number of parallel jobs', default=1)
    parser.add_argument('--test_f', type=str, help='path to test folder')
    parser.add_argument('--ofd', type=str, help='out base folder')

    args = parser.parse_args()
    train_file = args.ptf
    root, _ = os.path.split(train_file)
    ncomps = args.ncomp
    T = args.T
    test_f = args.test_f
    ofd = args.ofd
    njobs = args.njobs
    
    # compute the pca
    # 1) collect the train data for PCA
    pca_train_data = load_obj('.', 'pca_train_data')
    if pca_train_data is None:
        pca_train_data = get_pca_train_data_ldrs(root, train_file, T, N=1000) # get the features to compute pca on
        save_obj(pca_train_data, '.', 'pca_train_data')
        
    print('pca train shape ', pca_train_data.shape)
    # 2) perform the PCA
    pca_dict = load_obj('.', f'pca_model_{T}_{ncomps}')
    if pca_dict is None:
        scalar = StandardScaler().fit(pca_train_data)
        X_train_norm = scalar.transform(pca_train_data)
        pca = PCA(n_components=ncomps, copy=True, whiten=False, 
                  svd_solver='auto', tol=0.0, iterated_power='auto', random_state=None)
        pca = pca.fit(X_train_norm)
        pca_dict = {'pca_model': pca, 'pca_scalar': scalar}
        save_obj(pca_dict, '.', f'pca_model_{T}_{ncomps}')
    
    pca_train_data = []
    
    # transform all the files to transform to pca
    full_struct = []    
    for dirname, dirnames, filenames in os.walk(test_f):

        # if there are mp4 files then extract embeddings from this folder
        files = [v for v in os.listdir(dirname) if v.endswith('.npy')]  # get all the videos
        for vi in range(len(files)):
            
            fl_n = os.path.splitext(files[vi])[0] # the base file name
            if test_f[-1] == '/':
                out_fldr = os.path.join(ofd, dirname[len(test_f):]) # backslash
            else:
                out_fldr = os.path.join(ofd, dirname[len(test_f)+1:]) # no backlash in basefolder name
            
            os.makedirs(out_fldr, exist_ok=True)
            out_file = os.path.join(out_fldr, fl_n + '.npy')
            
            if not os.path.exists(out_file):
                full_struct.append((os.path.join(dirname, files[vi]), out_file, pca_dict))

    # run the jobs in parallel
    num_cores = multiprocessing.cpu_count()
    njobs = np.min([num_cores, njobs])
    print('Number of Cores {} \n'.format(num_cores))
    print('total processes {}'.format(len(full_struct)))
    Parallel(n_jobs=njobs, verbose=20)(delayed(one_file_pca)(*full_struct[c]) for c in range(len(full_struct)))
