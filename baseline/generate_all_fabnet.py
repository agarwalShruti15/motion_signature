"""
Two seperate features for correlation and feat var. Can choose to do either or both

"""

import numpy as np
import os
from itertools import combinations
import multiprocessing
from joblib import Parallel, delayed


def get_second_col_df(in_feat_df, j, vid_len, first_col, sec_col):
    # normalize for correlation
    feat_df = in_feat_df[j:j+vid_len, :].copy()
    feat_df = feat_df - np.mean(feat_df, axis=0, keepdims=True)
    feat_df = feat_df / np.linalg.norm(feat_df, axis=0, keepdims=True)
    
    return feat_df[:, first_col], feat_df[:, sec_col]


def get_feats(bs_fldr, infldr, infile, obj_fldr, cor_feat, vid_len=300, ovrlp_win=5):
    
    first_col = np.array(cor_feat)[:, 0]; sec_col = np.array(cor_feat)[:, 1]   
    cur_feat = np.load(os.path.join(bs_fldr, infldr, infile))
    if len(cur_feat.shape)>2:
        cur_feat = cur_feat[:, :, 0].copy()
    
    #normalize the fabnet to unit sphere
    cur_feat = cur_feat/np.linalg.norm(cur_feat, axis=1, keepdims=True)
    
    if cur_feat.shape[0]>(vid_len+ovrlp_win):
        frame_rng = np.arange(0, len(cur_feat)-vid_len, ovrlp_win)        
        for j in frame_rng:
            f, s = get_second_col_df(cur_feat, j, vid_len, first_col, sec_col)
            tmp_corr = np.sum(f * s, axis=0)
            np.save(os.path.join(obj_fldr, os.path.splitext(infile)[0] + f'_{j}.npy'), tmp_corr)
    else:
        print('Length smaller than', vid_len, infldr, infile)

if __name__ == '__main__':

    #input files to compare
    njobs = 30
    bs_fldr = '/data/home/shruti/voxceleb/fabnet/leaders' 
    out_bs_fldr = '/data/home/shruti/voxceleb/'
    vid_len_all = [int(10*30), 100]
    vid_fldrs = ['bo', 'bo_UWfake', 'bo_faceswap',
                 'hc', 'hc_faceswap', 'hc_imposter', 'diff_test',
                 'ew', 'ew_faceswap', 'ew_imposter',
                 'bs', 'bs_faceswap', 'bs_imposter', 
                 'bo_imposter', 'dt_week', 'dt_rndm', 
                 'dt_faceswap', 'dt_imposter', 'br', 
                 'pb', 'kh', 'jb', 'cb', 'trump_fake']
    
    ovrlp_win = 5
    all_feat_comb = list(combinations(range(256), 2))    #these correlation features to consider
    cor_feat = np.array([[j[0], j[1]] for j in all_feat_comb])
    
    full_struct = []
    for vid_len in vid_len_all:
        
        #get training features and fit a model
        for f in range(len(vid_fldrs)):
            
            out_fldr = os.path.join(out_bs_fldr, f'fabnet_corr_{vid_len}', vid_fldrs[f])
            os.makedirs(out_fldr, exist_ok=True)

            npy_files = [f for f in os.listdir(os.path.join(bs_fldr, vid_fldrs[f])) if f.endswith('.npy')]; npy_files.sort()

            for infile in npy_files:
                if not os.path.exists(os.path.join(out_fldr, os.path.splitext(infile)[0] + '.npy')):
                    full_struct.append((bs_fldr, vid_fldrs[f], infile, out_fldr, cor_feat, vid_len, 5))
                                       
    # run the jobs in parallel
    num_cores = multiprocessing.cpu_count()
    njobs = np.min([num_cores, njobs])
    print('Number of Cores {} \n'.format(num_cores))
    print('total processes {}'.format(len(full_struct)))
    Parallel(n_jobs=njobs, verbose=20)(delayed(get_feats)(*full_struct[c]) for c in range(len(full_struct)))
