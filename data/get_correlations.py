"""
Two seperate features for correlation and feat var. Can choose to do either or both

"""

import numpy as np
import os
from itertools import combinations
import multiprocessing
from joblib import Parallel, delayed
import argparse


def get_second_col_df(in_feat_df, j, vid_len, first_col, sec_col, unit_sphere):
    # normalize for correlation
    feat_df = in_feat_df[j:j+vid_len, :].copy()
    if unit_sphere:
        feat_df = feat_df / np.linalg.norm(feat_df, axis=1, keepdims=True)
    feat_df = feat_df - np.nanmean(feat_df, axis=0, keepdims=True)
    feat_df = feat_df / np.linalg.norm(feat_df, axis=0, keepdims=True)
    
    return feat_df[:, first_col], feat_df[:, sec_col]


def get_feats(infldr, fl_n, out_file, vid_len, ovrlp_win, unit_sphere):
    
    cur_feat = np.load(os.path.join(infldr, fl_n + '.npy'))
    if len(cur_feat.shape)>2:
        cur_feat = cur_feat[:, :, 0].copy()
    
    #these correlation features to consider
    all_feat_comb = list(combinations(range(cur_feat.shape[1]), 2))
    cor_feat = np.array([[j[0], j[1]] for j in all_feat_comb])        
    first_col = np.array(cor_feat)[:, 0]; sec_col = np.array(cor_feat)[:, 1] 
        
    if cur_feat.shape[0]>(vid_len+ovrlp_win):
        frame_rng = np.arange(0, len(cur_feat)-vid_len, ovrlp_win)
        out_feat = np.zeros((len(frame_rng), len(cor_feat)))
        for j in range(len(frame_rng)):            
            f, s = get_second_col_df(cur_feat, frame_rng[j], vid_len, first_col, sec_col, unit_sphere)
            out_feat[j, :] = np.sum(f * s, axis=0)
            
        np.save(out_file, out_feat)
    else:
        print('Length smaller than', vid_len, infldr, fl_n)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--bsfldr', type=str, help='base folder with features for which correlation needs to be computed', default='data')
    parser.add_argument('--njobs', type=int, help='how many jobs to run per folder', default=1)
    parser.add_argument('--ofd', type=str, help='base path to output the feature files')
    parser.add_argument('--vid_len', type=np.int32, help='number of frames over which corr is computed', default=100)
    parser.add_argument('--ow', type=np.int32, help='frames shift for next corr window', default=5)
    parser.add_argument('--us', type=np.int32, help='unit sphere normalization')    

    args = parser.parse_args()
    njobs = args.njobs
    bs_fldr = args.bsfldr
    vid_len = args.vid_len
    ovrlp_win = args.ow
    ofd = args.ofd
    unit_sphere = args.us==1
    
    print(bs_fldr, vid_len, ofd, unit_sphere)
    
    # collect all the files to process
    full_struct = []    
    for dirname, dirnames, filenames in os.walk(bs_fldr):

        # if there are mp4 files then extract embeddings from this folder
        files = [v for v in os.listdir(dirname) if v.endswith('.npy')]  # get all the videos
        
        for vi in range(len(files)):
            
            fl_n = os.path.splitext(files[vi])[0] # the base file name
            if bs_fldr[-1] == '/':
                out_fldr = os.path.join(ofd, dirname[len(bs_fldr):]) # backslash
            else:
                out_fldr = os.path.join(ofd, dirname[len(bs_fldr)+1:]) # no backlash in basefolder name
            os.makedirs(out_fldr, exist_ok=True)
            
            out_file = os.path.join(out_fldr, fl_n + '.npy')
            
            if not os.path.exists(out_file):
                full_struct.append((dirname, fl_n, out_file, vid_len, ovrlp_win, unit_sphere))
                                       
    # run the jobs in parallel
    num_cores = multiprocessing.cpu_count()
    njobs = np.min([num_cores, njobs])
    print('Number of Cores {} \n'.format(num_cores))
    print('total processes {}'.format(len(full_struct)))
    Parallel(n_jobs=njobs, verbose=20)(delayed(get_feats)(*full_struct[c]) for c in range(len(full_struct)))
