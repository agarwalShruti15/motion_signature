import os
import natsort
import argparse
import multiprocessing
from joblib import Parallel, delayed
import numpy as np


def resave_one_vid(cur_vid, cur_out_vid):
    
    os.system('ffmpeg -i {} -qp 40 -r 25 -y -hide_banner -loglevel panic {}'.format(cur_vid, cur_out_vid))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--bsfldr', type=str, help='base folder with features for which correlation needs to be computed', default='data')
    parser.add_argument('--njobs', type=int, help='how many jobs to run per folder', default=1)
    parser.add_argument('--ofd', type=str, help='base path to output the feature files')
    parser.add_argument('--path_file', type=str, help='path file with relative paths')
    
    args = parser.parse_args()
    njobs = args.njobs
    bs_fldr = args.bsfldr
    path_file = args.path_file
    ofd = args.ofd
    
    # params
    print(bs_fldr, ofd, path_file)
    
    # create the base output directory
    os.makedirs(ofd, exist_ok=True)
    
    # collect all the video files to process
    full_struct = []
    with open(path_file, 'r')  as f:
            paths = f.readlines()
            
    for p in paths:
            [dirname, vid_file] = os.path.split(p)
            
            # if there are mp4 files then extract embeddings from this folder
            fl_n = os.path.splitext(vid_file)[0] # the base file name
            
            # outfile
            os.makedirs(os.path.join(ofd, dirname), exist_ok=True)
            out_file = os.path.join(ofd, dirname, fl_n + '.mp4')                
            #out_file = os.path.join(ofd, '_'.join(dirname.split('/')) + '_' + fl_n + '.npy')
            
            if not os.path.exists(out_file):
                    full_struct.append((os.path.join(bs_fldr, dirname, fl_n + '.mp4'), out_file))
                    
    # run the jobs in parallel
    num_cores = multiprocessing.cpu_count()
    njobs = np.min([num_cores, njobs])
    print('Number of Cores {} \n'.format(num_cores))
    print('total processes {}'.format(len(full_struct)))
    Parallel(n_jobs=njobs, verbose=20)(delayed(resave_one_vid)(*full_struct[c]) for c in range(len(full_struct)))