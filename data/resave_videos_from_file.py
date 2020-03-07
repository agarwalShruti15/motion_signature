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
    parser.add_argument('--path_file', type=str, help='the file names to execute on this machine, all is None', default=None)           
    parser.add_argument('--ofd', type=str, help='base path to output the feature files')
    
    args = parser.parse_args()
    bs_fldr = args.bsfldr
    ofd = args.ofd
    paths_file = args.path_file
    
    # create the base output directory
    os.makedirs(ofd, exist_ok=True)
    
    # collect all the video files to process
    full_struct = []
    with open(paths_file, 'r')  as f:
        paths = f.readlines()

    for p in paths:
        [dirname, vid_file] = os.path.split(p)

        # if there are mp4 files then extract embeddings from this folder
        fl_n = os.path.splitext(vid_file)[0] # the base file name
        out_fldr = os.path.join(ofd,dirname)
        
        os.makedirs(out_fldr, exist_ok=True)
        out_file = os.path.join(out_fldr, fl_n + '.mp4')
        if not os.path.exists(out_file):
            full_struct.append((os.path.join(bs_fldr, dirname, fl_n + '.mp4'), out_file))
                    
    # run the jobs in parallel
    for c in range(len(full_struct)):
        print(c, len(full_struct))
        resave_one_vid(*full_struct[c])