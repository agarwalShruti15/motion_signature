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
    
    args = parser.parse_args()
    njobs = args.njobs
    bs_fldr = args.bsfldr
    ofd = args.ofd
        
    # create the base output directory
    os.makedirs(ofd, exist_ok=True)
    
    # collect all the video files to process
    full_struct = []
    for dirname, dirnames, filenames in os.walk(bs_fldr):

        # if there are mp4 files then extract embeddings from this folder
        vid_files = [v for v in os.listdir(dirname) if v.endswith('.mp4')]  # get all the videos
        for vi in range(len(vid_files)):
            fl_n = os.path.splitext(vid_files[vi])[0] # the base file name

            # outfile
            if bs_fldr[-1] == '/':
                out_fldr = os.path.join(ofd, dirname[len(bs_fldr):]) # backslash
            else:
                out_fldr = os.path.join(ofd, dirname[len(bs_fldr)+1:]) # no backlash in basefolder name

            os.makedirs(out_fldr, exist_ok=True)
            out_file = os.path.join(out_fldr, fl_n + '.mp4')
            if not os.path.exists(out_file):
                full_struct.append((os.path.join(dirname, fl_n + '.mp4'), out_file))
                    
    # run the jobs in parallel
    num_cores = multiprocessing.cpu_count()
    njobs = np.min([num_cores, njobs])
    print('Number of Cores {} \n'.format(num_cores))
    print('total processes {}'.format(len(full_struct)))
    Parallel(n_jobs=njobs, verbose=20)(delayed(resave_one_vid)(*full_struct[c]) for c in range(len(full_struct)))