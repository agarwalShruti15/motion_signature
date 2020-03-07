import os
import natsort
import argparse
import multiprocessing
from joblib import Parallel, delayed
import numpy as np
import pickle

def resave_one_vid(cur_vid, cur_out_vid):
    
    os.system('ffmpeg -i {} -qp 40 -r 25 -vf scale="iw/1:ih/2" -y -hide_banner -loglevel panic {}'.format(cur_vid, cur_out_vid))


def load_obj(path):
    if not path:
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--bsfldr', type=str, help='base folder with features for which correlation needs to be computed', default='data')
    parser.add_argument('--file_dict_path', type=str, help='path to file dictionary', default='data')
    parser.add_argument('--ofd', type=str, help='base path to output the feature files')
    
    args = parser.parse_args()
    bs_fldr = args.bsfldr
    file_dict_path = args.file_dict_path
    ofd = args.ofd
    
    print(file_dict_path)
    file_dict = load_obj(file_dict_path)
    # create the base output directory
    os.makedirs(ofd, exist_ok=True)
    
    # collect all the video files to process
    full_struct = []
    for r_or_f in ['real', 'fake']:
        
        for ids in file_dict[r_or_f].keys():
            
            for f in file_dict[r_or_f][ids]:
                
                fl_n = os.path.splitext(f.split('/')[-1])[0] + '.mp4'# the base file name
                out_fldr = os.path.join(ofd, '/'.join(f.split('/')[:-1]))
                os.makedirs(out_fldr, exist_ok=True)
                out_file = os.path.join(out_fldr, fl_n)
                if not os.path.exists(out_file):
                    print((os.path.join(bs_fldr, '/'.join(f.split('/')[:-1]), fl_n ), out_file))
                    full_struct.append((os.path.join(bs_fldr, '/'.join(f.split('/')[:-1]), fl_n), out_file))
                                       
                                       
    # run the jobs in parallel
    for c in range(len(full_struct)):
        print(c, len(full_struct))
        resave_one_vid(*full_struct[c])