import os
import argparse
import utils as u
from models_multiview import FrontaliseModelMasks_wider
import torch
import multiprocessing
from joblib import Parallel, delayed
import time
import numpy as np

"""
Usage: python -W ignore extract_subset_fabnet.py --bsfldr '/data/home/shruti/voxceleb/videos/vox2_mp4/dev/mp4' --ofd '/data/home/shruti/voxceleb/fabnet/vox2_mp4' --njobs 60 --openface 'OpenFace-master/build/bin' --fnmodel 'release/nv2_curriculum.pth' --path_file 'utils/data_subset_0.txt' > out 2>error

this file extracts the face embeddings for each mp4 file present in the base directory and sub directories. It saves the embeddings into a csv format and saves it in the same folder as input mp4 file.

# requirements: It needs the OpenFace bin folder (and all the model files) in OpenFaceBin folder at the same level as this directory
# It also need the fab net model downloaded from http://www.robots.ox.ac.uk/~vgg/research/unsup_learn_watch_faces/release_bmvc_fabnet.zip and kept in release folder"""

num_additional_ids=32

def boolean_string(s):
        if s not in {'False', 'True'}:
                raise ValueError('Not a valid boolean string')
        return s == 'True'

if __name__ == '__main__':

        #read all the txt file
        parser = argparse.ArgumentParser(description='')
        parser.add_argument('--bsfldr', type=str, help='path to folder with mp4 files', default='data')
        parser.add_argument('--njobs', type=int, help='how many jobs to run per folder', default=1)
        parser.add_argument('--openface', type=str, help='path to openface build/bin folder', default='OpenFace-master/build/bin')    
        parser.add_argument('--fnmodel', type=str, help='path to fab-net model', default='release')    
        parser.add_argument('--ofd', type=str, help='base path to output the feature files')
        parser.add_argument('--path_file', type=str, help='the file names to execute on this machine, all is None', default=None)
        parser.add_argument('--gpu', type=boolean_string, help='use gpu or not', default=True)

        args = parser.parse_args()
        bs_fldr = args.bsfldr
        njobs = args.njobs
        openface_path = args.openface
        fabnet_path = args.fnmodel
        ofd = args.ofd
        paths_file = args.path_file
        gpu = args.gpu
        
        # create the base output directory
        os.makedirs(ofd, exist_ok=True)
        
        # load the model for frame embeddings
        emb_model = FrontaliseModelMasks_wider(3, inner_nc=u.emb_n, num_additional_ids=num_additional_ids)
        if gpu:
                emb_model.load_state_dict(torch.load(fabnet_path)['state_dict'])
        else:
                emb_model.load_state_dict(torch.load(fabnet_path, map_location='cpu')['state_dict'])
        emb_model.eval()
        
        # collect all the video files to process
        full_struct = []
        with open(paths_file, 'r')  as f:
                paths = f.readlines()
                
        for p in paths:
                [dirname, vid_file] = os.path.split(p)
                
                # if there are mp4 files then extract embeddings from this folder
                fl_n = os.path.splitext(vid_file)[0] # the base file name
                out_file = os.path.join(ofd, '_'.join(dirname.split('/')) + '_' + fl_n + '.npy')
                if not os.path.exists(out_file):
                        full_struct.append((os.path.join(bs_fldr, dirname), fl_n, out_file, emb_model, openface_path))
                        
        # run the jobs in parallel
        start_time = time.time()
        num_cores = multiprocessing.cpu_count()
        njobs = np.min([num_cores, njobs])
        print('Number of Cores {} \n'.format(num_cores))
        print('total processes {}'.format(len(full_struct)))
        Parallel(n_jobs=njobs, verbose=20)(delayed(u.fabnet_one_vid)(*full_struct[c]) for c in range(len(full_struct)))
        print('end time {} seconds'.format(time.time() - start_time))	