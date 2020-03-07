import os
import argparse
import utils as u
import multiprocessing
from joblib import Parallel, delayed
import time
import numpy as np

from models_multiview import FrontaliseModelMasks_wider
from vggface_model import vgg_face_dag
import torch

"""
Usage: python -W ignore extract_subset_fabnet.py --bsfldr '/data/home/shruti/voxceleb/videos/vox2_mp4/dev/mp4' --ofd '/data/home/shruti/voxceleb/vgg/vox2_mp4' --njobs 60 --openface 'OpenFace-master/build/bin' --fnmodel 'VGG_FACE.t7' --path_file 'utils/data_subset_0.txt'

this file extracts the face embeddings for each mp4 file present in the base directory and sub directories. It saves the embeddings into a csv format and saves it in the same folder as input mp4 file.

# requirements: It needs the OpenFace bin folder (and all the model files) in OpenFaceBin folder at the same level as this directory
# It also need the vgg net model downloaded from http://www.robots.ox.ac.uk/~vgg/software/vgg_face/  kept in the folder"""

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
    parser.add_argument('--of', type=str, help='base path to output the fabnet files')
    parser.add_argument('--ov', type=str, help='base path to output the vgg files')
    parser.add_argument('--path', type=str, help='path to video files txt')

    args = parser.parse_args()
    bs_fldr = args.bsfldr
    njobs = args.njobs
    openface_path = args.openface
    o_fab = args.of
    o_vgg = args.ov
    path_file = args.path
    
    vgg_path = 'vgg_face_dag.pth'
    fab_path = 'release/nv2_curriculum.pth'
    
    
    # create the base output directory
    os.makedirs(o_fab, exist_ok=True)
    os.makedirs(o_vgg, exist_ok=True)
    
    # load the model for frame embeddings
    vgg_model = vgg_face_dag(vgg_path)
    if torch.cuda.is_available():
        torch.device('cuda:0')
        vgg_model.cuda() #assumes that you're using GPU
    vgg_model.eval()
    
    # load the model for frame embeddings
    fab_model = FrontaliseModelMasks_wider(3, inner_nc=u.emb_n, num_additional_ids=num_additional_ids)
    if torch.cuda.is_available():
        fab_model.load_state_dict(torch.load(fab_path)['state_dict'])
    else:
        fab_model.load_state_dict(torch.load(fab_path, map_location='cpu')['state_dict'])
    fab_model.eval()

    # collect all the video files to process
    # collect all the video files to process
    full_struct = []
    with open(path_file, 'r')  as f:
        paths = f.readlines()

    for p in paths:
        [dirname, vid_file] = os.path.split(p)
        fl_n = os.path.splitext(vid_file)[0] # the base file name
        
        fab_outfile = os.path.join(o_fab, dirname, fl_n + '.npy')
        vgg_outfile = os.path.join(o_vgg, dirname, fl_n + '.npy')

        if not os.path.exists(fab_outfile) or not os.path.exists(vgg_outfile):
            full_struct.append((os.path.join(bs_fldr, dirname), fl_n, vgg_model, fab_model, openface_path, vgg_outfile, fab_outfile))
                    
    # run the jobs in parallel
    start_time = time.time()
    num_cores = multiprocessing.cpu_count()
    njobs = np.min([num_cores, njobs])
    print('Number of Cores {} \n'.format(num_cores))
    print('total processes {}'.format(len(full_struct)))
    Parallel(n_jobs=njobs, verbose=20)(delayed(u.vgg_fab_one_vid)(*full_struct[c]) for c in range(len(full_struct)))
    print('end time {} seconds'.format(time.time() - start_time))