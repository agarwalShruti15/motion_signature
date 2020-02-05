import os
import argparse
import utils as u
import multiprocessing
from joblib import Parallel, delayed
import time
import numpy as np

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
    parser.add_argument('--fnmodel', type=str, help='path to vgg pretrained model', default='release')    
    parser.add_argument('--ofd', type=str, help='base path to output the feature files')
    parser.add_argument('--gpu', type=boolean_string, help='use gpu or not', default=True)

    args = parser.parse_args()
    bs_fldr = args.bsfldr
    njobs = args.njobs
    openface_path = args.openface
    vgg_path = args.fnmodel
    ofd = args.ofd
    gpu = args.gpu

    # create the base output directory
    os.makedirs(ofd, exist_ok=True)
    
    # load the model for frame embeddings
    vgg_model = vgg_face_dag(vgg_path)
    if torch.cuda.is_available():
        torch.device('cuda:0')
        vgg_model.cuda() #assumes that you're using GPU
    vgg_model.eval()    

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
            out_file = os.path.join(out_fldr, fl_n + '.npy')

            if not os.path.exists(out_file):
                print(out_file)
                full_struct.append((dirname, fl_n, out_file, vgg_model, openface_path, gpu))
                
    # run the jobs in parallel
    start_time = time.time()
    num_cores = multiprocessing.cpu_count()
    njobs = np.min([num_cores, njobs])
    print('Number of Cores {} \n'.format(num_cores))
    print('total processes {}'.format(len(full_struct)))
    Parallel(n_jobs=njobs, verbose=20)(delayed(u.vgg_one_vid)(*full_struct[c]) for c in range(len(full_struct)))
    print('end time {} seconds'.format(time.time() - start_time))	
