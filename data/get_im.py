import os
import argparse
import multiprocessing
from joblib import Parallel, delayed
import time
import numpy as np
from synthetic_fakes import utils
import dlib
import cv2 
from keras.applications import imagenet_utils

"""
Usage: python -W ignore get_au.py --bsfldr '/data/home/shruti/voxceleb/videos/vox2_mp4/dev/mp4' --ofd '/data/home/shruti/voxceleb/vgg/vox2_mp4' --njobs 60 --openface 'OpenFace-master/build/bin' --fnmodel 'VGG_FACE.t7' --path_file 'utils/data_subset_0.txt'

this file extracts the face embeddings for each mp4 file present in the base directory and sub directories. It saves the embeddings into a csv format and saves it in the same folder as input mp4 file.

# requirements: It needs the OpenFace bin folder (and all the model files) in OpenFaceBin folder at the same level as this directory
# It also need the vgg net model downloaded from http://www.robots.ox.ac.uk/~vgg/software/vgg_face/  kept in the folder"""


def one_vid_feat(in_fldr, fl_n, out_file, front_face_detector, lmark_predictor):
    
    # load the model for frame embeddings
    cur_vid = os.path.join(in_fldr, fl_n + '.mp4')
    assert not os.path.exists(out_file)
    resize = (224, 224)
       
    try:
        
        vidcap = cv2.VideoCapture(cur_vid)
        
        out_im = []
        ret, frame = vidcap.read()
        i = 0
        while ret:
            
            # Cut out head region
            faces = utils.align(frame[:, :, (2,1,0)], front_face_detector, lmark_predictor, scale=0)
            if len(faces) == 0:
                ret, frame = vidcap.read()
                continue
            else:
                trans_matrix, point = faces[0]           
            
            ims, _ = utils.cut_head([frame], point)
            im = cv2.resize(ims[0], (resize[0], resize[1]))
            image = np.expand_dims(im, axis=0)            
            image = imagenet_utils.preprocess_input(image)
                        
            out_im.append(image)
            ret, frame = vidcap.read()
            i = i+1
        
        out_emb = np.vstack(out_im)
        # save the output
        np.save(out_file, out_emb.astype(np.float32))
        
        vidcap.release()
        
    except Exception as e:
        print('{} {}'.format(cur_vid, e))
    

def _set_up_dlib(path):
    # self.cnn_face_detector = dlib.cnn_face_detection_model_v1(path + '/mmod_human_face_detector.dat')
    front_face_detector = dlib.get_frontal_face_detector()
    lmark_predictor = dlib.shape_predictor(path + '/shape_predictor_68_face_landmarks.dat')
    
    return front_face_detector, lmark_predictor

if __name__ == '__main__':

    #read all the txt file
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--bsfldr', type=str, help='path to folder with mp4 files', default='data')
    parser.add_argument('--njobs', type=int, help='how many jobs to run per folder', default=1)
    parser.add_argument('--ofd', type=str, help='base path to output the feature files')
    parser.add_argument('--path_file', type=str, help='the file names to execute on this machine, all is None', default=None)

    args = parser.parse_args()
    bs_fldr = args.bsfldr
    njobs = args.njobs
    ofd = args.ofd
    paths_file = args.path_file
    
    # collect all the video files to process
    full_struct = []
    with open(paths_file, 'r')  as f:
        paths = f.readlines()

    front_face_detector, lmark_predictor = _set_up_dlib('synthetic_fakes/models')

    for p in paths:
        
        [dirname, vid_file] = os.path.split(p)

        # if there are mp4 files then extract embeddings from this folder
        fl_n = os.path.splitext(vid_file)[0] # the base file name

        # create the base output directory
        os.makedirs(os.path.join(ofd, dirname), exist_ok=True)
        out_file = os.path.join(ofd, dirname, fl_n + '.npy')
            
        if not os.path.exists(out_file):
            full_struct.append((os.path.join(bs_fldr, dirname), fl_n, out_file, front_face_detector, lmark_predictor))

    # run the jobs in parallel
    start_time = time.time()
    num_cores = multiprocessing.cpu_count()
    njobs = np.min([num_cores, njobs])
    print('Number of Cores {} \n'.format(num_cores))
    print('total processes {}'.format(len(full_struct)))
    Parallel(n_jobs=njobs, verbose=20)(delayed(one_vid_feat)(*full_struct[c]) for c in range(len(full_struct)))
    print('end time {} seconds'.format(time.time() - start_time))	
