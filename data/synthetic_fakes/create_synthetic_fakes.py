"""
Exposing DeepFake Videos By Detecting Face Warping Artifacts
Yuezun Li, Siwei Lyu
https://arxiv.org/abs/1811.00656
"""
import numpy as np
import utils as lib
import dlib
import proc_vid
import time
import multiprocessing
from joblib import Parallel, delayed
import os

def generate_one_file_fake(in_vid_file, out_path, front_face_detector, lmark_predictor):
    
    # get all the video frames
    imgs, frame_num, fps, width, height = proc_vid.parse_vid(in_vid_file)
    
    # keep only polygon
    rnd2 = np.random.uniform()
    keep_polygon = False
    if rnd2 < 0.5:
        keep_polygon = True

    affine_imgs = []
    for i in range(frame_num):
        
        faces = lib.align(imgs[i][:, :, (2,1,0)], front_face_detector, lmark_predictor, scale=0)
        if len(faces) == 0:
            affine_imgs.append(imgs[i])
            continue
        else:
            trans_matrix, point = faces[0]           

            # Affine warp face area back
            size = np.arange(64, 128, dtype=np.int32)
            c = np.random.randint(0, len(size))
            new_im = lib._face_blur(imgs[i], trans_matrix, size=size[c])
            """if keep_polygon:            
                # Only retain a minimal polygon mask
                part_mask = lib.get_face_mask(imgs[i].shape[:2], point)
                # Select specific blurred part
                new_im = lib._select_part_to_blur(imgs[i], new_im, part_mask)"""
            
            affine_imgs.append(new_im)
            
    proc_vid.gen_vid(out_path, affine_imgs, fps, width=width, height=height)

def _set_up_dlib(path):
    # self.cnn_face_detector = dlib.cnn_face_detection_model_v1(path + '/mmod_human_face_detector.dat')
    front_face_detector = dlib.get_frontal_face_detector()
    lmark_predictor = dlib.shape_predictor(path + '/shape_predictor_68_face_landmarks.dat')
    
    return front_face_detector, lmark_predictor


if __name__ == '__main__':
        
    # base folder
    bs_fldr = '/data/home/shruti/voxceleb/videos/leaders/'
    sbfldrs = ['bo', 'bs', 'hc', 'ew', 'dt_rndm', 'br', 'cb', 'jb', 'pb', 'kh']
    N = 100 # number of synthetic fake videos to create for each subfldrs
    out_bs_fldr = bs_fldr
    njobs = 5
    
    # get the dlib models
    front_face_detector, lmark_predictor = _set_up_dlib('models')
    
    # collect all the jobs
    full_struct = []
    for sb in sbfldrs:
        
        vid_files = [f for f in sorted(os.listdir(os.path.join(bs_fldr, sb))) if f.endswith('.mp4')]
        vid_files = vid_files[:N].copy()
        
        for f in vid_files:
            
            cur_out_file = os.path.join(out_bs_fldr, sb + '_syn_fake', f)
            os.makedirs(os.path.join(out_bs_fldr, sb + '_syn_fake'), exist_ok=True)
            cur_in_file = os.path.join(bs_fldr, sb, f)
            if not os.path.exists(cur_out_file):
                full_struct.append((cur_in_file, cur_out_file, front_face_detector, lmark_predictor))
    
    # run the jobs in parallel
    start_time = time.time()
    num_cores = multiprocessing.cpu_count()
    njobs = np.min([num_cores, njobs])
    print('Number of Cores {} \n'.format(num_cores))
    print('total processes {}'.format(len(full_struct)))
    Parallel(n_jobs=njobs, verbose=20)(delayed(generate_one_file_fake)(*full_struct[c]) for c in range(len(full_struct)))
    print('end time {} seconds'.format(time.time() - start_time))	    
