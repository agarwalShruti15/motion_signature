import os
import multiprocessing
from joblib import Parallel, delayed
import numpy as np
import time
import argparse
from models_multiview import FrontaliseModelMasks_wider
import torch
import cv2
import dlib
from imutils import face_utils
import pandas as pd
from torchvision.transforms import Resize, Compose, ToTensor
from PIL import Image
import shutil
import matplotlib.pyplot as plt

emb_n = 256 # size of the output embedding 
crop_sz = 256 # the size of the image to input to the embedding model

num_additional_ids=32
model_file = 'release/nv2_curriculum.pth'
emb_nm = ['vid_emb' + str(i) for i in range(emb_n)]

def save_openface_frame(in_vid, out_fldr, size):

    #save the frames
    os.makedirs(out_fldr, exist_ok=True)
    cmd = './OpenFaceBin/FeatureExtraction -f {} -simsize {} -simalign -nomask -nobadaligned -out_dir {} > out.txt'.format(in_vid,size,out_fldr)
    os.system(cmd)
    
    return os.path.join(out_fldr, in_vid.split('/')[-1].split('.')[0] + '_aligned')


def get_video_frame_emb(frame, emb_model):
    
    try:
        # crop image and resize
        crop_im = Image.fromarray(frame)
    
        transform = Compose([ToTensor()])
        out_emb = emb_model.encoder(transform(crop_im).unsqueeze(0)).squeeze().detach().numpy()
        
    except Exception as e:
        out_emb = np.zeros((emb_n, ))
        print(e)
            
    return out_emb

def single_extract(in_fldr, fl_n, out_fldr, emb_model):
    out_file = os.path.join(out_fldr, fl_n + '.csv')  # output csv
    cur_vid = os.path.join(in_fldr, fl_n + '.mp4')
    tmp_fldr = 'aligned_faces'
    
    if not os.path.exists(out_file):
        try:
            
            frm_fldr = save_openface_frame(cur_vid, tmp_fldr, crop_sz)
            filenms = [f for f in os.listdir(frm_fldr) if f.endswith('.bmp')]
            
            # init the df
            out_df = pd.DataFrame(data=np.zeros((len(filenms), len(emb_nm))), columns=emb_nm)
                        
            for i in range(len(out_df)):
                filenm = 'frame_det_00_{0:06d}.bmp'.format(i+1)
                cur_frame = plt.imread(os.path.join(frm_fldr, filenm))
                out_df.loc[i, emb_nm] = get_video_frame_emb(cur_frame, emb_model)
                        
            shutil.rmtree(frm_fldr)
            os.remove(os.path.join(tmp_fldr, fl_n + '_of_details.txt'))
            os.remove(os.path.join(tmp_fldr, fl_n + '.csv'))            
            out_df.to_csv(out_file)
            
        except Exception as e:
            print('{} {}'.format(cur_vid, e))

def folder_extract(in_fldr, out_fldr, njobs=10):
    
    #fldrs = [os.path.join(bs_fldr, f) for f in os.listdir(bs_fldr) if os.path.isdir(os.path.join(bs_fldr, f))]
    os.makedirs(out_fldr, exist_ok=True)
    
    # load the model for frame embeddings
    emb_model = FrontaliseModelMasks_wider(3, inner_nc=emb_n, num_additional_ids=num_additional_ids)
    emb_model.load_state_dict(torch.load(model_file, map_location='cpu')['state_dict'])
    emb_model.eval()
    
    vid_files = [v for v in os.listdir(in_fldr) if v.endswith('.mp4')]  # get all the videos
    full_struct = np.zeros((len(vid_files),), dtype=np.object)
    i = 0
    for v in vid_files:
        full_struct[i] = (in_fldr, v.split('.')[0], out_fldr, emb_model)
        i = i+1

    start_time = time.time()
    num_cores = multiprocessing.cpu_count()
    njobs = np.min([num_cores, njobs])
    print('Number of Cores {} \n'.format(num_cores))
    print('total processes {}'.format(len(full_struct)))
    Parallel(n_jobs=njobs, verbose=3)(delayed(single_extract)(*full_struct[c]) for c in range(len(full_struct)))
    print('end time {} seconds'.format(time.time() - start_time))
