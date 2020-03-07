import numpy as np
import matplotlib.pyplot as plt
import os
import utils as u
import result_gen_utils as ru
import pandas as pd
import seaborn as sns
import multiprocessing
from joblib import Parallel, delayed
import natsort
import time
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import manifold
import moviepy.editor as mpe
import dlib


class Repo_TSNE(object):
    
    """class to get the repo per label"""
    def __init__(self, bs_fldr, frames, step, pool_func, N_per_id):
        super(Repo_TSNE, self).__init__()
        
        self.bs_fldr = bs_fldr
        self.N_per_id = N_per_id
        
        self.embDF = None # to store the embeddings
        
        self.frames = frames
        self.step = step
        self.pool_func = pool_func
        
        # initialize dlib's face detector (CNN-based) and then create
        # the facial landmark predictor and the face aligner
        self.detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')
        self.sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    
        
    def read_feat_file(self, f):
        
        if not os.path.exists(os.path.join(self.bs_fldr, f)):
            path, file = os.path.split(f)
            if file[0] == '_':
                f = os.path.join(path, file[1:])
            else:
                f = os.path.join(path, '_' + file)
        try:
            feat = np.load(os.path.join(self.bs_fldr, f))    
        except Exception as e:
            print(os.path.join(self.bs_fldr, f))
            return None

        if len(feat.shape)>2:
            feat = feat[:,:,0].copy()
        if len(feat) < (self.frames+ self.step):
            return None
        
        # pool the features according to pool_params
        # for vgg pool all 100 frames, whereas for resnet3D pool only 84 frames. 
        # for then step to another window to pool. The pool function is given in the params
        col_feat = u.im2col_sliding_strided(feat, (self.frames, feat.shape[1]), stepsize=self.step).T
        tmp = self.pool_func(np.reshape(col_feat, (col_feat.shape[0], self.frames, feat.shape[1])), axis=1)
        
        # normalize
        tmp = tmp - np.mean(tmp, axis=1, keepdims=True)
        tmp = tmp / np.linalg.norm(tmp, axis=1, keepdims=True) # normalize
        
        if self.step>1:
            frame_num = np.arange(0, self.step*len(tmp), self.step)
        else:
            frame_num = np.arange(0, 5*len(tmp), 5)
        
        feat = []
        col_feat = []
        
        out_df = pd.DataFrame(data=frame_num, columns=['frame_num'])
        out_df['fileName'] = f
        out_df['feat'] = tmp.tolist()
        
        tmp = []
        return out_df
        
        
    # n1, n2 gives the part of the video to use
    def add_ids(self, in_file_dict):
        
        all_keys = list(in_file_dict.keys()) # list of all identities
        all_keys.sort()
        
        for k in range(len(all_keys)):
                        
            id_feat = {}
            for f in in_file_dict[all_keys[k]]:
                
                tmp = self.read_feat_file(f)
                if tmp is not None:
                    id_feat[f] = tmp.copy()
                    tmp = []
    
            all_id_feat = pd.concat(list(id_feat.values()), ignore_index=True, sort=False)
            id_feat = []
            
            if self.N_per_id > 0 and self.N_per_id < len(all_id_feat):
                all_id_feat = all_id_feat.sample(n=self.N_per_id, random_state=0).copy()
                
            all_id_feat['actualLabel_vgg'] = all_keys[k]
            all_id_feat['actualLabel_behav'] = ru.get_behav_id(all_id_feat)
            

            # add the labels and embeddings
            if self.embDF is None:
                self.embDF = all_id_feat.copy()
            else:
                self.embDF = pd.concat((self.embDF, all_id_feat.copy()), ignore_index=True, sort=False)
                
            all_id_feat = []
            
    # return the face image cropped from a frame in the video
    def extract_face_images(self, fl_nm, frame_no=-1, time=-1):
        
        # read the frame
        video = mpe.VideoFileClip(os.path.join(self.bs_fldr, fl_nm))
        if time>=0:
            frame = video.get_frame(time) # get the frame at t=2 seconds
        if frame_no>=0:
            frame = video.get_frame(frame_no/video.fps) # get the frame at t=2 seconds
        
        # might need to change the image format to PIL image
        
        # crop and align the face to 256x256
        dets = self.detector(frame, 1)
        faces = dlib.full_object_detections()
        for detection in dets:
            faces.append(self.sp(frame, detection))
    
        # Get the aligned face images
        # Optionally: 
        # images = dlib.get_face_chips(img, faces, size=160, padding=0.25)
        face_image = dlib.get_face_chips(frame, faces, size=256)        
        
        return face_image[0]
    
    def compute_tsne(self):
        
        # compute the t-sne from this emb space and keep it for generating plots
        X_test = np.array(list(self.embDF['feat']))
        tsne = manifold.TSNE(n_components=2, perplexity=20.0,
                             early_exaggeration=12.0, learning_rate=200.0,
                             n_iter=1000, n_iter_without_progress=300,
                             min_grad_norm=1e-07, metric='euclidean',
                             init='random', verbose=0, random_state=777,
                             method='barnes_hut', angle=0.5)
        print('tsne train {}'.format(X_test.shape))
        Y = tsne.fit_transform(X_test)
        self.embDF['tsne'] = Y.tolist()
        
    
    def plot_tsne(self, in_flnm, ax):
        
        Y = np.array(list(self.embDF['tsne']))        

        lbl = np.unique(np.array(self.embDF['actualLabel_vgg']))
        
        # plot all the points
        for k in range(len(lbl)):
            c_l = np.sum(self.embDF['actualLabel_vgg']==lbl[k])
            color = np.zeros((c_l, 3))
            color[:, 0] = 0.8; color[:, 1] = 0.8; color[:, 2] = 0.8
            plt.scatter(Y[self.embDF['actualLabel_vgg']==lbl[k], 0], 
                        Y[self.embDF['actualLabel_vgg']==lbl[k], 1], marker='o', c=color)
            
        # plot the src, behav, file in g, b, r
        fl_idx = self.embDF['fileName']==in_flnm
        cur_fID = np.array(self.embDF.loc[fl_idx, 'actualLabel_vgg'])[0]
        cur_bID = np.array(self.embDF.loc[fl_idx, 'actualLabel_behav'])[0]
        
        # 
        print(cur_bID, cur_fID)
        face_idx = np.logical_and(self.embDF['actualLabel_vgg']==cur_fID, np.logical_not(fl_idx))
        behav_idx = np.logical_and(self.embDF['actualLabel_vgg']==cur_bID, np.logical_not(fl_idx))        
        
        print(np.sum(face_idx), np.sum(behav_idx), )
        plt.scatter(Y[fl_idx, 0], Y[fl_idx, 1], marker='^', c='r', label=r'fake')
        plt.scatter(Y[face_idx, 0], Y[face_idx, 1], marker='^', c='g', label=r'source')
        plt.scatter(Y[behav_idx, 0], Y[behav_idx, 1], marker='^', c='b', label=r'target')
        """plt.legend(loc='upper center', 
                       bbox_to_anchor=(0.5, 1.1), ncol=6, 
                       fancybox=True, shadow=False, fontsize=14)"""
        
        plt.xticks([])
        plt.yticks([])
        ax.axis('off')
        plt.tight_layout(pad=0, h_pad=None, w_pad=None, rect=None)