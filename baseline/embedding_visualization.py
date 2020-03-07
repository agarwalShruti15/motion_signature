# -*- coding: utf-8 -*-


import os
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import utils as u
import cv2

def get_image(flnm, fr_no):
    
    #Open the video file
    cap = cv2.VideoCapture(flnm)
    
    # get the frame
    time_length = 30.0
    fps=25
    frame_seq = 749
    frame_no = (frame_seq /(time_length*fps))
    
    # get the face
    # resize the frame to 224
    # return the face image
    
    
    
    
    
    #Set frame_no in range 0.0-1.0
    #In this example we have a video of 30 seconds having 25 frames per seconds, thus we have 750 frames.
    #The examined frame must get a value from 0 to 749.
    #For more info about the video flags see here: https://stackoverflow.com/questions/11420748/setting-camera-parameters-in-opencv-python
    #Here we select the last frame as frame sequence=749. In case you want to select other frame change value 749.
    #BE CAREFUL! Each video has different time length and frame rate. 
    #So make sure that you have the right parameters for the right video!
    
    
    #The first argument of cap.set(), number 2 defines that parameter for setting the frame selection.
    #Number 2 defines flag CV_CAP_PROP_POS_FRAMES which is a 0-based index of the frame to be decoded/captured next.
    #The second argument defines the frame number in range 0.0-1.0
    cap.set(2,frame_no);
    
    #Read the next frame from the video. If you set frame 749 above then the code will return the last frame.
    ret, frame = cap.read()
    
    #Set grayscale colorspace for the frame. 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    
    
    return None

def load_DFDC_files(path):
    
    out_file_list = []
    orig_fldr = 'DFDC_orig'
    fake_fldr = 'DFDC_fake'
    N = 10 # the number of different faceswaps (this many number of unique faceswaps)
    all_orig = np.array(u.load_file_names(path, orig_fldr))
    id_lbl = np.unique([f.split('/')[-1].split('_')[0] for f in all_orig])  # get the unique ids
    print(f'Total IDs: {len(id_lbl)}')
    
    all_fake = np.array(u.load_file_names(path, fake_fldr))
    all_fake = [f for f in all_fake if f.split('/')[-1].split('_')[0] in id_lbl and f.split('/')[-1].split('_')[1] in id_lbl]
    np.random.shuffle(all_fake)
    
    ids_included = []
    cnt = 0
    # in a for loop collect one faceswap file, 
    for i in range(len(all_fake)):
        
        if cnt == N:
            break
        
        fid, bid = all_fake[i].split('/')[-1].split('_')[:2]   # pick the face id and behav id corresponding to the faceswap 
        # and it's behav and face id in the real video, 
        if not np.isin(fid, ids_included) and not np.isin(bid, ids_included):
            
            print(fid, bid)
            ids_included = ids_included + [fid, bid]
            
            # select the real videos of fid and behav id
            orig_bid_file = os.path.join(orig_fldr, '_'.join(all_fake[i].split('/')[-1].split('_')[1:]))
            np.random.shuffle(all_orig)
            orig_fid_file = [f for f in all_orig if fid in f][0]
            
            print(all_fake[i], orig_bid_file, orig_fid_file)
            out_file_list = out_file_list + [all_fake[i], orig_bid_file, orig_fid_file]# add the real and fake video names corresponding to the embeddings
            cnt = cnt + 1
          
    return out_file_list
    

def embedding_load(path, file_names, params):
    # Given the list of file paths relative to the bsfldr, 
    out_emb = {}
    out_file_name = {}
    out_nm_to_num_dict = {}
    
    # this function returns the embeddings and name of the file list
    for i in range(len(file_names)):
        
        print(file_names[i])
        out_emb[i] = np.load(os.path.join(path, file_names[i]))
        
        # pool the embeddings
        col_feat = u.im2col_sliding_strided(out_emb[i], (params['frames'], out_emb[i].shape[1]), stepsize=params['step']).T
        out_emb[i] = np.mean(np.reshape(col_feat, (col_feat.shape[0], params['frames'], out_emb[i].shape[1])), axis=1)
                
        out_file_name[i] = np.array([file_names[i] for j in range(len(out_emb[i]))])
        out_nm_to_num_dict[file_names[i]] = i
    
    return np.vstack(list(out_emb.values())), np.concatenate(list(out_file_name.values()), axis=0), out_nm_to_num_dict

print(tf.__version__)

PATH = os.getcwd()

LOG_DIR = PATH + '/VGG_DFDC_logs'
os.makedirs(LOG_DIR, exist_ok=True)

# get file list
file_list = load_DFDC_files('/data/home/shruti/voxceleb/vgg/leaders/')
feature_vectors, fl_lbls, lblfile_to_num = embedding_load('/data/home/shruti/voxceleb/vgg/leaders/', file_list, {'frames':100, 'step':5})
num_of_samples=feature_vectors.shape[0]
print(f'Number of features: {num_of_samples}')
features = tf.Variable(feature_vectors, name='features')

metadata_file = open(os.path.join(LOG_DIR, 'metadata_VGG_DFDC.tsv'), 'w')
metadata_file.write('Class\tName\n')
for i in range(num_of_samples):
    metadata_file.write(f'{lblfile_to_num[fl_lbls[i]]}\t{fl_lbls[i]}\n')
metadata_file.close()

#%%
with tf.Session() as sess:
    saver = tf.train.Saver([features])

    sess.run(features.initializer)
    saver.save(sess, os.path.join(LOG_DIR, 'VGG_DFDC.ckpt'))
    
    config = projector.ProjectorConfig()
    # One can add multiple embeddings.
    embedding = config.embeddings.add()
    embedding.tensor_name = features.name
    # Link this tensor to its metadata file (e.g. labels).
    embedding.metadata_path = os.path.join(LOG_DIR, 'metadata_VGG_DFDC.tsv')
    # Saves a config file that TensorBoard will read during startup.
    projector.visualize_embeddings(tf.summary.FileWriter(LOG_DIR), config)