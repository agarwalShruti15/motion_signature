import os
import numpy as np
from torchvision.transforms import Compose, ToTensor, Resize
from PIL import Image
import shutil
import matplotlib.pyplot as plt

import torch

crop_sz = 256 # the size of the image to input to the embedding model
emb_n = 256 # size of the output embedding 
vgg_emb_n = 4096 # dimension of VGG embedding

def save_openface_frame(in_vid, out_fldr, size, openface_path):

    #save the frames
    os.makedirs(out_fldr, exist_ok=True)
    cmd = './{}/FeatureExtraction -f {} -simsize {} -simalign -nomask -out_dir {} | grep -m 1 -o "abc" | grep -o "123"'.format(openface_path, in_vid,size,out_fldr)
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

def fabnet_one_vid(in_fldr, fl_n, out_file, emb_model, openface_path):
    cur_vid = os.path.join(in_fldr, fl_n + '.mp4')
    
    tmp_fldr = '_'.join(in_fldr.split('/')[-2:]) + '_' + fl_n # create a folder which is unique to this file 
    
    assert not os.path.exists(out_file)
    try:
        
        frm_fldr = save_openface_frame(cur_vid, tmp_fldr, crop_sz, openface_path)
        filenms = [f for f in os.listdir(frm_fldr) if f.endswith('.bmp')]
        
        # init the array
        out_emb = np.zeros((len(filenms), emb_n))
                    
        for i in range(len(out_emb)):
            filenm = 'frame_det_00_{0:06d}.bmp'.format(i+1)
            if os.path.exists(os.path.join(frm_fldr, filenm)):
                cur_frame = plt.imread(os.path.join(frm_fldr, filenm))
                out_emb[i, :] = get_video_frame_emb(cur_frame, emb_model)
            else:
                print(cur_vid, i)
        
        # clear the aligned images
        shutil.rmtree(tmp_fldr)
        
        # save the output
        np.save(out_file, out_emb.astype(np.float32))
        
    except Exception as e:
        print('{} {}'.format(cur_vid, e))

loader = Compose([Resize(224), ToTensor()])               
def vgg_one_vid(in_fldr, fl_n, out_file, emb_model, openface_path, gpu):
    
    # load the model for frame embeddings
    cur_vid = os.path.join(in_fldr, fl_n + '.mp4')
    tmp_fldr = '_'.join(in_fldr.split('/')[-2:]) + '_' + fl_n # create a folder which is unique to this file 
    
    assert not os.path.exists(out_file)
    try:
        
        frm_fldr = save_openface_frame(cur_vid, tmp_fldr, crop_sz, openface_path)
        filenms = [f for f in os.listdir(frm_fldr) if f.endswith('.bmp')]
        
        # init the array
        out_emb = np.zeros((len(filenms), vgg_emb_n)).astype(np.float32)
                    
        for i in range(len(out_emb)):
            filenm = 'frame_det_00_{0:06d}.bmp'.format(i+1)
            if os.path.exists(os.path.join(frm_fldr, filenm)):
                
                # get vgg emb
                im = Image.fromarray(plt.imread(os.path.join(frm_fldr, filenm)))
                
                im = loader(im).view(1, 3, 224, 224).float()
                im -= torch.Tensor(np.array([129.1863, 104.7624, 93.5940])).float().view(1, 3, 1, 1)
                if gpu:
                    image = image.cuda()  #assumes that you're using GPU                   
                    
                out_emb[i, :] = emb_model(im).squeeze().cpu().detach().numpy().copy()
                
            else:
                print(cur_vid, i)
        
        # clear the aligned images
        shutil.rmtree(tmp_fldr)
        
        # save the output
        np.save(out_file, out_emb.astype(np.float32))
        
    except Exception as e:
        print('{} {}'.format(cur_vid, e))
