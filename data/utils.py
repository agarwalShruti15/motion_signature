
import os
import numpy as np
from torchvision.transforms import Compose, ToTensor, Resize
from PIL import Image
import shutil
import matplotlib.pyplot as plt
import cv2
import torchvision.transforms as transforms
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


def compose_transforms(meta, resize=256, center_crop=True,
                       override_meta_imsize=False):
    """Compose preprocessing transforms for model
    The imported models use a range of different preprocessing options,
    depending on how they were originally trained. Models trained in MatConvNet
    typically require input images that have been scaled to [0,255], rather
    than the [0,1] range favoured by PyTorch.
    Args:
        meta (dict): model preprocessing requirements
        resize (int) [256]: resize the input image to this size
        center_crop (bool) [True]: whether to center crop the image
        override_meta_imsize (bool) [False]: if true, use the value of `resize`
           to select the image input size, rather than the properties contained
           in meta (this option only applies when center cropping is not used.
    Return:
        (transforms.Compose): Composition of preprocessing transforms
    """
    normalize = transforms.Normalize(mean=meta['mean'], std=meta['std'])
    im_size = meta['imageSize']
    assert im_size[0] == im_size[1], 'expected square image size'
    if center_crop:
        transform_list = [transforms.Resize(resize),
                          transforms.CenterCrop(size=(im_size[0], im_size[1]))]
    else:
        if override_meta_imsize:
            im_size = (resize, resize)
        transform_list = [transforms.Resize(size=(im_size[0], im_size[1]))]
    transform_list += [transforms.ToTensor()]
    if meta['std'] == [1, 1, 1]:  # common amongst mcn models
        transform_list += [lambda x: x * 255.0]
    transform_list.append(normalize)
    return transforms.Compose(transform_list)


def vgg_one_vid(in_fldr, fl_n, out_file, emb_model, openface_path, gpu):
    
    # load the model for frame embeddings
    cur_vid = os.path.join(in_fldr, fl_n + '.mp4')
    tmp_fldr = '_'.join(in_fldr.split('/')[-2:]) + '_' + fl_n # create a folder which is unique to this file 
    
    preproc_transforms = compose_transforms(meta=emb_model.meta, center_crop=False )    
    
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
                im = cv2.cvtColor(cv2.imread(os.path.join(frm_fldr, filenm)), cv2.COLOR_BGR2RGB)
                im = Image.fromarray(im)
                im = preproc_transforms(im).view(1, 3, 224, 224).float()
                if torch.cuda.is_available():
                    im = im.cuda()  #assumes that you're using GPU                                   
            
                #get the embeddings
                out_emb[i, :] = emb_model(im).squeeze().cpu().detach().numpy().copy()           
            else:
                print(cur_vid, i)
        
        # clear the aligned images
        shutil.rmtree(tmp_fldr)
        
        # save the output
        np.save(out_file, out_emb.astype(np.float32))
        
    except Exception as e:
        print('{} {}'.format(cur_vid, e))
