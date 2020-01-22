import numpy as np
import os
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import re

""" usage:python 
extracts all the numpy files within a base folder. The extraction function should do the following:

1) pick files with atleast T number of frames. 
2) create a dictionary with file paths relative to the basefolder and the corresponding label key

the function should be added to the function dictionary and called via arguement
"""
def load_dict_file(bs_fldr, prefex):
    
    # read the file and collect in label dict 
    txt_files = [f for f in os.listdir(bs_fldr) if f[:len(prefex)] == prefex and f.endswith('.txt')]
    label_dict = {}
    for fd in txt_files:
        
        with open(os.path.join(bs_fldr, fd), 'r') as f:
            for line in f:
                _path, _label = re.split(r",| ", line.strip())
                if _label not in list(label_dict.keys()):
                    label_dict[_label] = []
                    
                label_dict[_label].extend([_path])
    
    return label_dict

def get_leader_file(bsfldr, T):
    
    subfldrs = np.array(['bo', 'br', 'bs', 'cb', 'dt', 'ew', 'hc', 'jb', 'kh', 'pb'])
    labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    
    #     
    label_file_dict = {}
    for i in range(len(labels)):
        
        cur_files = os.listdir(os.path.join(bsfldr, subfldrs[i])) # the files 
        label_file_dict[labels[i]] = []
        for f in cur_files:
            
            if f.endswith('.npy'): # consider only npy files
                
                num_fr = len(np.load(os.path.join(bsfldr, subfldrs[i], f))) # assumpstion that frame is size[0]
                if num_fr>=T: # consider only when the frames are greater than T
                    label_file_dict[labels[i]].extend([os.path.join(subfldrs[i], f)])
                    
        # verbose
        print(subfldrs[i], labels[i], len(label_file_dict[labels[i]]))    
    return label_file_dict

def get_leader_file_with_synfake(bsfldr, T):
      
    subfldrs = np.array(['bo', 'br', 'bs', 'cb', 'dt', 'ew', 'hc', 'jb', 'kh', 'pb', 
                         'bo_syn_fake', 'dt_rndm_syn_fake', 'hc_syn_fake', 'bs_syn_fake', 'ew_syn_fake', 
                         'br_syn_fake', 'cb_syn_fake', 'jb_syn_fake', 'kh_syn_fake', 'pb_syn_fake'])
    labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])    
    
    #     
    label_file_dict = {}
    for i in range(len(labels)):
        
        cur_files = os.listdir(os.path.join(bsfldr, subfldrs[i])) # the files 
        label_file_dict[labels[i]] = []
        for f in cur_files:
            
            if f.endswith('.npy'): # consider only npy files
                
                num_fr = len(np.load(os.path.join(bsfldr, subfldrs[i], f))) # assumpstion that frame is size[0]
                if num_fr>=T: # consider only when the frames are greater than T
                    label_file_dict[labels[i]].extend([os.path.join(subfldrs[i], f)])
                    
        # verbose
        print(subfldrs[i], labels[i], len(label_file_dict[labels[i]]))    
    return label_file_dict


def get_vgg_leader_file_faceswap(bsfldr, T):
    
    # TODO: This is a HACK1 because the labels have same prefix
    base_vid_fldr = '/data/home/shruti/voxceleb/videos/leaders/'
    
    subfldrs = np.array(['bo', 'br', 'bs', 'cb', 'dt', 'ew', 'hc', 'jb', 'kh', 'pb', 
                         'bo_faceswap', 'dt_faceswap', 'hc_faceswap', 'bs_faceswap', 'ew_faceswap'])
    fldr2lbl = {'bo': 0, 'br': 1, 'bs': 2, 'cb': 3, 'dt': 4, 'ew': 5, 'hc': 6, 'jb': 7, 'kh': 8, 'pb': 9, 
                         'bo_faceswap':10, 'dt_faceswap':11, 'hc_faceswap':12, 'bs_faceswap':13, 'ew_faceswap':14}
    
    all_files = []
    for s in subfldrs:
        
        all_files.extend([ np.array([s, s + '_' + os.path.splitext(f)[0] + '.npy']) for 
                           f in os.listdir(os.path.join(base_vid_fldr, s)) if f.endswith('.mp4') ])
    
    all_files = np.array(all_files)
    # the first 7 letters are of the ids, this is used as labels 
    [uniq_labels, uniq_ids, uniq_cnts] = np.unique(all_files[:, 0], return_inverse=True, return_counts=True)
    print(f'number of labels {len(uniq_labels)}')
    
    # create the dictionary
    label_file_dict = {}
    for i in range(len(uniq_labels)):
        
        cur_lbl_file = all_files[uniq_ids==i, 1]
        validx = np.zeros((len(cur_lbl_file), ), dtype=np.bool)
        for j in range(len(cur_lbl_file)):
            
            if cur_lbl_file[j].endswith('.npy'): # consider only npy files
                try:
                    num_fr = np.load(os.path.join(bsfldr, cur_lbl_file[j])).shape # assumpstion that frame is size[0]
                    if num_fr[0]>=T: # consider only when the frames are greater than T
                        validx[j] = True
                except Exception as e:
                    print(f'error: {cur_lbl_file[j]} {e}')
        
        if fldr2lbl[uniq_labels[i]] not in label_file_dict.keys():
            label_file_dict[fldr2lbl[uniq_labels[i]]] = cur_lbl_file[validx]
        else:
            label_file_dict[fldr2lbl[uniq_labels[i]]] = np.concatenate((label_file_dict[fldr2lbl[uniq_labels[i]]], 
                                                                        cur_lbl_file[validx]), axis=0)
            
        print(f'id: {uniq_labels[i]} lbl: {fldr2lbl[uniq_labels[i]]} len: {len(label_file_dict[fldr2lbl[uniq_labels[i]]])}')

    return label_file_dict


def get_voxceleb_file(bsfldr, T):
    
    all_files = np.array([np.array([f[:7], f]) for f in os.listdir(bsfldr) if f.endswith('.npy')])
        
    # the first 7 letters are of the ids, this is used as labels 
    [uniq_labels, uniq_ids, uniq_cnts] = np.unique(all_files[:, 0], return_inverse=True, return_counts=True)
    print(f'number of labels {len(uniq_labels)}')
    
    # create the dictionary
    label_file_dict = {}
    for i in range(len(uniq_labels)):
        
        cur_lbl_file = all_files[uniq_ids==i, 1]
        validx = np.zeros((len(cur_lbl_file), ), dtype=np.bool)
        for j in range(len(cur_lbl_file)):
            
            if cur_lbl_file[j].endswith('.npy'): # consider only npy files
                try:
                    num_fr = len(np.load(os.path.join(bsfldr, cur_lbl_file[j]))) # assumpstion that frame is size[0]
                    if num_fr>=T: # consider only when the frames are greater than T
                        validx[j] = True                    
                except Exception as e:
                    print(f'error: {cur_lbl_file[j]} {e}')
                    
        label_file_dict[i] = cur_lbl_file[validx]
        print(f'id: {uniq_labels[i]} lbl: {i} len: {len(label_file_dict[i])}')

    return label_file_dict

def get_vgg_leader_file(bsfldr, T):
    
    # TODO: This is a HACK1 because the labels have same prefix
    base_vid_fldr = '/data/home/shruti/voxceleb/videos/leaders/'
    
    subfldrs = np.array(['bo', 'br', 'bs', 'cb', 'dt', 'ew', 'hc', 'jb', 'kh', 'pb'])
    fldr2lbl = {'bo': 0, 'br': 1, 'bs': 2, 'cb': 3, 'dt': 4, 'ew': 5, 'hc': 6, 'jb': 7, 'kh': 8, 'pb': 9}
    
    all_files = []
    for s in subfldrs:
        
        all_files.extend([ np.array([s, s + '_' + os.path.splitext(f)[0] + '.npy']) for 
                           f in os.listdir(os.path.join(base_vid_fldr, s)) if f.endswith('.mp4') ])
    
    all_files = np.array(all_files)
    # the first 7 letters are of the ids, this is used as labels 
    [uniq_labels, uniq_ids, uniq_cnts] = np.unique(all_files[:, 0], return_inverse=True, return_counts=True)
    print(f'number of labels {len(uniq_labels)}')
    
    # create the dictionary
    label_file_dict = {}
    for i in range(len(uniq_labels)):
        
        cur_lbl_file = all_files[uniq_ids==i, 1]
        validx = np.zeros((len(cur_lbl_file), ), dtype=np.bool)
        for j in range(len(cur_lbl_file)):
            
            if cur_lbl_file[j].endswith('.npy'): # consider only npy files
                try:
                    num_fr = np.load(os.path.join(bsfldr, cur_lbl_file[j])).shape # assumpstion that frame is size[0]
                    if num_fr[0]>=T: # consider only when the frames are greater than T
                        validx[j] = True
                except Exception as e:
                    print(f'error: {cur_lbl_file[j]} {e}')
        
        if fldr2lbl[uniq_labels[i]] not in label_file_dict.keys():
            label_file_dict[fldr2lbl[uniq_labels[i]]] = cur_lbl_file[validx]
        else:
            label_file_dict[fldr2lbl[uniq_labels[i]]] = np.concatenate((label_file_dict[fldr2lbl[uniq_labels[i]]], 
                                                                        cur_lbl_file[validx]), axis=0)
            
        print(f'id: {uniq_labels[i]} lbl: {fldr2lbl[uniq_labels[i]]} len: {len(label_file_dict[fldr2lbl[uniq_labels[i]]])}')

    return label_file_dict

def get_test_file(bsfldr, T):
    
    subfldrs = np.array(['bo'])
    labels = np.array([0])
    
    #     
    label_file_dict = {}
    for i in range(len(labels)):
        
        cur_files = os.listdir(os.path.join(bsfldr, subfldrs[i])) # the files 
        label_file_dict[labels[i]] = []
        for f in cur_files:
            
            if f.endswith('.npy'): # consider only npy files
                
                num_fr = len(np.load(os.path.join(bsfldr, subfldrs[i], f))) # assumpstion that frame is size[0]
                if num_fr>=T: # consider only when the frames are greater than T
                    label_file_dict[labels[i]].extend([os.path.join(subfldrs[i], f)])
                    
        # verbose
        print(subfldrs[i], labels[i], len(label_file_dict[labels[i]]))    
    return label_file_dict


# the label file dictionary, 
# the number of samples
# the number of files from each label in a sample
def bootstrap_mean_std(bsfldr, label_file_dict, s_n, s_sz):
    
    out_mean = None
    out_std = None
    for i in range(s_n):
        
        feat_array = {}
        c = 0
        for k in label_file_dict.keys():
            
            # pick the random number of files for this label
            rndm_fls = np.random.choice(label_file_dict[k], s_sz)
            for f in rndm_fls:
                
                cur_feat = np.load(os.path.join(bsfldr, f))
                if len(cur_feat)<1:
                    continue
                    
                if len(cur_feat.shape)>2: # if there are more than one channel
                    feat_array[c] = cur_feat[:, :, 0].copy()
                else:
                    feat_array[c] = cur_feat.copy()
                    
                c = c+1

        # the sample mean, std
        cur_sample = np.concatenate(list(feat_array.values()), axis=0)
        cur_sample = cur_sample / np.linalg.norm(cur_sample, axis=1, keepdims=True)
        
        cur_mean = np.mean(cur_sample, axis=0)
        cur_std = np.std(cur_sample, axis=0)
        if i==0:
            out_mean = np.zeros((s_n, len(cur_mean)))
            out_std = np.zeros((s_n, len(cur_std)))
            
        out_mean[i, :] = cur_mean.copy()
        out_std[i, :] = cur_std.copy()
        
    
    # save the distribution of bootstrap sample means and standard deviation
    if 0:
        
        for f_id in range(out_mean.shape[1]):
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 7))
            sns.distplot(out_mean[:, f_id], ax=ax[0])
            ax[0].set_title('mean')
            sns.distplot(out_std[:, f_id], ax=ax[1])
            ax[1].set_title('std')
            plt.draw()
            plt.savefig(f'{f_id}.png')
    
    return np.mean(out_mean, axis=0), np.std(out_std, axis=0)

str2func = {'leaders': get_leader_file, 'test': get_test_file, 'voxceleb': get_voxceleb_file, 
            'vgg_leader': get_vgg_leader_file, 'leader_with_faceswap': get_vgg_leader_file_faceswap,
            'leader_with_synfake': get_leader_file_with_synfake}

exclude_files = ['25GOnaY8ZCY', '2AFpAATHXtc', '3vPdtajOJfw', 
                 'E3gfMumXCjI', 'EAZIHIiuhrc', 'k4OZOTaf3lk']

if __name__ == '__main__':
    
    #read all the txt file
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--bsfldr', type=str, help='path to folder with npy files')
    parser.add_argument('--T', type=int, help='the minimum number of frames the file should have', default=150)
    parser.add_argument('--func', type=str, help='the file extraction function')
    parser.add_argument('--outfile', type=str, help='the out file prefix')
    parser.add_argument('--tp', type=float, help='the ratio of test files', default=0.1)
    parser.add_argument('--vp', type=float, help='the ratio of val files', default=0.1)
    parser.add_argument('--comp_mean', type=str, help='whether to recompute mean (Y/N)', default='N')
    parser.add_argument('--sn', type=np.int32, help='number of bootstrapping samples', default=50)
    parser.add_argument('--s_sz', type=np.int32, help='number of files per label class', default=10)    
    
    args = parser.parse_args()
    test_perc = args.tp # the number of files to include in testing
    val_perc = args.vp # the number of files to include in validation
    outfile = args.outfile
    bsfldr = args.bsfldr
    func = args.func
    T = args.T
    comp_mean = args.comp_mean
    s_n = args.sn
    s_sz = args.s_sz
    
    # collect all the files first
    label_file_dict = str2func[func](bsfldr, T)

    print(len(list(label_file_dict.keys())))
    
    if comp_mean == 'Y':
        print('computing mean/std')
        # compute the mean and std of the files via bootrapping 
        [mean, std] = bootstrap_mean_std(bsfldr, label_file_dict, s_n = s_n, s_sz = s_sz)
        # mean and standard deviation file save
        np.save(os.path.join(bsfldr, outfile + f'_{T}_mean.npy'), mean)
        np.save(os.path.join(bsfldr, outfile + f'_{T}_std.npy'), std)
    
    if test_perc>0:
        
        train_fd = open(os.path.join(bsfldr, outfile + f'_{T}_train.txt'), 'w')
        val_fd = open(os.path.join(bsfldr, outfile + f'_{T}_val.txt'), 'w')
        test_fd = open(os.path.join(bsfldr, outfile + f'_{T}_test.txt'), 'w')
        
        for k in label_file_dict.keys():
            
            # considering sorted to ensure disjoint train and test videos
            files = np.sort(label_file_dict[k])
            train_len = int((1-test_perc-val_perc)*len(files))
            val_len = int(val_perc*len(files))
            for i in range(train_len):
                train_fd.write('{},{}\n'.format(files[i], k))
            
            for i in range(train_len, train_len+val_len):
                val_fd.write('{},{}\n'.format(files[i], k))
                
            for i in range(train_len+val_len, len(files)):
                test_fd.write('{},{}\n'.format(files[i], k))
                
        train_fd.close()
        val_fd.close()
        test_fd.close()
            
    else:
        
        fd = open(os.path.join(bsfldr, outfile + f'_{T}_full.txt'), 'w')
        
        for k in label_file_dict.keys():
            
            # considering sorted to ensure disjoint train and test videos
            files = np.sort(label_file_dict[k])
            for i in range(len(files)):
                fd.write('{},{}\n'.format(files[i], k))
            
        fd.close()