import numpy as np
import os
import argparse
import seaborn as sns
import matplotlib.pyplot as plt

""" usage:python 
extracts all the numpy files within a base folder. The extraction function should do the following:

1) pick files with atleast T number of frames. 
2) create a dictionary with file paths relative to the basefolder and the corresponding label key

the function should be added to the function dictionary and called via arguement
"""
def get_leader_file(bsfldr, T):
    
    subfldrs = np.array(['bo', 'br', 'bs', 'cb', 'dt_week', 'dt_rndm', 'ew', 'hc', 'jb', 'kh', 'pb'])
    labels = np.array([0, 1, 2, 3, 4, 4, 5, 6, 7, 8, 9])
    
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
                if len(cur_feat.shape)>2: # if there are more than one channel
                    feat_array[c] = cur_feat[:, :, 0]
                else:
                    feat_array[c] = cur_feat
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
    
    return np.mean(out_mean, axis=0), np.mean(out_std, axis=0)

str2func = {'leaders': get_leader_file, 'test': get_test_file}

if __name__ == '__main__':
    
    #read all the txt file
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--bsfldr', type=str, help='path to folder with npy files')
    parser.add_argument('--T', type=int, help='the minimum number of frames the file should have', default=150)
    parser.add_argument('--func', type=str, help='the file extraction function')
    parser.add_argument('--outfile', type=str, help='the out file prefix')
    parser.add_argument('--tp', type=float, help='the ratio of test files', default=0.1)
    parser.add_argument('--comp_mean', type=str, help='whether to recompute mean (Y/N)', default='N')
    
    args = parser.parse_args()
    test_perc = args.tp # the number of files to include in testing
    outfile = args.outfile
    bsfldr = args.bsfldr
    func = args.func
    T = args.T
    comp_mean = args.comp_mean
    
    # collect all the files first
    label_file_dict = str2func[func](bsfldr, T)
    
    if comp_mean == 'Y':
        # compute the mean and std of the files via bootrapping 
        [mean, std] = bootstrap_mean_std(bsfldr, label_file_dict, s_n = 100, s_sz = 10)
        # mean and standard deviation file save
        np.save(os.path.join(bsfldr, outfile + f'_{T}_mean.npy'), mean)
        np.save(os.path.join(bsfldr, outfile + f'_{T}_std.npy'), std)
    
    if test_perc>0:
        
        train_fd = open(os.path.join(bsfldr, outfile + f'_{T}_train.txt'), 'w')
        test_fd = open(os.path.join(bsfldr, outfile + f'_{T}_test.txt'), 'w')
        
        for k in label_file_dict.keys():
            
            # considering sorted to ensure disjoint train and test videos
            files = np.sort(label_file_dict[k])
            train_len = int((1-test_perc)*len(files))
            for i in range(train_len):
                train_fd.write('{},{}\n'.format(files[i], k))
            
            for i in range(train_len, len(files)):
                test_fd.write('{},{}\n'.format(files[i], k))
                
        train_fd.close()
        test_fd.close()
            
    else:
        
        fd = open(os.path.join(bsfldr, outfile + f'_{T}_full.txt'), 'w')
        
        for k in label_file_dict.keys():
            
            # considering sorted to ensure disjoint train and test videos
            files = np.sort(label_file_dict[k])
            for i in range(len(files)):
                fd.write('{},{}\n'.format(files[i], k))
            
        fd.close()