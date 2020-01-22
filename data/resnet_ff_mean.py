import numpy as np
import os
import re

def load_dict_file(infile):
    
    # read the file and collect in label dict 
    label_dict = []
    with open(infile, 'r') as f:
        for line in f:
            _path, _ = re.split(r",| ", line.strip())                
            label_dict.extend([_path])
    
    return label_dict


bs_fldr = '/data/home/shruti/voxceleb/resnet3D/ff'
train_files = load_dict_file('utils/ff_train.txt')

print(len(train_files))

np.random.seed(777)
train_files = np.random.choice(train_files, 1000, replace=False)
out_results = {}
for f in train_files:
    
    [dirname, vid_file] = os.path.split(f)
    fl_n = os.path.splitext(vid_file)[0]
    
    if os.path.exists(os.path.join(bs_fldr, dirname, fl_n + '.npy')):
        out_results[f] = np.load(os.path.join(bs_fldr, dirname, fl_n + '.npy'))
    else:
        out_results[f] = np.load(os.path.join(bs_fldr, dirname, '_' + fl_n + '.npy'))
        

mean  = np.mean(np.vstack(list(out_results.values())), axis=0)
std = np.std(np.vstack(list(out_results.values())), axis=0)

print(np.min(mean), np.max(mean))
print(np.min(std), np.max(std))
print(mean.shape, std.shape)

np.save(os.path.join(bs_fldr, 'ff_train_mean.npy'), mean)
np.save(os.path.join(bs_fldr, 'ff_train_std.npy'), std)