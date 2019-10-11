""" generate fake random files to test the mean and std measurement """

import os
import numpy as np

bsfldr = '/data/home/shruti/voxceleb/fabnet/test_samples/br'
samples = 100
feat = 10

frames = np.random.choice(np.arange(148, 200), samples, replace=True)

os.makedirs(bsfldr, exist_ok=True)

means = np.arange(feat)[:, np.newaxis].T + 5
stds = np.arange(feat)[:, np.newaxis].T + 2

for i in range(samples):
    
    cur_data = (np.random.randn(frames[i], feat)*stds)+means
    print(cur_data.shape)
    np.save(os.path.join(bsfldr, str(i) + '.npy'), cur_data)
