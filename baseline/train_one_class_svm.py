""" combine the transcript file with the landmark, au csv file

"""

import numpy as np
import utils as u
import os
import argparse
import multiprocessing
from joblib import Parallel, delayed

def one_class_svm(bsfldr, infiles, out_fld, out_file, ncomps, gamma=0.1, nu=0.1):
    
    X_train = {}
    for f in infiles:
        
        X_train[f] = np.load(os.path.join(bsfldr, f))[:, :ncomps]
        X_train[f] = X_train[f][np.sum(np.isnan(X_train[f]), axis=1)<1, :].copy()
    
    X_train = np.vstack(list(X_train.values()))
    print(out_file, X_train.shape)
    svm_model = u.train_ovr(X_train, gamma=gamma, nu=nu)
    u.save_obj(svm_model, out_fld, out_file)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--bsfldr', type=str, help='base folder of features')
    parser.add_argument('--tf', type=str, help='path to file with train file names relative to the base folder')
    parser.add_argument('--ncomp', type=np.int32, help='number of features to train the model')
    parser.add_argument('--ofd', type=str, help='out folder to save the trained model')
    parser.add_argument('--gamma', type=np.float32, help='gamma for svm', default=0.1)
    parser.add_argument('--nu', type=np.float32, help='nu for svm', default=0.1)
    
    args = parser.parse_args()
    bs_fldr = args.bsfldr
    train_file = args.tf
    ncomp = args.ncomp
    ofd = args.ofd
    gamma = args.gamma
    nu = args.nu
    
    os.makedirs(ofd, exist_ok=True)
    
    # read the train file and get the unique model files
    file_dict = u.load_dict_file(train_file)
    njobs = len(file_dict.keys())
    
    # transform all the files to transform to pca
    full_struct = []    
    for k in file_dict.keys():
        print(k, len(file_dict[k]))
        full_struct.append((bs_fldr, file_dict[k], ofd, k, ncomp, gamma, nu))
                                       
    # run the jobs in parallel
    num_cores = multiprocessing.cpu_count()
    njobs = np.min([num_cores, njobs])
    print('Number of Cores {} \n'.format(num_cores))
    print('total processes {}'.format(len(full_struct)))
    Parallel(n_jobs=njobs, verbose=20)(delayed(one_class_svm)(*full_struct[c]) for c in range(len(full_struct)))