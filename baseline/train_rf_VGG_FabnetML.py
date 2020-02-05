""" combine the transcript file with the landmark, au csv file

"""

import numpy as np
import utils as u
import os
import utils as u
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid

np.random.seed(0)
def train_SVC(X_train, y_train, X_val, y_val, do_cv=False):

    # Fit your data on the scaler object
    scaler = StandardScaler()
    scaler = scaler.fit(X_train)
    X_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_val)

    #init best params to default values
    clf = RandomForestClassifier(n_estimators=100, max_depth=2,
                                 random_state=0)
    
    svm_model = {}
    svm_model['scaler'] = scaler
    svm_model['model'] = clf
    
    # select the params
    print(X_scaled.shape, X_test_scaled.shape)
    #idx = np.random.choice(range(len(X_scaled)), size=4000, replace=False)
    #X_scaled = X_scaled[idx, :].copy()
    #y_train = y_train[idx]
    #idx = np.random.choice(range(len(X_test_scaled)), size=np.min([4000, len(X_test_scaled)]), replace=False)
    #X_test_scaled = X_test_scaled[idx, :].copy()
    #y_val = y_val[idx]
    
    print(np.sum(y_train==1), np.sum(y_train==0))
    print(np.sum(y_val==1), np.sum(y_val==0))
    svm_model['model'].fit(np.vstack((X_scaled, X_test_scaled)), np.concatenate((y_train, y_val), axis=0))

    return svm_model

def im2col_sliding_strided(A, BSZ, stepsize=1):
    # Parameters
    m,n = A.shape
    s0, s1 = A.strides
    nrows = m-BSZ[0]+1
    ncols = n-BSZ[1]+1
    shp = BSZ[0],BSZ[1],nrows,ncols
    strd = s0,s1,s0,s1

    out_view = np.lib.stride_tricks.as_strided(A, shape=shp, strides=strd)
    return out_view.reshape(BSZ[0]*BSZ[1],-1)[:,::stepsize]
    
# train and test split
def train_test():

    train_ldr_dict = u.load_dict_file('/data/home/shruti/voxceleb/motion_signature/data/utils/ff_train.txt')
    val_ldr_dict = u.load_dict_file('/data/home/shruti/voxceleb/motion_signature/data/utils/ff_val.txt')
    test_ldr_dict = u.load_dict_file('/data/home/shruti/voxceleb/motion_signature/data/utils/ff_test.txt')
    return train_ldr_dict, val_ldr_dict, test_ldr_dict

def build_FF_Ref_two_feat(bs_fldr_f1, bs_fldr_f2, file_dict, pool_params1, pool_params2, N=-1):
    
    all_keys = np.sort(list(file_dict.keys()))
    full_emb_df = {}
    for k in range(len(all_keys)):

        all_feat = {}
        for f in file_dict[all_keys[k]]:

            tmp1 = None
            tmp2 = None
            try:
                if bs_fldr_f1 is not None:
                         
                    if os.path.exists(os.path.join(bs_fldr_f1, f)):
                        feat1 = np.load(os.path.join(bs_fldr_f1, f))
                    else:
                        path, file = os.path.split(f)
                        feat1 = np.load(os.path.join(bs_fldr_f1, path, '_' + file))
                        
                    if len(feat1.shape)>2:
                        feat1 = feat1[:,:,0].copy() 
                                        
                    col_feat1 = im2col_sliding_strided(feat1, (pool_params1['frames'], 
                                                               feat1.shape[1]), 
                                                       stepsize=pool_params1['step']).T
                    tmp1 = pool_params1['pool_func'](np.reshape(col_feat1, (col_feat1.shape[0], 
                                                                            pool_params1['frames'], 
                                                                            feat1.shape[1])), axis=1) 
                    
                if bs_fldr_f2 is not None:
        
                    if os.path.exists(os.path.join(bs_fldr_f2, f)):
                        feat2 = np.load(os.path.join(bs_fldr_f2, f))
                    else:
                        path, file = os.path.split(f)
                        feat2 = np.load(os.path.join(bs_fldr_f2, path, '_' + file))
                    if len(feat2.shape)>2:
                        feat2 = feat2[:,:,0].copy()
                    if len(feat2) < (pool_params2['frames']+pool_params2['step']):
                        continue
                    
                    col_feat2 = im2col_sliding_strided(feat2, (pool_params2['frames'], feat2.shape[1]), 
                                                       stepsize=pool_params2['step']).T
                    tmp2 = pool_params2['pool_func'](np.reshape(col_feat2, (col_feat2.shape[0], 
                                                                            pool_params2['frames'], 
                                                                            feat2.shape[1])), axis=1)
            except Exception as e:
                print(f, e)
                continue

            if tmp1 is not None and tmp2 is not None:
                sz = np.min([len(tmp1), len(tmp2)])
                all_feat[f] = np.hstack((tmp1[:sz, :], tmp2[:sz, :]))
            else:
                all_feat[f] = tmp1.copy() if tmp1 is not None else tmp2.copy()
        
        all_feat = np.vstack(list(all_feat.values()))
        if N > 0:
            np.random.seed(seed=0)
            idx = np.random.choice(range(len(all_feat)), np.min([N, len(all_feat)]), replace=False)
            all_feat = all_feat[idx, :].copy()
        cur_df = pd.DataFrame(data=np.zeros((len(all_feat), 1), dtype=np.int32)+k, columns=['label'])
        
        cur_df['name'] = all_keys[k]
        cur_df['emb'] = all_feat.tolist()
        full_emb_df[all_keys[k]] = cur_df        
    
    return full_emb_df

def get_balanced_data(in_repo, in_names, lbls):
    
    print(in_names[lbls==1])
    pos_X = np.array(list(in_repo[in_names[lbls==1][0]]['emb']))
    per_file_cnt = int(np.ceil(len(pos_X)/np.sum(lbls==0)))
    
    full_neg = {}
    for i in np.argwhere(lbls==0).ravel():
        print(in_names[i])
        cur_neg = np.array(list(in_repo[in_names[i]]['emb']))
        np.random.seed(0)
        idx = np.random.choice(range(len(cur_neg)), np.min([len(cur_neg), per_file_cnt]), replace=False)
        full_neg[i] = cur_neg[idx, :].copy()
    
    full_neg = np.vstack(list(full_neg.values()))
    
    return np.vstack((pos_X, full_neg)), np.concatenate((np.ones((len(pos_X), )), np.zeros((len(full_neg), ))), axis=0)

def train_forgery_detector(bs_fldr1, bs_fldr2, pool_params1, pool_params2, do_cv=True):
    
    train_files, val_files, test_files = train_test()
    
    # these are the table values
    train_ldrs = np.array(['FF_orig', 'FF_Deepfakes', 'FF_FaceSwap', 'FF_Face2Face', 'FF_NeuralTextures'])
    lbls = np.array([1, 0, 0, 0, 0])
    
    train_repo = build_FF_Ref_two_feat(bs_fldr1, bs_fldr2, 
                                       {f: train_files[f] for f in train_ldrs}, 
                                       pool_params1, pool_params2)
    val_repo = build_FF_Ref_two_feat(bs_fldr1, bs_fldr2, 
                                       {f: val_files[f] for f in train_ldrs}, 
                                       pool_params1, pool_params2)
    test_repo = build_FF_Ref_two_feat(bs_fldr1, bs_fldr2, 
                                       {f: test_files[f] for f in train_ldrs}, 
                                       pool_params1, pool_params2)    
    
    # train the SVM classifier 
    X_train, y_train = get_balanced_data(train_repo, train_ldrs, lbls)
    X_val, y_val = get_balanced_data(val_repo, train_ldrs, lbls)
    
    # train SVM classifier with best params
    n1 = '_'.join(bs_fldr1.split('/')) if bs_fldr1 is not None else ''
    n2 = '_'.join(bs_fldr2.split('/')) if bs_fldr2 is not None else ''
    svm_model = u.load_obj('.', n1 + '_' + n2)
    if svm_model is None:
        svm_model = train_SVC(X_train, y_train, X_val, y_val, do_cv=do_cv)
        u.save_obj(svm_model, '.', n1 + '_' + n2 )
    
    # test on the test data
    i = 0
    for f in train_ldrs:
        
        X_test = svm_model['scaler'].transform(np.array(list(test_repo[f]['emb'])))
        y_test = np.zeros((len(X_test), ), dtype=np.int32) + lbls[i]
        y_pred = svm_model['model'].predict(X_test)
        
        print(f'{f} acc: {np.sum(y_test == y_pred)/len(y_pred)}')
        i = i+1
        
    return svm_model

if __name__ == '__main__':
    
    ### Face Forensics
    svm_model = train_forgery_detector('/data/home/shruti/voxceleb/fabnet_metric',
                               '/data/home/shruti/voxceleb/vgg/leaders', 
                               {'frames': 1, 'step': 1, 'pool_func': np.mean, 'a': 1 }, 
                               {'frames': 100, 'step': 5, 'pool_func': np.mean, 'a': 1 }, do_cv=False)