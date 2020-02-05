import numpy as np
import os
import utils as u
import pandas as pd
import multiprocessing
from joblib import Parallel, delayed

def load_file_names(bs_fldr, in_fldr):
    return [os.path.join(in_fldr, f) for f in os.listdir(os.path.join(bs_fldr, in_fldr)) if f.endswith('.npy')]

# model data
def get_model_params():
        
    models = {}
    models['bs_fldr'] = '/data/home/shruti/voxceleb/svm_model'
    models['features'] = ['aus_corr_100', 'fabnet_corr_100_190']
    models['feat_bsfldr'] = ['/data/home/shruti/voxceleb/aus_corr_100', 
                             '/data/home/shruti/voxceleb/fabnet_corr_100_pca2000']
    models['feat_comp'] = [190, 190]
    models['leaders'] = ['bo', 'br', 'bs', 'cb', 'dt', 'ew', 'hc', 'jb', 'kh', 'pb']
    models['test_cases'] = [[['br', 'bs', 'cb', 'dt', 'ew', 'hc', 'jb', 'kh', 'pb', 'FF_orig'], ['bo_faceswap'], 
                             ['bo_imposter'], ['bo_UWfake']], 
                            [['bo', 'bs', 'cb', 'dt', 'ew', 'hc', 'jb', 'kh', 'pb', 'FF_orig']],
                           [['bo', 'br', 'cb', 'dt', 'ew', 'hc', 'jb', 'kh', 'pb', 'FF_orig'], ['bs_faceswap'], 
                             ['bs_imposter']],
                           [['bo', 'br', 'bs', 'dt', 'ew', 'hc', 'jb', 'kh', 'pb', 'FF_orig']],
                           [['bo', 'br', 'bs', 'cb', 'ew', 'hc', 'jb', 'kh', 'pb', 'FF_orig'], ['dt_faceswap'], 
                             ['dt_imposter'] ],
                           [['bo', 'br', 'bs', 'cb', 'dt', 'hc', 'jb', 'kh', 'pb', 'FF_orig'], ['ew_faceswap'], 
                             ['ew_imposter']],
                           [['bo', 'br', 'bs', 'cb', 'dt', 'ew', 'jb', 'kh', 'pb', 'FF_orig'], ['hc_faceswap'], 
                             ['hc_imposter'] ],
                           [['bo', 'br', 'bs', 'cb', 'dt', 'ew', 'hc', 'kh', 'pb', 'FF_orig'], ['jb_faceswap'], 
                             ['jb_imposter'] ],
                           [['bo', 'br', 'bs', 'cb', 'dt', 'ew', 'hc', 'jb', 'pb', 'FF_orig']],
                           [['bo', 'br', 'bs', 'cb', 'dt', 'ew', 'hc', 'jb', 'kh', 'FF_orig']]
                           ]
    
    return models

def one_parmam_test(svm_model, out_fldr, out_file, feat_bsfldr, pos_feat_files, neg_feat_files, ncomps):
        
        print(feat_bsfldr)        
        pos_pred = u.test_svm_model(feat_bsfldr, pos_feat_files, ncomps, svm_model)
        neg_pred = u.test_svm_model(feat_bsfldr, neg_feat_files, ncomps, svm_model)
                
        pred = np.log(np.concatenate([pos_pred, neg_pred], axis=0))
        lbl = np.concatenate([np.ones_like(pos_pred), np.zeros_like(neg_pred)])
        pd.DataFrame(data=np.concatenate((pred[:, np.newaxis], lbl[:, np.newaxis]), axis=1), 
                     columns=['prob', 'label']).to_csv('{}/{}.csv'.format(out_fldr, out_file))
        test_auc = u.save_fig(pred, out_fldr, np.sum(lbl==1), np.sum(lbl==0), out_file, lbl, ' ')
        print('\t \t {}:{} auc:{}'.format(out_file, len(neg_pred), test_auc))

if __name__ == '__main__':

    # model data 
    models = get_model_params()
    
    # test data
    test_label_dict = u.load_dict_file('/data/home/shruti/voxceleb/motion_signature/data/utils/leaders_100_test.txt')
    
    # run all cases for test
    full_struct = []
    for f in range(len(models['features'])):
        
        print(models['features'][f])
        for l in range(len(models['leaders'])):
            
            out_fldr = os.path.join(models['bs_fldr'], models['features'][f], models['leaders'][l])
            os.makedirs(out_fldr, exist_ok=True)
            svm_model = u.load_obj(os.path.join(models['bs_fldr'], models['features'][f]), models['leaders'][l])
            print('\t Loaded SVM model {}'.format(models['leaders'][l]))
            
            pos_feat_files = test_label_dict[models['leaders'][l]]            
            for t in range(len(models['test_cases'][l])):
                
                cases = models['test_cases'][l][t]
                neg_feat_files = np.concatenate([test_label_dict[x] if x in test_label_dict.keys() 
                                        else load_file_names(models['feat_bsfldr'][f], x) for x in cases], 
                                       axis=0)
                
                out_file = '_'.join(cases)
                full_struct.append((svm_model, out_fldr, out_file, models['feat_bsfldr'][f], 
                                    pos_feat_files, neg_feat_files, models['feat_comp'][f]))
                
    # run the jobs in parallel
    num_cores = multiprocessing.cpu_count()
    njobs = np.min([int(num_cores/2), len(full_struct)])
    print('Number of Cores {} \n'.format(num_cores))
    print('total processes {}'.format(len(full_struct)))
    Parallel(n_jobs=njobs, verbose=20)(delayed(one_parmam_test)(*full_struct[c]) for c in range(len(full_struct)))
