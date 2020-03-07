import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import re

def load_dict_file(infile, join_bsfldr=False, inbsfldr=''):
    
    # read the file and collect in label dict 
    label_dict = {}
    with open(infile, 'r') as f:
        for line in f:
            _path, _ = re.split(r",| ", line.strip())
            _label = _path.split('/')[0]
            if _label not in list(label_dict.keys()):
                label_dict[_label] = []
            
            if join_bsfldr:
                label_dict[_label].extend([os.path.join(inbsfldr, _path)])
            else:
                label_dict[_label].extend([_path])
    
    return label_dict

def save_obj(obj, fldr, name ):
    os.makedirs(fldr, exist_ok=True)
    with open(os.path.join(fldr, name + '.pkl'), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(fldr, name ):
    if not os.path.exists(os.path.join(fldr, name + '.pkl')):
        return None
    with open(os.path.join(fldr, name + '.pkl'), 'rb') as f:
        return pickle.load(f)

def train_ovr(X_train, gamma=0.01, nu=0.01):

    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import OneClassSVM

    # Fit your data on the scaler object
    scaler = StandardScaler()
    scaler = scaler.fit(X_train)
    X_scaled = scaler.transform(X_train)

    #init best params to default values
    best_param = {}
    best_param['gamma'] = gamma
    best_param['kernel'] = 'rbf'
    best_param['nu'] = nu

    #SVM model
    clf = OneClassSVM(cache_size=1000)
    print(X_train.shape, best_param)
    svm_model = {}
    svm_model['scaler'] = scaler
    svm_model['model'] = clf.set_params(**best_param)
    
    #perform cross-validation on a small train set
    idx = np.random.choice(range(len(X_scaled)), size=np.min([30000, len(X_scaled)]), replace=False)
    svm_model['model'].fit(X_scaled[idx, :])

    return svm_model


def get_bins_edges(feats, bins):
    s_counts, s_bin_edges = np.histogram(feats, bins=bins)
    s_counts = s_counts/np.sum(s_counts)
    s_bin_edges = s_bin_edges[:-1]
    return s_counts, s_bin_edges


def plotROC(pred, lbl, out_file_name):
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt
    fpr, tpr, thresholds = roc_curve(lbl, pred)
    roc_auc = auc(fpr, tpr)
    fig = plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
    plt.legend(fontsize=20)
    plt.grid(b=True, axis='both')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.axis('equal')
    plt.xlim([0.0, 1.025])
    plt.ylim([0.0, 1.025])
    plt.draw()
    fig.savefig(out_file_name)   # save the figure to file
    plt.close(fig)

    out_df = pd.DataFrame(data=np.concatenate((fpr[:, np.newaxis], tpr[:, np.newaxis], thresholds[:, np.newaxis]), axis=1),
                 columns= ['fpr', 'tpr', 'thresholds'])
    out_df.to_csv(os.path.splitext(out_file_name)[0] + '_data.csv')
    return roc_auc

def test_svm_model(bs_fldr, infiles, ncomps, svm_model):
    
    X_test = {}
    for f in infiles:
        
        X_test[f] = np.load(os.path.join(bs_fldr, f))[:, :ncomps]
        X_test[f] = X_test[f][np.sum(np.isnan(X_test[f]), axis=1)<1, :].copy()
        
    all_feat = np.vstack(list(X_test.values()))
    return svm_model['model'].score_samples(svm_model['scaler'].transform(all_feat))

def remove_keys(cur_dict):
    cur_dict.pop('__header__', None)
    cur_dict.pop('__version__', None)
    cur_dict.pop('__globals__', None)
    return cur_dict

#plot the histogram and roc of the prediction probabibility
def save_fig(pred_prob, fig_fldr, real_n, fake_n, fig_nm, y_in, l_name):

    plt.figure()

    nn_id = np.logical_not(np.logical_or(np.isinf(pred_prob), np.isnan(pred_prob)))
    pred_prob = pred_prob[nn_id]
    y_in = y_in[nn_id]

    bins = np.linspace(np.min(pred_prob), np.max(pred_prob), 100)
    s_counts, s_bin_edges = get_bins_edges(pred_prob[0:real_n], bins)
    plt.bar(s_bin_edges, s_counts, width=s_bin_edges[1] - s_bin_edges[0], color='g', alpha=0.7)
    s_counts, s_bin_edges = get_bins_edges(pred_prob[real_n:], bins)
    plt.bar(s_bin_edges, s_counts, width=s_bin_edges[1] - s_bin_edges[0], color='r', alpha=0.7)
    plt.xlabel('score {}'.format(l_name), fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title('{} green count{} red count{}'.format(fig_nm, real_n, fake_n))
    plt.draw()
    plt.savefig('{}/{}_hist.png'.format(fig_fldr, fig_nm))
    plt.close()
    return plotROC(pred_prob, y_in, '{}/{}_roc.png'.format(fig_fldr, fig_nm))


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

def load_file_names(bs_fldr, in_fldr, join_bsfldr=False):
    if join_bsfldr:
        return [os.path.join(bs_fldr, in_fldr, f) for f in os.listdir(os.path.join(bs_fldr, in_fldr)) if f.endswith('.npy')]
    else:
        return [os.path.join(in_fldr, f) for f in os.listdir(os.path.join(bs_fldr, in_fldr)) if f.endswith('.npy')]


GG_train_files = ['__talking_against_wall', 
                  '__outside_talking_still_laughing', 
                  '__talking_angry_couch',
                  '__podium_speech_happy',
                  '__kitchen_still']
GG_real_test_files = ['__outside_talking_pan_laughing',
                      '__kitchen_pan']
GG_fake_test_files = ['__talking_against_wall', 
                      '__outside_talking_still_laughing', 
                      '__talking_angry_couch',
                      '__podium_speech_happy', 
                      '__outside_talking_pan_laughing',
                      '__kitchen_pan',
                      '__kitchen_still']

# get all the identity specific train and test files for all available dataset
def train_test_all_ids(bs_fldr):
    
    # REPO files are all the real files
    train_dict = load_dict_file('/data/home/shruti/voxceleb/motion_signature/data/utils/leaders_100_train.txt', join_bsfldr=False, inbsfldr=bs_fldr) # leaders repo file
    for k in ['bo_imposter', 'bs_imposter', 'ew_imposter', 'dt_imposter', 'hc_imposter', 'jb_imposter']:   # imposter videos
        train_dict[k] = load_file_names(bs_fldr, k, join_bsfldr=False)
        
    ff_orig_files = load_file_names(bs_fldr, 'FF_orig', join_bsfldr=False)
    for k in range(1000):
        cur_name = '{0:03d}'.format(k)
        df = [f for f in ff_orig_files if cur_name + '.npy' in f]
        if len(df)>0:
            train_dict['FF_'+cur_name] = df
        
    
    # TEST files, two dict for real and fake
    test_dict = {}; test_dict['real'] = {}; test_dict['fake'] = {}; 
    #REAL
    test_dict['real'] = load_dict_file('/data/home/shruti/voxceleb/motion_signature/data/utils/leaders_100_test.txt', join_bsfldr=False, inbsfldr=bs_fldr)
    for k in range(1000):
        cur_name = '{0:03d}'.format(k)
        df = [f for f in ff_orig_files if cur_name + '.npy' in f]
        if len(df)>0:
            test_dict['real']['FF_'+cur_name] = df
    
    #FAKE:leaders
    fake_name = ['bo_faceswap', 'bs_faceswap', 'ew_faceswap', 'dt_faceswap', 'hc_faceswap', 'jb_faceswap']
    lbl = ['bo', 'bs', 'ew', 'dt', 'hc', 'jb']
    for i in range(len(lbl)):
        test_dict['fake'][lbl[i]] = load_file_names(bs_fldr, fake_name[i], join_bsfldr=False)
    
    #FAKE: FaceForensics
    for ff_fldr in ['FF_Deepfakes', 'FF_FaceSwap']:            
        ff_files = load_file_names(bs_fldr, ff_fldr, join_bsfldr=False)
        for k in range(1000):
            cur_name = '{0:03d}'.format(k)
            df = [f for f in ff_files if cur_name + '.npy' in f]
            if len(df)>0:            
                if 'FF_'+cur_name not in list(test_dict['fake'].keys()):
                    test_dict['fake']['FF_'+cur_name] = df
                else:
                    test_dict['fake']['FF_'+cur_name] = test_dict['fake']['FF_'+cur_name] + df
                
    # google dataset original videos divide the original in train and test
    # identify the fake indentites 
    GG_orig_allfiles = load_file_names(bs_fldr, 'GG_orig', join_bsfldr=False)
    GG_fake_allfiles = load_file_names(bs_fldr, 'GG_fake', join_bsfldr=False)
    
    id_lbl = np.unique([f.split('/')[-1][:2] for f in GG_orig_allfiles])  # get the unique ids
    for k in id_lbl:
        
        #real Train
        orig_files = [f for f in GG_orig_allfiles if f.split('/')[-1][:2]==k and 
                      np.any([x in f for x in GG_train_files])] # current file label
        orig_files.sort()
        train_dict['GG_'+k] = orig_files.copy()
        
        #real Test
        orig_test_files = [f for f in GG_orig_allfiles if f.split('/')[-1][:2]==k and np.any([x in f for x in GG_real_test_files])] # current file label
        test_dict['real']['GG_'+k] = orig_test_files.copy()
        
        #fake
        fake_files = [f for f in GG_fake_allfiles if (f'_{k}_' in f.split('/')[-1][2:]) and np.any([x in f for x in GG_fake_test_files])] # current file label
        fake_files.sort()
        test_dict['fake']['GG_'+k] = fake_files.copy()    
    
    # DFDC dataset original videos divide the original in train and test
    # identify the fake indentites 
    DFDC_orig_allfiles = load_file_names(bs_fldr, 'DFDC_orig', join_bsfldr=False)
    DFDC_fake_allfiles = load_file_names(bs_fldr, 'DFDC_fake', join_bsfldr=False)
    
    id_lbl = np.unique([f.split('/')[-1].split('_')[0] for f in DFDC_orig_allfiles])  # get the unique ids
    # remove fake files where either the face or the behavior id is not in the original dataset
    print(f'before {len(DFDC_fake_allfiles)}')
    DFDC_fake_allfiles = [f for f in DFDC_fake_allfiles if f.split('/')[-1].split('_')[0] in id_lbl and f.split('/')[-1].split('_')[1] in id_lbl]
    print(f'after {len(DFDC_fake_allfiles)}')
    
    for k in id_lbl:
        
        #real Train
        orig_files = [f for f in DFDC_orig_allfiles if f.split('/')[-1].split('_')[0]==k] # current file label
        np.random.shuffle(orig_files)
        tr_n = int(0.8*len(orig_files))
        train_dict['DFDC_' + k] = orig_files[:tr_n].copy()
        
        #real Test
        test_dict['real']['DFDC_'+k] = orig_files[tr_n:].copy()
        
        #fake
        test_dict['fake']['DFDC_'+k] = [f for f in DFDC_fake_allfiles if f.split('/')[-1].split('_')[0]==k] # current file label
    
        # CELEB-DF dataset
    CDF_orig_allfiles = load_file_names(bs_fldr, 'celeb_real', join_bsfldr=False)
    CDF_fake_allfiles = load_file_names(bs_fldr, 'celeb_fake', join_bsfldr=False)
    
    id_lbl = np.unique([f.split('/')[-1].split('_')[0] for f in CDF_orig_allfiles])  # get the unique ids    
    for k in id_lbl:
        
        #real Train
        orig_files = [f for f in CDF_orig_allfiles if f.split('/')[-1].split('_')[0]==k] # current file label
        train_dict['CDF_' + k] = orig_files.copy()
        
        #real Test
        test_dict['real']['CDF_'+k] = orig_files.copy()
        
        #fake
        test_dict['fake']['CDF_'+k] = [f for f in CDF_fake_allfiles if f.split('/')[-1].split('_')[1]==k] # current file label
        
    # add new test videos for Jennifer Lawerence
    #train_dict['CDF_id12'] = train_dict['CDF_id12'] + [f for f in jen_l_v if 'pW7TbJJMVak_0' not in f and 'real.npy' not in f]    
    test_dict['real']['CDF_id12'] = test_dict['real']['CDF_id12'] + ['jennifer_l/real.npy', 'jennifer_l/pW7TbJJMVak_0.npy'] #
    
    
    # IN wild
    # steve bushemi and jennifer lawrence
    train_dict['inwild_sb'] = load_file_names(bs_fldr, 'steve_b', join_bsfldr=False)    
    
    # steve faceswap
    test_dict['fake']['inwild_sb'] = load_file_names(bs_fldr, 'steve_faceswap', join_bsfldr=False)
    
    
    # in wild dataset
    inwild_orig_allfiles = load_file_names(bs_fldr, 'inwild_orig', join_bsfldr=False)
    inwild_fake_allfiles = load_file_names(bs_fldr, 'inwild_fake', join_bsfldr=False)
    
    id_lbl = np.unique([f.split('/')[-1].split('_')[0] for f in inwild_orig_allfiles])  # get the unique ids
    for k in id_lbl:
        
        #real Train
        orig_files = [f for f in inwild_orig_allfiles if f.split('/')[-1].split('_')[0]==k] # current file label
        
        # divide in train and test, there are four identities (billie ellish, angela office, bill hader, tom crusie)
        # for bill hader and angela office remove the test-fake utterance from the repo and add it to the test set instead
        #bh_real_test = ['bh_zS1Aee2X3Yc_5', 'bh_zS1Aee2X3Yc_6', 'bh_zS1Aee2X3Yc_8']
        bh_real_test = ['bh_zS1Aee2X3Yc_2', 'bh_zS1Aee2X3Yc_3', 'bh_zS1Aee2X3Yc_4', 'bh_XGr_yXtJ170', 'bh_zS1Aee2X3Yc_10'] #,
        
        if k == 'bh':
            tr_files = [f for f in orig_files if not np.any([x in f for x in bh_real_test])]
            ts_files = [f for f in orig_files if np.any([x in f for x in bh_real_test])]
            train_dict['inwild_' + k] = tr_files.copy()
            test_dict['real']['inwild_'+k] = ['inwild_orig/bh_XGr_yXtJ170_3.npy', 
                                              'inwild_orig/bh_XGr_yXtJ170_5.npy', 
                                              'inwild_orig/bh_XGr_yXtJ170_6.npy', 
                                              'inwild_orig/bh_XGr_yXtJ170_8.npy', 
                                              'inwild_orig/bh_XGr_yXtJ170_10.npy'] #ts_files.copy() #
            
        if k == 'an':
            tr_files = [f for f in orig_files if not np.any([x in f for x in ['an_w8TeV93Ji7M_9']])]
            ts_files = [f for f in orig_files if np.any([x in f for x in ['an_w8TeV93Ji7M_9']])]
            train_dict['inwild_' + k] = tr_files.copy()
            test_dict['real']['inwild_'+k] = ts_files.copy()            
            
        if k == 'be':
            train_dict['inwild_' + k] = orig_files.copy()
            test_dict['fake']['inwild_'+k] = ['inwild_fake/an_be_03.npy'] # current file label  
            
        if k == 'tc':
            train_dict['inwild_' + k] = orig_files.copy()
            test_dict['fake']['inwild_'+k] = ['inwild_fake/bh_tc_01.npy'] # current file label  
    
    return train_dict, test_dict



# get all the identity specific train and test files for all available dataset
def train_test_half_vids(bs_fldr):
    
    # REPO files are all the real files
    train_dict = load_dict_file('/data/home/shruti/voxceleb/motion_signature/data/utils/leaders_100_train.txt', join_bsfldr=False, inbsfldr=bs_fldr) # leaders repo file
    for k in ['bo_imposter', 'bs_imposter', 'ew_imposter', 'dt_imposter', 'hc_imposter', 'jb_imposter']:   # imposter videos
        train_dict[k] = load_file_names(bs_fldr, k, join_bsfldr=False)
    test_dict = {}; test_dict['real'] = {}; test_dict['fake'] = {}
    #REAL
    test_dict['real'] = load_dict_file('/data/home/shruti/voxceleb/motion_signature/data/utils/leaders_100_test.txt', join_bsfldr=False, inbsfldr=bs_fldr)    
    #FAKE:leaders
    fake_name = ['bo_faceswap', 'bs_faceswap', 'ew_faceswap', 'dt_faceswap', 'hc_faceswap', 'jb_faceswap']
    lbl = ['bo', 'bs', 'ew', 'dt', 'hc', 'jb']
    for i in range(len(lbl)):
        test_dict['fake'][lbl[i]] = load_file_names(bs_fldr, fake_name[i], join_bsfldr=False)
        
    
    # Face-Forensics
    ff_orig_files = load_file_names(bs_fldr, 'FF_orig', join_bsfldr=False)
    for k in range(1000):
        cur_name = '{0:03d}'.format(k)
        df = [f for f in ff_orig_files if cur_name + '.npy' in f]
        if len(df)>0:
            train_dict['FF_'+cur_name] = df    
    for k in range(1000):
        cur_name = '{0:03d}'.format(k)
        df = [f for f in ff_orig_files if cur_name + '.npy' in f]
        if len(df)>0:
            test_dict['real']['FF_'+cur_name] = df
    #FAKE: FaceForensics
    for ff_fldr in ['FF_Deepfakes', 'FF_FaceSwap']:            
        ff_files = load_file_names(bs_fldr, ff_fldr, join_bsfldr=False)
        for k in range(1000):
            cur_name = '{0:03d}'.format(k)
            df = [f for f in ff_files if cur_name + '.npy' in f]
            if len(df)>0:            
                if 'FF_'+cur_name not in list(test_dict['fake'].keys()):
                    test_dict['fake']['FF_'+cur_name] = df
                else:
                    test_dict['fake']['FF_'+cur_name] = test_dict['fake']['FF_'+cur_name] + df
                
    # google dataset original videos divide the original in train and test
    # identify the fake indentites 
    GG_orig_allfiles = load_file_names(bs_fldr, 'GG_orig', join_bsfldr=False)
    GG_fake_allfiles = load_file_names(bs_fldr, 'GG_fake', join_bsfldr=False)
    
    id_lbl = np.unique([f.split('/')[-1][:2] for f in GG_orig_allfiles])  # get the unique ids
    for k in id_lbl:
        
        #real Train
        orig_files = [f for f in GG_orig_allfiles if f.split('/')[-1][:2]==k and 
                      np.any([x in f for x in GG_fake_test_files])] # current file label
        orig_files.sort()
        train_dict['GG_'+k] = orig_files.copy()
        
        #real Test
        orig_test_files = [f for f in GG_orig_allfiles if f.split('/')[-1][:2]==k and np.any([x in f for x in GG_fake_test_files])] # current file label
        test_dict['real']['GG_'+k] = orig_test_files.copy()
        
        #fake
        fake_files = [f for f in GG_fake_allfiles if (f'_{k}_' in f.split('/')[-1][2:]) and np.any([x in f for x in GG_fake_test_files])] # current file label
        fake_files.sort()
        test_dict['fake']['GG_'+k] = fake_files.copy()
    
    
    # DFDC dataset original videos divide the original in train and test
    # identify the fake indentites 
    DFDC_orig_allfiles = load_file_names(bs_fldr, 'DFDC_orig', join_bsfldr=False)
    DFDC_fake_allfiles = load_file_names(bs_fldr, 'DFDC_fake', join_bsfldr=False)
    
    id_lbl = np.unique([f.split('/')[-1].split('_')[0] for f in DFDC_orig_allfiles])  # get the unique ids
    
    # remove fake files where either the face or the behavior id is not in the original dataset
    print(f'before {len(DFDC_fake_allfiles)}')
    DFDC_fake_allfiles = [f for f in DFDC_fake_allfiles if f.split('/')[-1].split('_')[0] in id_lbl and f.split('/')[-1].split('_')[1] in id_lbl]
    print(f'after {len(DFDC_fake_allfiles)}')
    
    for k in id_lbl:
        
        #real Train
        orig_files = [f for f in DFDC_orig_allfiles if f.split('/')[-1].split('_')[0]==k] # current file label
        train_dict['DFDC_' + k] = orig_files.copy()
        
        #real Test
        test_dict['real']['DFDC_'+k] = orig_files.copy()
        
        #fake
        test_dict['fake']['DFDC_'+k] = [f for f in DFDC_fake_allfiles if f.split('/')[-1].split('_')[0]==k] # current file label
    
    # CELEB-DF dataset
    CDF_orig_allfiles = load_file_names(bs_fldr, 'celeb_real', join_bsfldr=False)
    CDF_fake_allfiles = load_file_names(bs_fldr, 'celeb_fake', join_bsfldr=False)
    
    id_lbl = np.unique([f.split('/')[-1].split('_')[0] for f in CDF_orig_allfiles])  # get the unique ids    
    for k in id_lbl:
        
        #real Train
        orig_files = [f for f in CDF_orig_allfiles if f.split('/')[-1].split('_')[0]==k] # current file label
        train_dict['CDF_' + k] = orig_files.copy()
        
        #real Test
        test_dict['real']['CDF_'+k] = orig_files.copy()
        
        #fake
        test_dict['fake']['CDF_'+k] = [f for f in CDF_fake_allfiles if f.split('/')[-1].split('_')[1]==k] # current file label
        
    # add new test videos for Jennifer Lawerence
    jen_l_v = load_file_names(bs_fldr, 'jennifer_l', join_bsfldr=False)
    # remove pW7TbJJMVak.mp4 file which is ground-truth from 
    #train_dict['CDF_id12'] = train_dict['CDF_id12'] + [f for f in jen_l_v if 'pW7TbJJMVak_0' not in f and 'real.npy' not in f]    
    test_dict['real']['CDF_id12'] = test_dict['real']['CDF_id12'] + ['jennifer_l/real.npy', 'jennifer_l/pW7TbJJMVak_0.npy'] #
    
    
    # IN wild
    # steve bushemi and jennifer lawrence
    train_dict['inwild_sb'] = load_file_names(bs_fldr, 'steve_b', join_bsfldr=False)    
    
    # steve faceswap
    test_dict['fake']['inwild_sb'] = load_file_names(bs_fldr, 'steve_faceswap', join_bsfldr=False)
    
    
    # in wild dataset
    inwild_orig_allfiles = load_file_names(bs_fldr, 'inwild_orig', join_bsfldr=False)
    inwild_fake_allfiles = load_file_names(bs_fldr, 'inwild_fake', join_bsfldr=False)
    
    id_lbl = np.unique([f.split('/')[-1].split('_')[0] for f in inwild_orig_allfiles])  # get the unique ids
    for k in id_lbl:
        
        #real Train
        orig_files = [f for f in inwild_orig_allfiles if f.split('/')[-1].split('_')[0]==k] # current file label
        
        # divide in train and test, there are four identities (billie ellish, angela office, bill hader, tom crusie)
        # for bill hader and angela office remove the test-fake utterance from the repo and add it to the test set instead
        #bh_real_test = ['bh_zS1Aee2X3Yc_5', 'bh_zS1Aee2X3Yc_6', 'bh_zS1Aee2X3Yc_8']
        bh_real_test = ['bh_zS1Aee2X3Yc_2', 'bh_zS1Aee2X3Yc_3', 'bh_zS1Aee2X3Yc_4', 'bh_XGr_yXtJ170', 'bh_zS1Aee2X3Yc_10'] #,
        
        if k == 'bh':
            tr_files = [f for f in orig_files if not np.any([x in f for x in bh_real_test])]
            ts_files = [f for f in orig_files if np.any([x in f for x in bh_real_test])]
            train_dict['inwild_' + k] = tr_files.copy()
            test_dict['real']['inwild_'+k] = ['inwild_orig/bh_XGr_yXtJ170_3.npy', 
                                              'inwild_orig/bh_XGr_yXtJ170_5.npy', 
                                              'inwild_orig/bh_XGr_yXtJ170_6.npy', 
                                              'inwild_orig/bh_XGr_yXtJ170_8.npy', 
                                              'inwild_orig/bh_XGr_yXtJ170_10.npy'] #ts_files.copy() #
            
        if k == 'an':
            tr_files = [f for f in orig_files if not np.any([x in f for x in ['an_w8TeV93Ji7M_9']])]
            ts_files = [f for f in orig_files if np.any([x in f for x in ['an_w8TeV93Ji7M_9']])]
            train_dict['inwild_' + k] = tr_files.copy()
            test_dict['real']['inwild_'+k] = ts_files.copy()            
            
        if k == 'be':
            train_dict['inwild_' + k] = orig_files.copy()
            test_dict['fake']['inwild_'+k] = ['inwild_fake/an_be_03.npy'] # current file label  
            
        if k == 'tc':
            train_dict['inwild_' + k] = orig_files.copy()
            test_dict['fake']['inwild_'+k] = ['inwild_fake/bh_tc_01.npy'] # current file label  

    return train_dict, test_dict



def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    from mpl_toolkits import axes_grid1
 
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)
