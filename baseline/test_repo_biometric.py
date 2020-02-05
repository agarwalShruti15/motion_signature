import numpy as np
import matplotlib.pyplot as plt
import os
import utils as u
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import multiprocessing
from joblib import Parallel, delayed
import natsort

def load_file_names(bs_fldr, in_fldr, join_bsfldr=False):
    if join_bsfldr:
        return [os.path.join(bs_fldr, in_fldr, f) for f in os.listdir(os.path.join(bs_fldr, in_fldr)) if f.endswith('.npy')]
    else:
        return [os.path.join(in_fldr, f) for f in os.listdir(os.path.join(bs_fldr, in_fldr)) if f.endswith('.npy')]

# build the repo
# collect the files with which repo needs to build    
# construct a class with these file names to read the embeddings
# I need the embeddings, each embedding should have a label and a name, filename
# for each label get N embeddings to keep in the repo

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

def build_ref(bs_fldr, file_dict, pool_params, N):

    # for each unique label
    full_emb_df = {}
    all_keys = np.sort(list(file_dict.keys()))
    for k in range(len(all_keys)):

        all_feat = {}
        for f in file_dict[all_keys[k]]:
            
            try:                 
                
                feat = np.load(os.path.join(bs_fldr, f))
                if len(feat.shape)>2:
                    feat = feat[:,:,0].copy()
                if len(feat) < (pool_params['frames']+pool_params['step']):
                    continue
                
                # pool the features according to pool_params
                # for vgg pool all 100 frames, whereas for resnet3D pool only 84 frames. 
                # for then step to another window to pool. The pool function is given in the params
                col_feat = im2col_sliding_strided(feat, (pool_params['frames'], feat.shape[1]), stepsize=pool_params['step']).T
                tmp = pool_params['pool_func'](np.reshape(col_feat, (col_feat.shape[0], pool_params['frames'], feat.shape[1])), axis=1)
                all_feat[f] = tmp.copy()
                
            except Exception as e:
                print(f)

        all_feat = np.vstack(list(all_feat.values()))
        if N > 0:
            np.random.seed(seed=0)
            idx = np.random.choice(range(len(all_feat)), np.min([N, len(all_feat)]), replace=False)
            all_feat = all_feat[idx, :].copy()
        cur_df = pd.DataFrame(data=np.zeros((len(all_feat), 1), dtype=np.int32)+k, columns=['label'])
        cur_df['name'] = all_keys[k]
        cur_df['emb'] = all_feat.tolist()
        full_emb_df[k] = cur_df

    # then for each file read all the files 
    return pd.concat(list(full_emb_df.values()), ignore_index=True, sort=False)


def build_FF_Ref(bs_fldr, file_dict, pool_params, N):
    
    # for each unique label
    full_emb_train_df = {}
    full_emb_test_df = {}
    
    all_keys = np.sort(list(file_dict.keys()))
    for k in range(len(all_keys)):

        train_feat = {}
        test_feat = {}
        for f in file_dict[all_keys[k]]:

            feat = np.load(os.path.join(bs_fldr, f))
            if len(feat.shape)>2:
                feat = feat[:,:,0].copy()            
            if len(feat) < (pool_params['frames']+pool_params['step']):
                continue
            # pool the features according to pool_params
            # for vgg pool all 100 frames, whereas for resnet3D pool only 84 frames. 
            # for then step to another window to pool. The pool function is given in the params
            col_feat = im2col_sliding_strided(feat, (pool_params['frames'], feat.shape[1]), stepsize=pool_params['step']).T
            tmp = pool_params['pool_func'](np.reshape(col_feat, (col_feat.shape[0], pool_params['frames'], feat.shape[1])), axis=1)
            
            n1 = int(N*len(tmp))
            train_feat[f] = tmp[:n1, :].copy()
            test_feat[f] = tmp[n1:, :].copy()
            
        full_emb_train_df[all_keys[k]] = train_feat.copy()
        full_emb_test_df[all_keys[k]] = test_feat.copy()

    # then for each file read all the files 
    return full_emb_train_df, full_emb_test_df

# train and test split
def train_test_full_repo(bs_fldr):

    train_ldr_dict = u.load_dict_file('/data/home/shruti/voxceleb/motion_signature/data/utils/leaders_100_train.txt')
    test_ldr_dict = u.load_dict_file('/data/home/shruti/voxceleb/motion_signature/data/utils/leaders_100_test.txt')
    train_add_name = ['bo_imposter', 'bs_imposter', 'ew_imposter', 'dt_imposter', 'hc_imposter', 'jb_imposter', 'FF_orig']

    # add folders to the test label dict
    fake_name = ['bo_faceswap', 'bo_UWfake', 'bs_faceswap', 'ew_faceswap', 'dt_faceswap', 'hc_faceswap', 'jb_faceswap']

    # test and train fakes
    for k in fake_name:
        cur_files = load_file_names(bs_fldr, k)
        cur_files = np.sort(cur_files)
        test_ldr_dict[k] = cur_files.copy()
        
    # test and train fakes
    for k in train_add_name:
        cur_files = load_file_names(bs_fldr, k)
        cur_files = np.sort(cur_files)
        train_ldr_dict[k] = cur_files.copy() 
        
    return train_ldr_dict, test_ldr_dict

def train_test_FF(bs_fldr):
    
    # add folders to the test label dict
    fake_name = ['FF_orig', 'FF_Deepfakes', 'FF_FaceSwap', 'FF_Face2Face', 'FF_NeuralTextures']
    
    train_ldr_dict = {}
    # test and train fakes
    for k in fake_name:
        cur_files = load_file_names(bs_fldr, k)
        train_ldr_dict[k] = np.sort(cur_files)
        
    return train_ldr_dict

def get_FF_src2behav_dict(bsfldr, fldr):
    
    # read all the filenames
    # if only one underscore then same mapping
    # if two underscores then different mapping
    cur_files = load_file_names(bsfldr, fldr)
    
    out_dict = {}
    for i in cur_files:
        
        lbls = i.split('/')[-1]
        if lbls[0] == '_':
            lbls = lbls[1:]
        
        lbls = lbls.split('.')[0].split('_')
        if len(lbls)==2:
            out_dict[int(lbls[1])] = int(lbls[0])
        else:
            out_dict[int(lbls[0])] = int(lbls[0])
        
    return out_dict

def get_leaders_lblToName_dict():
    
    # read all the filenames
    # if only one underscore then same mapping
    # if two underscores then different mapping
    # the name of the leader to a number mapping
    label_dict = {0:'bo', 1:'br', 2:'bs', 3:'cb', 4:'dt', 5:'ew', 6:'hc', 7:'jb', 8:'kh', 9:'pb', 
                  10:'bo_imposter', 11:'bs_imposter', 12:'ew_imposter', 13:'dt_imposter', 14:'hc_imposter', 
                  15:'jb_imposter', 16:'FF_orig'}
    return label_dict


def one_cos_dist(a, b):
    cur_sim = np.sum(a * b, axis=1)
    return np.max(cur_sim), np.argmax(cur_sim)

def one_cos_dist_merge(a1, b1, a2, b2, a, b):
    cur_sim1 = np.sum(a1 * b1, axis=1)
    cur_sim2 = np.sum(a2 * b2, axis=1)    
    sim_mat = (a*cur_sim1 + b*cur_sim2)
    return np.max(sim_mat), np.argmax(sim_mat)

def one_cos_dist_discrim(a1, b1, a2, b2, a, b):
    cur_sim1 = np.sum(a1 * b1, axis=1)
    cur_sim2 = np.sum(a2 * b2, axis=1)
    return cur_sim2[np.argmax(cur_sim1)], np.argmax(cur_sim1)
    
def get_dist_one(train_emb, test_emb):
    
    # normalize the embeddings
    train_emb = train_emb - np.mean(train_emb, axis=1, keepdims=True)
    train_emb = train_emb/ np.linalg.norm(train_emb, axis=1, keepdims=True)
    
    test_emb = test_emb - np.mean(test_emb, axis=1, keepdims=True)
    test_emb = test_emb/ np.linalg.norm(test_emb, axis=1, keepdims=True)
    
    sim_mat = np.matmul(train_emb, test_emb.T)
        
    return sim_mat
    
# get the cosine similarity
def get_results(train_repo, test_repo, parallel=False):
    
    train_emb = np.array(list(train_repo['emb']))
    test_emb = np.array(list(test_repo['emb']))
    
    # normalize the embeddings
    train_emb = train_emb - np.mean(train_emb, axis=1, keepdims=True)
    train_emb = train_emb/ np.linalg.norm(train_emb, axis=1, keepdims=True)
    
    test_emb = test_emb - np.mean(test_emb, axis=1, keepdims=True)
    test_emb = test_emb/ np.linalg.norm(test_emb, axis=1, keepdims=True)
    
    if parallel:
        
        full_struct = []
        for i in range(len(test_emb)):#
            full_struct.append((train_emb, test_emb[[i], :]))
        
        num_cores = 70#multiprocessing.cpu_count()
        #print('Number of Cores {} \n'.format(num_cores))
        #print('total processes {}'.format(len(full_struct)))
        max_ans = Parallel(n_jobs=num_cores, verbose=1)(delayed(one_cos_dist)(*full_struct[c]) for c in range(len(full_struct)))
        max_ans = np.array(max_ans)
        max_val = max_ans[:, 0]
        max_id = max_ans[:, 1].astype(np.int32)
        
    else:
        
        sim_mat = np.matmul(train_emb, test_emb.T)
        max_val = np.max(sim_mat, axis=0)
        max_id = np.argmax(sim_mat, axis=0)
    
    result_df = pd.DataFrame(data=[test_repo['name'].iloc[i] for i in range(len(max_id))], columns=['true_name'])
    result_df['pred_name'] = [train_repo['name'].iloc[i] for i in max_id]
    result_df['pred_label'] = [train_repo['label'].iloc[i] for i in max_id]
    result_df['dist'] = [max_val[i] for i in range(len(max_id))]
    
    return result_df

def get_current_label(infile):
    
    #if the file name has two numbers then last number is label
    #if there is only one number then original 
    lbls = infile.split('/')[-1]
    if lbls[0] == '_':
        lbls = lbls[1:]
    
    lbls = lbls.split('.')[0].split('_')
    if len(lbls)==2:
        out_lbl = int(lbls[1]) 
    else:
        out_lbl = int(lbls[0])
    
    return out_lbl


def get_leaders_label(infile):
    
    # the name of the leader to a number mapping
    label_dict = {'bo': 0, 'br': 1, 'bs': 2, 'cb': 3, 'dt': 4, 'ew': 5, 'hc': 6, 'jb': 7, 'kh': 8, 'pb': 9, 
                         'bo_faceswap':0, 'dt_faceswap':4, 'hc_faceswap':6, 'bs_faceswap':2, 'ew_faceswap':5, 
                         'bo_imposter':10, 'bs_imposter':11, 'ew_imposter':12, 'dt_imposter':13, 
                         'hc_imposter':14, 'jb_imposter':15, 'FF_orig':16, 'bo_UWfake':0, 'jb_faceswap':7}
    
    lbls = infile.split('/')[0]
    return label_dict[lbls]
    

### Face Forensics
def test_FF_repo_biometric(bs_fldr, pool_params):
    
    train_files = train_test_FF(bs_fldr)

    # these are the table values
    train_ldrs = ['FF_orig', 'FF_Deepfakes', 'FF_FaceSwap', 'FF_Face2Face', 'FF_NeuralTextures']
    train_repo, test_repo = build_FF_Ref(bs_fldr, {f: train_files[f] for f in train_ldrs}, pool_params, 0.5)
    
    train_cases = 'FF_orig'
    test_names = ['FF_Deepfakes', 'FF_FaceSwap', 'FF_Face2Face', 'FF_NeuralTextures']
    
    # src embedding
    src_emb = train_repo[train_cases]
    
    # file names, which should be 1000 in number and should also correspond to the identities
    src_fl_names = list(src_emb.keys())
    print(f'number of source ids : {len(list(src_fl_names))}')
    src_lbls = natsort.natsorted(src_fl_names)
    
    results = {}
    
    # these are all the test keys
    for k in train_repo.keys():
        
        cur_test_emb = test_repo[k]
        
        results[k] = {}
        results[k]['dist'] = []
        results[k]['pred lbl'] = []
        results[k]['actual lbl'] = []
        for j in cur_test_emb.keys():
            
            # get the actual label
            cur_act_lbl = get_current_label(j)
            max_sim = np.array([-np.inf for x in range(len(cur_test_emb[j]))])
            pred_label = np.array([0 for x in range(len(cur_test_emb[j]))])
            actual_label = np.array([cur_act_lbl for x in range(len(cur_test_emb[j]))])
            
            for i in src_lbls:
                
                sim_mat = cosine_similarity(src_emb[i], cur_test_emb[j], dense_output=True)
                cur_max_sim = np.max(sim_mat, axis=0)
                chng_id = cur_max_sim>max_sim
                max_sim[chng_id] = cur_max_sim[chng_id]
                pred_label[chng_id] = int(i.split('/')[-1].split('.')[0])
            
            results[k]['dist'] = results[k]['dist'] + list(max_sim)
            results[k]['pred lbl'] = results[k]['pred lbl'] + list(pred_label)
            results[k]['actual lbl'] = results[k]['actual lbl'] + list(actual_label)
    
    acc = {}
    # the labels of pos class
    for k in train_repo.keys():
        
        acc[k] = len(np.argwhere(np.array(results[k]['pred lbl']) == np.array(results[k]['actual lbl'])).ravel())/len(results[k]['actual lbl'])
        
    return results, acc
    

### Leaders
def test_leaders_repo_biometric(bs_fldr, pool_params):
    
    train_files, test_files = train_test_full_repo(bs_fldr)

    # these are the table values
    train_repo = build_ref(bs_fldr, train_files, pool_params, 5000)
    test_repo = build_ref(bs_fldr, test_files, pool_params, 5000)
    
    # file names, which should be 1000 in number and should also correspond to the identities
    src_fl_names = list(train_files.keys())
    print(f'number of source ids : {len(list(src_fl_names))}')
    src_lbls = natsort.natsorted(src_fl_names)
    
    results = {}
    for j in test_files.keys():
        
        results[j] = {}
        results[j]['dist'] = []
        results[j]['pred lbl'] = []
        results[j]['actual lbl'] = []
        
        cur_test_emb = test_repo[np.array(test_repo['name'])==j]
        
        # get the actual label
        cur_act_lbl = get_leaders_label(j)
        max_sim = np.array([-np.inf for x in range(len(cur_test_emb))])
        pred_label = np.array([0 for x in range(len(cur_test_emb))])
        actual_label = np.array([cur_act_lbl for x in range(len(cur_test_emb))])
        
        for i in src_lbls:
            
            cur_train_emb = train_repo[np.array(train_repo['name'])==i]
            cur_res = get_results(cur_train_emb, cur_test_emb)
            cur_max_sim = np.array(cur_res['dist'])
            
            chng_id = cur_max_sim>max_sim
            max_sim[chng_id] = cur_max_sim[chng_id]
            pred_label[chng_id] = get_leaders_label(i)
        
        results[j]['dist'] = results[j]['dist'] + list(max_sim)
        results[j]['pred lbl'] = results[j]['pred lbl'] + list(pred_label)
        results[j]['actual lbl'] = results[j]['actual lbl'] + list(actual_label)
    
    acc = {}
    # the labels of pos class
    for k in test_files.keys():
        
        acc[k] = len(np.argwhere(np.array(results[k]['pred lbl']) == np.array(results[k]['actual lbl'])).ravel())/len(results[k]['actual lbl'])
        
    return results, acc





#----------------------------------------ID specific Functions ---------------------------------------


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
                      '__kitchen_pan']


# get all the identity specific train and test files for all available dataset
def train_test_all_ids(bs_fldr):
    
    # REPO files are all the real files
    train_dict = u.load_dict_file('/data/home/shruti/voxceleb/motion_signature/data/utils/leaders_100_train.txt', join_bsfldr=False, inbsfldr=bs_fldr) # leaders repo file
    for k in ['bo_imposter', 'bs_imposter', 'ew_imposter', 'dt_imposter', 'hc_imposter', 'jb_imposter']:   # imposter videos
        train_dict[k] = load_file_names(bs_fldr, k, join_bsfldr=False)
    for k in range(1000):
        cur_name = '{0:03d}'.format(k)
        train_dict['FF_'+cur_name] = [os.path.join('FF_orig', cur_name + '.npy')]
        
    
    # TEST files, two dict for real and fake
    test_dict = {}; test_dict['real'] = {}; test_dict['fake'] = {}; 
    #REAL
    test_dict['real'] = u.load_dict_file('/data/home/shruti/voxceleb/motion_signature/data/utils/leaders_100_test.txt', join_bsfldr=False, inbsfldr=bs_fldr)
    for k in range(1000):
        cur_name = '{0:03d}'.format(k)
        test_dict['real']['FF_'+cur_name] = [os.path.join('FF_orig', cur_name + '.npy')]
    
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
            assert len(df)==1, f'{cur_name} {df} {ff_files}'
            
            if cur_name not in list(test_dict['fake'].keys()):
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
        
    # steve bushemi and jennifer lawrence
    train_dict['steve_b'] = load_file_names(bs_fldr, 'steve_b', join_bsfldr=False)
    train_dict['jen_l'] = load_file_names(bs_fldr, 'jennifer_l', join_bsfldr=False)
    # remove pW7TbJJMVak.mp4 file which is ground-truth from 
    train_dict['jen_l'] = [f for f in train_dict['jen_l'] if 'pW7TbJJMVak.npy' not in f]
    
    # steve faceswap
    test_dict['fake']['steve_b'] = load_file_names(bs_fldr, 'steve_faceswap', join_bsfldr=False)
    test_dict['real']['jen_l'] = ['jennifer_l/pW7TbJJMVak.npy']
        
    return train_dict, test_dict


def build_repo(bs_fldr, file_names, params, in_dict={}):
    
    all_keys = np.sort(list(file_names.keys())) # list of all identities
    for k in range(len(all_keys)):

        id_feat = {}
        for f in file_names[all_keys[k]]:
            
            if not os.path.exists(os.path.join(bs_fldr, f)):
                path, file = os.path.split(f)
                if file[0] == '_':
                    f = os.path.join(path, file[1:])
                else:
                    f = os.path.join(path, '_' + file)
            try:
                feat = np.load(os.path.join(bs_fldr, f))    
            except Exception as e:
                print(os.path.join(bs_fldr, f), all_keys[k])
                continue
                
            if len(feat.shape)>2:
                feat = feat[:,:,0].copy()
            if len(feat) < (params['frames']+params['step']):
                continue
            # pool the features according to pool_params
            # for vgg pool all 100 frames, whereas for resnet3D pool only 84 frames. 
            # for then step to another window to pool. The pool function is given in the params
            col_feat = im2col_sliding_strided(feat, (params['frames'], feat.shape[1]), stepsize=params['step']).T
            tmp = params['pool_func'](np.reshape(col_feat, (col_feat.shape[0], params['frames'], feat.shape[1])), axis=1)

            n1 = int(params['n1']*len(tmp))
            n2 = int(params['n2']*len(tmp))
            id_feat[f] = tmp[n1:n2+1, :].copy()
        
        all_id_feat = np.vstack(list(id_feat.values()))
        if params['N'] > 0:
            np.random.seed(seed=0)
            idx = np.random.choice(range(len(all_id_feat)), np.min([params['N'], len(all_id_feat)]), replace=False)
            all_id_feat = all_id_feat[idx, :].copy()

        in_dict[all_keys[k]] = all_id_feat.copy()

    # then for each file read all the files 
    return in_dict



# the majority voting using k neighbours
def get_repo_dist(in_repo, test_emb, k=1):
    
    max_sim = np.array([-np.inf for x in range(len(test_emb))])
    pred_label = np.array([np.str() for x in range(len(test_emb))], dtype=object)
    nn_id_tst = np.sum(np.isnan(test_emb), axis=1)<1
    if not np.any(nn_id_tst):
        return None, None
    
    for i in in_repo.keys():
        
        # remove nan from repo and test_emb
        nn_id = np.sum(np.isnan(in_repo[i]), axis=1)<1
        if not np.any(nn_id_tst):
            continue
        
        sim_mat = cosine_similarity(in_repo[i][nn_id, :], test_emb[nn_id_tst, :], dense_output=True)
        print(sim_mat.shape)
        
        # majority 
        if k>1:
            cur_max_sim = np.mean(-np.sort(-sim_mat, axis=0)[:k, :], axis=0)
        else:
            cur_max_sim = np.max(sim_mat, axis=0)
            
        chng_id = cur_max_sim>max_sim
        max_sim[chng_id] = cur_max_sim[chng_id]
        pred_label[chng_id] = i
        
    return max_sim, pred_label


def build_repo_per_file(bs_fldr, file_names, params, in_dict={}):
    
    all_keys = np.sort(list(file_names.keys())) # list of all identities
    if 'concat' not in list(params.keys()):
        params['concat'] = True
        
    for k in range(len(all_keys)):

        id_feat = {}
        n_p_f = -1
        if params['N']>0: # the number of samples per file
            n_p_f = np.ceil(params['N']/len(file_names[all_keys[k]]))
        n_p_f = int(n_p_f)
        
        for f in file_names[all_keys[k]]:
            
            if not os.path.exists(os.path.join(bs_fldr, f)):
                path, file = os.path.split(f)
                if file[0] == '_':
                    f = os.path.join(path, file[1:])
                else:
                    f = os.path.join(path, '_' + file)
            try:
                feat = np.load(os.path.join(bs_fldr, f))    
            except Exception as e:
                print(os.path.join(bs_fldr, f), all_keys[k])
                continue
                
            if len(feat.shape)>2:
                feat = feat[:,:,0].copy()
            if len(feat) < (params['frames']+params['step']):
                continue
            # pool the features according to pool_params
            # for vgg pool all 100 frames, whereas for resnet3D pool only 84 frames. 
            # for then step to another window to pool. The pool function is given in the params
            col_feat = im2col_sliding_strided(feat, (params['frames'], feat.shape[1]), stepsize=params['step']).T
            tmp = params['pool_func'](np.reshape(col_feat, (col_feat.shape[0], params['frames'], feat.shape[1])), axis=1)

            n1 = int(params['n1']*len(tmp))
            n2 = int(params['n2']*len(tmp))
            tmp = tmp[n1:n2, :].copy()
            
            # sample 
            if n_p_f > 0 and len(tmp)>n_p_f:
                np.random.seed(seed=0)
                idx = np.random.choice(range(len(tmp)), n_p_f, replace=False)
                tmp = tmp[idx, :].copy()
            
            id_feat[f] = tmp.copy()
        
        
        if params['concat']:
            in_dict[all_keys[k]] = np.vstack([id_feat[f] for f in file_names[all_keys[k]]])
        else:
            in_dict[all_keys[k]] = id_feat.copy()

    # then for each file read all the files 
    return in_dict


def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    import matplotlib.pyplot as plt
    from mpl_toolkits import axes_grid1
 
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)