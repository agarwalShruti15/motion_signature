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

def load_file_names(bs_fldr, in_fldr):
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

def build_ref_two_feat(bs_fldr_f1, bs_fldr_f2, file_dict, pool_params1, pool_params2, N):

    # for each unique label
    full_emb_df = {}
    all_keys = np.sort(list(file_dict.keys()))
    for k in range(len(all_keys)):

        all_feat1 = {}
        all_feat2 = {}
        for f in file_dict[all_keys[k]]:

            feat1 = np.load(os.path.join(bs_fldr_f1, f))
            if len(feat1.shape)>2:
                feat1 = feat1[:,:,0].copy()
                
            feat2 = np.load(os.path.join(bs_fldr_f2, f))
            if len(feat2.shape)>2:
                feat2 = feat2[:,:,0].copy()
                
            if len(feat1) < (pool_params1['frames']+pool_params1['step']):
                continue
            
            if len(feat2) < (pool_params2['frames']+pool_params2['step']):
                continue
            
            # pool the features according to pool_params
            # for vgg pool all 100 frames, whereas for resnet3D pool only 84 frames. 
            # for then step to another window to pool. The pool function is given in the params
            col_feat1 = im2col_sliding_strided(feat1, (pool_params1['frames'], 
                                                       feat1.shape[1]), 
                                               stepsize=pool_params1['step']).T
            tmp1 = pool_params1['pool_func'](np.reshape(col_feat1, (col_feat1.shape[0], 
                                                                   pool_params1['frames'], 
                                                                   feat1.shape[1])), axis=1)
            
            col_feat2 = im2col_sliding_strided(feat2, (pool_params2['frames'], 
                                                       feat2.shape[1]), 
                                               stepsize=pool_params2['step']).T
            tmp2 = pool_params2['pool_func'](np.reshape(col_feat2, (col_feat2.shape[0], 
                                                                   pool_params2['frames'], 
                                                                   feat2.shape[1])), axis=1)
            
            sz = np.min([len(tmp1), len(tmp2)])
            #if sz != len(tmp2) or sz != len(tmp1):
            #    print(sz, len(tmp1), len(tmp2))
            
            all_feat1[f] = tmp1[:sz, :].copy()
            all_feat2[f] = tmp2[:sz, :].copy()

        all_feat1 = np.vstack(list(all_feat1.values()))
        all_feat2 = np.vstack(list(all_feat2.values()))
        
        cur_df = pd.DataFrame(data=np.zeros((len(all_feat1), 1), dtype=np.int32)+k, columns=['label'])
        cur_df['name'] = all_keys[k]
        cur_df['emb1'] = all_feat1.tolist()
        cur_df['emb2'] = all_feat2.tolist()
        full_emb_df[k] = cur_df
        if N > 0:
            np.random.seed(seed=0)
            full_emb_df[k] = full_emb_df[k].sample(n=np.min([N, len(full_emb_df[k])]), replace=False, random_state=0).copy()

    # then for each file read all the files 
    return pd.concat(list(full_emb_df.values()), ignore_index=True, sort=False)



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


def build_FF_Ref_two_feat(bs_fldr_f1, bs_fldr_f2, file_dict, pool_params1, pool_params2, N):
    
    # for each unique label
    full_emb_train_df = {}
    full_emb_train_df['emb1'] = {}
    full_emb_train_df['emb2'] = {}
    full_emb_test_df = {}
    full_emb_test_df['emb1'] = {}
    full_emb_test_df['emb2'] = {}
    all_keys = np.sort(list(file_dict.keys()))
    for k in range(len(all_keys)):

        all_feat1 = {}
        all_feat2 = {}
        for f in file_dict[all_keys[k]]:
            
            if os.path.exists(os.path.join(bs_fldr_f1, f)):
                feat1 = np.load(os.path.join(bs_fldr_f1, f))
            else:
                path, file = os.path.split(f)
                f = os.path.join(path, '_' + file)                
                feat1 = np.load(os.path.join(bs_fldr_f1, f))
            if len(feat1.shape)>2:
                feat1 = feat1[:,:,0].copy()

            if os.path.exists(os.path.join(bs_fldr_f2, f)):
                feat2 = np.load(os.path.join(bs_fldr_f2, f))
            else:
                path, file = os.path.split(f)
                f = os.path.join(path, '_' + file)
                feat2 = np.load(os.path.join(bs_fldr_f2, f))
            if len(feat2.shape)>2:
                feat2 = feat2[:,:,0].copy()

            if len(feat1) < (pool_params1['frames']+pool_params1['step']):
                continue

            if len(feat2) < (pool_params2['frames']+pool_params2['step']):
                continue

            # pool the features according to pool_params
            # for vgg pool all 100 frames, whereas for resnet3D pool only 84 frames. 
            # for then step to another window to pool. The pool function is given in the params
            col_feat1 = im2col_sliding_strided(feat1, (pool_params1['frames'], 
                                                       feat1.shape[1]), 
                                               stepsize=pool_params1['step']).T
            tmp1 = pool_params1['pool_func'](np.reshape(col_feat1, (col_feat1.shape[0], 
                                                                    pool_params1['frames'], 
                                                                   feat1.shape[1])), axis=1)

            col_feat2 = im2col_sliding_strided(feat2, (pool_params2['frames'], 
                                                       feat2.shape[1]), 
                                               stepsize=pool_params2['step']).T
            tmp2 = pool_params2['pool_func'](np.reshape(col_feat2, (col_feat2.shape[0], 
                                                                    pool_params2['frames'], 
                                                                   feat2.shape[1])), axis=1)

            sz = np.min([len(tmp1), len(tmp2)])
            #if sz != len(tmp2) or sz != len(tmp1):
            #    print(sz, len(tmp1), len(tmp2))

            all_feat1[f] = tmp1[:sz, :].copy()
            all_feat2[f] = tmp2[:sz, :].copy()

        full_emb_train_df['emb1'][all_keys[k]] = {f: all_feat1[f][:int(N*len(all_feat1[f])), :] for f in all_feat1.keys()}
        full_emb_train_df['emb2'][all_keys[k]] = {f: all_feat2[f][:int(N*len(all_feat2[f])), :] for f in all_feat2.keys()}
        full_emb_test_df['emb1'][all_keys[k]] = {f: all_feat1[f][int(N*len(all_feat1[f])):, :] for f in all_feat1.keys()}
        full_emb_test_df['emb2'][all_keys[k]] = {f: all_feat2[f][int(N*len(all_feat2[f])):, :] for f in all_feat2.keys()}        
        
    return full_emb_train_df, full_emb_test_df

# train and test split

# train and test split
def train_test(bs_fldr):

    train_ldr_dict = u.load_dict_file('/data/home/shruti/voxceleb/motion_signature/data/utils/leaders_100_train.txt')
    test_ldr_dict = u.load_dict_file('/data/home/shruti/voxceleb/motion_signature/data/utils/leaders_100_test.txt')

    # add folders to the test label dict
    fake_name = ['bo_imposter', 'bo_faceswap', 'bo_UWfake', 
                  'bs_faceswap', 'bs_imposter', 'ew_faceswap', 
                  'ew_imposter', 'dt_faceswap', 'dt_imposter', 
                  'hc_faceswap', 'hc_imposter', 
                  'jb_faceswap', 'jb_imposter', 'FF_orig']

    # test and train fakes
    for k in fake_name:
        cur_files = load_file_names(bs_fldr, k)
        cur_files = np.sort(cur_files)
        
        # partition
        n1 = int(0.*len(cur_files))
        n2 = len(cur_files) - n1
        
        train_ldr_dict[k] = cur_files[:n1]
        test_ldr_dict[k] = cur_files[n1:]
        
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

def get_results_2feat(train_repo, test_repo, a, b, parallel=False):
    
    train_emb1 = np.array(list(train_repo['emb1']))
    test_emb1 = np.array(list(test_repo['emb1']))
    train_emb2 = np.array(list(train_repo['emb2']))
    test_emb2 = np.array(list(test_repo['emb2']))    
    
    if not parallel:
        sim1 = get_dist_one(train_emb1, test_emb1)        
        sim2 = get_dist_one(train_emb2, test_emb2)
        
        #sim_mat = (a*sim1 + b*sim2)
        #max_val = np.max(sim_mat, axis=0)
        max_id = np.argmax(sim1, axis=0)
        max_val = np.array([sim2[max_id[x], x] for x in range(len(max_id))])
        
    else:
        
        
        # normalize the embeddings
        train_emb1 = train_emb1 - np.mean(train_emb1, axis=1, keepdims=True)
        train_emb1 = train_emb1/ np.linalg.norm(train_emb1, axis=1, keepdims=True)
        test_emb1 = test_emb1 - np.mean(test_emb1, axis=1, keepdims=True)
        test_emb1 = test_emb1/ np.linalg.norm(test_emb1, axis=1, keepdims=True)
        
        train_emb2 = train_emb2 - np.mean(train_emb2, axis=1, keepdims=True)
        train_emb2 = train_emb2/ np.linalg.norm(train_emb2, axis=1, keepdims=True)
        test_emb2 = test_emb2 - np.mean(test_emb2, axis=1, keepdims=True)
        test_emb2 = test_emb2/ np.linalg.norm(test_emb2, axis=1, keepdims=True)
        
        full_struct = []
        for i in range(len(test_emb1)):#
            full_struct.append((train_emb1, test_emb1[[i], :], train_emb2, test_emb2[[i], :], a, b))
        
        num_cores = 70#int(multiprocessing.cpu_count())
        #print('Number of Cores {} \n'.format(num_cores))
        #print('total processes {}'.format(len(full_struct)))
        #max_ans = Parallel(n_jobs=num_cores, verbose=1)(delayed(one_cos_dist_merge)(*full_struct[c]) for c in range(len(full_struct)))
        max_ans = Parallel(n_jobs=num_cores, verbose=1)(delayed(one_cos_dist_discrim)(*full_struct[c]) for c in range(len(full_struct)))
        max_ans = np.array(max_ans)
        max_val = max_ans[:, 0]
        max_id = max_ans[:, 1].astype(np.int32)
                    
    
    result_df = pd.DataFrame(data=[test_repo['name'].iloc[i] for i in range(len(max_id))], columns=['true_name'])
    result_df['pred_name'] = [train_repo['name'].iloc[i] for i in max_id]
    result_df['pred_label'] = [train_repo['label'].iloc[i] for i in max_id]
    result_df['dist'] = [max_val[i] for i in range(len(max_id))]
    
    return result_df

    
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



### Face Forensics
def test_FF_half(bs_fldr, pool_params):
    
    train_files = train_test_FF(bs_fldr)

    # these are the table values
    train_ldrs = ['FF_orig', 'FF_Deepfakes', 'FF_FaceSwap', 'FF_Face2Face', 'FF_NeuralTextures']
    train_repo, test_repo = build_FF_Ref(bs_fldr, {f: train_files[f] for f in train_ldrs}, pool_params, 0.5)
    
    train_cases = 'FF_orig'
    test_names = ['FF_Deepfakes', 'FF_FaceSwap', 'FF_Face2Face', 'FF_NeuralTextures']
    
    results = {}
    src_emb = np.vstack(list(train_repo[train_cases].values()))
    
    np.random.seed(seed=0)
    idx = np.random.choice(range(len(src_emb)), np.min([5000, len(src_emb)]), replace=False)
    src_emb = src_emb[idx, :].copy()
    print(f'source embedding {src_emb.shape}')
    for k in train_repo.keys():
        
        cur_emb = test_repo[k]
        results[k] = {}
        for j in cur_emb.keys():
            
            sim_mat = cosine_similarity(src_emb, 
                                        cur_emb[j],
                                        dense_output=True)
            results[k][j] = np.max(sim_mat, axis=0)
            
        sns.distplot(np.concatenate(list(results[k].values()), axis=0), label=k)
        
    plt.legend()
    plt.show()

    # for each case
    auc_results = np.zeros((1, len(test_names))) + np.nan
    fpr_tpr_t = {}
    pos_dist = np.concatenate(list(results[train_cases].values()), axis=0)
    for t in range(len(test_names)):

        neg_dist = np.concatenate(list(results[test_names[t]].values()), axis=0)
        pred = (np.concatenate([pos_dist, neg_dist], axis=0))
        lbl = np.concatenate([np.ones_like(pos_dist), np.zeros_like(neg_dist)])
        fpr, tpr, thresholds = roc_curve(lbl, pred)
        auc_results[0, t] = auc(fpr, tpr)
        fpr_tpr_t[test_names[t]] = (fpr, tpr, thresholds)
        
    return auc_results, fpr_tpr_t

### Leaders Unseen Identities
def test_unseen_identities(bs_fldr, pool_params, parallel=False):

    train_files, test_files = train_test(bs_fldr)
    leg = ['Diff People', 'Faceswap', 'Imposter']
    
    # these are the table values
    train_ldrs = ['bo', 'bs', 'ew', 'hc', 'dt', 'jb', 'cb', 'pb', 'kh', 'br']
    test_names = ['diff', 'faceswap', 'imposter']
    test_ldrs =  [[['FF_orig'], ['bo_faceswap'], ['bo_imposter']],
                  [['FF_orig'], ['bs_faceswap'], ['bs_imposter']],
                  [['FF_orig'], ['ew_faceswap'], ['ew_imposter']],
                  [['FF_orig'], ['hc_faceswap'], ['hc_imposter'] ],
                  [['FF_orig'], ['dt_faceswap'], ['dt_imposter'] ],
                  [['FF_orig'], ['jb_faceswap'], ['jb_imposter']],
                  [['FF_orig']],
                  [['FF_orig']],
                  [['FF_orig']], 
                  [['FF_orig']]
                 ]
    
    test_ldrs_all =  ['bo', 'br', 'bs', 'cb', 'dt', 'ew', 'hc', 'jb', 'kh', 'pb', 'FF_orig', 
                  'bo_faceswap', 'bo_imposter', 'bs_faceswap', 'bs_imposter', 'dt_faceswap', 'dt_imposter', 
                  'ew_faceswap', 'ew_imposter', 'hc_faceswap', 'hc_imposter', 'jb_imposter', 'jb_faceswap']    
    train_repo_all = build_ref(bs_fldr, {f: train_files[f] for f in train_ldrs}, pool_params, 5000)
    test_repo_all = build_ref(bs_fldr, {f: test_files[f] for f in test_ldrs_all}, pool_params, 5000)

    # for each case
    results = np.zeros((len(train_ldrs), len(test_names))) + np.nan
    for l in range(len(train_ldrs)):

        train_repo = train_repo_all[train_repo_all['name'] == train_ldrs[l]]
        pos_test_repo = test_repo_all[test_repo_all['name'] == train_ldrs[l]]
        pos_result = get_results(train_repo, pos_test_repo, parallel=parallel)
        if train_ldrs[l] == 'jb':
            sns.distplot(np.array(pos_result['dist']), label='jb real')
        
        for t in range(len(test_ldrs[l])):

            neg_test_repo = test_repo_all[test_repo_all['name'].isin(test_ldrs[l][t])]
            neg_results = get_results(train_repo, neg_test_repo, parallel=parallel)

            pred = np.concatenate([np.array(pos_result['dist']), 
                                   np.array(neg_results['dist'])], axis=0)
            lbl = np.concatenate([np.ones_like(np.array(pos_result['dist'])), 
                                  np.zeros_like(np.array(neg_results['dist']))])
            fpr, tpr, thresholds = roc_curve(lbl, pred)
            results[l, t] = auc(fpr, tpr)
            #print(f'{train_ldrs[l]} {test_names[t]} auc: {results[l, t]}')
            if train_ldrs[l] == 'jb':
                sns.distplot(np.array(neg_results['dist']), label=leg[t])
                
        if train_ldrs[l] == 'jb':
            plt.xlabel('Cosine Similarity')
            plt.legend()
            plt.savefig('jb_nocomp.png')
            plt.show()                        

    results_df = pd.DataFrame(data=results, columns=test_names)
    results_df['leader_name'] = train_ldrs
    return results_df

### Leaders Multiple Identities

def test_multiple_identities(bs_fldr, pool_params, parallel=False):
    
    train_files, test_files = train_test(bs_fldr)

    # these are the table values
    train_ldrs = ['bo', 'bs', 'ew', 'hc', 'dt', 'jb', 'cb', 'pb', 'kh', 'br']
    test_ldrs =  ['bo', 'br', 'bs', 'cb', 'dt', 'ew', 'hc', 'jb', 'kh', 'pb', 'FF_orig', 
                  'bo_faceswap', 'bo_imposter', 'bs_faceswap', 'bs_imposter', 'dt_faceswap', 'dt_imposter', 
                  'ew_faceswap', 'ew_imposter', 'hc_faceswap', 'hc_imposter', 'jb_imposter', 'jb_faceswap']
    
    train_repo = build_ref(bs_fldr, {f: train_files[f] for f in train_ldrs}, pool_params, 5000)
    test_repo = build_ref(bs_fldr, {f: test_files[f] for f in test_ldrs}, pool_params, 5000)
    
    full_results = get_results(train_repo, test_repo, parallel=parallel)
    
    # these are the table values
    test_names = ['diff', 'faceswap', 'imposter']
    test_cases =  [[['FF_orig'], ['bo_faceswap'], ['bo_imposter']],
                  [['FF_orig'], ['bs_faceswap'], ['bs_imposter']],
                  [['FF_orig'], ['ew_faceswap'], ['ew_imposter']],
                  [['FF_orig'], ['hc_faceswap'], ['hc_imposter'] ],
                  [['FF_orig'], ['dt_faceswap'], ['dt_imposter'] ],
                  [['FF_orig'], ['jb_faceswap'], ['jb_imposter']],
                  [['FF_orig']],
                  [['FF_orig']],
                  [['FF_orig']], 
                  [['FF_orig']] ]

    # for each case
    results = np.zeros((len(train_ldrs), len(test_names))) + np.nan
    for l in range(len(train_ldrs)):

        pos_dist = np.array(full_results['dist'][full_results['true_name']==train_ldrs[l]])
        if train_ldrs[l] == 'jb':
            sns.distplot(pos_dist, label='jb real')        
        for t in range(len(test_cases[l])):
            
            neg_dist = np.array(full_results['dist'][full_results['true_name'].isin(test_cases[l][t])])

            pred = np.concatenate([pos_dist, neg_dist], axis=0)
            lbl = np.concatenate([np.ones_like(pos_dist), np.zeros_like(neg_dist)])
            fpr, tpr, thresholds = roc_curve(lbl, pred)
            results[l, t] = auc(fpr, tpr)
            #print(f'{train_ldrs[l]} {test_names[t]} auc: {results[l, t]}')
            if train_ldrs[l] == 'jb':
                sns.distplot(neg_dist, label=test_ldrs[l][t][0])
                
        if train_ldrs[l] == 'jb':
            plt.legend()
            plt.show()            

    results_df = pd.DataFrame(data=results, columns=test_names)
    results_df['leader_name'] = train_ldrs
    
    # pivot table
    confusion_mat = full_results.pivot_table(values='dist', 
                                             index='true_name', columns='pred_name', 
                                             aggfunc=lambda x: len(x))
    confusion_mat = confusion_mat/np.nansum(np.array(confusion_mat), axis=1, keepdims=True)
    
    return results_df, confusion_mat



def test_unseen_identities_2feat(bs_fldr1, bs_fldr2, pool_params1, pool_params2, parallel=False):

    train_files, test_files = train_test(bs_fldr1)

    # these are the table values
    train_ldrs = ['bo', 'bs', 'ew', 'hc', 'dt', 'jb', 'cb', 'pb', 'kh', 'br']
    test_names = ['diff', 'faceswap', 'imposter']
    test_ldrs =  [[['FF_orig'], ['bo_faceswap'], ['bo_imposter']],
                  [['FF_orig'], ['bs_faceswap'], ['bs_imposter']],
                  [['FF_orig'], ['ew_faceswap'], ['ew_imposter']],
                  [['FF_orig'], ['hc_faceswap'], ['hc_imposter'] ],
                  [['FF_orig'], ['dt_faceswap'], ['dt_imposter'] ],
                  [['FF_orig'], ['jb_faceswap'], ['jb_imposter']],
                  [['FF_orig']],
                  [['FF_orig']],
                  [['FF_orig']], 
                  [['FF_orig']]
                 ]
    
    test_ldrs_all =  ['bo', 'br', 'bs', 'cb', 'dt', 'ew', 'hc', 'jb', 'kh', 'pb', 'FF_orig', 
                  'bo_faceswap', 'bo_imposter', 'bs_faceswap', 'bs_imposter', 'dt_faceswap', 'dt_imposter', 
                  'ew_faceswap', 'ew_imposter', 'hc_faceswap', 'hc_imposter', 'jb_imposter', 'jb_faceswap']    
    train_repo_all = build_ref_two_feat(bs_fldr1, bs_fldr2, {f: train_files[f] for f in train_ldrs}, pool_params1, pool_params2, 5000)
    test_repo_all = build_ref_two_feat(bs_fldr1, bs_fldr2, {f: test_files[f] for f in test_ldrs_all}, pool_params1, pool_params2, 5000)

    # for each case
    results = np.zeros((len(train_ldrs), len(test_names))) + np.nan
    for l in range(len(train_ldrs)):

        train_repo = train_repo_all[train_repo_all['name'] == train_ldrs[l]]
        pos_test_repo = test_repo_all[test_repo_all['name'] == train_ldrs[l]]
        pos_result = get_results_2feat(train_repo, pos_test_repo, pool_params1['a'], pool_params2['a'], parallel=parallel)
        
        for t in range(len(test_ldrs[l])):

            neg_test_repo = test_repo_all[test_repo_all['name'].isin(test_ldrs[l][t])]
            neg_results = get_results_2feat(train_repo, neg_test_repo, pool_params1['a'], pool_params2['a'], parallel=parallel)
            pred = np.concatenate([np.array(pos_result['dist']), 
                                   np.array(neg_results['dist'])], axis=0)
            lbl = np.concatenate([np.ones_like(np.array(pos_result['dist'])), 
                                  np.zeros_like(np.array(neg_results['dist']))])
            fpr, tpr, thresholds = roc_curve(lbl, pred)
            results[l, t] = auc(fpr, tpr)
            #print(f'{train_ldrs[l]} {test_names[t]} auc: {results[l, t]}')

    results_df = pd.DataFrame(data=results, columns=test_names)
    results_df['leader_name'] = train_ldrs
    return results_df

### Leaders Multiple Identities
def test_multiple_identities_2feat(bs_fldr1, bs_fldr2, pool_params1, pool_params2, parallel=False):
    
    train_files, test_files = train_test(bs_fldr1)

    # these are the table values
    train_ldrs = ['bo', 'bs', 'ew', 'hc', 'dt', 'jb', 'cb', 'pb', 'kh', 'br']
    test_ldrs =  ['bo', 'br', 'bs', 'cb', 'dt', 'ew', 'hc', 'jb', 'kh', 'pb', 'FF_orig', 
                  'bo_faceswap', 'bo_imposter', 'bs_faceswap', 'bs_imposter', 'dt_faceswap', 'dt_imposter', 
                  'ew_faceswap', 'ew_imposter', 'hc_faceswap', 'hc_imposter', 'jb_imposter', 'jb_faceswap']
    
    train_repo = build_ref_two_feat(bs_fldr1, bs_fldr2, {f: train_files[f] for f in train_ldrs}, pool_params1, pool_params2, 5000)
    test_repo = build_ref_two_feat(bs_fldr1, bs_fldr2, {f: test_files[f] for f in test_ldrs}, pool_params1, pool_params2, 5000)
    
    print(np.unique(np.array(test_repo['name'])))
    
    full_results = get_results_2feat(train_repo, test_repo, pool_params1['a'], pool_params2['a'], parallel=parallel)
    
    # these are the table values
    test_names = ['diff', 'faceswap', 'imposter']
    test_cases =  [[['FF_orig'], ['bo_faceswap'], ['bo_imposter']],
                  [['FF_orig'], ['bs_faceswap'], ['bs_imposter']],
                  [['FF_orig'], ['ew_faceswap'], ['ew_imposter']],
                  [['FF_orig'], ['hc_faceswap'], ['hc_imposter'] ],
                  [['FF_orig'], ['dt_faceswap'], ['dt_imposter'] ],
                  [['FF_orig'], ['jb_faceswap'], ['jb_imposter']],
                  [['FF_orig']],
                  [['FF_orig']],
                  [['FF_orig']], 
                  [['FF_orig']] ]

    # for each case
    results = np.zeros((len(train_ldrs), len(test_names))) + np.nan
    for l in range(len(train_ldrs)):

        pos_dist = np.array(full_results['dist'][full_results['true_name']==train_ldrs[l]])        
        for t in range(len(test_cases[l])):
            
            neg_dist = np.array(full_results['dist'][full_results['true_name'].isin(test_cases[l][t])])

            pred = np.concatenate([pos_dist, neg_dist], axis=0)
            lbl = np.concatenate([np.ones_like(pos_dist), np.zeros_like(neg_dist)])
            fpr, tpr, thresholds = roc_curve(lbl, pred)
            results[l, t] = auc(fpr, tpr)
            #print(f'{train_ldrs[l]} {test_names[t]} auc: {results[l, t]}')

    results_df = pd.DataFrame(data=results, columns=test_names)
    results_df['leader_name'] = train_ldrs
    
    # pivot table
    confusion_mat = full_results.pivot_table(values='dist', 
                                             index='true_name', columns='pred_name', 
                                             aggfunc=lambda x: len(x))
    confusion_mat = confusion_mat/np.nansum(np.array(confusion_mat), axis=1, keepdims=True)
    
    return results_df, confusion_mat



### Face Forensics
def test_FF_half_2feat(bs_fldr1, bs_fldr2, pool_params1, pool_params2):
    
    train_files = train_test_FF(bs_fldr1)

    # these are the table values
    train_ldrs = ['FF_orig', 'FF_Deepfakes', 'FF_FaceSwap', 'FF_Face2Face', 'FF_NeuralTextures']
    train_repo, test_repo = build_FF_Ref_two_feat(bs_fldr1, bs_fldr2, 
                                                  {f: train_files[f] for f in train_ldrs}, 
                                                  pool_params1, pool_params2, 0.5)
    
    train_cases = 'FF_orig'
    test_names = ['FF_Deepfakes', 'FF_FaceSwap', 'FF_Face2Face', 'FF_NeuralTextures']
    
    results = {}
    src_emb1 = np.vstack(list(train_repo['emb1'][train_cases].values()))
    src_emb2 = np.vstack(list(train_repo['emb2'][train_cases].values()))
    
    np.random.seed(seed=0)
    idx = np.random.choice(range(len(src_emb1)), np.min([5000, len(src_emb1)]), replace=False)
    src_emb1 = src_emb1[idx, :].copy()
    src_emb2 = src_emb2[idx, :].copy()
    print(f'source embedding {src_emb1.shape}')
    for k in test_repo['emb1'].keys():
        
        cur_emb1 = test_repo['emb1'][k]
        cur_emb2 = test_repo['emb2'][k]
        results[k] = {}
        for j in cur_emb1.keys():
            
            sim_mat1 = cosine_similarity(src_emb1, 
                                        cur_emb1[j],
                                        dense_output=True)
            sim_mat2 = cosine_similarity(src_emb2, 
                                        cur_emb2[j],
                                        dense_output=True)
            
            
            results[k][j] = np.max(pool_params1['a']*sim_mat1 + pool_params2['a']*sim_mat2, axis=0)
            
        sns.distplot(np.concatenate(list(results[k].values()), axis=0), label=k)
        
    plt.legend()
    plt.show()

    # for each case
    auc_results = np.zeros((1, len(test_names))) + np.nan
    fpr_tpr_t = {}
    pos_dist = np.concatenate(list(results[train_cases].values()), axis=0)
    for t in range(len(test_names)):

        neg_dist = np.concatenate(list(results[test_names[t]].values()), axis=0)
        pred = (np.concatenate([pos_dist, neg_dist], axis=0))
        lbl = np.concatenate([np.ones_like(pos_dist), np.zeros_like(neg_dist)])
        fpr, tpr, thresholds = roc_curve(lbl, pred)
        auc_results[0, t] = auc(fpr, tpr)
        fpr_tpr_t[test_names[t]] = (fpr, tpr, thresholds)
        
    return auc_results, fpr_tpr_t


def test_FF_half_2feat_follow(bs_fldr1, bs_fldr2, pool_params1, pool_params2):
    
    train_files = train_test_FF(bs_fldr1)

    # these are the table values
    train_ldrs = ['FF_orig', 'FF_Deepfakes', 'FF_FaceSwap', 'FF_Face2Face', 'FF_NeuralTextures']
    train_repo, test_repo = build_FF_Ref_two_feat(bs_fldr1, bs_fldr2, 
                                                  {f: train_files[f] for f in train_ldrs}, 
                                                  pool_params1, pool_params2, 0.5)
    
    train_cases = 'FF_orig'
    test_names = ['FF_Deepfakes', 'FF_FaceSwap', 'FF_Face2Face', 'FF_NeuralTextures']
    
    results = {}    
    for k in test_repo['emb1'].keys():
        
        train_emb1 = train_repo['emb1'][train_cases]
        train_emb2 = train_repo['emb2'][train_cases]
                
        test_emb1 = np.vstack(list(test_repo['emb1'][k].values()))
        test_emb2 = np.vstack(list(test_repo['emb2'][k].values()))
        
        test_flnm = list(train_emb1.keys())
        max_id_emb1 = np.zeros((len(test_emb1), ), np.int32)
        max_val_emb1 = np.zeros((len(test_emb1), ))
        for i in range(len(test_flnm)):
            
            train_flm = test_flnm[i].split('_')[0] + '.npy'
            sim_mat1 = cosine_similarity(train_emb1[train_flm], test_emb1, dense_output=True)
            cur_max_val = np.max(sim_mat1, axis=0)
            max_id_emb1[cur_max_val > max_val_emb1] = i
            max_val_emb1[cur_max_val > max_val_emb1] = cur_max_val[cur_max_val > max_val_emb1]
            
        
        results[k] = np.zeros((len(test_emb2), ))
        for i in range(len(test_emb2)):
            
            sim_mat2 = cosine_similarity(train_emb2[test_flnm[max_id_emb1[i]]], test_emb2[[i], :], dense_output=True)
            results[k][i] = np.max(sim_mat2, axis=0)
        
            
        sns.distplot(results[k], label=k)
        
    plt.legend()
    plt.show()

    # for each case
    auc_results = np.zeros((1, len(test_names))) + np.nan
    fpr_tpr_t = {}
    pos_dist = results[train_cases]
    for t in range(len(test_names)):

        neg_dist = results[test_names[t]]
        pred = (np.concatenate([pos_dist, neg_dist], axis=0))
        lbl = np.concatenate([np.ones_like(pos_dist), np.zeros_like(neg_dist)])
        fpr, tpr, thresholds = roc_curve(lbl, pred)
        auc_results[0, t] = auc(fpr, tpr)
        fpr_tpr_t[test_names[t]] = (fpr, tpr, thresholds)
        
    return auc_results, fpr_tpr_t    


### Leaders Unseen Identities
def test_comp_unseen_identities(bs_fldr, pool_params, parallel=False):

    leg = ['Diff People', 'Faceswap', 'Imposter']
    train_files, test_files = train_test(bs_fldr)

    # these are the table values
    train_ldrs = ['bo', 'bs', 'ew', 'hc', 'dt', 'jb', 'cb', 'pb', 'kh', 'br']
    test_names = ['diff', 'faceswap', 'imposter']
    test_ldrs =  [[['FF_orig'], ['bo_faceswap'], ['bo_imposter']],
                  [['FF_orig'], ['bs_faceswap'], ['bs_imposter']],
                  [['FF_orig'], ['ew_faceswap'], ['ew_imposter']],
                  [['FF_orig'], ['hc_faceswap'], ['hc_imposter'] ],
                  [['FF_orig'], ['dt_faceswap'], ['dt_imposter'] ],
                  [['FF_orig'], ['jb_faceswap'], ['jb_imposter']],
                  [['FF_orig']],
                  [['FF_orig']],
                  [['FF_orig']], 
                  [['FF_orig']]
                 ]
    
    test_ldrs_all =  ['bo', 'br', 'bs', 'cb', 'dt', 'ew', 'hc', 'jb', 'kh', 'pb', 'FF_orig', 
                  'bo_faceswap', 'bo_imposter', 'bs_faceswap', 'bs_imposter', 'dt_faceswap', 'dt_imposter', 
                  'ew_faceswap', 'ew_imposter', 'hc_faceswap', 'hc_imposter', 'jb_imposter', 'jb_faceswap']    
    train_repo_all = build_ref(bs_fldr, {f: train_files[f] for f in train_ldrs}, pool_params, 5000)
    test_repo_all = build_ref(os.path.join(bs_fldr, 'compression'), {f: test_files[f] for f in test_ldrs_all}, pool_params, 5000)

    # for each case
    results = np.zeros((len(train_ldrs), len(test_names))) + np.nan
    for l in range(len(train_ldrs)):

        train_repo = train_repo_all[train_repo_all['name'] == train_ldrs[l]]
        pos_test_repo = test_repo_all[test_repo_all['name'] == train_ldrs[l]]
        pos_result = get_results(train_repo, pos_test_repo, parallel=parallel)
        if train_ldrs[l] == 'jb':
            sns.distplot(np.array(pos_result['dist']), label='jb real')        
        
        for t in range(len(test_ldrs[l])):

            neg_test_repo = test_repo_all[test_repo_all['name'].isin(test_ldrs[l][t])]
            neg_results = get_results(train_repo, neg_test_repo, parallel=parallel)

            pred = np.concatenate([np.array(pos_result['dist']), 
                                   np.array(neg_results['dist'])], axis=0)
            lbl = np.concatenate([np.ones_like(np.array(pos_result['dist'])), 
                                  np.zeros_like(np.array(neg_results['dist']))])
            fpr, tpr, thresholds = roc_curve(lbl, pred)
            results[l, t] = auc(fpr, tpr)
            #print(f'{train_ldrs[l]} {test_names[t]} auc: {results[l, t]}')
            if train_ldrs[l] == 'jb':
                sns.distplot(np.array(neg_results['dist']), label=leg[t])            
                
        if train_ldrs[l] == 'jb':
            plt.xlabel('Cosine Similarity')
            plt.legend()
            plt.savefig('jb_comp.png')
            plt.show()

    results_df = pd.DataFrame(data=results, columns=test_names)
    results_df['leader_name'] = train_ldrs
    return results_df

### Leaders Multiple Identities

def test_comp_multiple_identities(bs_fldr, pool_params, parallel=False):
    
    train_files, test_files = train_test(bs_fldr)

    # these are the table values
    train_ldrs = ['bo', 'bs', 'ew', 'hc', 'dt', 'jb', 'cb', 'pb', 'kh', 'br']
    test_ldrs =  ['bo', 'br', 'bs', 'cb', 'dt', 'ew', 'hc', 'jb', 'kh', 'pb', 'FF_orig', 
                  'bo_faceswap', 'bo_imposter', 'bs_faceswap', 'bs_imposter', 'dt_faceswap', 'dt_imposter', 
                  'ew_faceswap', 'ew_imposter', 'hc_faceswap', 'hc_imposter', 'jb_imposter', 'jb_faceswap']
    
    train_repo = build_ref(bs_fldr, {f: train_files[f] for f in train_ldrs}, pool_params, 5000)
    test_repo = build_ref(os.path.join(bs_fldr, 'compression'), {f: test_files[f] for f in test_ldrs}, pool_params, 5000)
    
    full_results = get_results(train_repo, test_repo, parallel=parallel)
    
    # these are the table values
    test_names = ['diff', 'faceswap', 'imposter']
    test_cases =  [[['FF_orig'], ['bo_faceswap'], ['bo_imposter']],
                  [['FF_orig'], ['bs_faceswap'], ['bs_imposter']],
                  [['FF_orig'], ['ew_faceswap'], ['ew_imposter']],
                  [['FF_orig'], ['hc_faceswap'], ['hc_imposter'] ],
                  [['FF_orig'], ['dt_faceswap'], ['dt_imposter'] ],
                  [['FF_orig'], ['jb_faceswap'], ['jb_imposter']],
                  [['FF_orig']],
                  [['FF_orig']],
                  [['FF_orig']], 
                  [['FF_orig']] ]

    # for each case
    results = np.zeros((len(train_ldrs), len(test_names))) + np.nan
    for l in range(len(train_ldrs)):

        pos_dist = np.array(full_results['dist'][full_results['true_name']==train_ldrs[l]])        
        for t in range(len(test_cases[l])):
            
            neg_dist = np.array(full_results['dist'][full_results['true_name'].isin(test_cases[l][t])])

            pred = np.concatenate([pos_dist, neg_dist], axis=0)
            lbl = np.concatenate([np.ones_like(pos_dist), np.zeros_like(neg_dist)])
            fpr, tpr, thresholds = roc_curve(lbl, pred)
            results[l, t] = auc(fpr, tpr)
            #print(f'{train_ldrs[l]} {test_names[t]} auc: {results[l, t]}')

    results_df = pd.DataFrame(data=results, columns=test_names)
    results_df['leader_name'] = train_ldrs
    
    # pivot table
    confusion_mat = full_results.pivot_table(values='dist', 
                                             index='true_name', columns='pred_name', 
                                             aggfunc=lambda x: len(x))
    confusion_mat = confusion_mat/np.nansum(np.array(confusion_mat), axis=1, keepdims=True)
    
    return results_df, confusion_mat