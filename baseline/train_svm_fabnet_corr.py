""" combine the transcript file with the landmark, au csv file

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utils as u
import os


fps = 30
rs = 777
np.random.seed(rs)
obj_bsfldr = 'obj'
feat_fldrs = ['fabnet_corr_100_30', 'fabnet_corr_100', 'fabnet_corr_100_190']
reslt_prefixs = ['results/fabnet_corr_100_30', 'results/fabnet_corr_100', 'results/fabnet_corr_100_190']
gammas = [0.1]
nus = [0.1]
do_cv = False

#plot the histogram and roc of the prediction probabibility
def save_fig(pred_prob, fig_fldr, real_n, fake_n, fig_nm, y_in, l_name):

    plt.figure()

    print(pred_prob.shape)
    print(y_in.shape)

    nn_id = np.logical_not(np.isnan(pred_prob))
    pred_prob = pred_prob[nn_id]
    y_in = y_in[nn_id]

    bins = np.linspace(np.min(pred_prob), np.max(pred_prob), 100)
    s_counts, s_bin_edges = u.get_bins_edges(pred_prob[0:real_n], bins)
    plt.bar(s_bin_edges, s_counts, width=s_bin_edges[1] - s_bin_edges[0], color='g', alpha=0.7)
    s_counts, s_bin_edges = u.get_bins_edges(pred_prob[real_n:], bins)
    plt.bar(s_bin_edges, s_counts, width=s_bin_edges[1] - s_bin_edges[0], color='r', alpha=0.7)
    plt.xlabel('score {}'.format(l_name), fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title('{} green count{} red count{}'.format(fig_nm, real_n, fake_n))
    plt.draw()
    plt.savefig('{}/{}_hist.png'.format(fig_fldr, fig_nm))
    plt.close()
    return u.plotROC(pred_prob, y_in, '{}/{}_roc.png'.format(fig_fldr, fig_nm))


def get_train_test_split(in_dataset, in_ratio):

    #split the dataset using the length of the database
    all_keys = list(in_dataset.keys())
    
    if type(in_dataset[all_keys[0]]).__module__ == np.__name__:
        arr_data = np.vstack(list(in_dataset.values()))
    else:
        arr_data = pd.concat(list(in_dataset.values()), ignore_index=True, sort=False)

    #full length
    full_len = len(arr_data)

    #train:test 90:10
    train_len = int(full_len * in_ratio)
    all_keys = np.sort(np.array(all_keys))
    srt_id = list(range(len(all_keys)))#select random idx
    np.random.seed(rs)

    i = 0
    cur_len = 0
    if train_len == full_len:
        i = len(srt_id)
        cur_len = full_len
    else:
        while 1:
            if (cur_len + len(in_dataset[all_keys[srt_id[i]]])) >= train_len:
                break

            cur_len += len(in_dataset[all_keys[srt_id[i]]])
            i += 1

    train_id = srt_id[0:i]
    test_id = srt_id[i:]

    fv_X_train = {}
    for i in train_id:
        fv_X_train[all_keys[i]] = np.array(in_dataset[all_keys[i]])

    fv_X_test = {}
    for i in test_id:
        fv_X_test[all_keys[i]] = np.array(in_dataset[all_keys[i]])

    #print('desired: {} split_length {}/{} = {} Clips: Tr{}/Ts{}'.format(in_ratio, cur_len, full_len, cur_len/full_len, len(fv_X_train), len(fv_X_test)))

    return fv_X_train, fv_X_test

def get_dataset_equal(train, in_fldr):

    train_lbl1 = {}
    test_lbl1 = {}
    max_tr_count = 0
    max_ts_count = 0
    r = {}
    for i in np.argwhere(train['lbl']==1).ravel():

        db = u.load_obj(in_fldr, train['db'][i])
        try:
            cur_tr, cur_ts = get_train_test_split(u.remove_keys(db), train['perc'][i])
        except:
            assert 'dt' in train['db'][i], 'this should only be for donald trump' # Sometimes only one of the dt_week/dt_rndm is present
            print(f'{db} is NoneType')
            continue
        r.update(cur_ts)

        if len(cur_tr) > 0:
            train_lbl1[train['db'][i]] = np.vstack(list(cur_tr.values()))
            if max_tr_count<len(train_lbl1[train['db'][i]]):
                max_tr_count = len(train_lbl1[train['db'][i]])

        if len(cur_ts) > 0:
            test_lbl1[train['db'][i]] = np.vstack(list(cur_ts.values()))
            if max_ts_count<len(test_lbl1[train['db'][i]]):
                max_ts_count = len(test_lbl1[train['db'][i]])

    #print('train len{} test len{}'.format(max_tr_count, max_ts_count))
    X_train = None
    y_train = None
    #balance the label 1 database: Train
    for k in list(train_lbl1.keys()):
        ridx = np.random.choice(np.arange(len(train_lbl1[k])), max_tr_count, replace=True)
        if X_train is None:
            X_train = train_lbl1[k][ridx, :]
            y_train = np.zeros((len(X_train), )) + 1
        else:
            farr = train_lbl1[k][ridx, :]
            X_train = np.vstack((X_train, farr))
            y_train = np.concatenate((y_train, np.zeros((len(farr), )) + 1), axis=0)

    X_test = None
    y_test = None
    #balance the label 1 database: Test
    for k in list(test_lbl1.keys()):
        ridx = np.random.choice(np.arange(len(test_lbl1[k])), max_tr_count, replace=True)
        if X_test is None:
            X_test = test_lbl1[k][ridx, :]
            y_test = np.zeros((len(X_test), )) + 1
        else:
            farr = test_lbl1[k][ridx, :]
            X_test = np.vstack((X_test, farr))
            y_test = np.concatenate((y_test, np.zeros((len(farr), )) + 1), axis=0)

    #print('lbl1 Xtrain{}, yTrain{}, XTest{}, yTest{}'.format(len(X_train), len(y_train), len(X_test), len(y_test)))

    #get the label zero dataset with balancing
    train_blnc0 = int(len(X_train)/(np.sum(np.logical_and(train['lbl']==0, train['perc']>0))))
    test_blnc0 = int(len(X_test)/(np.sum(np.logical_and(train['lbl']==0, train['perc']<1))))
    f = {}
    for i in np.argwhere(train['lbl']==0).ravel():

        db = u.load_obj(in_fldr, train['db'][i])
        cur_tr, cur_ts = get_train_test_split(u.remove_keys(db), train['perc'][i])
        f.update(cur_ts)

        if len(cur_tr) > 0:
            farr = np.vstack(list(cur_tr.values()))
            ridx = np.random.choice(np.arange(len(farr)), train_blnc0, replace=True)
            X_train = np.vstack((X_train, farr[ridx, :]))
            y_train = np.concatenate((y_train, np.zeros((len(farr[ridx, :]), )) + 0), axis=0)

        if len(cur_ts) > 0:
            farr = np.vstack(list(cur_ts.values()))
            ridx = np.random.choice(np.arange(len(farr)), test_blnc0, replace=True)
            X_test = np.vstack((X_test, farr[ridx, :]))
            y_test = np.concatenate((y_test, np.zeros((len(farr[ridx, :]), )) + 0), axis=0)
            

    #print('lbl10 Xtrain{}, yTrain{}, XTest{}, yTest{}'.format(len(X_train), len(y_train), len(X_test), len(y_test)))

    train_name = None
    test_name = None
    for i in range(len(train['db'])):

        if train_name is None and train['perc'][i] > 0:
            train_name = ''.join(train['db'][i].split('_'))
        else:
            if train['perc'][i] > 0:
                train_name = train_name + '_' + ''.join(train['db'][i].split('_'))

        if test_name is None and train['perc'][i] < 1:
            test_name = ''.join(train['db'][i].split('_'))
        else:
            if train['perc'][i] < 1:
                test_name = test_name + '_' + ''.join(train['db'][i].split('_'))

    #print('train:{} test: real{} fake:{}'.format(len(y_train), np.sum(y_test==1), np.sum(y_test==0)))

    fv_y_test = np.concatenate((np.ones((len(r), )), np.zeros((len(f), ))), axis=0)
    fv_X_test = r
    fv_X_test.update(f)

    return X_train, y_train, X_test, y_test, fv_X_test, fv_y_test, train_name, test_name

def get_all_train_cases():

    train_all = {}
    train_all['bernie'] =  {} #leader name
    train_all['bernie']['db'] = np.array(['bs', 'diff_test']) #, 'bs_imposter', 'bs_faceswap'
    train_all['bernie']['lbl'] = np.array([1, 0])
    train_all['bernie']['perc'] = np.array([0.9, 0.1])
    train_all['bernie_imp'] = {}
    train_all['bernie_imp']['db'] = np.array(['bs', 'diff_test', 'bs_imposter']) #, 'bs_imposter', 'bs_faceswap'
    train_all['bernie_imp']['lbl'] = np.array([1, 0, 0])
    train_all['bernie_imp']['perc'] = np.array([0.9, 1, 0])
    train_all['bernie_fc'] = {}
    train_all['bernie_fc']['db'] = np.array(['bs', 'diff_test', 'bs_faceswap']) #, 'bs_imposter', 'bs_faceswap'
    train_all['bernie_fc']['lbl'] = np.array([1, 0, 0])
    train_all['bernie_fc']['perc'] = np.array([0.9, 1, 0])

    train_all['ew'] = {} #leader name
    train_all['ew']['db'] = np.array(['ew', 'diff_test']) #
    train_all['ew']['lbl'] = np.array([1, 0])
    train_all['ew']['perc'] = np.array([0.9, 0.1])
    train_all['ew_imp'] = {}
    train_all['ew_imp']['db'] = np.array(['ew', 'diff_test', 'ew_imposter']) #
    train_all['ew_imp']['lbl'] = np.array([1, 0, 0])
    train_all['ew_imp']['perc'] = np.array([0.9, 1, 0])
    train_all['ew_fc'] = {}
    train_all['ew_fc']['db'] = np.array(['ew', 'diff_test', 'ew_faceswap']) #
    train_all['ew_fc']['lbl'] = np.array([1, 0, 0])
    train_all['ew_fc']['perc'] = np.array([0.9, 1, 0])

    train_all['hc'] = {}#leader name
    train_all['hc']['db'] = np.array(['hc', 'diff_test'])
    train_all['hc']['lbl'] = np.array([1, 0])
    train_all['hc']['perc'] = np.array([0.9, 0.1])
    train_all['hc_fc'] = {}
    train_all['hc_fc']['db'] = np.array(['hc', 'diff_test', 'hc_faceswap'])
    train_all['hc_fc']['lbl'] = np.array([1, 0, 0])
    train_all['hc_fc']['perc'] = np.array([0.9, 1, 0])
    train_all['hc_imp'] = {}
    train_all['hc_imp']['db'] = np.array(['hc', 'diff_test', 'hc_imposter'])
    train_all['hc_imp']['lbl'] = np.array([1, 0, 0])
    train_all['hc_imp']['perc'] = np.array([0.9, 1, 0])

    train_all['trump'] =  {} #leader name
    train_all['trump']['db'] = np.array(['dt_week', 'dt_rndm', 'diff_test'])
    train_all['trump']['lbl'] = np.array([1, 1, 0])
    train_all['trump']['perc'] = np.array([0.9, 0.9, 0.1])
    train_all['trump_imp'] = {}
    train_all['trump_imp']['db'] = np.array(['dt_week', 'dt_rndm', 'diff_test', 'dt_imposter']) 
    train_all['trump_imp']['lbl'] = np.array([1, 1, 0, 0])
    train_all['trump_imp']['perc'] = np.array([0.9, 0.9, 1, 0])
    train_all['trump_fc'] = {}
    train_all['trump_fc']['db'] = np.array(['dt_week', 'dt_rndm', 'diff_test', 'dt_faceswap']) 
    train_all['trump_fc']['lbl'] = np.array([1, 1, 0, 0])
    train_all['trump_fc']['perc'] = np.array([0.9, 0.9, 1, 0])
    
    train_all['jb'] = {} #leader name
    train_all['jb']['db'] = np.array(['jb', 'diff_test'])
    train_all['jb']['lbl'] = np.array([1, 0])
    train_all['jb']['perc'] = np.array([0.9, 0.1])
    
    train_all['pb'] = {} #leader name
    train_all['pb']['db'] = np.array(['pb', 'diff_test'])
    train_all['pb']['lbl'] = np.array([1, 0])
    train_all['pb']['perc'] = np.array([0.9, 0.1])
    
    train_all['br'] = {} #leader name
    train_all['br']['db'] = np.array(['br', 'diff_test'])
    train_all['br']['lbl'] = np.array([1, 0])
    train_all['br']['perc'] = np.array([0.9, 0.1])
    
    train_all['cb'] = {} #leader name
    train_all['cb']['db'] = np.array(['cb', 'diff_test'])
    train_all['cb']['lbl'] = np.array([1, 0])
    train_all['cb']['perc'] = np.array([0.9, 0.1])    
    
    train_all['kh'] = {} #leader name
    train_all['kh']['db'] = np.array(['kh', 'diff_test'])
    train_all['kh']['lbl'] = np.array([1, 0])
    train_all['kh']['perc'] = np.array([0.9, 0.1])        

    train_all['obama'] = {} #leader name
    train_all['obama']['db'] = np.array(['bo', 'diff_test'])#, 'bo_UWfake', 'bo_imposter']) #
    train_all['obama']['lbl'] = np.array([1, 0])
    train_all['obama']['perc'] = np.array([0.9, 0.1])
    train_all['obama_imp'] = {}
    train_all['obama_imp']['db'] = np.array(['bo', 'diff_test', 'bo_imposter'])#, 'bo_UWfake', 'bo_imposter']) #
    train_all['obama_imp']['lbl'] = np.array([1, 0, 0])
    train_all['obama_imp']['perc'] = np.array([0.9, 1, 0])
    train_all['obama_fc'] = {}
    train_all['obama_fc']['db'] = np.array(['bo', 'diff_test', 'bo_faceswap'])#, 'bo_UWfake', 'bo_imposter']) #
    train_all['obama_fc']['lbl'] = np.array([1, 0, 0])
    train_all['obama_fc']['perc'] = np.array([0.9, 1, 0])
    
    """    train_all['obama_UWFake'] = {}
    train_all['obama_UWFake']['db'] = np.array(['bo', 'diff_test', 'bo_UWfake'])#, 'bo_UWfake', 'bo_imposter']) #
    train_all['obama_UWFake']['lbl'] = np.array([1, 0, 0])
    train_all['obama_UWFake']['perc'] = np.array([0.9, 1, 0])    
    train_all['trump_lipsyn'] = {}
    train_all['trump_lipsyn']['db'] = np.array(['dt_week', 'dt_rndm', 'diff_test', 'trump_fake']) 
    train_all['trump_lipsyn']['lbl'] = np.array([1, 1, 0, 0])
    train_all['trump_lipsyn']['perc'] = np.array([0.9, 0.9, 1, 0])
"""

    return train_all


if __name__ == '__main__':

        
        for g in range(len(gammas)):
            
            for f in range(len(feat_fldrs)):
                
                reslt_prefix = reslt_prefixs[f] + '_' + str(gammas[g]) + '/'
                print('obj_fldr : {} \nreslt_prefix : {} \n feat_fldr: {} \n'.format(obj_bsfldr,
                                                                     reslt_prefix, feat_fldrs[f]))
                train_all = get_all_train_cases() 
                for l_name in train_all.keys():
            
                    train = train_all[l_name]
                    print(l_name)
                    #get the dataset
                    X_train, y_train, X_test, y_test, fv_X_test, fv_y_test, train_name, test_name = get_dataset_equal(train, 
                                                                                                                      os.path.join(obj_bsfldr, 
                                                                                                                                   feat_fldrs[f]))
                    fig_fldr = '{}_{}'.format(reslt_prefix, train_name)
                    os.makedirs(fig_fldr, exist_ok=True)
                    print(X_train.shape)
            
                    #train the model
                    linear_model = u.load_obj(fig_fldr, 'model_{}'.format(train_name))
                    if linear_model is None:
                        print('training...')
                        linear_model = u.train_ovr(np.array(X_train)[y_train==1, :], np.array(X_train), y_train, gamma=gammas[g], nu=nus[g], do_cv=do_cv)
                        #train auc
                        pred_prob = linear_model['model'].score_samples(linear_model['scaler'].transform(np.array(X_train)[y_train==1, :]))
                        prob_rng = [np.quantile(pred_prob, q=0.02), np.quantile(pred_prob, q=0.98)]
                        linear_model['prob_rng'] = prob_rng
                        u.save_obj(linear_model, fig_fldr, 'model_{}'.format(train_name))
            
                    #testing: 10-second clip
                    prob_rng = linear_model['prob_rng']
                    pred_prob = linear_model['model'].score_samples(linear_model['scaler'].transform(np.array(X_test)))
                    pred_prob = (pred_prob - prob_rng[0])/prob_rng[1] #normalize
                    pd.DataFrame(data=np.concatenate((pred_prob[:, np.newaxis], y_test[:, np.newaxis]), axis=1), columns=['prob', 'label']).to_csv('{}/{}.csv'.format(fig_fldr, test_name))
                    test_auc = save_fig(pred_prob, fig_fldr, np.sum(y_test==1), np.sum(y_test==0), test_name, y_test, l_name)
                    print('{} auc:{}'.format(test_name, test_auc))
