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
