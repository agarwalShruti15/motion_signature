import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import natsort
from itertools import combinations

CONF = 0.9
fps = 30
mfcc_n = 20

def save_obj(obj, fldr, name ):
    os.makedirs(fldr, exist_ok=True)
    with open(os.path.join(fldr, name + '.pkl'), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(fldr, name ):
    if not os.path.exists(os.path.join(fldr, name + '.pkl')):
        return None
    with open(os.path.join(fldr, name + '.pkl'), 'rb') as f:
        return pickle.load(f)

def get_ngram_dataframe(in_data_frame, n):
    cur_list = in_data_frame['word'].str.lower().tolist()
    if(n>1):
        ngrm = list(zip(*[cur_list[i:] for i in range(n)]))
        strt = in_data_frame['start'].tolist()[:-(n-1)]
        endt = in_data_frame['end'].tolist()[(n-1):]
    else:
        ngrm = cur_list
        strt = in_data_frame['start'].tolist()
        endt = in_data_frame['end'].tolist()
    return pd.DataFrame(data={'ngram': ngrm, 'start': strt, 'end': endt})

def get_ngrams_from_lines(in_lines, in_df, col_nm, ngrams):

    l_n = len(in_lines)

    out_df = in_df.copy()
    for n in range(ngrams):
        out_df[col_nm + str(n+1)] = [[] for i in range(len(out_df))]

    for l in range(l_n):

        wrd_data = in_lines[l].split(',')
        cur_wrds = ''.join(wrd_data[:-2])  # remove the last two csv values
        cur_strT = float(wrd_data[-2]); cur_endT = float(wrd_data[-1])
        for n in range(ngrams):

            if (l+n) < l_n:
                if n == 0:
                    wrds = cur_wrds
                    strT = np.round(cur_strT * fps) + 1
                    endT = np.round(cur_endT * fps) + 1
                else:
                    wrd_data = in_lines[l+n].split(',')
                    cur_wrds = ''.join(wrd_data[:-2])  # remove the last two csv values
                    cur_endT = float(wrd_data[-1])
                    wrds = wrds + ' ' + cur_wrds
                    endT = np.round(cur_endT * fps) + 1

                idx = np.logical_and(out_df['frame'] >= strT, out_df['frame'] <= endT)
                word_frms = np.array(out_df.loc[idx, col_nm + str(n+1)])
                new_series = pd.Series(data=[np.append(w, wrds) for w in word_frms])
                out_df.loc[idx, col_nm + str(n+1)] = np.array(new_series)

                # print(np.array([tuple(np.append(w, wrds)) for w in word_frms]).shape)
                # print(out_df.loc[idx, col_nm + str(n+1)].shape)

    return out_df

def get_phone_df(infile):

    phone_file = infile + '_phone.txt'
    if not os.path.exists(phone_file):
        return None
    # phone file
    fd = open(phone_file, 'r')
    lines = fd.readlines()
    fd.close()

    l_n = len(lines)
    if l_n == 0:  # empty transcript checks
        return None

    wrds = []
    strts = np.zeros((l_n, ), dtype=np.float)
    ends = np.zeros((l_n, ), dtype=np.float)
    for l in range(l_n):

        wrd_data = lines[l].split(',')
        wrds = wrds + [''.join(wrd_data[:-2])]  # remove the last two csv values
        strts[l] = float(wrd_data[-2]); ends[l] = float(wrd_data[-1])

    out_df = pd.DataFrame(data=wrds, columns=['phones'])
    out_df['start'] = strts
    out_df['end'] = ends
    out_df['file'] = infile

    return out_df


def get_trnscrpt_phone(in_df, trnscrpt_file, phone_file, ngram):

    out_df = in_df.copy()

    try:
        # transcript file
        fd = open(trnscrpt_file, 'r')
        lines = fd.readlines()
        fd.close()
        if len(lines) == 0:  # empty transcript checks
            return None

        out_df = get_ngrams_from_lines(lines, out_df, 'words', ngram)

        # phone file
        fd = open(phone_file, 'r')
        lines = fd.readlines()
        fd.close()
        if len(lines) == 0:  # empty transcript checks
            return None

        out_df = get_ngrams_from_lines(lines, out_df, 'phones', ngram)

    except Exception as e:
        print(e)
        return None

    return out_df

#read landmark(,au etc.)
def lndmrk_to_df(infile):
    cur_df = pd.read_csv(infile)
    clean_df = cur_df.loc[cur_df[' confidence']>CONF]

    x = (np.array(clean_df.loc[:, ' x_0':' x_67'])-np.mean(np.array(clean_df.loc[:, ' x_0':' x_67'])))/np.std(np.array(clean_df.loc[:, ' x_0':' x_67']))
    y = (np.array(clean_df.loc[:, ' y_0':' y_67'])-np.mean(np.array(clean_df.loc[:, ' y_0':' y_67'])))/np.std(np.array(clean_df.loc[:, ' y_0':' y_67']))
    clean_df.loc[:, ' x_0':' x_67'] = x.copy()
    clean_df.loc[:, ' y_0':' y_67'] = y.copy()

    return clean_df

#correct for head rotation
def alignLndmrks_withcsv(csv_file, verbose=False):

    x = np.array(csv_file.loc[:, ' X_0':' X_67'])
    y = np.array(csv_file.loc[:, ' Y_0':' Y_67'])
    z = np.array(csv_file.loc[:, ' Z_0':' Z_67'])

    r_x = np.array(csv_file.loc[:, ' pose_Rx'])
    r_y = np.array(csv_file.loc[:, ' pose_Ry'])
    r_z = np.array(csv_file.loc[:, ' pose_Rz'])

    x_new = x * (np.cos(r_z)*np.cos(r_y))[:, np.newaxis] \
            + y * (np.cos(r_z)*np.sin(r_y)*np.sin(r_x) + np.sin(r_z)*np.cos(r_x))[:, np.newaxis] \
            + z * (np.sin(r_z)*np.sin(r_x) - np.cos(r_z)*np.sin(r_y)*np.cos(r_x))[:, np.newaxis]
    y_new = -x * (np.sin(r_z)*np.cos(r_y))[:, np.newaxis] \
            + y * (np.cos(r_z)*np.cos(r_x) - np.sin(r_z)*np.sin(r_y)*np.sin(r_x))[:, np.newaxis] \
            + z * (np.sin(r_z)*np.sin(r_y)*np.cos(r_x) + np.cos(r_z)*np.sin(r_x))[:, np.newaxis]

    y_new = -y_new

    #x_new = x.copy(); y_new = -y.copy()

    #for every row find t_x, t_y, theta, and scale
    l_e_x = np.mean(x_new[:, 36:42], axis=1)
    l_e_y = np.mean(y_new[:, 36:42], axis=1)
    r_e_x = np.mean(x_new[:, 42:48], axis=1)
    r_e_y = np.mean(y_new[:, 42:48], axis=1)

    #translate
    x = x_new - l_e_x[:, np.newaxis]
    y = y_new - l_e_y[:, np.newaxis]
    r_e_x = r_e_x - l_e_x
    r_e_y = r_e_y - l_e_y
    l_e_x = l_e_x - l_e_x
    l_e_y = l_e_y - l_e_y

    #rotate theta, assumption r_e_x is positive
    cos_theta = r_e_x / np.sqrt(r_e_x**2 + r_e_y**2)
    sin_theta = np.sqrt(1 - cos_theta**2)
    sin_theta[r_e_y<0] = -sin_theta[r_e_y<0]

    x_new = x * cos_theta[:, np.newaxis] + y * sin_theta[:, np.newaxis]
    y_new = y * cos_theta[:, np.newaxis] - x * sin_theta[:, np.newaxis]
    x = x_new
    y = y_new
    #for every row find t_x, t_y, theta, and scale
    l_e_x = np.mean(x_new[:, 36:42], axis=1)
    l_e_y = np.mean(y_new[:, 36:42], axis=1)
    r_e_x = np.mean(x_new[:, 42:48], axis=1)
    r_e_y = np.mean(y_new[:, 42:48], axis=1)

    #scale
    x = x / r_e_x[:, np.newaxis]
    y = y / r_e_x[:, np.newaxis]
    l_e_x = l_e_x / r_e_x
    l_e_y = l_e_y / r_e_x
    r_e_y = r_e_y / r_e_x
    r_e_x = r_e_x / r_e_x


    if verbose:
        import matplotlib.pyplot as plt
        for i in range(len(l_e_y)):
            plt.clf()
            plt.scatter(x[i, :], y[i, :], c='b', marker='.')
            plt.scatter(l_e_x[i], l_e_y[i], c='r', marker='.')
            plt.scatter(r_e_x[i], r_e_y[i], c='r', marker='.')
            plt.draw()
            plt.pause(0.001) #Note this correction

    out_ar = dict()
    out_ar['x'] = x
    out_ar['y'] = y

    return out_ar

#landmark(,au etc.) dataframe
def combine_lip_opening(in_lndmrk_df):

    cur_feat = alignLndmrks_withcsv(in_lndmrk_df)
    d_hr = ((cur_feat['x'][:, 48]-cur_feat['x'][:, 54])**2)+((cur_feat['y'][:, 48]-cur_feat['y'][:, 54])**2)
    d_vert = ((cur_feat['x'][:, 51]-cur_feat['x'][:, 57])**2)+((cur_feat['y'][:, 51]-cur_feat['y'][:, 57])**2)
    l_eye_vert = (((cur_feat['x'][:, 37]+cur_feat['x'][:, 38])/2 -
                   (cur_feat['x'][:, 40]+cur_feat['x'][:, 41])/2)**2) + \
                 (((cur_feat['y'][:, 37]+cur_feat['y'][:, 38])/2 -
                   (cur_feat['y'][:, 40]+cur_feat['y'][:, 41])/2)**2)

    r_eye_vert = (((cur_feat['x'][:, 43]+cur_feat['x'][:, 44])/2 -
                   (cur_feat['x'][:, 47]+cur_feat['x'][:, 46])/2)**2) + \
                 (((cur_feat['y'][:, 43]+cur_feat['y'][:, 44])/2 -
                   (cur_feat['y'][:, 47]+cur_feat['y'][:, 46])/2)**2)

    out_df = in_lndmrk_df.copy()
    out_df['lip_hor'] = d_hr
    out_df['lip_ver'] = d_vert

    out_df['l_eye_vert'] = l_eye_vert
    out_df['r_eye_vert'] = r_eye_vert

    out_df.loc[:, ' x_0':' x_67'] = cur_feat['x'].copy()
    out_df.loc[:, ' y_0':' y_67'] = cur_feat['y'].copy()

    return out_df

#landmark(,au etc.) csv file, transcript file
def tmp_get_full_df(infile, ngram=2):

    csv_files = infile + '.csv'
    flac_file = infile + '.flac'
    trnscrpt_file = infile + '.txt'
    phone_file = infile + '_phone.txt'

    if not os.path.exists(csv_files) or \
            not os.path.exists(flac_file) or \
            not os.path.exists(trnscrpt_file) or \
            not os.path.exists(phone_file):
        return None

    # read transcript and landmark csv
    lndmrk_df = lndmrk_to_df(csv_files)

    # combine the transcript and landmark files
    out_df = combine_lip_opening(lndmrk_df)

    # put the transcript information into the frames
    out_df = get_trnscrpt_phone(out_df, trnscrpt_file, phone_file, ngram)
    if out_df is not None:
        out_df['file'] = infile
        # put the transcript information into the frames
        out_df = get_trnscrpt_phone(out_df, trnscrpt_file, phone_file, ngram)
    return out_df

def prune_segments(in_df, in_segs):

    diff_dist = 1

    # valid id is needed when the frames from two video files were in the change
    frms = np.array(in_df['frame'])
    valid_id = np.zeros((len(in_df),), dtype=np.bool)
    valid_id[:-diff_dist] = (frms[diff_dist:] - frms[:-diff_dist]) == diff_dist

    idx = np.logical_not(np.logical_or(np.sum(in_segs < 0, axis=1) > 0, np.sum(in_segs >= len(in_df), axis=1) > 0))
    in_segs = in_segs[idx, :].copy()

    idx = np.sum(np.logical_not(valid_id[in_segs]), axis=1) < 1

    return in_segs[idx, :]

def get_pauses(in_df):

    sil_len = 10 # minimum number of frames where the change in energy is low
    frm_pad = 20 # number of frames before and after the silence

    #find the sections with silence in the audio
    sil_seg = in_df['phones1'].apply(lambda x: 'sp' in x)
    sil_seg = np.abs(np.convolve(sil_seg, np.ones((sil_len,))/sil_len, 'same') - 1) < (10**-6) # erode to remove smaller sequence
    sil_seg = np.convolve(sil_seg, np.ones((sil_len,))/sil_len, 'same') > (10**-6) # dilate back

    # extract continous silence segments
    tmp_idx = np.logical_xor(sil_seg[1:], sil_seg[:-1])
    strt = np.argwhere(np.logical_and(np.concatenate(([True], tmp_idx), axis=0), sil_seg)).ravel()
    ends = np.argwhere(np.logical_and(np.concatenate((tmp_idx, [True]), axis=0), sil_seg)).ravel()

    assert np.sum(ends-strt<0) < 1, 'start greater than end'

    idx = (ends-strt+1)>=sil_len
    strt = strt[idx].copy(); ends = ends[idx].copy()
    if len(strt) < 1:
        return []

    pause_idx = np.array([np.concatenate((np.arange(strt[x] - frm_pad, strt[x]),
                           np.linspace(strt[x], ends[x], sil_len, dtype=np.int32),
                           np.arange(ends[x] + 1, ends[x] + frm_pad + 1)), axis=0) for x in range(len(strt))])
    return prune_segments(in_df, pause_idx)


def get_all_phone_feat(infldr, obj_bsfldr):

    csv_files = [os.path.join(infldr, f) for f in os.listdir(infldr) if f.endswith('.flac')]; csv_files.sort()
    n = len(csv_files)

    out_full_df = {}
    for i in range(n):
        filename, _ = os.path.splitext(csv_files[i])
        feat_df = get_phone_df(filename)

        if feat_df is not None:
            out_full_df[filename] = feat_df.copy()

    return out_full_df

def get_second_col_df(feat_df, j, vid_len, first_col, sec_col):
    return np.around(np.array(feat_df[first_col]), 7)[j:j+vid_len, :], np.around(np.array(feat_df[sec_col]), 7)[j:j+vid_len, :]


def get_one_feat(in_arr, pref, out_df, do_acc=True):
    in_arr = in_arr - np.min(in_arr) #zero minimum

    out_df[pref + '_featMean'] = np.mean(in_arr)
    if out_df[pref + '_featMean'] > 0:
        out_df[pref + '_featMean'] = np.log(out_df[pref + '_featMean'])
        
    out_df[pref + '_featVar'] = np.var(in_arr)
    if out_df[pref + '_featVar'] > 0:
        out_df[pref + '_featVar']  = np.log(out_df[pref + '_featVar'])


def get_binned_change(audio_df, feat):
    
    out_df = {}
        
    for au in feat:
        get_one_feat(np.array(audio_df[au]), au, out_df, do_acc=False)
    
    keys = natsort.natsorted(out_df.keys())
    return np.array([out_df[x] for x in keys])[:, np.newaxis], keys


def get_corr_per_frame(infile, feat1, feat2, vid_len, obj_bsfldr, overlapping, ovrlp_win, do_corr, do_var):

    #these correlation features to consider
    all_feat_comb = list(combinations(range(len(feat1)), 2))
    cor_feat = np.array([[feat1[j[0]], feat1[j[1]]] for j in all_feat_comb])
    col_names = [''.join(x) for x in cor_feat]

    if not overlapping:
        ovrlp_win = vid_len

    feat_df = load_obj(os.path.join(obj_bsfldr, infile.split('/')[-2]), infile.split('/')[-1])
    if feat_df is None:
        feat_df = tmp_get_full_df(infile, ngram=2)
        save_obj(feat_df, os.path.join(obj_bsfldr, infile.split('/')[-2]), infile.split('/')[-1])

    if feat_df is None or feat_df.shape[0]<(vid_len+ovrlp_win):
        return None

    first_col = np.array(cor_feat)[:, 0]; sec_col = np.array(cor_feat)[:, 1]
    frame_rng = np.arange(0, len(feat_df)-vid_len, ovrlp_win)
    tmp_corr = np.zeros((len(frame_rng), len(cor_feat)))
    tempo_feat = {}

    i = 0
    for j in frame_rng:

        f, s = get_second_col_df(feat_df, j, vid_len, first_col, sec_col)
        if do_corr:
            tmp_corr[i, :] = np.corrcoef(f.T, s.T)[0:len(cor_feat), len(cor_feat):].diagonal()
        if do_var:
            tempo_feat[i], tempo_keys = get_binned_change(feat_df.iloc[j:j+vid_len, :], feat2)
        i = i+1

    if do_corr and do_var:
        tmp_corr = tmp_corr[:i, :].copy()
        out_df = pd.DataFrame(tmp_corr, columns=col_names)
        tempo_feat_array = np.concatenate([tempo_feat[x] for x in range(i)], axis=1).T
        for x in range(tempo_feat_array.shape[1]):
            out_df[tempo_keys[x]] = tempo_feat_array[:, x]

    else:
        if do_corr:
            tmp_corr = tmp_corr[:i, :].copy()
            out_df = pd.DataFrame(tmp_corr, columns=col_names)
        else:
            if do_var:
                tempo_feat_array = np.concatenate([tempo_feat[x] for x in range(i)], axis=1).T
                out_df = pd.DataFrame(tempo_feat_array, columns=tempo_keys)
                out_df = out_df.dropna()
            else:
                raise ValueError
    #out_df['frame'] = feat_df['frame'].iloc[int(frame_rng + vid_len/2)]
    out_df = out_df.dropna()

    return out_df


def get_arg(kwargs, key, default, req):

    if req and (key not in kwargs.keys()):
        print('{} is required'.format(key))
        raise ValueError

    ret = default
    if key in kwargs.keys():
        ret = kwargs[key]

    return ret

def get_feats(infldr, feat1, feat2, **kwargs):

    csv_files = [os.path.join(infldr, f) for f in os.listdir(infldr) if f.endswith('.csv')]; csv_files.sort()

    #get the arguments
    vid_len =  kwargs.get('vid_len', 300)
    obj_bsfldr = kwargs.get('obj_bsfldr', 'obj')
    overlapping = kwargs.get('overlapping', True)
    ovrlp_win = kwargs.get('ovrlp_win', 1)
    do_corr = kwargs.get('do_corr', True)
    do_var = kwargs.get('do_var', True)
    verbose = kwargs.get('verbose', True)

    out_full_df = {}
    total_files = 0
    n = len(csv_files)

    for i in range(n):
        filename, _ = os.path.splitext(csv_files[i])
        cur_df = get_corr_per_frame(filename, feat1, feat2, vid_len, obj_bsfldr, overlapping, ovrlp_win, do_corr, do_var)

        if cur_df is None:
            continue

        out_full_df[filename] = cur_df.copy()
        if verbose:
            print('{} {} {}/{}'.format(filename, cur_df.shape, total_files, len(csv_files)))
        total_files = total_files+1

    print(total_files)
    return out_full_df

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


def train_ovr(X_train, X_val, y_val, gamma=0.01, nu=0.01, do_cv=False):

    from sklearn.model_selection import ParameterGrid
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import OneClassSVM

    # Fit your data on the scaler object
    scaler = StandardScaler()
    scaler = scaler.fit(X_train)
    X_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_val)

    #init best params to default values
    best_param = {}
    best_param['gamma'] = gamma
    best_param['kernel'] = 'rbf'
    best_param['nu'] = nu

    #SVM model
    clf = OneClassSVM(cache_size=1000)

    # if cross-validation
    if do_cv:

        #if no validation set then optimize for train dataset only
        if len(X_val)==0:
            X_val = X_train.copy()
            y_val = np.ones((len(X_val), ))

        gamma_range = 10. ** np.arange(-5, -1)
        nu = np.linspace(0.01, 0.2, 5)

        #cross validation
        grid = {
            'gamma': gamma_range,
            'kernel': ['rbf'],
            'nu': nu
        }

        best_param = {}; best_acc = 0
        X_test_scaled = scaler.transform(X_val)
        #perform cross-validation on a small train set
        idx = np.random.choice(range(len(X_scaled)), size=4000, replace=False)
        for z in ParameterGrid(grid):

            clf.set_params(**z)
            clf.fit(X_scaled[idx, :])
            y_pred = clf.predict(X_test_scaled)
            y_pred[y_pred<0] = 0

            acc = ((np.sum(np.logical_and(y_pred==0, y_val==y_pred)) /np.sum(y_val==0)) + \
                  (np.sum(np.logical_and(y_pred==1, y_val==y_pred)) /np.sum(y_val==1)))/2
            if acc>best_acc:
                print('gamma:{}, kernel:{}, nu:{}, acc:{} **'.format(z['gamma'],
                                                                     z['kernel'],
                                                                     z['nu'],
                                                                     acc))
                best_param = z
                best_acc = acc
            else:
                print('gamma:{}, kernel:{}, nu:{}, acc:{}'.format(z['gamma'],
                                                                 z['kernel'],
                                                                 z['nu'],
                                                                 acc))
    print(best_param)
    svm_model = {}
    svm_model['scaler'] = scaler
    svm_model['model'] = clf.set_params(**best_param)
    
    #perform cross-validation on a small train set
    idx = np.random.choice(range(len(X_scaled)), size=np.min([30000, len(X_scaled)]), replace=False)
    svm_model['model'].fit(X_scaled[idx, :])

    return svm_model


def remove_keys(cur_dict):
    cur_dict.pop('__header__', None)
    cur_dict.pop('__version__', None)
    cur_dict.pop('__globals__', None)
    return cur_dict
