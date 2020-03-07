import numpy as np
import utils as u
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import log_loss
from scipy.stats import norm


def GG_beh(fldr, fl_nm, in_act):
    
    # if folder is orig, then just return the in_act
    if 'orig' in fldr:
        return in_act
    else:
        return 'GG_' + fl_nm[:2]# else return the first id

def FF_beh(fldr, fl_nm, in_act):
    
    # if folder is orig, then just return the in_act
    if 'orig' in fldr:
        return in_act
    else:
        return 'FF_' + fl_nm[:3]# else return the first id

def ldr_beh(fldr, fl_nm, in_act):
    
    # if folder is orig, then just return the in_act
    if 'faceswap' in fldr:
        return in_act + '_imposter'
    else:
        if 'imposter' in fldr:
            return in_act
            
    return in_act
    
def stv_beh(fldr, fl_nm, in_act):
    
    # if folder is orig, then just return the in_act
    if fldr == 'jennifer_l' or fldr == 'steve_b':
        return in_act
    else:
        return 'jen_l'
    
def dfdc_beh(fldr, fl_nm, in_act):
    if 'orig' in fldr:
        return in_act
    else:
        return 'DFDC_' + fl_nm.split('_')[1]# else return the first id

def cdf_beh(fldr, fl_nm, in_act):
    if 'real' in fldr or 'jennifer_l' in fldr:
        return in_act
    else:
        return 'CDF_' + fl_nm.split('_')[0]# else return the first id
    
def inwild_beh(fldr, fl_nm, in_act):
    if 'orig' in fldr or 'steve_b' in fldr:
        return in_act
    else:
        if in_act == 'inwild_sb':
            return 'CDF_id12'
        return 'inwild_' + fl_nm.split('_')[0]# else return the first id

# function to give number of time the vgg/fabnet went to face, behaviour, other
def get_face_behave_id(in_name, in_act):
    
    fldr, fl_nm = in_name.split('/')
    pref_dict = {'GG': GG_beh, 'FF': FF_beh, 'bo': ldr_beh, 
            'bs': ldr_beh, 'dt': ldr_beh, 'ew': ldr_beh, 
            'hc': ldr_beh, 'jb': ldr_beh, 'steve_b': stv_beh, 
                 'jen_l': stv_beh, 'DFDC': dfdc_beh, 'CDF': cdf_beh, 'inwild': inwild_beh}
    
    # get the actual category
    all_pref = list(pref_dict.keys())
    cur_pref = [all_pref[i] for i in range(len(all_pref)) if all_pref[i] in in_act]
    
    assert len(cur_pref) == 1, 'only one prefix should be matched'
    
    return pref_dict[cur_pref[0]](fldr, fl_nm, in_act)

def get_pred_label(in_series):
    # label dict
    label_dict = u.load_obj('result_csv', 'name2label_dict')
    cur_arr = np.array(list(in_series))
    cur_id = np.argmax(cur_arr, axis=1)
    out_lbl = [k for i, (k, v) in enumerate(label_dict) if v==cur_id][0]
    return out_lbl

def get_behav_id(in_df):
    
    flnms = np.array(in_df['fileName'])
    face_id = np.array(in_df['actualLabel_vgg'])
    behav_id = np.array([get_face_behave_id(flnms[i], face_id[i]) for i in range(len(flnms))])
    return behav_id


#---------------------------------------------------------------------------------------

# get the combined probability and then take the one corresponding to fabnet
def get_comb_fab(in_df, mu_sig=None):
    
    sim_vgg = np.array(list(in_df['sim_vgg']))
    sim_fab = np.array(list(in_df['sim_fab']))
    
    if mu_sig == None:
        mult_prob = sim_vgg*sim_fab
    else:
        mult_prob = norm.cdf(sim_vgg, mu_sig[0], mu_sig[1])*norm.cdf(sim_fab, mu_sig[0], mu_sig[1])
        
    f_mx_id = np.argmax(sim_fab, axis=1)
    
    return np.array([mult_prob[x, f_mx_id[x]] for x in range(len(f_mx_id))])

def get_comb_vgg(in_df, mu_sig=None):
    
    sim_vgg = np.array(list(in_df['sim_vgg']))
    sim_fab = np.array(list(in_df['sim_fab']))
    
    if mu_sig == None:
        mult_prob = sim_vgg*sim_fab
    else:    
        mult_prob = norm.cdf(sim_vgg, mu_sig[0], mu_sig[1])*norm.cdf(sim_fab, mu_sig[0], mu_sig[1])
        
    f_mx_id = np.argmax(sim_vgg, axis=1)
    return np.array([mult_prob[x, f_mx_id[x]] for x in range(len(f_mx_id))])


def get_comb_scalevgg(in_df, mu_sig=None):
    
    sim_vgg = np.array(list(in_df['sim_vgg']))
    sim_fab = np.array(list(in_df['sim_fab']))
    
    if mu_sig == None:
        mult_prob = sim_vgg*sim_fab
    else:    
        mult_prob = norm.cdf(sim_vgg, mu_sig[0], mu_sig[1])*sim_fab
        
    return np.max(mult_prob, axis=1)

def get_comb_scalevggfab(in_df, mu_sigVGG=None, mu_sigFab=None ):
    
    sim_vgg = np.array(list(in_df['sim_vgg']))
    sim_fab = np.array(list(in_df['sim_fab']))
    
    if mu_sigVGG == None or mu_sigFab == None:
        mult_prob = sim_vgg*sim_fab
    else:    
        mult_prob = norm.cdf(sim_vgg, mu_sigVGG[0], mu_sigVGG[1])*norm.cdf(sim_fab, mu_sigFab[0], mu_sigFab[1])
        
    return np.max(mult_prob, axis=1)


def get_comb_scaleMaxVGGFab(in_df, mu_sigVGG=None, mu_sigFab=None):
    
    sim_vgg = np.array(list(in_df['sim_vgg']))
    sim_fab = np.array(list(in_df['sim_fab']))
    
    vgg_id = np.argmax(sim_vgg, axis=1)
    fab_id = np.argmax(sim_fab, axis=1)
    
    out = np.zeros((len(vgg_id), ))
    for i in range(len(vgg_id)):
        
        if mu_sigVGG == None or mu_sigFab == None:
            vgg_scr = sim_vgg[i, vgg_id[i]]
            fab_scr = sim_fab[i, fab_id[i]]
        else:
            vgg_scr = norm.cdf(sim_vgg[i, vgg_id[i]], mu_sigVGG[0], mu_sigVGG[1])
            fab_scr = norm.cdf(sim_fab[i, fab_id[i]], mu_sigFab[0], mu_sigFab[1])            
        
        if vgg_scr < 0.00001 and fab_scr < 0.00001:
            out[i] = np.nan
        else:
            if vgg_id[i] == fab_id[i]:
                out[i] = vgg_scr * fab_scr
            else:
                out[i] == 0
                
    return out

def get_comb_Or_scaleMaxVGGFab(in_df, mu_sigVGG=None, mu_sigFab=None):
    
    sim_vgg = np.array(list(in_df['sim_vgg']))
    sim_fab = np.array(list(in_df['sim_fab']))
    
    vgg_id = np.argmax(sim_vgg, axis=1)
    fab_id = np.argmax(sim_fab, axis=1)
    
    vgg_scr = np.max(sim_vgg, axis=1)
    fab_scr = np.max(sim_fab, axis=1)    
    if not (mu_sigVGG == None) and not (mu_sigFab == None):
        vgg_scr = norm.cdf(vgg_scr, mu_sigVGG[0], mu_sigVGG[1])
        fab_scr = norm.cdf(fab_scr, mu_sigFab[0], mu_sigFab[1])    
    
    out = np.zeros((len(vgg_id), ))
    out[fab_scr < 0.00001] = np.nan
    out[vgg_id == fab_id] = vgg_scr[vgg_id == fab_id] * fab_scr[vgg_id == fab_id]
                
    return out


#---------------------------------------------------------------------------------------

# only vgg dist
def get_max_vgg(in_df, mu_sig=None):
    
    sim_vgg = np.array(list(in_df['sim_vgg']))
    
    if mu_sig == None:
        out = np.max(sim_vgg, axis=1)
    else:
        out = norm.cdf(np.max(sim_vgg, axis=1), mu_sig[0], mu_sig[1])
    
    return out

def get_max_fab(in_df, mu_sig=None):
    
    sim_vgg = np.array(list(in_df['sim_fab']))
    
    if mu_sig == None:
        out = np.max(sim_vgg, axis=1)
    else:
        out = norm.cdf(np.max(sim_vgg, axis=1), mu_sig[0], mu_sig[1])
    
    return out

#---------------------------------------------------------------------------------------

# distance of vgg from face id
def get_vgg_face_id(in_df):
    
    label_dict = u.load_obj('result_csv', 'name2label_dict')
    face_id = np.array(in_df['actualLabel_vgg'])
    
    sim_vgg = np.array(list(in_df['sim_vgg']))
    out_arr = np.zeros((len(sim_vgg), ))
    
    for i in range(len(out_arr)):
        out_arr[i] = sim_vgg[i, label_dict[face_id[i]]]
    
    return out_arr

def get_vgg_behav_id(in_df):
    
    label_dict = u.load_obj('result_csv', 'name2label_dict')
    behav_id = np.array(in_df['actualLabel_behav'])
    
    sim_vgg = np.array(list(in_df['sim_vgg']))
    out_arr = np.zeros((len(sim_vgg), ))
    
    for i in range(len(out_arr)):
        out_arr[i] = sim_vgg[i, label_dict[behav_id[i]]]
    
    return out_arr

def get_fab_face_id(in_df):
    
    label_dict = u.load_obj('result_csv', 'name2label_dict')
    face_id = np.array(in_df['actualLabel_vgg'])
    
    sim_vgg = np.array(list(in_df['sim_fab']))
    out_arr = np.zeros((len(sim_vgg), ))
    
    for i in range(len(out_arr)):
        out_arr[i] = sim_vgg[i, label_dict[face_id[i]]]
    
    return out_arr

def get_fab_behav_id(in_df):
    
    label_dict = u.load_obj('result_csv', 'name2label_dict')
    behav_id = np.array(in_df['actualLabel_behav'])
    
    sim_vgg = np.array(list(in_df['sim_fab']))
    out_arr = np.zeros((len(sim_vgg), ))
    
    for i in range(len(out_arr)):
        out_arr[i] = sim_vgg[i, label_dict[behav_id[i]]]
    
    return out_arr

#---------------------------------------------------------------------------------------


def get_consistency(in_df):
    
    face_id = np.array(in_df['actualLabel_vgg'])
    behav_id = np.array(in_df['actualLabel_behav'])
    vgg_id = get_pred_label(in_df['sim_vgg'])
    fab_id = get_pred_label(in_df['sim_fab'])
    
    out = np.zeros((len(vgg_id), ))
    out[vgg_id==fab_id] = 1
    
    return out


#---------------------------------------------------------------------------------------


# compute the threshold 
def compute_threshold(in_dict, in_keys, feat_nm, n=1000, cutoff = 0.05):
    
    comb_df = {}
    for k in in_keys:
        # pick only the actual label, and feat nm
        comb_df[k] = in_dict[k][['actualLabel_vgg', feat_nm]].dropna().sample(n=np.min([n, len(in_dict[k])]), 
                                                                     random_state=0, replace=False)
    
    # join all the dfs
    smple = np.array(pd.concat(list(comb_df.values()), ignore_index=True, sort=False)[feat_nm])
    smple = np.sort(smple)
    
    # out the threshold which give 95% quantile accuracy on this data
    i1 = int(np.ceil(len(smple)*cutoff))
    i2 = int(np.floor(len(smple)*cutoff))
    
    return (smple[i1] + smple[i2])/2


# compute distribution
def compute_mean_std(in_dict, in_keys, feat_nm, n=1000):
    
    comb_df = {}
    for k in in_keys:
        # pick only the actual label, and feat nm
        comb_df[k] = in_dict[k][['actualLabel_vgg', feat_nm]].dropna().sample(n=np.min([n, len(in_dict[k])]), 
                                                                     random_state=0, replace=False)
    # join all the dfs
    smple = np.array(pd.concat(list(comb_df.values()), ignore_index=True, sort=False)[feat_nm])
    
    return np.mean(smple), np.std(smple)


# compute the accuracies
def compute_accuracy(in_df, in_feat, in_thres):
    cur_arr = np.array(in_df[in_feat].dropna())
    return np.sum(cur_arr>in_thres)/len(cur_arr), (len(in_df)-len(cur_arr))/len(in_df)
