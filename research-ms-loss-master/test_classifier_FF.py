import torch
import sys
from ret_benchmark.config import cfg
from ret_benchmark.modeling import build_model
import os
import numpy as np
import pickle
import argparse
from sklearn.metrics import roc_curve, auc
import re

sys.path.insert(1, '/data/home/shruti/voxceleb/motion_signature/research-ms-loss-master/')

# get all the embedding
def load_obj(fldr, name ):
    if not os.path.exists(os.path.join(fldr, name + '.pkl')):
        return None
    with open(os.path.join(fldr, name + '.pkl'), 'rb') as f:
        return pickle.load(f)

# save the obj
def save_obj(obj, fldr, name ):
    os.makedirs(fldr, exist_ok=True)
    with open(os.path.join(fldr, name + '.pkl'), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

## load model

def load_model(cfgpath):
    
    cfg.merge_from_file(cfgpath)
    # load the model
    if torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cuda"
        #device = torch.device("cuda")
    else:
        cfg.MODEL.DEVICE = "cpu"
        #device = torch.device("cpu")
    
    print(cfg.MODEL.DEVICE)
    model = build_model(cfg)
    
    model.to(cfg.MODEL.DEVICE)
    model.eval()
    return model

# get metric embedding
def FabNet(in_feat, data_params):
    
    fab_emb = in_feat/np.linalg.norm(in_feat, axis=1, keepdims=True)
    return (fab_emb-data_params['mean'])/data_params['std']

def get_prob(infldr, fl_n, metric_model, data_params):
    
    cur_feat = np.load(os.path.join(infldr, fl_n + '.npy'))
    if len(cur_feat.shape)>2:
        cur_feat = cur_feat[:, :, 0].copy()
    
    #these correlation features to consider
    
    if cur_feat.shape[0]>(data_params['T']+data_params['ow']):
        
        frame_rng = np.arange(0, len(cur_feat)-data_params['T'], data_params['ow'])
        out_feat = np.zeros((len(frame_rng), data_params['emb_sz']))
        for j in range(len(frame_rng)):
            
            cur_data = data_params['data_loader_func'](cur_feat[frame_rng[j]:frame_rng[j]+data_params['T'], :], data_params)
            cur_data = np.reshape(cur_data, (1, 1, cur_data.shape[0], cur_data.shape[1]))
            if torch.cuda.is_available():
                tensor_data = torch.from_numpy(cur_data).cuda().float()
            else:
                tensor_data = torch.from_numpy(cur_data).float()
                
            # pass the model
            with torch.no_grad():
                out_feat[j, :] = metric_model(tensor_data).data.cpu().numpy().ravel()
                out_feat[j, :] = np.exp(out_feat[j, :])/np.sum(np.exp(out_feat[j, :]))
                
        return out_feat
    
    else:
        print('Length smaller than', data_params['T'], infldr, fl_n)
        return []
        
        
# dataset loader
data_loader_dict = {'FabNet': FabNet}


def load_dict_file(infile):
    
    # read the file and collect in label dict 
    label_dict = {}
    with open(infile, 'r') as f:
        for line in f:
            _path, _ = re.split(r",| ", line.strip())
            _label = _path.split('/')[0]
            if _label not in list(label_dict.keys()):
                label_dict[_label] = []
                
            label_dict[_label].extend([_path])
    
    return label_dict

# train and test split
def train_test():

    train_ldr_dict = load_dict_file('/data/home/shruti/voxceleb/motion_signature/data/utils/ff_train.txt')
    val_ldr_dict = load_dict_file('/data/home/shruti/voxceleb/motion_signature/data/utils/ff_val.txt')
    test_ldr_dict = load_dict_file('/data/home/shruti/voxceleb/motion_signature/data/utils/ff_test.txt')
    return train_ldr_dict, val_ldr_dict, test_ldr_dict


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--bsfldr', type=str, help='base folder with features for which correlation needs to be computed', default='data')
    parser.add_argument('--cfg', type=str, help='path to config file')
    parser.add_argument('--ow', type=np.int32, help='frames shift for next corr window', default=5)
    
    args = parser.parse_args()
    bs_fldr = args.bsfldr
    cfg_path = args.cfg
    ovrlp_win = args.ow
    
    train_files, val_files, test_files = train_test()
    
    # params
    print(bs_fldr, cfg_path)
    
    # load the model for frame embeddings
    metric_model = load_model(cfg_path)
    
    # mean and std
    data_params = {}
    data_params['data_loader_func'] = data_loader_dict[cfg.INPUT.DATA_LOADER]
    data_params['mean'] = np.reshape(np.load(cfg.INPUT.MEAN), (1, -1))
    data_params['std'] = np.reshape(np.load(cfg.INPUT.STD), (1, -1))
    data_params['T'] = cfg.INPUT.FRAME_LENGTH
    data_params['ow'] = ovrlp_win
    data_params['emb_sz'] = cfg.MODEL.HEAD.DIM
    
    # these are the table values
    train_ldrs = np.array(['FF_orig', 'FF_Deepfakes', 'FF_FaceSwap', 'FF_Face2Face', 'FF_NeuralTextures'])
    lbls = np.array([1, 0, 0, 0, 0])    
    
    # collect all the files to process
    out_results = {}
    for k in train_ldrs:
        out_results[k] = {}
        for f in test_files[k]:
            
            [dirname, vid_file] = os.path.split(f)
            fl_n = os.path.splitext(vid_file)[0]
            
            if os.path.exists(os.path.join(bs_fldr, dirname, fl_n + '.npy')):
                out_results[k][f] = get_prob(os.path.join(bs_fldr, dirname), fl_n, metric_model, data_params)
            else:
                out_results[k][f] = get_prob(os.path.join(bs_fldr, dirname), '_' + fl_n, metric_model, data_params)
                    
    # ROC
    pos_pred = np.vstack(list(out_results['FF_orig'].values()))[:, 1]
    for k in ['FF_Deepfakes', 'FF_FaceSwap', 'FF_Face2Face', 'FF_NeuralTextures']:
        
        neg_pred = np.vstack(list(out_results[k].values()))[:, 1]
        
        fpr, tpr, thresholds = roc_curve(np.concatenate((np.ones((len(pos_pred), ), dtype=np.int32), 
                                                          np.zeros((len(neg_pred), ), dtype=np.int32)), axis=0), 
                                         np.concatenate((pos_pred, neg_pred), axis=0))
        roc_auc = auc(fpr, tpr)
        
        print(f'{k} acc: {roc_auc}')
