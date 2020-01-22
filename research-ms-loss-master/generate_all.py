import torch
import sys
from ret_benchmark.config import cfg
from ret_benchmark.modeling import build_model
import os
import numpy as np
import pickle
import argparse
import multiprocessing
from joblib import Parallel, delayed

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

def get_feats(infldr, fl_n, out_file, metric_model, data_params):
    
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
            
        np.save(out_file, out_feat)
    else:
        print('Length smaller than', data_params['T'], infldr, fl_n)
        
        
# dataset loader
data_loader_dict = {'FabNet': FabNet}

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--bsfldr', type=str, help='base folder with features for which correlation needs to be computed', default='data')
    parser.add_argument('--njobs', type=int, help='how many jobs to run per folder', default=1)
    parser.add_argument('--ofd', type=str, help='base path to output the feature files')
    parser.add_argument('--cfg', type=str, help='path to config file')
    parser.add_argument('--ow', type=np.int32, help='frames shift for next corr window', default=5)
    
    args = parser.parse_args()
    njobs = args.njobs
    bs_fldr = args.bsfldr
    cfg_path = args.cfg
    ovrlp_win = args.ow
    ofd = args.ofd
    
    # params
    print(bs_fldr, ofd, cfg_path)
    
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
    
    # collect all the files to process
    full_struct = []
    for dirname, dirnames, filenames in os.walk(bs_fldr):

        # if there are mp4 files then extract embeddings from this folder
        files = [v for v in os.listdir(dirname) if v.endswith('.npy')]  # get all the videos
        
        for vi in range(len(files)):
            
            fl_n = os.path.splitext(files[vi])[0] # the base file name
            if bs_fldr[-1] == '/':
                out_fldr = os.path.join(ofd, dirname[len(bs_fldr):]) # backslash
            else:
                out_fldr = os.path.join(ofd, dirname[len(bs_fldr)+1:]) # no backlash in basefolder name
            os.makedirs(out_fldr, exist_ok=True)
            
            out_file = os.path.join(out_fldr, fl_n + '.npy')
            
            if not os.path.exists(out_file):
                full_struct.append((dirname, fl_n, out_file, metric_model, data_params ))
                                       
    # run the jobs in parallel
    num_cores = multiprocessing.cpu_count()
    njobs = np.min([num_cores, njobs])
    print('Number of Cores {} \n'.format(num_cores))
    print('total processes {}'.format(len(full_struct)))
    Parallel(n_jobs=njobs, verbose=20)(delayed(get_feats)(*full_struct[c]) for c in range(len(full_struct)))