import os
import argparse
import multiprocessing
from joblib import Parallel, delayed
import time
import numpy as np
import pandas as pd

"""
Usage: python -W ignore get_au.py --bsfldr '/data/home/shruti/voxceleb/videos/vox2_mp4/dev/mp4' --ofd '/data/home/shruti/voxceleb/vgg/vox2_mp4' --njobs 60 --openface 'OpenFace-master/build/bin' --fnmodel 'VGG_FACE.t7' --path_file 'utils/data_subset_0.txt'

this file extracts the face embeddings for each mp4 file present in the base directory and sub directories. It saves the embeddings into a csv format and saves it in the same folder as input mp4 file.

# requirements: It needs the OpenFace bin folder (and all the model files) in OpenFaceBin folder at the same level as this directory
# It also need the vgg net model downloaded from http://www.robots.ox.ac.uk/~vgg/software/vgg_face/  kept in the folder"""

feat_nm = [' AU01_r', ' AU02_r', ' AU04_r', ' AU05_r', ' AU06_r'
            , ' AU07_r', ' AU09_r', ' AU10_r', ' AU12_r', ' AU14_r'
            , ' AU15_r', ' AU17_r', ' AU20_r', ' AU23_r', ' AU25_r'
            , ' AU26_r', ' pose_Rx', ' pose_Rz', 'lip_ver', 'lip_hor']


#read landmark(,au etc.)
def lndmrk_to_df(infile):
    cur_df = pd.read_csv(infile)
    clean_df = cur_df.loc[cur_df[' confidence']>0.0]

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

    out_df = in_lndmrk_df.copy()
    out_df['lip_hor'] = d_hr
    out_df['lip_ver'] = d_vert

    return out_df

#landmark(,au etc.) csv file
def tmp_get_full_df(infile):

    csv_files = infile + '.csv'

    if not os.path.exists(csv_files):
        return None

    # read transcript and landmark csv
    lndmrk_df = lndmrk_to_df(csv_files)

    # combine the transcript and landmark files
    out_df = combine_lip_opening(lndmrk_df)

    return out_df

def one_vid_feat(in_fldr, fl_n, out_file):
    cur_csv = os.path.join(in_fldr, fl_n )
    
    assert not os.path.exists(out_file)
    try:
        
        full_df = tmp_get_full_df(cur_csv)
        if full_df is None:
            return
        out_emb = np.array(full_df[feat_nm])
        
        print(f'{fl_n}: {out_emb.shape}')
        
        # save the output
        np.save(out_file, out_emb.astype(np.float32))
        
    except Exception as e:
        print('{} {}'.format(cur_csv, e))
    

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

if __name__ == '__main__':

    #read all the txt file
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--bsfldr', type=str, help='path to folder with mp4 files', default='data')
    parser.add_argument('--njobs', type=int, help='how many jobs to run per folder', default=1)
    parser.add_argument('--ofd', type=str, help='base path to output the feature files')
    parser.add_argument('--path_file', type=str, help='the file names to execute on this machine, all is None', default=None)

    args = parser.parse_args()
    bs_fldr = args.bsfldr
    njobs = args.njobs
    ofd = args.ofd
    paths_file = args.path_file

    # create the base output directory
    os.makedirs(ofd, exist_ok=True)
    
    # collect all the video files to process
    full_struct = []
    with open(paths_file, 'r')  as f:
        paths = f.readlines()

    for p in paths:
        [dirname, vid_file] = os.path.split(p)

        # if there are mp4 files then extract embeddings from this folder
        fl_n = os.path.splitext(vid_file)[0] # the base file name
        out_file = os.path.join(ofd, '_'.join(dirname.split('/')) + '_' + fl_n + '.npy')
        if not os.path.exists(out_file):
            full_struct.append((os.path.join(bs_fldr, dirname), fl_n, out_file))

    # run the jobs in parallel
    start_time = time.time()
    num_cores = multiprocessing.cpu_count()
    njobs = np.min([num_cores, njobs])
    print('Number of Cores {} \n'.format(num_cores))
    print('total processes {}'.format(len(full_struct)))
    Parallel(n_jobs=njobs, verbose=20)(delayed(one_vid_feat)(*full_struct[c]) for c in range(len(full_struct)))
    print('end time {} seconds'.format(time.time() - start_time))	
