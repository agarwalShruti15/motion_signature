"""
Two seperate features for correlation and feat var. Can choose to do either or both

"""

import numpy as np
import utils as u
import pandas as pd
import os

fps = 30
np.random.seed(0)


feat1 = [' AU01_r', ' AU02_r', ' AU04_r', ' AU05_r', ' AU06_r'
            , ' AU07_r', ' AU09_r', ' AU10_r', ' AU12_r', ' AU14_r'
            , ' AU15_r', ' AU17_r', ' AU20_r', ' AU23_r', ' AU25_r'
            , ' AU26_r', ' pose_Rx', ' pose_Rz', 'lip_ver', 'lip_hor']
feat2 = [' AU01_r', ' AU02_r', ' AU04_r', ' AU05_r', ' AU06_r'
            , ' AU07_r', ' AU09_r', ' AU10_r', ' AU12_r', ' AU14_r'
            , ' AU15_r', ' AU17_r', ' AU20_r', ' AU23_r', ' AU25_r'
            , ' AU26_r', ' AU45_r', ' pose_Rx', ' pose_Ry', ' pose_Rz'
            , 'audio_rmse', 'audio_pitch']

if __name__ == '__main__':

    #input files to compare
    bs_fldr = '/data/home/shruti/voxceleb/videos/leaders' #207565
    """, 'jb' """
    vid_fldrs = ['bo', 'bo_UWfake', 'bo_faceswap',
                 'hc', 'hc_faceswap', 'hc_imposter', 'diff_test',
                 'ew', 'ew_faceswap', 'ew_imposter',
                 'bs', 'bs_faceswap', 'bs_imposter', 
                 'bo_imposter', 'dt_week', 'dt_rndm', 
                 'dt_faceswap', 'dt_imposter', 'br', 'pb', 'kh', 'jb', 'cb', 'trump_fake']
    ovrlp_win = 5
    u.CONF = 0.9

    out_obj_fldr = 'obj'
    vid_len_all = [int(10*30), 100] #[int(2*30), int(5*30), int(10*30), int(15*30), int(20*30)] #30 fps [int(10*30)]
    doVar_doCor = [[False, True]]
    out_fldr = ['only_cor']


    for o in range(len(out_fldr)):

        for vid_len in vid_len_all:

            #get training features and fit a model
            for f in range(len(vid_fldrs)):
                cache_corr_file =  '{}_{}'.format(vid_len, vid_fldrs[f])
                cur_train_feat = u.load_obj('{}/{}'.format(out_obj_fldr, out_fldr[o]), cache_corr_file)
                if cur_train_feat is None:
                    cur_train_feat = u.get_feats(os.path.join(bs_fldr, vid_fldrs[f]), feat1, feat2, vid_len=vid_len,
                                                     obj_bsfldr=out_obj_fldr, overlapping=True, ovrlp_win=ovrlp_win,
                                                     verbose=True, do_var=doVar_doCor[o][0], do_corr=doVar_doCor[o][1])
                    u.save_obj(cur_train_feat, '{}/{}'.format(out_obj_fldr, out_fldr[o]), cache_corr_file)

                print('{} {}'.format(vid_fldrs[f], len(pd.concat(list(cur_train_feat.values()), ignore_index=True, sort=False))))
