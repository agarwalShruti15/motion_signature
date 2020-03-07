import os
import argparse
import utils as u
import multiprocessing
from joblib import Parallel, delayed
import time
import numpy as np
import shutil

"""
Usage: python -W ignore extract_subset_fabnet.py --bsfldr '/data/home/shruti/voxceleb/videos/vox2_mp4/dev/mp4' --ofd '/data/home/shruti/voxceleb/vgg/vox2_mp4' --njobs 60 --openface 'OpenFace-master/build/bin' --fnmodel 'VGG_FACE.t7' --path_file 'utils/data_subset_0.txt'

this file extracts the face embeddings for each mp4 file present in the base directory and sub directories. It saves the embeddings into a csv format and saves it in the same folder as input mp4 file.

# requirements: It needs the OpenFace bin folder (and all the model files) in OpenFaceBin folder at the same level as this directory
# It also need the vgg net model downloaded from http://www.robots.ox.ac.uk/~vgg/software/vgg_face/  kept in the folder"""
def one_face_align(in_vid, out_file, openface_path, tmp_fldr):

    #save the frames
    os.makedirs(tmp_fldr, exist_ok=True)
    cmd = './{}/FeatureExtraction -f {} -simsize 256 -simalign -nomask -out_dir {} | grep -m 1 -o "abc" | grep -o "123"'.format(openface_path, in_vid, tmp_fldr)
    os.system(cmd)
    folder_name = os.path.join(tmp_fldr, in_vid.split('/')[-1].split('.')[0] + '_aligned')
    
    # create a video at 25 fps and remove the folder
    mov_cmd = f'ffmpeg -i {folder_name}/frame_det_00_%06d.bmp -c:v libx264 -vf fps=25 -pix_fmt yuv420p -qp 20 {out_file}'
    os.system(mov_cmd)
    
    # clear the aligned images
    shutil.rmtree(tmp_fldr)


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

if __name__ == '__main__':

    #read all the txt file
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--bsfldr', type=str, help='path to folder with mp4 files', default='data')
    parser.add_argument('--njobs', type=int, help='how many jobs to run per folder', default=1)
    parser.add_argument('--openface', type=str, help='path to openface build/bin folder', default='OpenFace-master/build/bin')    
    parser.add_argument('--ofd', type=str, help='base path to output the feature files')

    args = parser.parse_args()
    bs_fldr = args.bsfldr
    njobs = args.njobs
    openface_path = args.openface
    ofd = args.ofd

    # create the base output directory
    os.makedirs(ofd, exist_ok=True)
    
    # collect all the video files to process
    full_struct = []
    for dirname, dirnames, filenames in os.walk(bs_fldr):

        # if there are mp4 files then extract embeddings from this folder
        vid_files = [v for v in os.listdir(dirname) if v.endswith('.mp4')]  # get all the videos
        for vi in range(len(vid_files)):
            fl_n = os.path.splitext(vid_files[vi])[0] # the base file name

            # outfile
            if bs_fldr[-1] == '/':
                out_fldr = os.path.join(ofd, dirname[len(bs_fldr):]) # backslash
            else:
                out_fldr = os.path.join(ofd, dirname[len(bs_fldr)+1:]) # no backlash in basefolder name

            os.makedirs(out_fldr, exist_ok=True)
            out_file = os.path.join(out_fldr, fl_n + '.mp4')

            if not os.path.exists(out_file):
                tmp_fldr = '_'.join(dirname.split('/')[-2:]) + '_' + fl_n
                full_struct.append((os.path.join(dirname, fl_n + '.mp4'), out_file, openface_path, tmp_fldr))
                
    # run the jobs in parallel
    start_time = time.time()
    num_cores = multiprocessing.cpu_count()
    njobs = np.min([num_cores, njobs])
    print('Number of Cores {} \n'.format(num_cores))
    print('total processes {}'.format(len(full_struct)))
    Parallel(n_jobs=njobs, verbose=20)(delayed(one_face_align)(*full_struct[c]) for c in range(len(full_struct)))
    print('end time {} seconds'.format(time.time() - start_time))	
