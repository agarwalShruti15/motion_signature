import os
import numpy as np

"""
Usage: python -W ignore extract_fabnet.py --bsfldr <basefolder with mp4> --njobs 10 --openface <path to openface build/bin> --fnmodel <path to fabnet model file>

this file extracts the face embeddings for each mp4 file present in the base directory and sub directories. It saves the embeddings into a csv format and saves it in the same folder as input mp4 file.

# requirements: It needs the OpenFace bin folder (and all the model files) in OpenFaceBin folder at the same level as this directory
# It also need the fab net model downloaded from http://www.robots.ox.ac.uk/~vgg/research/unsup_learn_watch_faces/release_bmvc_fabnet.zip and kept in release folder"""

if __name__ == '__main__':

    #read all the txt file
    bs_fldr = '/data_efs/home/shruti/voxceleb/videos/vox2_mp4/dev/mp4'
    paths_file = 'data_subset'
    part_num = 10

    # collect all the video files to process
    full_struct = []
    for dirname, dirnames, filenames in os.walk(bs_fldr):
        vid_files = [v for v in os.listdir(dirname) if v.endswith('.mp4')]  # get all the videos
        for vi in range(len(vid_files)):
            if bs_fldr[-1] == '/':
                full_struct.append(os.path.join(dirname[len(bs_fldr):], vid_files[vi]))
            else:
                full_struct.append(os.path.join(dirname[len(bs_fldr)+1:], vid_files[vi]))
    
    # total files
    N = len(full_struct)
    print('Total files {}'.format(N))
    
    # sort the files
    full_struct = np.sort(np.array(full_struct))
    
    # number of files per subset
    f_per_sub = int(np.floor(N/part_num))
    strt = 0
    for s in range(part_num):
        
        out_file = paths_file + '_' + str(s) + '.txt'
        with open(out_file, 'w') as f:
            
            file_len = f_per_sub
            if s == part_num-1:
                file_len = len(full_struct[strt:])
                for j in np.arange(strt, strt+file_len):
                    f.write("%s\n" % full_struct[j])
            else:
                for j in np.arange(strt, strt+f_per_sub):
                    f.write("%s\n" % full_struct[j])
                
            print(f'{out_file} {file_len}')
            strt = strt + f_per_sub