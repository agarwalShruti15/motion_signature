import os
import argparse
import utils as u

"""this file extracts the face embeddings for each mp4 file present in the base directory and sub directories. It saves the embeddings into a csv format and saves it in the same folder as input mp4 file.

# requirements: It needs the OpenFace bin folder (and all the model files) in OpenFaceBin folder at the same level as this directory
# It also need the fab net model downloaded from http://www.robots.ox.ac.uk/~vgg/research/unsup_learn_watch_faces/release_bmvc_fabnet.zip and kept in release folder"""

if __name__ == '__main__':
    
    #read all the txt file
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--bsfldr', type=str, help='path to folder with mp4 files', default='data')
    parser.add_argument('--njobs', type=int, help='how many jobs to run per folder', default=1)
    
    args = parser.parse_args()
    bs_fldr = args.bsfldr
    njobs = args.njobs
    
    for dirname, dirnames, filenames in os.walk(bs_fldr):
        
        # if there are mp4 files then extract embeddings from this folder
        vid_files = [v for v in os.listdir(dirname) if v.endswith('.mp4')]  # get all the videos
        if len(vid_files)>0:
            print(dirname)
            u.folder_extract(dirname, dirname, njobs=njobs)