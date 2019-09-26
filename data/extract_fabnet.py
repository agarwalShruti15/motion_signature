import os
import argparse
import utils as u

"""
Usage: python -W ignore extract_fabnet.py --bsfldr <basefolder with mp4> --njobs 10 --openface <path to openface build/bin> --fnmodel <path to fabnet model file>

this file extracts the face embeddings for each mp4 file present in the base directory and sub directories. It saves the embeddings into a csv format and saves it in the same folder as input mp4 file.

# requirements: It needs the OpenFace bin folder (and all the model files) in OpenFaceBin folder at the same level as this directory
# It also need the fab net model downloaded from http://www.robots.ox.ac.uk/~vgg/research/unsup_learn_watch_faces/release_bmvc_fabnet.zip and kept in release folder"""


if __name__ == '__main__':
    
    #read all the txt file
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--bsfldr', type=str, help='path to folder with mp4 files', default='data')
    parser.add_argument('--njobs', type=int, help='how many jobs to run per folder', default=1)
    parser.add_argument('--openface', type=str, help='path to openface build/bin folder', default='OpenFace-master/build/bin')    
    parser.add_argument('--fnmodel', type=str, help='path to fab-net model', default='release') 
    
    args = parser.parse_args()
    bs_fldr = args.bsfldr
    njobs = args.njobs
    openface_path = args.openface
    fabnet_path = args.fnmodel
    
    for dirname, dirnames, filenames in os.walk(bs_fldr):
        
        # if there are mp4 files then extract embeddings from this folder
        vid_files = [v for v in os.listdir(dirname) if v.endswith('.mp4')]  # get all the videos
        if len(vid_files)>0:
            print(dirname)
            u.folder_extract(dirname, dirname, openface_path, fabnet_path, njobs=njobs)