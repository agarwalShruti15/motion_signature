import os
import argparse
import utils as u
from models_multiview import FrontaliseModelMasks_wider
import torch
import multiprocessing
from joblib import Parallel, delayed
import time
import numpy as np

"""
Usage: python -W ignore extract_fabnet.py --bsfldr <basefolder with mp4> --njobs 10 --openface <path to openface build/bin> --fnmodel <path to fabnet model file>

this file extracts the face embeddings for each mp4 file present in the base directory and sub directories. It saves the embeddings into a csv format and saves it in the same folder as input mp4 file.

# requirements: It needs the OpenFace bin folder (and all the model files) in OpenFaceBin folder at the same level as this directory
# It also need the fab net model downloaded from http://www.robots.ox.ac.uk/~vgg/research/unsup_learn_watch_faces/release_bmvc_fabnet.zip and kept in release folder"""

num_additional_ids=32

if __name__ == '__main__':

	#read all the txt file
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('--bsfldr', type=str, help='path to folder with mp4 files', default='data')
	parser.add_argument('--njobs', type=int, help='how many jobs to run per folder', default=1)
	parser.add_argument('--openface', type=str, help='path to openface build/bin folder', default='OpenFace-master/build/bin')    
	parser.add_argument('--fnmodel', type=str, help='path to fab-net model', default='release')    
	parser.add_argument('--ofd', type=str, help='base path to output the feature files')

	args = parser.parse_args()
	bs_fldr = args.bsfldr
	njobs = args.njobs
	openface_path = args.openface
	fabnet_path = args.fnmodel
	ofd = args.ofd
	
	# create the base output directory
	os.makedirs(ofd, exist_ok=True)
	
	# load the model for frame embeddings
	emb_model = FrontaliseModelMasks_wider(3, inner_nc=u.emb_n, num_additional_ids=num_additional_ids)
	emb_model.load_state_dict(torch.load(fabnet_path)['state_dict'])
	emb_model.eval()	
	
	# collect all the video files to process
	full_struct = []
	for dirname, dirnames, filenames in os.walk(bs_fldr):

		# if there are mp4 files then extract embeddings from this folder
		vid_files = [v for v in os.listdir(dirname) if v.endswith('.mp4')]  # get all the videos
		for vi in range(len(vid_files)):
			fl_n = os.path.splitext(vid_files[vi])[0] # the base file name
			out_file = os.path.join(ofd, '_'.join(dirname[len(bs_fldr)+1:].split('/')) + '_' + fl_n + '.npy')
			full_struct.append((dirname, fl_n, out_file, emb_model, openface_path))
			
	# run the jobs in parallel
	start_time = time.time()
	num_cores = multiprocessing.cpu_count()
	njobs = np.min([num_cores, njobs])
	print('Number of Cores {} \n'.format(num_cores))
	print('total processes {}'.format(len(full_struct)))
	Parallel(n_jobs=njobs, verbose=10)(delayed(u.fabnet_one_vid)(*full_struct[c]) for c in range(len(full_struct)))
	print('end time {} seconds'.format(time.time() - start_time))	