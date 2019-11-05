import os
import numpy as np

"""
Creates text file with filename to further extract features.

"""

if __name__ == '__main__':

    #read all the txt file
    #bs_fldr = '/data/home/shruti/voxceleb/videos/vox2_test_mp4/mp4'
    bs_fldr = '/data/home/shruti/voxceleb/videos/leaders'
    paths_file = 'data_leaders_with_fakes'
    part_num = 1

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