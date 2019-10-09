import numpy as np
import os

def get_label_file(bsfldr):
    
    subfldrs = ['bo/', 'br/', 'bs/', 'cb/', 'dt_week/', 'dt_rndm/', 'ew/', 'hc/', 'jb/', 'kh/', 'pb/']
    subfldrs = np.sort(subfldrs)
    labels = np.arange(len(subfldrs))
    label_file_dict = {}
    for i in range(len(labels)):
        
        label_file_dict[labels[i]] = [os.path.join(subfldrs[i], f) for f in os.listdir(os.path.join(bsfldr, subfldrs[i])) if f.endswith('.npy') ]
        print(subfldrs[i], labels[i], label_file_dict[labels[i]])
    
    return label_file_dict

if __name__ == '__main__':
    
    divide_test = True
    test_perc = 0.1
    outfile = 'leaders'
    bsfldr = '/data/home/shruti/voxceleb/fabnet/leaders'
    
    # collect all the files first
    label_file_dict = get_label_file(bsfldr)    
    
    if divide_test:
        
        train_fd = open(os.path.join(bsfldr, outfile + '_train.txt'), 'w')
        test_fd = open(os.path.join(bsfldr, outfile + '_test.txt'), 'w')
        
        for k in label_file_dict.keys():
            
            # considering sorted to ensure disjoint train and test videos
            files = np.sort(label_file_dict[k])
            train_len = int((1-test_perc)*len(files))
            for i in range(train_len):
                train_fd.write('{}, {}\n'.format(files[i], k))
            
            for i in range(train_len, len(files)):
                test_fd.write('{}, {}\n'.format(files[i], k))
                
        train_fd.close()
        test_fd.close()
            
    else:
        
        fd = open(os.path.join(bsfldr, outfile + '.txt'), 'w')
        
        for k in label_file_dict.keys():
            
            # considering sorted to ensure disjoint train and test videos
            files = np.sort(label_file_dict[k])
            for i in range(len(files)):
                fd.write('{}, {}\n'.format(files[i], k))
            
                
        fd.close()
