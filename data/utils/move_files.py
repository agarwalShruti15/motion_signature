import os 
import shutil

bs_fdlr = '/data/home/shruti/voxceleb'
out_fldr = '/data/home/shruti/voxceleb/temp'

os.makedirs(out_fldr, exist_ok=True)

with open('mv_files.txt', 'r')  as f:
        paths = f.readlines()
        
for i in paths:
        
        i = i.strip()
        print(os.path.join(bs_fdlr, i), os.path.join(out_fldr, '?'.join(i.split('/')) ))
        shutil.move(os.path.join(bs_fdlr, i), os.path.join(out_fldr, '?'.join(i.split('/')) ))