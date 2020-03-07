# motion_signature
Motion signature based soft-biometric for people.

## steps for installation (tested only on Ubuntu 18.04)

1) download data using data/download_data.sh

2) activate python virtual env. I have tested the code on python3.6

3) run sh setup.sh

4) cd data/

5) download the VGG, Fabnet and Fabnet Metric Learning models for feature extraction. 
The models can be found at: https://www.dropbox.com/sh/lfon3rvjvt6uatk/AAAALG6B07faPSPjY3GZRHXXa?dl=0


## Feature Generation

### VGG and FAb-Net feature:

Feature Extraction File: data/extract_vgg_fabnet.py
Usage: cd data/
python -W ignore extract_vgg_fabnet.py \ 
--bsfldr <basefolder with mp4 files> \ 
--njobs 4 \
--openface <Openface Binary Folder> \
--fnmodel <Path to VGG Model : download from .> \
--of  <output base folder for fabnet features>
--ov  <output base folder for VGG features>

### Behavior-Net Features

Feature Extraction File: research-ms-loss-master/generate_all.py
Usage: 
1) cd research-ms-loss-master/
2) Update the pretrained model path in config file : 
MODEL:PRETRIANED_PATH

3) python -W ignore generate_all.py \ 
--bsfldr <basefolder with mp4> \ 
--njobs 4 \
--ofd  <output base folder>
--cfg configs/ms_resnet101_fabnet_vox.yaml
--ow 5

  
## training

1) Activate the virtual environment (source /data/opt/voxceleb/bin/activate)
2) cd research-ms-loss-master
3) sh run_cub.sh
python tools/main.py --cfg configs/ms_resnet101_fabnet_vox.yaml > log_file.txt

There are example configuration files given for metric learning task in research-ms-loss-master/configs/ms_resnet101_fabnet_vox.yaml