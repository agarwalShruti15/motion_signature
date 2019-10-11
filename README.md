# motion_signature
Motion signature based soft-biometric for people.

## steps for installation (tested only on Ubuntu 18.04)

1) download data using data/download_data.sh

2) activate python virtual env. I have tested the code on python3.6

3) run sh setup.sh

4) cd data/

5) python -W ignore extract_fabnet.py --bsfldr <basefolder with mp4> --njobs 10 --openface <path to openface build/bin> --fnmodel <path to fabnet model file>
    
  Example usage is given in run_face_emb.sh
  
  python -W ignore extract_fabnet.py --bsfldr 'vox2_test_mp4' --njobs 10 --openface 'OpenFace-master/build/bin' --fnmodel 'release/nv2_curriculum.pth'
  
## training

```
sh run_cub.sh
```
