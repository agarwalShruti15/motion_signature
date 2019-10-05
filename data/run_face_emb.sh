# don't include a slash in the end of bsfldr
python -W ignore extract_fabnet.py --bsfldr '/data/home/shruti/voxceleb/videos/vox2_test_mp4/mp4' --ofd '/data/home/shruti/voxceleb/fabnet/vox2_test_mp4'--njobs 30 --openface 'OpenFace-master/build/bin' --fnmodel 'release/nv2_curriculum.pth'
python -W ignore extract_fabnet.py --bsfldr '/data/home/shruti/voxceleb/videos/vox2_mp4/mp4' --ofd '/data/home/shruti/voxceleb/fabnet/vox2_mp4' --njobs 30 --openface 'OpenFace-master/build/bin' --fnmodel 'release/nv2_curriculum.pth'
