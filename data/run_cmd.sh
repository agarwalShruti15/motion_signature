
#python resave_videos.py --bsfldr /data/home/shruti/voxceleb/videos/leaders/ --njobs 1 --ofd /data/home/shruti/voxceleb/videos/compression/
#CUDA_VISIBLE_DEVICES=0 python extract_fabnet.py --bsfldr /data/home/shruti/voxceleb/videos/compression/ --njobs 4 --openface OpenFace-master/build/bin/ --fnmodel release/nv2_curriculum.pth --ofd /data/home/shruti/voxceleb/fabnet/compression/
#CUDA_VISIBLE_DEVICES=0 python extract_vgg.py --bsfldr /data/home/shruti/voxceleb/videos/compression/ --njobs 4 --openface OpenFace-master/build/bin/ --fnmodel vgg_face_dag.pth --ofd /data/home/shruti/voxceleb/vgg/compression/
#CUDA_VISIBLE_DEVICES=0 python extract_fabnet.py --bsfldr /data/home/shruti/voxceleb/videos/leaders/ --njobs 4 --openface OpenFace-master/build/bin/ --fnmodel release/nv2_curriculum.pth --ofd /data/home/shruti/voxceleb/fabnet/leaders/
#CUDA_VISIBLE_DEVICES=0 python extract_vgg.py --bsfldr /data/home/shruti/voxceleb/videos/leaders/ --njobs 4 --openface OpenFace-master/build/bin/ --fnmodel vgg_face_dag.pth --ofd /data/home/shruti/voxceleb/vgg/leaders/

CUDA_VISIBLE_DEVICES=0 python extract_vgg_fabnet.py --bsfldr /data/home/shruti/voxceleb/videos/leaders/ --njobs 4 --openface OpenFace-master/build/bin/ --of /data/home/shruti/voxceleb/fabnet/leaders/ --ov /data/home/shruti/voxceleb/vgg/leaders/
