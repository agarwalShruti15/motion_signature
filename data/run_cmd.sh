
#python resave_videos.py --bsfldr /data/home/shruti/voxceleb/videos/leaders/ --njobs 1 --ofd /data/home/shruti/voxceleb/videos/compression/
#CUDA_VISIBLE_DEVICES=0 python extract_fabnet.py --bsfldr /data/home/shruti/voxceleb/videos/compression/ --njobs 4 --openface OpenFace-master/build/bin/ --fnmodel release/nv2_curriculum.pth --ofd /data/home/shruti/voxceleb/fabnet/compression/
#CUDA_VISIBLE_DEVICES=0 python extract_vgg.py --bsfldr /data/home/shruti/voxceleb/videos/compression/ --njobs 4 --openface OpenFace-master/build/bin/ --fnmodel vgg_face_dag.pth --ofd /data/home/shruti/voxceleb/vgg/compression/
CUDA_VISIBLE_DEVICES=0 python extract_fabnet.py --bsfldr /data/home/shruti/voxceleb/videos/leaders/ --njobs 4 --openface OpenFace-master/build/bin/ --fnmodel release/nv2_curriculum.pth --ofd /data/home/shruti/voxceleb/fabnet/leaders/
CUDA_VISIBLE_DEVICES=0 python extract_vgg.py --bsfldr /data/home/shruti/voxceleb/videos/leaders/ --njobs 4 --openface OpenFace-master/build/bin/ --fnmodel vgg_face_dag.pth --ofd /data/home/shruti/voxceleb/vgg/leaders/

#python get_correlations.py --bsfldr /data/home/shruti/voxceleb/aus/leaders/ --njobs 90 --ofd /data/home/shruti/voxceleb/aus_corr_100 --vid_len 100 --ow 5 --us 0
#python get_correlations.py --bsfldr /data/home/shruti/voxceleb/fabnet/leaders/ --njobs 90 --ofd /data/home/shruti/voxceleb/fabnet_corr_100 --vid_len 100 --ow 5 --us 1
#python get_correlations.py --bsfldr /data/home/shruti/voxceleb/aus/leaders/ --njobs 90 --ofd /data/home/shruti/voxceleb/aus_corr_300 --vid_len 300 --ow 5 --us 0
#python get_correlations.py --bsfldr /data/home/shruti/voxceleb/fabnet/leaders/ --njobs 90 --ofd /data/home/shruti/voxceleb/fabnet_corr_300 --vid_len 300 --ow 5 --us 1
#python get_pca_features.py --ptf /data/home/shruti/voxceleb/fabnet/vox2_mp4/voxceleb_100_train.txt --ncomp 2000 --T 100 --njobs 90 --test_f /data/home/shruti/voxceleb/fabnet_corr_100/ --ofd /data/home/shruti/voxceleb/fabnet_corr_100_pca2000/
