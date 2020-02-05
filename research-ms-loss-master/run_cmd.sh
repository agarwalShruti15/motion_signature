python generate_all.py --bsfldr /data/home/shruti/voxceleb/fabnet/leaders/ --njobs 4 --ofd /data/home/shruti/voxceleb/fabnet_metric/ --cfg configs/ms_resnet101_fabnet_vox.yaml --ow 5 
python generate_all.py --bsfldr /data/home/shruti/voxceleb/fabnet/leaders/ --njobs 4 --ofd /data/home/shruti/voxceleb/fabnet_metric25/ --cfg configs/ms_resnet101_fabnet_vox_frame25.yaml --ow 5
python generate_all.py --bsfldr /data/home/shruti/voxceleb/fabnet/leaders/ --njobs 4 --ofd /data/home/shruti/voxceleb/fabnet_metric50/ --cfg configs/ms_resnet101_fabnet_vox_frame50.yaml --ow 5
python generate_all.py --bsfldr /data/home/shruti/voxceleb/fabnet/leaders/ --njobs 4 --ofd /data/home/shruti/voxceleb/fabnet_metric75/ --cfg configs/ms_resnet101_fabnet_vox_frame75.yaml --ow 5
