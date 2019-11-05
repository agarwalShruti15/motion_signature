# 
CUDA_VISIBLE_DEVICES=0 python tools/main.py --cfg configs/class_resnet50_fabnet_ldrs.yaml > class_resnet50_fabnet_ldrs.txt
CUDA_VISIBLE_DEVICES=0 python tools/main.py --cfg configs/ms_resnet50_fabnet_ldrs.yaml > ms_resnet50_fabnet_ldrs.txt