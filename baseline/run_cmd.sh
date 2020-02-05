python train_one_class_svm.py --bsfldr /data/home/shruti/voxceleb/aus_corr_100 \
--tf /data/home/shruti/voxceleb/motion_signature/data/utils/leaders_100_train.txt \
--ncomp 190 \
--ofd /data/home/shruti/voxceleb/svm_model/aus_corr_100 \
--gamma 0.1 \
--nu 0.1

python train_one_class_svm.py --bsfldr /data/home/shruti/voxceleb/fabnet_corr_100_pca2000 \
--tf /data/home/shruti/voxceleb/motion_signature/data/utils/leaders_100_train.txt \
--ncomp 190 \
--ofd /data/home/shruti/voxceleb/svm_model/fabnet_corr_100_190 \
--gamma 0.1 \
--nu 0.1

python train_one_class_svm.py --bsfldr /data/home/shruti/voxceleb/fabnet_corr_100_pca2000 \
--tf /data/home/shruti/voxceleb/motion_signature/data/utils/leaders_100_train.txt \
--ncomp 256 \
--ofd /data/home/shruti/voxceleb/svm_model/fabnet_corr_100_256 \
--gamma 0.1 \
--nu 0.1

python train_one_class_svm.py --bsfldr /data/home/shruti/voxceleb/fabnet_corr_100_pca2000 \
--tf /data/home/shruti/voxceleb/motion_signature/data/utils/leaders_100_train.txt \
--ncomp 512 \
--ofd /data/home/shruti/voxceleb/svm_model/fabnet_corr_100_512 \
--gamma 0.1 \
--nu 0.1

python train_one_class_svm.py --bsfldr /data/home/shruti/voxceleb/fabnet_corr_100_pca2000 \
--tf /data/home/shruti/voxceleb/motion_signature/data/utils/leaders_100_train.txt \
--ncomp 1024 \
--ofd /data/home/shruti/voxceleb/svm_model/fabnet_corr_100_1024 \
--gamma 0.1 \
--nu 0.1

python train_one_class_svm.py --bsfldr /data/home/shruti/voxceleb/fabnet_corr_100_pca2000 \
--tf /data/home/shruti/voxceleb/motion_signature/data/utils/leaders_100_train.txt \
--ncomp 2000 \
--ofd /data/home/shruti/voxceleb/svm_model/fabnet_corr_100_2000 \
--gamma 0.1 \
--nu 0.1
