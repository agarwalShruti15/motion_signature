# data download and unzip (un comment to get the data)
#sh data/download_data.sh
unzip data/vox2_mp4.zip -d data/vox2_mp4
rm data/vox2_mp4.zip
unzip data/vox2_test_mp4.zip -d data/vox2_test_mp4
rm data/vox2_test_mp4.zip

# Openface UBUNTU installation
cd data
wget https://github.com/TadasBaltrusaitis/OpenFace/archive/master.zip
unzip master.zip
rm master.zip
cd OpenFace-master
sh download_models.sh
sudo ./install.sh
cd ../../

# fab-net model installation
cd data
wget http://www.robots.ox.ac.uk/~vgg/research/unsup_learn_watch_faces/release_bmvc_fabnet.zip
unzip release_bmvc_fabnet.zip
rm release_bmvc_fabnet.zip
cd ../

# python libraries installation
pip install -r requirements.txt