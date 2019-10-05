# data download and unzip (un comment to get the data)
#sh data/download_data.sh
#unzip data/vox2_mp4.zip -d data/vox2_mp4
#rm data/vox2_mp4.zip
#unzip data/vox2_test_mp4.zip -d data/vox2_test_mp4
#rm data/vox2_test_mp4.zip

# Openface UBUNTU installation, it assume we already have opencv and dlib installed
cd data
wget https://github.com/TadasBaltrusaitis/OpenFace/archive/master.zip
unzip master.zip
rm master.zip
cd OpenFace-master
sh download_models.sh
echo "Installing OpenFace..."
mkdir -p build
cd build
cmake -D CMAKE_CXX_COMPILER=g++-8 -D CMAKE_C_COMPILER=gcc-8 -D CMAKE_BUILD_TYPE=RELEASE ..
make
cd ../../

# fab-net model installation
cd data
wget http://www.robots.ox.ac.uk/~vgg/research/unsup_learn_watch_faces/release_bmvc_fabnet.zip
unzip release_bmvc_fabnet.zip
rm release_bmvc_fabnet.zip
cd ../

# python libraries installation
pip install -r requirements.txt
