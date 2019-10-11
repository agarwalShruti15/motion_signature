import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

SHOW_IMAGES=True
sample_rate_show_images = 1


data_filename = 'deepfake_embedding_features.npy'
labels_filename = 'deepfake_embedding_labels.npy'
if SHOW_IMAGES:
    images_filename = 'deepfake_embedding_images.npy'

#data_filename = 'deepfake_vgg_embedding_train.npy'
#labels_filename = 'deepfake_vgg_embedding_train_labels.npy'
#if SHOW_IMAGES:
#    images_filename = 'deepfake_vgg_embedding_train_images.npy'


X = np.load(data_filename)
Y = np.load(labels_filename)
if SHOW_IMAGES:
    images = np.load(images_filename)

#import pdb ; pdb.set_trace()
ind_fake = np.where(Y==1)
ind_real = np.where(Y==0)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)
#import pdb ; pdb.set_trace()
cm = plt.cm.get_cmap('RdYlBu')
xy = range(20)
ax.scatter(X[ind_real,0], X[ind_real, 1], c='b', cmap='viridis')
ax.scatter(X[ind_fake,0], X[ind_fake, 1], c='r', cmap='viridis')
ax.axis('tight')
plt.axis('tight')


if SHOW_IMAGES:
    xx = X[ind_real, 0].squeeze()
    yy = X[ind_real, 1].squeeze()
    images2 = images[ind_real, :, :, :].squeeze()
    idx=0
    for x0, y0, img in zip(xx, yy, images2):
        if idx % sample_rate_show_images == 0:
            ab = AnnotationBbox(OffsetImage(img, zoom=0.15), (x0+0.5, y0+0.5), xycoords='data', frameon=True, pad=0.1, bboxprops =dict(edgecolor='blue'))
            ax.add_artist(ab)
        idx+=1
    print('plotted {} real images'.format(idx))

    xx = X[ind_fake, 0].squeeze()
    yy = X[ind_fake, 1].squeeze()
    images2 = images[ind_fake, :, :, :].squeeze()
    idx=0
    for x0, y0, img in zip(xx, yy, images2):
        if idx % sample_rate_show_images == 0:
            ab = AnnotationBbox(OffsetImage(img, zoom=0.15), (x0+0.5, y0+0.5), xycoords='data', frameon=True, pad=0.1, bboxprops =dict(edgecolor='red'))
            ax.add_artist(ab)
        idx+=1
    print('plotted {} fake images'.format(idx))

ax.autoscale()
plt.show()
plt.savefig('embedding-2.png')

