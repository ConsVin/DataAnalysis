# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 09:58:19 2017

@author: Const
"""


import matplotlib.pyplot as plt
import pylab
import skimage
from skimage.io import imread
from sklearn.cluster import KMeans
import numpy as np
import math

def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    d = codebook.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image

def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


image = imread('parrots.jpg')
image = skimage.img_as_float(image)
pylab.imshow(image)
a = image.shape
w, h, d = original_shape = tuple(image.shape);
X = image.reshape((w*h,d));

for n in range(1,20):                  
    kmeans = KMeans(init='k-means++',random_state=241,n_clusters=n).fit(X);
    labels = kmeans.predict(X)
    ImageRestored = recreate_image(kmeans.cluster_centers_, labels, w, h);
    ImgPsnr = psnr(ImageRestored,image)                              
    plt.imshow(ImageRestored)
    print('PSNR(%d) = %5.3f.' % (n, ImgPsnr))

#image_clust=labels.reshape(a);                      
#pylab.imshow(image_clust)  

print('Done')