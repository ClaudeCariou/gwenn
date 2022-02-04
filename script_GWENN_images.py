# -*- coding: utf-8 -*-
"""
(Weighted Mode) Graph WatershEd using Nearest Neighbors (GWENN)

GWENN is an unsupervised clustering algorithm based on nearest-neighbor (NN)
pointwise density estimation.

The only parameter of the method is the number of NNs K.
For high K, the number of clusters decreases.

Please cite:
    C. Cariou and K. Chehdi, "A new k-nearest neighbor density-based clustering
    method and its application to hyperspectral images," 2016 IEEE International
    Geoscience and Remote Sensing Symposium (IGARSS), 2016, pp. 6161-6164,
    https://doi.org/10.1109/IGARSS.2016.7730609

    Cariou, C.; Le Moan, S.; Chehdi, K. Improving K-Nearest Neighbor Approaches
    for Density-Based Pixel Clustering in Hyperspectral Remote Sensing Images.
    Remote Sens. 2020, 12, 3745. https://doi.org/10.3390/rs12223745

Created on Fri Jan 28 18:41:44 2022

@author: admincariou
"""

import numpy as np

np.random.seed(1234)
from sklearn.neighbors import NearestNeighbors
from scipy import stats
from sklearn.utils.extmath import weighted_mode

import matplotlib.pyplot as plt
import load_imgs
import time

start_time = time.time()



# # Check
# data = np.random.rand(64,64,3)
# data = np.append(data,np.random.rand(64,64,3)+2.,axis=1)
# data = data*10
# im = data

# Real images
impath = "./images" #only one png image
im = load_imgs.load_imgs(impath)
im = im[0]


plt.close("all")

K = 500     #<<<<<<<<<<<<<<< change K if necessary

sizx, sizy, nbands = np.shape(im)

N = sizx*sizy

plt.figure()
plt.imshow(np.uint8(im))

data = np.reshape(im,(N,nbands))

nbrs = NearestNeighbors(n_neighbors=K, algorithm='ball_tree').fit(data)
dists, neigh = nbrs.kneighbors(data)

dens = np.sum(dists,axis=1)

v = np.argsort(dens)

labs = np.zeros(N, dtype=np.integer)
P = np.zeros(N, dtype=np.integer)

labs[v[0]] = v[0]

for k in range(1,N):
    P[v[k-1]] = 1;
    B = neigh[v[k]]
    temp = P[B]
    temp = B[temp==1]

    if temp.any():
        #idx = stats.mode(labs[temp])                   #initial GWENN (IGARSS paper)
        idx = weighted_mode(labs[temp],1/dens[temp])    #WM-GWENN (better) (RS paper)
        labs[v[k]] = idx[0]
    else:
        labs[v[k]] = v[k]

uniq,idx = np.unique(labs, return_inverse=True)
NC = len(uniq)

print("Number of clusters = ", NC)

imexmplrs = data[np.uint32(labs)]//4
imexmplrs[uniq] = np.array([255, 255, 255])
imexplrs = np.reshape(imexmplrs,(sizx, sizy, nbands))
plt.figure()
plt.imshow(np.uint8(imexplrs))

imrec = np.reshape(data[np.uint32(labs)],(sizx, sizy, nbands))
plt.figure()
plt.imshow(np.uint8(imrec))

MSE = np.mean(np.square(imrec-im))
PSNR = 10*np.log10(255**2/MSE)
print("MSE  = ", MSE)
print("PSNR = ", PSNR)

print("--- Elapsed: %s seconds ---" % (time.time() - start_time))