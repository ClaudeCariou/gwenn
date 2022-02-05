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

@author:    Claude Cariou - Univ Rennes/Enssat, CNRS-IETR/MULTIP - Lannion, France
            claude.cariou@univ-rennes1.fr
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy import stats
from sklearn.utils.extmath import weighted_mode

import matplotlib.pyplot as plt
import load_imgs
import time

# Real images
impath = "./images"
im = load_imgs.load_imgs(impath)

# Choose here the image you wish to segment
im = im[1]

plt.close("all")

K = 100     #<<<<<<<<<<<<<<< change the number of nearest neighbors here

sizx, sizy, nbands = np.shape(im)

N = sizx*sizy

plt.figure()
plt.imshow(np.uint8(im))
plt.title('Original image', fontdict=None, loc='center')

start_time = time.time()

# reshape image to data array
data = np.reshape(im,(N,nbands))

# get KNN distances and indices
nbrs = NearestNeighbors(n_neighbors=K, algorithm='ball_tree').fit(data)
dists, neigh = nbrs.kneighbors(data)

# estimate the inverse of the pointwise (local) density
invdens = np.sum(dists,axis=1) + 1e-16

# sort the density in descending order and get the coresponding indices
v = np.argsort(invdens)

# initialize the label array
labs = np.zeros(N, dtype=np.int32)
P = np.zeros(N, dtype=np.int32)

# set the point with highest density as the exemplar of the first cluster
labs[v[0]] = v[0]

# process the other data points in order of decreasing density
for k in range(1,N):
    P[v[k-1]] = 1;
    B = neigh[v[k]]
    temp = P[B]
    temp = B[temp==1]   # the set of current NNs which have been already processed

    if temp.any():      # if nonempty
        #idx = stats.mode(labs[temp])                   #initial GWENN (IGARSS paper)
        idx = weighted_mode(labs[temp],1./invdens[temp])    #WM-GWENN (better) (RS paper)
        labs[v[k]] = idx[0]
    else:               # if empty, create a new cluster with its exemplar
        labs[v[k]] = v[k]

# get the final exemplars and their number
exemplars = np.unique(labs)
NC = len(exemplars)

print("Number of clusters = ", NC)

# display results
imexmplrs = data[np.uint32(labs)]//4
imexmplrs[exemplars] = np.array([255, 255, 255])
imexplrs = np.reshape(imexmplrs,(sizx, sizy, nbands))
plt.figure()
plt.imshow(np.uint8(imexplrs))
plt.title('Cluster exemplars', fontdict=None, loc='center')

imseg = np.reshape(data[np.uint32(labs)],(sizx, sizy, nbands))
plt.figure()
plt.imshow(np.uint8(imseg))
plt.title('Segmented image', fontdict=None, loc='center')

MSE = np.mean(np.square(imseg-im))
PSNR = 10*np.log10(255**2/MSE)
print("MSE  = ", MSE)
print("PSNR = ", PSNR)

print("--- Elapsed: %s seconds ---" % (time.time() - start_time))
