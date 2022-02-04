# -*- coding: utf-8 -*-
"""
(Weighted Mode) Graph WatershEd using Nearest Neighbors (GWENN)

GWENN is an unsupervised clustering algorithm based on nearest-neighbor (NN)
pointwise density estimation.

The only parameter of the method is the number of NNs K.

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
from scipy import stats
from sklearn.utils.extmath import weighted_mode

def gwenn(dists, neigh, wm):
    N, K = np.shape(dists)
    dens = np.sum(dists,axis=1)

    v = np.argsort(dens)
    labs = np.zeros(N, dtype=np.integer)
    P = np.zeros(N, dtype=np.integer);

    labs[v[0]] = v[0]

    for k in range(1,N):
        P[v[k-1]] = 1;
        B = neigh[v[k]]
        temp = P[B]
        temp = B[temp==1]

        if temp.any():
            if wm:
                idx = weighted_mode(labs[temp],1/dens[temp])   #WM-GWENN (better) (RS paper)
            else:
                idx = stats.mode(labs[temp])                   #initial GWENN (IGARSS paper)
            labs[v[k]] = idx[0]
        else:
            labs[v[k]] = v[k]

    exemplars,_ = np.unique(labs, return_inverse=True)
    return labs, exemplars
