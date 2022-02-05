# -*- coding: utf-8 -*-
"""
Computes the distances and indices of K NNs.

Created on Fri Jan 28 18:41:44 2022

@author:    Claude Cariou - Univ Rennes/Enssat, CNRS-IETR/MULTIP - Lannion, France
            claude.cariou@univ-rennes1.fr
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors

def fn_D_fast_full(data, K):
    sizx, sizy, nbands = np.shape(data)
    N = sizx*sizy

    data = np.reshape(data,(N,nbands))  #np.transpose(data,(1,0,2))

    nbrs = NearestNeighbors(n_neighbors=K, algorithm='ball_tree').fit(data)
    dists, neigh = nbrs.kneighbors(data)
    return dists, neigh

def fn_D_fast_idx(data, idx, K):
    sizx, sizy, nbands = np.shape(data)
#    K = len(idx)
    N = sizx*sizy

    data = np.reshape(data,(N,nbands))
    tmp2 = data[idx]

    nbrs = NearestNeighbors(n_neighbors=K, algorithm='ball_tree').fit(tmp2)
    dists, neigh = nbrs.kneighbors(data,n_neighbors=K)
    neigh = idx[neigh]
    return dists, neigh
