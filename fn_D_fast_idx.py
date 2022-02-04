# -*- coding: utf-8 -*-
"""
Computes the distances and indices of K NNs.

Created on Fri Jan 28 18:41:44 2022

@author: admincariou
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors

def fn_D_fast_full(data, K):
    sizx, sizy, nbands = np.shape(data)
    N = sizx*sizy

    data = np.reshape(data,(N,nbands))

    nbrs = NearestNeighbors(n_neighbors=K, algorithm='ball_tree').fit(data)
    dists, neigh = nbrs.kneighbors(data)
    return dists, neigh
