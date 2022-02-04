# -*- coding: utf-8 -*-
"""
Multi-Resolution - Graph WatershEd using Nearest Neighbors (MR-GWENN)

This method performs unsupervised image segmentation.

GWENN is an unsupervised clustering algorithm based on nearest-neighbor (NN)
pointwise density estimation. In the present setting, GWENN is embedded in a
multiresolution framework using the Haar discrete wavelet transform (DWT).

One parameter is the number of NNs K (for GWENN)
For high K, the number of clusters decreases (default K=5)

The other parameter is the number of DWT levels nLevel. Please check the spatial
sizes of the second to last approximation image are both even.

Please cite:
    C. Cariou and K. Chehdi, "A new k-nearest neighbor density-based clustering
    method and its application to hyperspectral images," 2016 IEEE International
    Geoscience and Remote Sensing Symposium (IGARSS), 2016, pp. 6161-6164,
    https://doi.org/10.1109/IGARSS.2016.7730609

    Cariou, C.; Le Moan, S.; Chehdi, K. Improving K-Nearest Neighbor Approaches
    for Density-Based Pixel Clustering in Hyperspectral Remote Sensing Images.
    Remote Sens. 2020, 12, 3745. https://doi.org/10.3390/rs12223745

Created on Fri Feb 04 15:29:00 2022

@author: admincariou
"""



import pywt
import numpy as np
#np.random.seed(1234)
from sklearn.neighbors import NearestNeighbors
from NNsearch import fn_D_fast_full, fn_D_fast_idx
from gwenn import gwenn
import load_imgs
import matplotlib.pyplot as plt
import time


def show_exmplrs(im,labs):
    sizx, sizy, nbands = np.shape(im)
    data = np.reshape(im,(sizx*sizy,nbands))
    imexmplrs = data[np.uint32(labs)]/2
    idx = np.unique(labs)
    imexmplrs[idx] = np.array([255, 255, 255]) #imexmplrs[uniq]
    imexplrs = np.reshape(imexmplrs,(sizx, sizy, nbands))
    plt.figure()
    plt.imshow(np.uint8(imexplrs))
    plt.title('segmented image w/ exemplars', fontdict=None, loc='center')
    imseg = data[np.uint32(labs)]
    imseg = np.reshape(imseg,(sizx, sizy, nbands))
    return imseg

def interpolate_indices(idx,sizy):
    newidx = 2*(idx+(idx//sizy)*sizy)
    newidx = np.append(newidx,2*(idx+(idx//sizy)*sizy)+1)
    newidx = np.append(newidx,2*(idx+(idx//sizy)*sizy+sizy))
    newidx = np.append(newidx,2*(idx+(idx//sizy)*sizy+sizy)+1)
    return newidx

def my_dwt(im, nLevel):
    sizx, sizy, nbands = np.shape(im)
    im = np.transpose(im,(2,0,1))   # bands first
    startImage = im
    cA_dict = {}
    cH_dict = {}
    cV_dict = {}
    cD_dict = {}
    for level in range(nLevel):
        #print(level)
        (cAtmp, (cHtmp, cVtmp, cDtmp)) = pywt.dwt2(startImage, wavelet='haar')
        startImage = cAtmp
        cA_dict['layer_' + str(level)] = cAtmp
        cH_dict['layer_' + str(level)] = cHtmp
        cV_dict['layer_' + str(level)] = cVtmp
        cD_dict['layer_' + str(level)] = cDtmp

    return cA_dict, cH_dict, cV_dict, cD_dict


def MR_gwenn(im, nLevel=3, K=5):
    print('DWT')
    cA_dict, cH_dict, cV_dict, cD_dict = my_dwt(im, nLevel)

    fullRecon = cA_dict['layer_' + str(nLevel-1)]
    fullRecon = np.transpose(fullRecon,(1,2,0)) # bands last
    fullRecon_old = fullRecon

    for iLevel in range(nLevel-1,-1,-1):
        print('iLevel = ', iLevel)
        sizx, sizy, nbands = np.shape(fullRecon)
        if iLevel == nLevel-1:
            #print('fn_D_fast_full')
            distances, indices = fn_D_fast_full(fullRecon,K)
            labs, idx = gwenn(distances, indices, 1)
            print('Number of exemplars :', len(idx))
        else:
            #print('fn_D_fast_idx')
            newidx = interpolate_indices(idx,sizy)
            distances, indices = fn_D_fast_idx(fullRecon, newidx, K)
            labs, idx = gwenn(distances, indices, 1)
            print('Number of exemplars :', len(idx))

        print('IDWT')
        cHtmp = cH_dict['layer_' + str(iLevel)]
        cVtmp = cV_dict['layer_' + str(iLevel)]
        cDtmp = cD_dict['layer_' + str(iLevel)]
        fullRecon = np.transpose(fullRecon,(2,0,1)) # bands first
        fullRecon = pywt.idwt2((fullRecon, (cHtmp, cVtmp, cDtmp)), wavelet='haar')
        fullRecon = np.transpose(fullRecon,(1,2,0))
        # K = min(4*K,len(idx))
        # print('K = ', K)
        # plt.figure()
        # plt.imshow(np.uint8(fullRecon/2**iLevel))

    sizx, sizy, nbands = np.shape(fullRecon)
    newidx = interpolate_indices(idx,sizy)
    distances, indices = fn_D_fast_idx(fullRecon, newidx, K)
    labs, idx = gwenn(distances, indices, 1)
    imseg = show_exmplrs(fullRecon,labs)

    return imseg, idx, labs

def main():
    plt.close("all")

    impath = "./images" #only one png image in this folder
    im = load_imgs.load_imgs(impath)

    plt.figure()
    plt.imshow(np.uint8(im))
    plt.title('Original image', fontdict=None, loc='center')

    nLevel = 4
    K = 10

    start_time = time.time()
    imseg, idx, labs = MR_gwenn(im, nLevel, K)
    print("--- Elapsed: %s seconds ---" % (time.time() - start_time))

    plt.figure()
    plt.imshow(np.uint8(imseg))
    plt.title('Segmented image', fontdict=None, loc='center')
    MSE = np.mean(np.square(imseg-im))
    PSNR = 10*np.log10(255**2/MSE)
    print("MSE  = ", MSE)
    print("PSNR = ", PSNR)
    print('Number of clusters : ',len(idx))

main()

