# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 16:58:32 2021

@author:    Claude Cariou - Univ Rennes/Enssat, CNRS-IETR/MULTIP - Lannion, France
            claude.cariou@univ-rennes1.fr
"""

import cv2
import os
import glob
import numpy as np

def load_imgs(img_dir):
    data_path = os.path.join(img_dir,'*g')
    files = glob.glob(data_path)
    print(files)
    imgs = []
    for f1 in files:
        img = cv2.imread(f1)
        print('Image: ',f1, 'Size:', np.shape(img))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) #required to permute R & B
        imgs.append(img)
    imgs = np.array(imgs, dtype=object)
    return imgs