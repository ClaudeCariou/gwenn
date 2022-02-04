# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 16:58:32 2021

@author: admincariou
"""

import cv2
import os
import glob
import numpy as np

def load_imgs(img_dir):
#    img_dir = "C:/D/arvalis/OneDrive_1_24-06-2021/extracted_256x256" # Enter Directory of all images
    data_path = os.path.join(img_dir,'*g')
    files = glob.glob(data_path)
    print(files)
    # data = []
    for f1 in files:
        img = cv2.imread(f1)
        print(np.shape(img))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) #required to permute R & B
        #data.append(img)
    # data = np.array(data)
    # print(data.dtype)
    return img