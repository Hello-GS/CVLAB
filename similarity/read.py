import os
import random
import numpy as np
import shutil
import cv2
path = '/disk/11711603/LAB/Video/ans.txt'
file = open(path)
count = 0

crop_size = (512, 512)
for line in file:
    line = line.split('\n')[0]
    print(line)
    path1=line.split(';')[0]
    print(path1)
    path2 = line.split(';')[1]
    print(path2)
    str1 = path1.split('.')[0]

    str2 = path2.split('.')[0]

    fram1 = cv2.imread(path1)

    fram1 = cv2.resize(fram1, crop_size, interpolation=cv2.INTER_CUBIC)
    print(fram1.shape)
    str = '/disk/11711603/LAB/Video/ttt/'+str1.split('.')[0].replace('/','-')+'---'+str2.split('.')[0].replace('/','-')+'.png'
    fram2 = cv2.imread(path2)
    fram2 = cv2.resize(fram2, crop_size, interpolation=cv2.INTER_CUBIC)
    print(fram2.shape)
    frame = np.hstack((fram1,fram2))
    cv2.imwrite(str,frame)
    print(str)