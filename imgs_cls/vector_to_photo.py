import numpy as np
import cv2

path = '/disk/data/output_256_label0.txt'

file = open(path)
string=''
i = 0
for line in file:
    i+=1
    string = line.split('\t')[1][:51076]
    if i==256:
        break
string_list = []
for i in string:
    string_list.append(int(i)*32)

frame = np.array(string_list)
frame = np.reshape(frame, newshape=(226,226))
cv2.imwrite('/disk/data/test4.png',frame)

