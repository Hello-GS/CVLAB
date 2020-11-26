import numpy as np
import cv2

path = '/disk/data/output_256_label0.txt'

file = open(path)
string=''
for line in file:
    string = line.split('\t')[1][:51076]
    break
string_list = []
for i in string:
    string_list.append(int(i))

frame = np.array(string_list)
frame = np.reshape(frame, newshape=(226,226))
cv2.imwrite('/disk/data/test.png',frame)

