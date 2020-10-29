import configparser
import os
import cv2
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import matplotlib
import matplotlib.pyplot as plt
import imagehash
input_path = '/disk/data/total_incident/'

pic_length = 256



file = open('output_256_label0_Dhash.txt')
label_list = []
for i in file:
    photo = i.split('\t')[0]
    mid = i.split('\t')[1]
    tupe = (photo, mid)
    label_list.append(tupe)
file2 = open('output_256_label1_Dhash.txt')
for i in file2:
    photo = i.split('\t')[0]
    mid = i.split('\t')[1]
    tupe = (photo, mid)
    label_list.append(tupe)
print('#read file finish,length=', len(label_list))

result_photo_name = ''

path_file = open('output_path.txt')
path_list = []

for line in path_file:
    path_list.append(line[0:-1])
print('#read path finish,length =', len(path_list))
output = open('output_200_random_ans.txt', 'r+')
for i in range(200):
    random = np.random.randint(0, len(path_list) - 1)

    ans = 99999999999999999
    count = 0
    path = path_list[random]

    img = Image.open(path)
    img = img.resize((pic_length, pic_length), Image.ANTIALIAS)
    input_image = imagehash.dhash(img, pic_length)
    for i in label_list:
        count += 1
        if count % 1000 == 0:
            print('#has finish ', count)
        if i[0] == path.split('/')[-1]:
            continue
        temp =np.abs(input_image-imagehash.hex_to_hash(i[1][0:-1]))
        if temp < ans :
            ans = temp
            result_photo_name=i[0]
    print(str(path) + ';' + str(result_photo_name) + '\n')
    print('#analyse finish ,picture is ', result_photo_name)
