import configparser
import os
import cv2
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import matplotlib
import matplotlib.pyplot as plt

input_path = '/disk/data/total_incident/'

pic_length = 256


def aHash(inputImg):
    inputImg = cv2.resize(inputImg, (pic_length, pic_length))
    gray = cv2.cvtColor(inputImg, cv2.COLOR_BGR2GRAY)
    hash_str = ''
    for i in range(pic_length):
        for j in range(pic_length):
            if (128 - i) * (128 - i) + (128 - j) * (128 - j) <= 128 * 128:
                hash_str += str(int(gray[i, j] / 32))
    return hash_str


def cmpHash(hash1, hash2, level):
    n = 0
    if len(hash1) != len(hash2):
        return -1
    for index in range(len(hash1)):
        n += abs(int(hash1[index]) - int(hash2[index]))
        if n > level:
            break
    return n


label_list = []
file = open('output_256_label1.txt')

for i in file:
    photo = i.split('\t')[0]
    mid = i.split('\t')[1]
    tupe = (photo, mid)
    label_list.append(tupe)
file2 = open('output_256_label0.txt')
for i in file2:
    photo = i.split('\t')[0]
    mid = i.split('\t')[1]
    tupe = (photo, mid)
    label_list.append(tupe)
print('#read file finish,length=', len(label_list))

result_photo_name = ''

path_file = open('../DHash/output_path.txt')
path_list = []

for line in path_file:
    path_list.append(line[0:-1])
print('#read path finish,length =', len(path_list))
for i in range(200):
    random = np.random.randint(0, len(path_list) - 1)
    ans = 99999999999999999
    count = 0
    # path = path_list[random]
    path = '/disk/data/total_incident/0/CME19960816141406/19960816_1455_c2_1024.jpg'
    print('#' + str(path))
    img = cv2.imread(path)
    hash = aHash(img)
    for i in label_list:
        count += 1
        if count % 1000 == 0:
            print('#has finish ', count)
        if i[0].split("/")[5] == path.split('/')[5]:
            continue
        if ans > cmpHash(hash, i[1][0:-1], ans):
            result_photo_name = i[0]
            ans = cmpHash(hash, i[1][0:-1], ans)
    print(str(path) + ';' + str(result_photo_name))
    print(ans)
    print('#analyse finish ,picture is ', result_photo_name)
