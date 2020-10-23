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
output_path = './output_path.txt'


if __name__ == '__main__':
    output = open(output_path, 'r+')
    flies = os.listdir(input_path + '0/')
    count = 0
    for file in flies:
        if file == '.DS_Store':
            continue
        graphs = os.listdir(input_path + '0/' + file)
        for graph in graphs:
            curpath = '/disk/data/total_incident/' + '0/' + file + '/' + graph
            output.write(curpath + '\n')
            count += 1
            if count % 1000 == 0:
                print("has finish" + str(count))
    flies = os.listdir(input_path + '1/')
    for file in flies:
        if file == '.DS_Store':
            continue
        graphs = os.listdir(input_path + '1/' + file)
        for graph in graphs:
            curpath = '/disk/data/total_incident/' + '1/' + file + '/' + graph
            output.write(curpath + '\n')
            count += 1
            if count % 1000 == 0:
                print("has finish" + str(count))

