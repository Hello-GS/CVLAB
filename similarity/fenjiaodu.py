import configparser
import os
import cv2
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import matplotlib
import matplotlib.pyplot as plt
import math
from similarity.testAccuracy import HashCalculator

input_path = '/disk/data/total_incident/'
pic_length = 256


def calculate_angle(i, j):
    if i ==0 and j==0:
        return 0
    print(i,j)
    x = i - 127
    y = j - 127
    if y < 0:
        return 360 - math.acos(x / math.sqrt(x * x + y * y) / math.pi * 180)
    elif y == 0:
        if x > +0:
            return 0
        else:
            return 180

    return math.acos(x / math.sqrt(x * x + y * y) / math.pi * 180)


def calculate_dest(i, j):
    return abs(i - 127) * abs(i - 127) + abs(j - 127) * abs(j - 127)

def calculateAngle():
    for x in range(256):
        for y in range(256):
            calculate_angle(x,y)

if __name__ == '__main__':
    calculate_angle(100,20)
    calculateAngle()
