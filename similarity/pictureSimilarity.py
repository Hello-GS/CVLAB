import configparser
import os
import cv2
from PIL import Image
import requests
from io import BytesIO
import yagmail

input_path = ''
output_path_label0 = ''
output_path_label1 = ''
user, password, receiver, host = '', '', [], ''
pic_length = 256


class HashCalculator:
    valid_index = set()

    def __init__(self):
        for i in range(256):
            for j in range(256):
                if (128 - i) * (128 - i) + (128 - j) * (128 - j) <= 128 * 128:
                    self.valid_index.add((i, j))

    def aHash(self, img, pic_size):
        img = cv2.resize(img, (pic_size, pic_size))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hash_str = ''
        for i in range(pic_size):
            for j in range(pic_size):
                if (i, j) in self.valid_index:
                    hash_str += str(int(gray[i, j] / 32))
        return hash_str


def read_config():
    config = configparser.ConfigParser()
    config.read('./config.ini')
    global user, password, receiver, input_path, output_path_label0, output_path_label1, host, pic_length
    input_path = config.get('data', 'data_path')
    output_path_label0 = config.get('data', 'output_path_label0')
    output_path_label1 = config.get('data', 'output_path_label1')
    user = config.get('email', 'user')
    password = config.get('email', 'password')
    receiver = config.get('email', 'receiver').split(',')
    host = config.get('email', 'host')
    pic_length = int(config.get('data', 'picture_length'))


def send_email(title, content):
    yagmail.SMTP(user=user, password=password, host=host).send(receiver, title, content)


def cmpHash(hash1, hash2):
    n = 0
    if len(hash1) != len(hash2):
        return -1
    for i in range(len(hash1)):
        n += abs(int(hash1[i]) - int(hash2[i]))
    return n


def getImageByUrl(url):
    html = requests.get(url, verify=False)
    image = Image.open(BytesIO(html.content))
    return image


def write_hash_feature(label):
    if label == 0:
        target_file = output_path_label0
    else:
        target_file = output_path_label1
    output_file = open(target_file, 'r+')
    all_flies = os.listdir(input_path + str(label) + '/')
    count = 0
    for file in all_flies:
        if file == '.DS_Store':
            continue
        graphs = os.listdir(input_path + str(label) + '/' + file)
        for graph in graphs:
            abs_path = '/disk/data/total_incident/' + str(label) + '/' + file + '/' + graph
            output_file.write(
                abs_path + '\t' + hashCalculator.aHash(img=cv2.imread(abs_path), pic_size=pic_length) + '\n')
            count += 1
            if count % 1000 == 0:
                print("has finish" + str(count))


if __name__ == '__main__':
    read_config()
    hashCalculator = HashCalculator()
    write_hash_feature(0)
    write_hash_feature(1)
    send_email('图片大小为' + str(pic_length) + 'ahash值已经计算完成',
               '对应路径为disk/11712504/fuck/similarity' + output_path_label0 + '\t' +
               'disk/11712504/fuck/similarity' + output_path_label1)
