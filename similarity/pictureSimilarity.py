import configparser
import os

import cv2
import yagmail

# from CVLAB.similarity.fenjiaodu import angleHashCalculator
from fenjiaodu import angleHashCalculator
input_path = ''
output_path_fre = ''
output_path_post = ''
user, password, receiver, host = '', '', [], ''
pic_length = 512


def Hash(test_pic, pic_size, index_set):
    test_pic = cv2.resize(test_pic, (pic_size, pic_size))
    gray = cv2.cvtColor(test_pic, cv2.COLOR_BGR2GRAY)
    hash_str = ''
    for (i, j) in index_set:
        hash_str += str(int(gray[i, j] / 32))
    return hash_str


def read_config():
    config = configparser.ConfigParser()
    config.read('./config.ini')
    global user, password, receiver, input_path, output_path_fre, output_path_post, host, pic_length
    input_path = config.get('data', 'data_path')
    output_path_fre = config.get('data', 'output_path_fre')
    output_path_post = config.get('data', 'output_path_post')
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


def write_hash_feature(pic_size, label):
    target_file = output_path_fre + str(pic_size) + output_path_post + str(label) + '.txt'
    print(target_file)
    output_file = open(target_file, 'a')
    all_flies = os.listdir(input_path + str(label) + '/')
    count = 0
    for file in all_flies:
        if file == '.DS_Store':
            continue
        graphs = os.listdir(input_path + str(label) + '/' + file)
        for graph in graphs:
            abs_path = '/disk/data/total_incident/' + str(label) + '/' + file + '/' + graph
            output_file.write(
                abs_path + '\t' + hashCalculator.calculateHash(fs=abs_path) + '\n')
            count += 1
            if count % 100 == 0:
                print("has finish", str(count))
    send_email('图片大小为' + str(pic_length) + 'ahash值已经计算完成',
               '对应路径为disk/11712504/fuck/similarity' + target_file)


if __name__ == '__main__':
    read_config()
    hashCalculator = angleHashCalculator(512)
    write_hash_feature(pic_size=pic_length, label=0)
    write_hash_feature(pic_size=pic_length, label=1)
