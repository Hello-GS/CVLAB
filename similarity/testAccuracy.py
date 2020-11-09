import configparser
import cv2
import numpy as np
import yagmail
import heapq

input_path = '/disk/data/total_incident/'

output_path_fre = ''
output_path_post = ''
user, password, receiver, host = '', '', [], ''
result_photo_name = ''
label_list_256 = []
label_list_32 = []
path_list = []


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


def cmpHash(hash1, hash2, level):
    n = 0
    if len(hash1) != len(hash2):
        return -1
    for index in range(len(hash1)):
        n += abs(int(hash1[index]) - int(hash2[index]))
        if n > level:
            break
    return n


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


def read_feature(pic_size, label):
    file = open('./output_' + str(pic_size) + '_label' + str(label) + '.txt')
    for line in file:
        cur = line.split('\t')
        if pic_size == 256:
            label_list_256.append((cur[0], cur[1]))
        else:
            label_list_32.append((cur[0], cur[1]))


def read_path():
    path_file = open('../DHash/output_path.txt')
    for line in path_file:
        path_list.append(line[0:-1])


def send_email(title, content):
    yagmail.SMTP(user=user, password=password, host=host).send(receiver, title, content)


def match(size):
    global result_photo_name
    for i in range(200):
        random = np.random.randint(0, len(path_list) - 1)
        ans = 99999999999999999
        count = 0
        path = path_list[random]
        print('#' + str(path))
        img = cv2.imread(path)
        hash = hashCalculator.aHash(img=img, pic_size=size)
        for i in label_list_256:
            count += 1
            if count % 1000 == 0:
                print('#has finish ', count)
            if i[0].split("/")[5] == path.split('/')[5]:
                continue
            cur_hash = cmpHash(hash, i[1][0:-1], ans)
            if ans > cur_hash:
                result_photo_name = i[0]
                ans = cur_hash
        print(str(path) + ';' + str(result_photo_name))
        print(ans)
        print('#analyse finish ,picture is ', result_photo_name)


hashCalculator = HashCalculator()
pre_ans = []
if __name__ == '__main__':
    read_config()
    read_feature(pic_size=256, label=0)
    read_feature(pic_size=256, label=1)
    read_feature(pic_size=32, label=0)
    read_feature(pic_size=32, label=1)
    print('read feature finish')
    read_path()
    print('read path finish,length =', len(path_list))

