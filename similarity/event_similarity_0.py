import configparser
import cv2
import numpy as np
import yagmail
import os
import pandas as pd
import numpy as np
from fenjiaodu import angleHashCalculator

input_path = '/disk/data/total_incident/'
output_path_fre = ''
output_path_post = ''
user, password, receiver, host = '', '', [], ''
result_photo_name = ''
label_list_256 = []
label_list_32 = []
path_list = []
result_path = ''


class PriorityQueue:

    def __init__(self):
        self._queue = []
        self._min = 99999999999999999

    def push(self, item, priority):
        if self._min > priority:
            self._queue.clear()
            self._queue.append(item)
            self._min = priority
        if self._min == priority:
            self._queue.append(item)

    def get_list(self):
        return self._queue


def Hash(test_pic, pic_size, index_set):
    test_pic = cv2.resize(test_pic, (pic_size, pic_size))
    gray = cv2.cvtColor(test_pic, cv2.COLOR_BGR2GRAY)
    hash_str = ''
    for (i, j) in index_set:
        hash_str += str(int(gray[i, j] / 32))
    return hash_str


class HashCalculator:
    valid_index_256 = []
    valid_index_32 = []

    def __init__(self):
        for i in range(256):
            for j in range(256):
                if (128 - i) * (128 - i) + (128 - j) * (128 - j) <= 128 * 128:
                    self.valid_index_256.append((i, j))
        for i in range(32):
            for j in range(32):
                if (16 - i) * (16 - i) + (16 - j) * (16 - j) <= 16 * 16:
                    self.valid_index_32.append((i, j))

    def aHash(self, test_pic, pic_size):
        if pic_size == 32:
            return Hash(test_pic=test_pic, pic_size=pic_size, index_set=self.valid_index_32)
        elif pic_size == 256:
            return Hash(test_pic=test_pic, pic_size=pic_size, index_set=self.valid_index_256)


def cmpHash(hash1, hash2, level):
    n = 0
    if len(hash1) != len(hash2):
        return -1
    if level == 99999999999999999:
        for index in range(len(hash1)):
            n += abs(int(hash1[index]) - int(hash2[index]))
        return n
    for index in range(len(hash1)):
        n += abs(int(hash1[index]) - int(hash2[index]))
        if n > level:
            break
    return n


def read_config():
    config = configparser.ConfigParser()
    config.read('./config.ini')
    global user, password, receiver, input_path, output_path_fre, output_path_post, host, result_path
    input_path = config.get('data', 'data_path')
    output_path_fre = config.get('data', 'output_path_fre')
    output_path_post = config.get('data', 'output_path_post')
    user = config.get('email', 'user')
    password = config.get('email', 'password')
    receiver = config.get('email', 'receiver').split(',')
    host = config.get('email', 'host')
    result_path = config.get('data', 'result_path')


def read_feature(pic_size, label):
    file = open(output_path_fre + str(pic_size) + output_path_post + str(label) + '.txt')
    for line in file:
        cur = line.split('\t')
        if pic_size == 512:
            label_list_256.append((cur[0], cur[1]))
        if pic_size == 256:
            label_list_256.append((cur[0], cur[1]))
        else:
            label_list_32.append((cur[0], cur[1]))
    file.close()


def read_path():
    path_file = open('/disk/11712501/CVLAB/DHash/output_path.txt')
    for line in path_file:
        path_list.append(line[0:-1])


def send_email(title, content):
    yagmail.SMTP(user=user, password=password, host=host).send(receiver, title, content)


def match(label_list):
    q = PriorityQueue()
    ans = 99999999999999999
    for label in label_list:
        if label[0].split('/')[5] == path.split('/')[5]:
            continue
        cur_hash = cmpHash(target_hash, label[1][0:-1], ans)
        q.push(label[0], cur_hash)
    return q.get_list()


def calculate_answer(size, pre_ans):
    answer = 99999999999999999
    answer_path = ''
    if len(pre_ans) == 1:
        return pre_ans[0]
    for candidate in pre_ans:
        cur_pic = cv2.imread(candidate)
        cur_hash = hashCalculator.aHash(test_pic=cur_pic, pic_size=size)
        cur_ans = cmpHash(target_hash, cur_hash, answer)
        if answer > cur_ans:
            answer_path = candidate
            answer = cur_ans
    return answer_path


if __name__ == '__main__':
    hashCalculator = HashCalculator()
    caculater512 = angleHashCalculator(512)
    read_config()
    # read_feature(pic_size=256, label=0)
    # read_feature(pic_size=256, label=1)
    # read_feature(pic_size=32, label=0)
    # read_feature(pic_size=32, label=1)

    read_feature(pic_size=512, label=0)
    print('read feature finish', flush=True)
    read_path()
    print('read path finish,length =', len(path_list), flush=True)

    # 1类
    # path0 = '/disk/data/total_incident/1'
    #
    # for event_0 in os.listdir(path0):
    #
    #     cur_month = event_0[3:9]
    #     for event_1 in os.listdir(path0):
    #         event_1_month = event_1[3:9]
    #         if cur_month == event_1_month:
    #             continue
    #         for file0 in os.listdir(path0+'/'+event_0):
    #             image_path0 = path0+'/'+event_0+'/'+file0
    #             for file1 in os.listdir(path0+'/'+event_1):
    #                 image_path1 = path0+'/'+event_1+'/'+file1

    # 1类
    # 所有的事件放在一个list中
    item_list = []
    print("13232")
    for item in label_list_256:
        item_name = item[0].split('/')[5]
        if item_list.__contains__(item_name) is False:
            item_list.append(item_name)
    print("has put all items in one list")
    item_similarity_map = np.zeros([len(item_list), len(item_list), 2], dtype=int)
    print("the length of the items: ", len(item_list))
    for item0 in label_list_256:
        image_path0 = item0[0]
        hash0 = item0[1][:-1]

        event_name0 = image_path0.split('/')[5]
        for item1 in label_list_256:
            image_path1 = item1[0]
            hash1 = item1[1][:-1]

            event_name1 = image_path1.split('/')[5]
            if event_name0[3:9] == event_name1[3:9]:
                continue

            item_similarity_map[item_list.index(event_name0)][item_list.index(event_name1)][0] += cmpHash(hash0, hash1,
                                                                                                      99999999999999999)
            item_similarity_map[item_list.index(event_name0)][item_list.index(event_name1)][1] += 1

        print("finish ", item0[0]," epoch ")


    for index1 in item_similarity_map:
        for index2 in index1:
            if index2[1]!= 0:
                index2[0] = index2[0]*1.0/index2[1]
    items_similarity_result = []

    for a in range(len(item_list)):
        ans = 9999999999
        index = a
        for b in range(len(item_list)):
            if ans > item_similarity_map[a][b][0] and item_similarity_map[a][b][0] !=0:
                ans = item_similarity_map[a][b][0]
                index = b
        result = item_list[a] +"/"+ item_list[index]
        items_similarity_result.append(result)
    print(item_list.index('CME'+items_similarity_result[0].split('CME')[2]))
    with open('item_similarity_0.txt', 'w') as f:
        for line in items_similarity_result:
            f.write(line+"\n")










    # for t in range(500):
    #     random = np.random.randint(0, len(path_list) - 1)
    #     path = path_list[random]
    #     # img = cv2.imread(path)
    #     # target_hash = hashCalculator.aHash(test_pic=img, pic_size=32)
    #     target_hash = caculater512.calculateHash(fs=path)  # 哈希值
    #
    #     print(path + ';' + calculate_answer(256, match(label_list_32)), flush=True)
    #     if t == 999:
    #         send_email('匹配结果完成' + str(t), result_path.removeprefix('./'))
