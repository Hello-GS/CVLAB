
from utils.misc import *
import pandas as pd
from sklearn.metrics.pairwise import  cosine_similarity
import cv2
import matplotlib as mpl
#mpl.use('Agg')
from matplotlib import pyplot as plt
import sys
import os

def find_topk_similar_images(image_name,path,image_path,k=5):

    df = pd.read_json(path)
    query_list = df['query event']
    event_list = df['similar events']
    query_list = list(query_list)


    index = query_list.index(image_name)
    similarity_incident = event_list[index]


    if (os.path.exists(image_path + '/0/' + image_name)):
        img_path = os.listdir(image_path + '/0/' + image_name)
        img = cv2.imread(image_path+ '/0/' + image_name+ '/' + img_path[0])
        img = img[:, :, [2, 1, 0]]
        title = image_name + "\nquery event"
        plt.subplot(2, 3, 1)
        plt.imshow(img)
        plt.title(title,fontdict={'fontsize':9})
        plt.xticks([])
        plt.yticks([])


    elif (os.path.exists(image_path + '/1/' + image_name)):
        img_path = os.listdir(image_path + '/1/' + image_name)
        img = cv2.imread(image_path+ '/1/' + image_name+ '/' + img_path[0])
        img = img[:, :, [2, 1, 0]]
        title = image_name + "\nquery event"
        plt.subplot(2, 3, 1)
        plt.imshow(img)
        plt.title(title,fontdict={'fontsize':9})
        plt.xticks([])
        plt.yticks([])

    else:
        print('--------------------------query path error--------------------------------')


    for i in range(5):
        if (os.path.exists(image_path + '/0/' + similarity_incident[i])):
            img_path = os.listdir(image_path + '/0/' + similarity_incident[i])
            img = cv2.imread(image_path + '/0/' + similarity_incident[i] + '/' + img_path[0])
            img = img[:, :, [2, 1, 0]]
            title =  similarity_incident[i] +'\nsimilar event'+str(i+1)
            plt.subplot(2,3,i+2)
            plt.imshow(img)
            plt.title(title, fontdict={'fontsize': 9})
            plt.xticks([])
            plt.yticks([])


        elif(os.path.exists(image_path + '/1/' + similarity_incident[i])):
            img_path = os.listdir(image_path + '/1/' + similarity_incident[i])
            img = cv2.imread(image_path + '/1/' + similarity_incident[i] + '/' + img_path[0])
            img = img[:, :, [2, 1, 0]]
            title = similarity_incident[i] + '\nsimilar event' + str(i + 1)
            plt.subplot(2,3,i+2)
            plt.imshow(img)
            plt.title(title, fontdict={'fontsize': 9})

            plt.xticks([])
            plt.yticks([])

        else:

            print('--------------------------gallery path error--------------------------------')
    plt.subplots_adjust(wspace=1)
    plt.show()


    #plt.savefig(sys.argv[1]+'.jpg')

image_name = sys.argv[1]

path =  '/home/xiaoxiaoyu/codes/imgs_cls/similar_event.json'
image_path = '/data/majian/cme/total_incident'
find_topk_similar_images(image_name,path,image_path)





