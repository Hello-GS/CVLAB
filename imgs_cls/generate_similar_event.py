from utils.misc import *
import pandas as pd
from sklearn.metrics.pairwise import  cosine_similarity
from matplotlib import pyplot as plt
import sys
import numpy as np
def generate_similar_events(query_path,gallery_path):
    df = pd.read_json(gallery_path) # train->
    gallery_id = df['gallery_id']
    gallery_vector = df['gallery_incident']

    df_query_incident = pd.read_json(query_path) #test->
    query_id = df_query_incident['query_id']
    query_vector = df_query_incident['query_incident']



    gallery_vector = np.array(list(gallery_vector), dtype=float)
    gallery_id = np.array(list(gallery_id))

    query_vector = np.array(list(query_vector), dtype=float)
    query_id = list(query_id)

    length = len(query_id)
    similar_events = []
    for i in range(length):
        id = query_id[i]
        vector = query_vector[i]
        similarity = cosine_similarity([vector], gallery_vector) #判断相似性标准 返回的是[[]]
        similarity = similarity.flatten() # 变成一维
        arg_index = np.argsort(similarity)
        arg_index = arg_index[::-1] # 从大到小  arg_index[-5:]
        similarity_incident = gallery_id[list(arg_index[0:5])]
        similar_events.append(similarity_incident)

    df_new = pd.DataFrame({'query event':query_id,'similar events':similar_events})
    df_new.to_csv('similar_event.csv')
    df_new.to_json('similar_event.json')

query_path = '/home/xiaoxiaoyu/codes/imgs_cls/query_incident.json'
gallery_path =  '/home/xiaoxiaoyu/codes/imgs_cls/gallery_incident.json'
generate_similar_events(query_path,gallery_path)
