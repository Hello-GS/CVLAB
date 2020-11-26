import pandas as pd
import numpy as np



def image_to_incident():

    print("---------------------transfrom gallery -------------------------------------")
    gallery = pd.read_json('gallery_CME.json')
    gallery_vector = gallery['gallery']
    gallery_incident_list = gallery['gallery_incident']
    length = len(gallery_incident_list) #多少个事件

    gallery_vector = np.array(list(gallery_vector), dtype=float)
    gallery_incident_id = np.array(list(gallery_incident_list)) #事件名字

    dic = {}
    for i in range(length):
        if gallery_incident_id[i] not in dic:
            dic[gallery_incident_id[i]] = [gallery_vector[i]]
        else:
            dic[gallery_incident_id[i]].append(gallery_vector[i])

    incident_id = []
    gallery_incident = []

    for incident, value in dic.items():
        incident_id.append(incident) #事件列表

        value = np.sum(np.array(value), axis=0) / len(value) #所有该事件里的图片的特征向量的平均，当作事件的特征向量
        gallery_incident.append(value)  #事件特征向量列表

    df = pd.DataFrame({'gallery_id': incident_id, 'gallery_incident': gallery_incident})
    df.to_json('gallery_incident.json')


    print("---------------------transfrom query -------------------------------------")
    query = pd.read_json('query_CME.json')
    query_vector = query['query']
    query_incident_list = query['query_incident']
    length = len(query_incident_list)

    query_vector = np.array(list(query_vector), dtype=float)
    query_incident_id = np.array(list(query_incident_list))

    dic = {}
    for i in range(length):
        if query_incident_id[i] not in dic:
            dic[query_incident_id[i]] = [query_vector[i]]
        else:
            dic[query_incident_id[i]].append(query_vector[i])

    incident_id = []
    query_incident = []

    for incident, value in dic.items():
        incident_id.append(incident)

        value = np.sum(np.array(value), axis=0) / len(value)
        query_incident.append(value)


    df = pd.DataFrame({'query_id': incident_id, 'query_incident': query_incident})
    df.to_json('query_incident.json')

image_to_incident()

