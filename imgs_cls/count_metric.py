from sklearn.metrics import classification_report
from utils.misc import *
import pandas as pd
import  numpy as np

from imgs_cls.utils.misc import count_event_metirc

label_file = "./data/cme_by_incident_cropped_split_label_list.csv"
pred_file = "./data/cme_by_incident_cropped_split_pred_list.csv"
data_path = '/data/majian/cme/total_incident'



df_label= pd.read_csv(label_file)
df_pred = pd.read_csv(pred_file)
preds = df_pred['type']
targets = df_label['type']
preds = np.array(preds,dtype=int)
targets= np.array(targets,dtype=int)
# print(targets)
# print("len(targets):", len(targets))
# print("len(preds):", len(preds))

incident_metric = classification_report(targets, preds, target_names=['0', '1'], output_dict=True)
print('--------------count image metircs----------------------------')
print(incident_metric['1'])

count_event_metirc(pred_file,data_path)

