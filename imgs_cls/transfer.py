# transfer image to video classification and get some metrics
#%%
import pandas as pd
import os
import math

CMEdata_dir = "/data/majian/cme/total_incident"
img_result_file = "./data/cme_by_incident_cropped_split_pred_list.csv"

df = pd.read_csv(img_result_file) # delim_whitespace=True
img_to_event = {}
event_to_cls = {}
event_to_numimg = {}
for clss in os.listdir(CMEdata_dir):
    for event in os.listdir(CMEdata_dir+"/"+clss):
        assert len(os.listdir(CMEdata_dir+"/"+clss+"/"+event)) != 0
        if len(os.listdir(CMEdata_dir+"/"+clss+"/"+event)) == 0:
            print(event)
        event_to_numimg[event] = len(os.listdir(CMEdata_dir+"/"+clss+"/"+event))
        for img in os.listdir(CMEdata_dir+"/"+clss+"/"+event):
            img_to_event[img] = event
            event_to_cls[event] = clss
#%%
events = [img_to_event[row.iloc[0][7:]] for _,row in df.iterrows()] # str[7:]
df['events'] = events
df['GT'] = [event_to_cls[i] for i in events]
df.to_csv("test.csv", index=False)
event_to_pred = df['type'].to_list()
event_to_predcls = {} # pred pos nums of each event 

for i in range(len(events)):
    if events[i] in event_to_predcls.keys():
        event_to_predcls[events[i]] += event_to_pred[i]
    else:
        event_to_predcls[events[i]] = event_to_pred[i]

labels = [int(event_to_cls[i]) for i in event_to_predcls.keys()]

# different strategies
AGG = True

if AGG:
    print("============Aggressive Strategy: hit by one===============")
    preds=[int(event_to_predcls[i] >= 1) for i in event_to_predcls.keys()] 
else:
    print("============Conservative Strategy: hit by all===============")
    preds=[int(event_to_predcls[i] >= math.ceil(event_to_numimg[i] / 2)) for i in event_to_predcls.keys()] 
# print([math.ceil(event_to_numimg[i] / 2) for i in event_to_predcls.keys()])
# or more conservative - all in
# preds=[int(event_to_predcls[i] == event_to_numimg[i]) for i in event_to_predcls.keys()]

assert len(labels) == len(preds)

from sklearn.metrics import precision_recall_fscore_support
print(precision_recall_fscore_support(labels, preds, average="binary"))

# %%
