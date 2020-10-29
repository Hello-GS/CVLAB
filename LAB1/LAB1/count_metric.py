from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


# label_file = "./data/cme_by_incident_cropped_split_label_list.csv"
# pred_file = "./submits/resnet50-model-sgd_bs64_lr_1e-2_CrossEntropy_resize512x512_liuzhumean_cropedfuxian_baseline.txt"


# label_file = "./data/cme_by_incident_cropped_split_label_list.csv"
# pred_file = "./submits/resnet34-model-sgd_bs64_lr_1e-2_CrossEntropy_resize512x512_mymean_cropfuxian_baseline.txt"


label_file = "./data/cme_by_incident_cropped_split_label_list.csv"
pred_file = "./submits/inceptionv4-model-sgd_bs64_lr_1e-2_CrossEntropy_resize512x512_mymean_cropfuxian_baseline.txt"


# label_file = "./data/ori_total_incident_split_label_list.csv"
# pred_file = "./submits/resnet50-model-sgd_bs64_lr_1e-2_CrossEntropy_resize512x512_mymean_oridatafuxian_baseline.txt"


# label_file = "./data/total_incident_diff_to_frame_label_list.csv"
# pred_file = "./submits/resnet34-model-sgd_bs64_lr_1e-2_CrossEntropy_resize512x512_mymean_oridifffuxian_baseline.txt"



preds = []
with open(pred_file, "r") as f:
    for i,line in enumerate(f.readlines()):
        # print(line)
        label = int(line.strip().split(" ")[-1])
        # print(label)
        preds.append(int(label))


targets = []
with open(label_file, "r") as f:
    for i,line in enumerate(f.readlines()):
        # print(line)
        if i > 0:
            label = int(line.strip().split(",")[-1])
            # print(label)
            targets.append(label)

# print(targets)
# print("len(targets):", len(targets))
# print("len(preds):", len(preds))

incident_metric = classification_report(targets, preds, target_names=['0', '1'], output_dict=True)
print(incident_metric)
acc = accuracy_score(targets, preds, normalize=False)
print(acc/len(targets))
