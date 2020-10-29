import os

# error_list = []
# with open("/home/majian/datasets/cme/cme_input_data.csv", "r") as f:
#     for line in f.readlines():
#         img = line.split(",")[0]
#         label = line.split(",")[1]
#         incident = line.split(",")[-1].strip()
#         if not os.path.exists(os.path.join("/home/majian/datasets/cme/total_incident", label, incident, img)):
#             print(img)
#             error_list.append(img)
#
#
# print(len(error_list))


repeat_list = []
import os
train_path = "/home/majian/datasets/cme/pytorch/train"
test_path = "/home/majian/datasets/cme/pytorch/test"
for label in os.listdir(train_path):
    for img in os.listdir(os.path.join(train_path, label)):
        if os.path.exists(os.path.join(test_path, label, img)):
            print(img)
            repeat_list.append(img)

print(len(repeat_list))