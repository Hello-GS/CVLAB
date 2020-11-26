import os
import torch
import shutil
import pandas as pd
from .optimizers import *
from imgs_cls.config import configs
from torch import optim as optim_t
from tqdm import tqdm
from glob import glob
from itertools import chain
import numpy as np
from sklearn.metrics import classification_report

def get_optimizer(model):
    if configs.optim == "adam":
        return optim_t.Adam(model.parameters(),
                            configs.lr,
                            betas=(configs.beta1,configs.beta2),
                            weight_decay=configs.wd)
    elif configs.optim == "radam":
        return RAdam(model.parameters(),
                    configs.lr,
                    betas=(configs.beta1,configs.beta2),
                    weight_decay=configs.wd)
    elif configs.optim == "ranger":
        return Ranger(model.parameters(),
                      lr = configs.lr,
                      betas=(configs.beta1,configs.beta2),
                      weight_decay=configs.wd)
    elif configs.optim == "over9000":
        return Over9000(model.parameters(),
                        lr = configs.lr,
                        betas=(configs.beta1,configs.beta2),
                        weight_decay=configs.wd)
    elif configs.optim == "ralamb":
        return Ralamb(model.parameters(),
                      lr = configs.lr,
                      betas=(configs.beta1,configs.beta2),
                      weight_decay=configs.wd)
    elif configs.optim == "sgd":
        return optim_t.SGD(model.parameters(),
                        lr = configs.lr,
                        momentum=configs.mom,
                        weight_decay=configs.wd)
    else:
        print("%s  optimizer will be add later"%configs.optim)

def save_checkpoint(state,is_best,is_best_loss, is_best_f1, is_best_test_f1):
    filename = configs.checkpoints + os.sep + configs.model_name + "-checkpoint.pth.tar"
    torch.save(state, filename)
    if is_best:
        message = filename.replace("-checkpoint.pth.tar", "-best_model.pth.tar")
        shutil.copyfile(filename, message)
    if is_best_loss:
        message = filename.replace("-checkpoint.pth.tar", "-best_loss.pth.tar")
        shutil.copyfile(filename, message)

    if is_best_f1:
        message = filename.replace("-checkpoint.pth.tar", "-best_f1.pth.tar")
        shutil.copyfile(filename, message)
    
    if is_best_test_f1:
        message = filename.replace("-checkpoint.pth.tar", "-best_test_f1.pth.tar")
        shutil.copyfile(filename, message)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def get_files(root,mode):
    if mode == "test":
        files = []
        for img in os.listdir(root):
            files.append(root + img)
        # files = pd.DataFrame({"filename":files})
        files = pd.DataFrame({"FileName":files})
        return files
    else:
        all_data_path, labels,incident_ids,image_name = [], [],[],[]
        image_folders = list(map(lambda x: root + x, os.listdir(root)))
        all_images = list(chain.from_iterable(list(map(lambda x: glob(x + "/*"), image_folders))))
        print("loading train dataset")
        for file in tqdm(all_images):
            all_data_path.append(file)
            labels.append(int(file.split(os.sep)[-2]))
            incident_ids.append(file.split(os.sep)[-1].split('_')[0])
            image_name.append(file.split(os.sep)[-1])
        # all_files = pd.DataFrame({"filename": all_data_path, "label": labels})\
        all_files = pd.DataFrame({"FileName": all_data_path, "label": labels,'incident_ids':incident_ids,
                                  "image_name":image_name})
        return all_files
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lrs = [5e-4, 1e-4, 1e-5, 1e-6]
    if epoch<=10:
        lr = lrs[0]
    elif epoch>10 and epoch<=16:
        lr = lrs[1]
    elif epoch>16 and epoch<=22:
        lr = lrs[2]
    else:
        lr = lrs[-1]
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def evaluation(pred, label, topk=(1,)):
    """
    get the measurement according to https://tianchi.aliyun.com/competition/entrance/531804/information
    pred : array (n, )
    label : array(n, )

    -----------
    return
    Four measurements in terms of the importance
    """
    maxk = max(topk)
    batch_size = label.size(0)

    _, pred = pred.topk(maxk, 1, True, True)
    pred = pred.t()
    # correct = pred.eq(label.view(1, -1).expand_as(pred))
    pred = pred.view(1, -1).squeeze()

    H = M = F = CN = 0

    for i in range(len(pred)):
        if pred[i] == 1 and label[i] == 1:
            H += 1
        elif pred[i] == 1 and label[i] == 0:
            F += 1
        elif pred[i] == 0 and label[i] == 1:
            M += 1
        elif pred[i] == 0 and label[i] == 0:
            CN += 1
    if (H + M) == 0:
        rec = 0
    else:
        rec = H / (H + M)
    if (H + F) == 0:
        prec = 0
    else:
        prec = H / (H + F)

    if (prec + rec) == 0:
        f1 = 0
    else:
        f1 = 2 * prec * rec / (prec + rec)

    if (H + F) == 0:
        far = 0
    else:
        far = F / (H + F)
    PC = (H + CN) / (H + M + F + CN)

    return f1 * 100, rec * 100, far * 100, PC * 100, prec * 100


def evaluation_new(pred, label, H, F, M, CN, topk=(1,)):
    """
    get the measurement according to https://tianchi.aliyun.com/competition/entrance/531804/information
    pred : array (n, )
    label : array(n, )

    -----------
    return
    Four measurements in terms of the importance
    """
    maxk = max(topk)
    batch_size = label.size(0)

    _, pred = pred.topk(maxk, 1, True, True)
    pred = pred.t()
    # correct = pred.eq(label.view(1, -1).expand_as(pred))
    pred = pred.view(1, -1).squeeze()

    for i in range(len(pred)):
        if pred[i] == 1 and label[i] == 1:
            H += 1
        elif pred[i] == 1 and label[i] == 0:
            F += 1
        elif pred[i] == 0 and label[i] == 1:
            M += 1
        elif pred[i] == 0 and label[i] == 0:
            CN += 1
    return H, F, M, CN


def cal_metric(H, F, M, CN):
    if (H + M) == 0:
        rec = 0
    else:
        rec = H / (H + M)
    if (H + F) == 0:
        prec = 0
    else:
        prec = H / (H + F)

    if (prec + rec) == 0:
        f1 = 0
    else:
        f1 = 2 * prec * rec / (prec + rec)

    if (H + F) == 0:
        far = 0
    else:
        far = F / (H + F)
    PC = (H + CN) / (H + M + F + CN)

    return f1 * 100, rec * 100, far * 100, PC * 100, prec * 100

def add_incident(pred,label,incident_id,dic_incident_label,dic_incident_pred, topk=(1,)):
    maxk = max(topk)

    _,pred = pred.topk(maxk,1,True,True)
    pred = pred.t()
    pred = pred.view(1,-1).squeeze()

    for i in range(len(pred)):
        if incident_id not in dic_incident_label:
            dic_incident_label[incident_id[i]] = [label[i]]
            dic_incident_pred[incident_id[i]] = [pred[i]]
        else:
            dic_incident_label[incident_id[i]].append(label[i])
            dic_incident_pred[incident_id[i]].append(pred[i])

def evaluate_incident(dic_incident_label,dic_incident_pred):
    H = M = F = CN =0
    for key,value in dic_incident_pred.items():
        pred_label  = np.sum(value)/np.float(len(value))
        ground_value = dic_incident_label[key]
        ground_label = np.sum(ground_value)/np.float(len(value))
        if pred_label>0:
            pred_label =1
        else:
            pred_label = 0
        if ground_label >0:
            ground_label =1
        else:
            ground_label = 0
        if pred_label == 1 and ground_label ==1:
            H+=1
        elif pred_label == 1 and ground_label == 0:
            F +=1
        elif pred_label == 0 and ground_label == 1:
            M +=1
        elif pred_label == 0 and ground_label == 0:
            CN += 1
    return H, M, F, CN
def get_incident_pred_lable(dic_incident_label,dic_incident_pred):
    all_incident_label = []
    all_incident_pred= []


    for key,value in dic_incident_pred.items():

        pred_label  = np.sum(value)/np.float(len(value))
        ground_value = dic_incident_label[key]
        ground_label = np.sum(ground_value)/np.float(len(ground_value))
        if pred_label>0:
            pred_label =1
        else:
            pred_label = 0
        if ground_label >0:
            ground_label =1
        else:
            ground_label = 0

        all_incident_label.append(ground_label)
        all_incident_pred.append(pred_label)
    return all_incident_pred,all_incident_label

def count_event_metirc(img_result_file,CMEdata_dir):
    df = pd.read_csv(img_result_file)  # delim_whitespace=True
    img_to_event = {}
    event_to_cls = {}
    event_to_numimg = {}
    for clss in os.listdir(CMEdata_dir):
        for event in os.listdir(CMEdata_dir + "/" + clss):
            assert len(os.listdir(CMEdata_dir + "/" + clss + "/" + event)) != 0
            if len(os.listdir(CMEdata_dir + "/" + clss + "/" + event)) == 0:
                print(event)
            event_to_numimg[event] = len(os.listdir(CMEdata_dir + "/" + clss + "/" + event))
            for img in os.listdir(CMEdata_dir + "/" + clss + "/" + event):
                img_to_event[img] = event
                event_to_cls[event] = clss
    # %%
    events = [img_to_event[row.iloc[0][7:]] for _, row in df.iterrows()]  # str[7:]
    df['events'] = events
    df['GT'] = [event_to_cls[i] for i in events]
    df.to_csv("test.csv", index=False)
    event_to_pred = df['type'].to_list()
    event_to_predcls = {}  # pred pos nums of each event

    for i in range(len(events)):
        if events[i] in event_to_predcls.keys():
            event_to_predcls[events[i]] += event_to_pred[i]
        else:
            event_to_predcls[events[i]] = event_to_pred[i]

    labels = [int(event_to_cls[i]) for i in event_to_predcls.keys()]

    # different strategies
    AGG = True

    if AGG:
        #print("============Aggressive Strategy: hit by one===============")
        preds = [int(event_to_predcls[i] >= 1) for i in event_to_predcls.keys()]
    else:
        #print("============Conservative Strategy: hit by all===============")
        preds = [int(event_to_predcls[i] >= math.ceil(event_to_numimg[i] / 2)) for i in event_to_predcls.keys()]
        # print([math.ceil(event_to_numimg[i] / 2) for i in event_to_predcls.keys()])
    # or more conservative - all in
    # preds=[int(event_to_predcls[i] == event_to_numimg[i]) for i in event_to_predcls.keys()]

    assert len(labels) == len(preds)

    print('--------------count event metircs----------------------------')
    print(classification_report(labels, preds, target_names=['0', '1'], output_dict=True)['1'])


'''
def count_event_metirc(label_file, pred_file):
    df_label = pd.read_csv(label_file)
    df_pred = pd.read_csv(pred_file)
    image_name = df_label['FileName']
    label  = df_label['type']
    pred = df_pred['type']


    length = len(image_name)
    path0 = '/data/majian/cme/total_incident/0'
    path1 = '/data/majian/cme/total_incident/1'
    dirs0 = os.listdir(path0)
    dirs1 = os.listdir(path1)

    dic_label = {}
    dic_pred = {}
    for i in range(length):
        success = False

        for file0 in dirs0:
            if (os.path.exists(path0 + '/' + file0 + '/' + image_name[i].split('/')[-1])):
                image_name[i] = file0
                success = True
                break
        if success == False:
            for file1 in dirs1:
                if (os.path.exists(path1 + '/' + file1 + '/' + image_name[i].split('/')[-1])):
                    image_name[i] = file1
                    break
    for i in range(length):
        if image_name[i] not in dic_label:
            dic_label[image_name[i]] = [label[i]]
            dic_pred[image_name[i]] = [pred[i]]

        else:
            dic_label[image_name[i]].append(label[i])
            dic_pred[image_name[i]].append(pred[i])
    pred_list = []
    label_list =[]
    for key,value in dic_label.items():
        label_i = np.sum(value)/len(value)
        pred_i = np.sum(dic_pred[key])/len(dic_pred[key])
        if label_i >0:
            label_list.append(1)
        else:
            label_list.append(0)
        if pred_i>0:
            pred_list.append(1)
        else:
            pred_list.append(0)
    metric_result = classification_report(label_list, pred_list, target_names=['0', '1'], output_dict=True)
    print(metric_result)
    '''













