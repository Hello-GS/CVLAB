import random
import warnings

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from PIL import ImageFile
from imgs_cls.models.model import get_gallery_model,Identity
from sklearn.model_selection import train_test_split
from utils.misc import *
from progress.bar import Bar
from imgs_cls.utils.reader import WeatherDataset, make_weights_for_balanced_classes
import pandas as pd
from sklearn.metrics.pairwise import  cosine_similarity
from imgs_cls.config import configs
import os
import pandas as pd
import numpy as np
from imgs_cls.utils.misc import get_files

'''
生成test和train数据集下的图片的特征向量，保存为query_CME.json和galllery_CME.json。
'''

# for train fp16
if configs.fp16:
    try:
        import apex
        from apex.parallel import DistributedDataParallel as DDP
        from apex.fp16_utils import *
        from apex import amp, optimizers
        from apex.multi_tensor_apply import multi_tensor_applier
    except ImportError:
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = configs.gpu_id


# set random seed
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(configs.seed)


# make dir for use
def makdir():
    if not os.path.exists(configs.checkpoints):
        os.makedirs(configs.checkpoints)
    if not os.path.exists(configs.log_dir):
        os.makedirs(configs.log_dir)
    if not os.path.exists(configs.submits):
        os.makedirs(configs.submits)


makdir()


def make_incident(query_image,query):
    length = len(query_image)
    # path0 = '/data/majian/cme/total_incident/0'
    # path1 = '/data/majian/cme/total_incident/1'

    path0 = '/disk/data/total_incident/0'
    path1 = '/disk/data/total_incident/1'
    dirs0 = os.listdir(path0)
    dirs1 = os.listdir(path1)

    for i in range(length):
        success = False

        for file0 in dirs0:
            if (os.path.exists(path0 + '/' + file0 + '/' + str(query_image[i]))):
                query[i] = file0
                success = True
                break
        if success == False:
            for file1 in dirs1:
                if (os.path.exists(path1 + '/' + file1 + '/' + str(query_image[i]))):
                    query[i] = file1
                    break
    return query

def main():
    configs.bs= 16
    # global best_acc
    # global best_loss
    # global best_f1
    num_of_1 = 0
    last_val_f1 = 0
    best_test_f1 = 0

    start_epoch = configs.start_epoch
    # set normalize configs for imagenet
    normalize_imgnet = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
    # normalize_imgnet = transforms.Normalize(mean=[0.50008379, 0.50008379, 0.50008379],
    #                                         std=[0.05762105, 0.05762105, 0.05762105])

    train_files = get_files(configs.dataset + "/train/", "train")

    # if step > 0:
    #     configs.lr = configs.lr * 0.1

    transform_train = transforms.Compose([
        # transforms.RandomCrop(),
        transforms.Resize(configs.input_size),
        # transforms.RandomResizedCrop(configs.input_size),
        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomVerticalFlip(p=0.5),
        # transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
        transforms.ToTensor(),
        # transforms.RandomErasing(),
        normalize_imgnet
    ])

    transform_val = transforms.Compose([
        transforms.Resize(configs.input_size),
        # transforms.CenterCrop(configs.input_size),
        transforms.ToTensor(),
        normalize_imgnet
    ])

    transform_test = transforms.Compose([
        transforms.Resize(configs.input_size),
        # transforms.CenterCrop(configs.input_size),
        transforms.ToTensor(),
        normalize_imgnet
    ])

    # Data loading code
    if configs.split_online:
        # use online random split dataset method
        total_files = get_files(configs.dataset, "train")
        train_files, val_files = train_test_split(total_files, test_size=0.1, stratify=total_files["label"])
        train_dataset = WeatherDataset(train_files, transform_train)
        val_dataset = WeatherDataset(val_files, transform_val)
    else:
        # use offline split dataset
        # train_files = get_files(configs.dataset + "/train/", "train")
        # print(train_files)
        test_files = get_files(configs.dataset + "/test/", "train")
        train_dataset = WeatherDataset(train_files, transform_train)
        test_dataset = WeatherDataset(test_files, transform_test)

        # 构造了三个dataset

        weights = make_weights_for_balanced_classes(train_files, configs.num_classes)
        weights = torch.DoubleTensor(weights)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, int(len(weights) * 1.0), replacement=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=configs.bs, sampler=sampler, shuffle=False,
        num_workers=configs.workers, pin_memory=True,
    )

    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=configs.bs, shuffle=True,
    #     num_workers=configs.workers, pin_memory=True,
    # )


    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=configs.bs, shuffle=False,
        num_workers=configs.workers, pin_memory=True
    )

    # get model
    model = get_gallery_model()


    model.cuda()


    if configs.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(configs.resume), 'Error: no checkpoint directory found!'
        configs.checkpoint = os.path.dirname(configs.resume)
        print(configs.checkpoint)
        checkpoint = torch.load(configs.resume)["state_dict"]

        trained_list = list(checkpoint.keys())

        #存储参数 dic
        dic_new = model.state_dict().copy()
        model_list = list(model.state_dict().keys())

        length = len(model_list)

        # why -2
        for i in range(length-2):
            dic_new[model_list[i]] = checkpoint[trained_list[i]]

        model.load_state_dict(dic_new)
        model.last_linear = Identity()

        if not os.path.exists('query_CME.json'):
            output_vector = torch.FloatTensor([])
            incident_list = []
            image_name_list = []
            #model.load_state_dict(checkpoint['state_dict'])
            #print(model)
            bar = Bar('generate query json:',max = len(test_loader))
            for batch_idx,(inputs,targets,incident_id_list,image_name) in enumerate(test_loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
                # compute output
                outputs = model(inputs)
                print(outputs.shape)

                #torch.cat()将两个tensor拼接 本质上还是outputs的叠加
                output_vector = torch.cat([output_vector,outputs.data.to("cpu")],dim =0)
                incident_list.extend(incident_id_list)
                image_name_list.extend(image_name)
                # 显示
                bar.suffix = '({batch}/{size})'.format(
                        batch = batch_idx +1,
                        size =len(test_loader))
                bar.next()
            output_list = output_vector.data.cpu().numpy().tolist() # 图片的特征向量
            #保存要查询的图片的路径， 输入仅仅是名字
            incident_list = make_incident(image_name_list,incident_list)


            df = pd.DataFrame({'query':output_list,  #特征向量
                               'query_incident':incident_list, #事件名字
                               'query_image':image_name_list}) #图片名字
            df.to_json('query_CME.json')


        if not os.path.exists('gallery_CME.json'):
            output_vector = torch.FloatTensor([])
            incident_list = []
            image_name_list = []
            # model.load_state_dict(checkpoint['state_dict'])
            # print(model)
            bar = Bar('generate gallery json:', max=len(train_loader))
            for batch_idx, (inputs, targets, incident_id_list, image_name) in enumerate(train_loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
                # compute output
                outputs = model(inputs)
                output_vector = torch.cat([output_vector, outputs.data.to("cpu")], dim=0)
                incident_list.extend(incident_id_list)
                image_name_list.extend(image_name)
                bar.suffix = '({batch}/{size})'.format(
                    batch=batch_idx + 1,
                    size=len(train_loader))
                bar.next()
            output_list = output_vector.data.cpu().numpy().tolist()
            incident_list = make_incident(image_name_list, incident_list)

            df = pd.DataFrame({'gallery': output_list,
                               'gallery_incident': incident_list,
                               'gallery_image': image_name_list})
            df.to_json('gallery_CME.json')

if __name__ == '__main__':
    main()