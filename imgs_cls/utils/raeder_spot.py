from torch.utils.data import Dataset
from PIL import Image
from astropy.io import fits
from numpy import uint8

class WeatherDataset(Dataset):
    # define dataset
    def __init__(self,label_list,transforms=None,mode="train"):
        super(WeatherDataset,self).__init__()
        self.label_list = label_list
        self.transforms = transforms
        self.mode = mode
        imgs = []
        if self.mode == "test":
            for index,row in label_list.iterrows():
                imgs.append((row["filename"]))
            self.imgs = imgs
        else:
            for index,row in label_list.iterrows():
                imgs.append((row["filename"],row["label"]))
            self.imgs = imgs
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self,index):
        if self.mode == "test":
            filename = self.imgs[index]
            img = fits.open(filename)
            img.verify('fix')
            img = img[1].data
            img = Image.fromarray(uint8(img))
            img = img.convert('RGB')
            # img = Image.open(filename).convert('RGB')
            img = self.transforms(img)
            return img,filename
        else:
            filename,label = self.imgs[index]
            img = fits.open(filename)
            img.verify('fix')
            img = img[1].data
            img = Image.fromarray(uint8(img))
            img = img.convert('RGB')
            # img = Image.open(filename).convert('RGB')
            img = self.transforms(img)
            return img,label


def make_weights_for_balanced_classes(images, nclasses):
    # print(images['label'].iloc[0])
    count = [0] * nclasses
    for i in range(len(images)):
       count[images['label'].iloc[i]] += 1

    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])  # 默认
        # weight_per_class[i] = float(count[i]) / N # idn
    weight = [0] * len(images)

    for i in range(len(images)):
        weight[i] = weight_per_class[images['label'].iloc[i]]
    # for idx, val in enumerate(images):
    #     weight[idx] = weight_per_class[val[1]]
    return weight

