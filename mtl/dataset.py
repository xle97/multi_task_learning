import torch
import cv2
import numpy as np
from torchvision import transforms
from torchvision.datasets import VisionDataset
from PIL import Image

class MultiTaskDataset(VisionDataset):
    def __init__(self, path, label, is_train=False):
        self.paths = path
        self.labels = label
        self.is_train = is_train
        if is_train:
            self.tfms = transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])
        else:
            self.tfms = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])


    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        #dealing with the image
        img = Image.open(self.paths[idx]).convert('RGB')
        # img = cv2.imread(self.paths[idx], cv2.IMREAD_COLOR)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.tfms(img)

        #dealing with the labels
        labels = self.labels[idx]
        sexy = torch.tensor(int(labels[0]), dtype=torch.int64)
        flag = torch.tensor(int(labels[1]), dtype=torch.int64)
        violence = torch.tensor(int(labels[2]), dtype=torch.int64)
        
        return img.data, (sexy, flag, violence)

    def show(self,idx):
        x,y = self.__getitem__(idx)
        sexy, flag, violence = y
        stds = np.array([0.229, 0.224, 0.225])
        means = np.array([0.485, 0.456, 0.406])
        img = ((x.numpy().transpose((1,2,0))*stds + means)*255).astype(np.uint8)

        cv2.imshow("{} {} {}".format(sexy.item(), flag.item(), violence.item()), img)

class SeparateDataset(VisionDataset):
    def __init__(self, path, label, is_train=False):
        self.paths = path
        self.labels = label
        self.is_train = is_train
        if is_train:
            self.tfms = transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])
        else:
            self.tfms = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        #dealing with the image
        img = Image.open(self.paths[idx]).convert('RGB')
        img = self.tfms(img)

        #dealing with the labels
        labels = self.labels[idx]
        target = torch.tensor(int(labels), dtype=torch.int64)
        return img.data, target

    def show(self,idx):
        pass