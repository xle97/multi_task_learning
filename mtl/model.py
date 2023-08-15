import torch
from torch import nn
import arch as model
from collections import OrderedDict


class MultiTaskModel(nn.Module):
    def __init__(self, arch):
        super(MultiTaskModel,self).__init__()
        self.encoder = model.__dict__[arch]()
        self.n_features =  self.encoder.fc_infeatures
        self.avgpool =nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Sequential(OrderedDict([
            ('bn1', nn.BatchNorm1d(self.n_features, momentum=0.01)),
            ('linear', nn.Linear(self.n_features,self.n_features)),
            ('relu1', nn.ReLU()),
            ('bn2', nn.BatchNorm1d(self.n_features, momentum=0.01)),
            ('final', nn.Linear(self.n_features, 5))]))
        self.fc2 = nn.Sequential(OrderedDict([
            ('bn1', nn.BatchNorm1d(self.n_features, momentum=0.01)),
            ('linear', nn.Linear(self.n_features,self.n_features)),
            ('relu1', nn.ReLU()),
            ('bn2', nn.BatchNorm1d(self.n_features, momentum=0.01)),
            ('final', nn.Linear(self.n_features, 7))]))
        self.fc3 = nn.Sequential(OrderedDict([
            ('bn1', nn.BatchNorm1d(self.n_features, momentum=0.01)),
            ('linear', nn.Linear(self.n_features,self.n_features)),
            ('relu1', nn.ReLU()),
            ('bn2', nn.BatchNorm1d(self.n_features, momentum=0.01)),
            ('final', nn.Linear(self.n_features, 9))]))


    def forward(self,x):

        x = self.encoder(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        sexy = self.fc1(x)
        flag = self.fc2(x)
        violence = self.fc3(x)

        return sexy, flag, violence