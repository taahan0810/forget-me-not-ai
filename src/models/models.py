import os
import torch
from torch import nn
import torch.nn.functional as F

class SNeurodCNN(nn.Module):
    def __init__(self):
        super(SNeurodCNN, self).__init__()

        self.conv1 = nn.Conv2d(1,32,3)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32,32,3)
        self.conv3 = nn.Conv2d(32,64,3)
        self.maxpool2 = nn.MaxPool2d(2)
        self.dense1 = nn.Linear(238144,500)
        self.dropout = nn.Dropout(0.5)
        self.dense2 = nn.Linear(500,2) # classifying AD and MCI

    def forward(self, x):

        x = self.maxpool1(nn.ReLU(self.conv1(x)))
        x = self.maxpool2(nn.ReLU(self.conv3(nn.ReLU(self.conv2(x)))))
        x = self.dropout(self.dense1(x))
        x = F.softmax(self.dense2(x))

        return x