import torch
import torch.functional as F
import matplotlib.pyplot as plt
import matplotlib

from torch import nn
from models.ModelWrapper import ModelWrapper

matplotlib.use('agg')


class EEGNet(ModelWrapper):
    def __init__(self):
        super(EEGNet, self).__init__()
        self.T = 512

        # Layer 1
        self.conv1 = nn.Conv2d(1, 16, (1, 16), padding=0)
        self.batchnorm1 = nn.BatchNorm2d(16, False)

        # Layer 2
        self.padding1 = nn.ZeroPad2d((16, 17, 0, 1))
        self.conv2 = nn.Conv2d(1, 4, (2, 32))
        self.batchnorm2 = nn.BatchNorm2d(4, False)
        self.pooling2 = nn.MaxPool2d(2, 4)

        # Layer 3
        self.padding2 = nn.ZeroPad2d((2, 1, 4, 3))
        self.conv3 = nn.Conv2d(4, 4, (8, 4))
        self.batchnorm3 = nn.BatchNorm2d(4, False)
        self.pooling3 = nn.MaxPool2d((2, 4))

        # FC Layer
        # NOTE: This dimension will depend on the number of timestamps per sample in your data.
        # I have 120 timepoints. 
        self.fc1 = nn.Linear(4 * 2 * 32, 3)

    def forward(self, x):
        # Layer 1
        x = x.permute(0, 2, 1)
        x = torch.unsqueeze(x, 1)
        x = F.elu(self.conv1(x))
        x = self.batchnorm1(x)
        x = F.dropout(x, 0.25)
        x = x.permute(0, 3, 1, 2)

        # Layer 2
        x = self.padding1(x)
        x = F.elu(self.conv2(x))
        x = self.batchnorm2(x)
        x = F.dropout(x, 0.25)
        x = self.pooling2(x)

        # Layer 3
        x = self.padding2(x)
        x = F.elu(self.conv3(x))
        x = self.batchnorm3(x)
        x = F.dropout(x, 0.25)
        x = self.pooling3(x)

        # FC Layer
        x = x.reshape(-1, 4 * 2 * 32)
        x = F.softmax(self.fc1(x), dim=-1)

        return x
