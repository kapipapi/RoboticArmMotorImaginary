import torch
import torch.nn.functional as F

from torch import nn
from models.ModelWrapper import ModelWrapper


class DeepConvNet(ModelWrapper):
    def __init__(self, n_classes=3):
        super(DeepConvNet, self).__init__(n_classes)
        self.model = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1,5), bias=False),
            nn.Conv2d(25, 25, kernel_size=(16,1),bias=False),
            nn.BatchNorm2d(25, eps=1e-05, momentum=0.1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1,3)),

            nn.Dropout(p=0.5),
            nn.Conv2d(25, 50, kernel_size=(1,5),bias=False),
            nn.BatchNorm2d(50, eps=1e-05, momentum=0.1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1,3)),

            nn.Dropout(p=0.5),
            nn.Conv2d(50, 100, kernel_size=(1,5),bias=False),
            nn.BatchNorm2d(100, eps=1e-05, momentum=0.1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1,3)),

            nn.Dropout(p=0.5),
            nn.Conv2d(100, 200, kernel_size=(1,5),bias=False),
            nn.BatchNorm2d(200, eps=1e-05, momentum=0.1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1,3)),

            nn.Flatten(),
            nn.Linear(2000, n_classes)
        )

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        out = self.model(x)
        out = F.softmax(out, dim=-1)
        return out
