import torch
import torch.nn.functional as F

from torch import nn
from models.ModelWrapper import ModelWrapper


class DeepConvNet(ModelWrapper):
    def __init__(self, n_output):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1, 5), bias=False),
            nn.Conv2d(25, 25, kernel_size=(16, 1), bias=False),
            nn.BatchNorm2d(25, eps=1e-05, momentum=0.1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.47),

            nn.Conv2d(25, 50, kernel_size=(1, 5), bias=False),
            nn.BatchNorm2d(50, eps=1e-05, momentum=0.1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.47),

            nn.Conv2d(50, 100, kernel_size=(1, 5), bias=False),
            nn.BatchNorm2d(100, eps=1e-05, momentum=0.1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.47),

            nn.Conv2d(100, 200, kernel_size=(1, 5), bias=False),
            nn.BatchNorm2d(200, eps=1e-05, momentum=0.1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.47),

            nn.Flatten(),
            nn.Linear(5600, n_output, bias=True)
        )

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        out = self.model(x)
        out = F.softmax(out, dim=-1)
        return out
