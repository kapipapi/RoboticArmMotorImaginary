import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ModelWrapper import ModelWrapper


class CustomPad(nn.Module):
    def __init__(self, padding):
        super().__init__()
        self.padding = padding

    def forward(self, x):
        return F.pad(x, self.padding)


class EEGInception(ModelWrapper):
    def __init__(self, input_time=4000, fs=512, ncha=16, filters_per_branch=8,
                 scales_time=(500, 250, 125), dropout_rate=0.25,
                 activation=nn.ELU(inplace=True), n_classes=3):
        super(EEGInception, self).__init__()

        input_samples = int(input_time * fs / 1000)
        scales_samples = [int(s * fs / 1000) for s in scales_time]

        # ========================== BLOCK 1: INCEPTION ========================== #
        self.inception1 = nn.ModuleList([
            nn.Sequential(
                CustomPad((0, 0, scales_sample // 2 - 1, scales_sample // 2,)),
                nn.Conv2d(1, filters_per_branch, (scales_sample, 1)),
                nn.BatchNorm2d(filters_per_branch),
                activation,
                nn.Dropout(dropout_rate),
                nn.Conv2d(filters_per_branch, filters_per_branch * 2,
                          (1, ncha), bias=False, groups=filters_per_branch),  # DepthwiseConv2D
                nn.BatchNorm2d(filters_per_branch * 2),
                activation,
                nn.Dropout(dropout_rate),
            ) for scales_sample in scales_samples
        ])

        self.avg_pool1 = nn.AvgPool2d((4, 1))

        # ========================== BLOCK 2: INCEPTION ========================== #
        self.inception2 = nn.ModuleList([
            nn.Sequential(
                CustomPad((0, 0, scales_sample // 8 -
                           1, scales_sample // 8,)),
                nn.Conv2d(
                    len(scales_samples) * 2 * filters_per_branch,
                    filters_per_branch, (scales_sample // 4, 1),
                    bias=False
                ),
                nn.BatchNorm2d(filters_per_branch),
                activation,
                nn.Dropout(dropout_rate),
            ) for scales_sample in scales_samples
        ])

        self.avg_pool2 = nn.AvgPool2d((2, 1))

        # ============================ BLOCK 3: OUTPUT =========================== #
        self.output = nn.Sequential(

            CustomPad((0, 0, 4, 3)),
            nn.Conv2d(
                24, filters_per_branch * len(scales_samples) // 2, (8, 1),
                bias=False

            ),
            nn.BatchNorm2d(filters_per_branch * len(scales_samples) // 2),
            activation,
            nn.AvgPool2d((2, 1)),
            nn.Dropout(dropout_rate),

            CustomPad((0, 0, 2, 1)),
            nn.Conv2d(
                12, filters_per_branch * len(scales_samples) // 4, (4, 1),
                bias=False

            ),
            nn.BatchNorm2d(filters_per_branch * len(scales_samples) // 4),
            activation,
            # nn.Dropout(dropout_rate),
            nn.AvgPool2d((2, 1)),
            nn.Dropout(dropout_rate),
        )

        self.dense = nn.Sequential(
            # nn.Linear(4 * 1 * 6, n_classes), # to zmieniłem bo sie rozjeżdżało
            nn.Linear(384, n_classes),
            nn.Softmax(1)
        )

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = x.permute(0, 1, 3, 2)
        x = torch.cat([net(x) for net in self.inception1], 1)
        x = self.avg_pool1(x)
        x = torch.cat([net(x) for net in self.inception2], 1)
        x = self.avg_pool2(x)
        x = self.output(x)
        x = torch.flatten(x, 1)
        x = self.dense(x)
        return x

