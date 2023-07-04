import torchmetrics
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from torch.optim import Adam
from torch.nn.functional import binary_cross_entropy, one_hot


class CustomPad(nn.Module):
    def __init__(self, padding):
        super().__init__()
        self.padding = padding

    def forward(self, x):
        return F.pad(x, self.padding)


class Inception(pl.LightningModule):
    def __init__(self, input_time=1000, fs=128, ncha=8, filters_per_branch=8,
                 scales_time=(500, 250, 125), dropout_rate=0.25,
                 activation=nn.ELU(inplace=True), n_classes=2):
        super(Inception, self).__init__()

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

        self.accuracy_train = torchmetrics.Precision(task="multiclass", average='macro', num_classes=3)
        self.accuracy_test = torchmetrics.Precision(task="multiclass", average='macro', num_classes=3)
        self.accuracy_val = torchmetrics.Precision(task="multiclass", average='macro', num_classes=3)

        self.f1 = torchmetrics.F1Score(task="multiclass", num_classes=3)
        self.confmat = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=3)
        self.roc = torchmetrics.ROC(task='multilabel', num_labels=3)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = x.permute((0, 1, 3, 2))
        x = torch.cat([net(x) for net in self.inception1], 1)  # concat
        x = self.avg_pool1(x)
        x = torch.cat([net(x) for net in self.inception2], 1)
        x = self.avg_pool2(x)
        x = self.output(x)
        x = torch.flatten(x, 1)
        x = self.dense(x)
        return x

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        data, label = batch

        # get predictions
        output = self(data)

        # convert for loss calculation
        label = one_hot(label, num_classes=3)

        # calculate loss
        loss = binary_cross_entropy(output, label.to(torch.float32))
        self.log("train_loss", loss)

        # calculate accuracy
        self.accuracy_train.update(output, label)
        self.log('train_acc', self.accuracy_train, on_epoch=True, on_step=True)

        # calculate f1 score
        self.f1.update(output, label)
        self.log('train_f1', self.f1)

        self.confmat.update(output, label)

        return loss

    def test_step(self, batch, batch_idx):
        data, label = batch

        # get predictions
        output = self(data)

        # convert for loss calculation
        label = one_hot(label, num_classes=3)

        # calculate loss
        loss = binary_cross_entropy(output, label.to(torch.float32))
        self.log("test_loss", loss)

        # calculate accuracy
        self.accuracy_test.update(output, label)
        self.log('test_acc', self.accuracy_test)

        # calculate f1 score
        self.f1.update(output, label)
        self.log('train_f1', self.f1)

    def validation_step(self, batch, batch_idx):
        data, label = batch

        # get predictions
        output = self(data)

        # convert for loss calculation
        label = one_hot(label, num_classes=3)

        # calculate loss
        loss = binary_cross_entropy(output, label.to(torch.float32))
        self.log("validation_loss", loss)

        # calculate accuracy
        self.accuracy_val.update(output, label)
        self.log('validation_acc', self.accuracy_val)

        # calculate f1 score
        self.f1.update(output, label)
        self.log('train_f1', self.f1, on_epoch=True)

    def on_train_epoch_end(self):
        cm = self.confmat.compute().detach().cpu().numpy()

        import seaborn as sn
        import pandas as pd
        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt

        fig, ax1 = plt.subplots(1)
        df_cm = pd.DataFrame(cm, index=[i for i in "012"],
                             columns=[i for i in "012"])
        sn.heatmap(df_cm, annot=True, ax=ax1)

        # add the confusion matrix to TensorBoard
        self.logger.experiment.add_figure("Confusion Matrix", fig, self.current_epoch)
