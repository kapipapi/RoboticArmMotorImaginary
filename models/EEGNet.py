import torch
import pytorch_lightning as pl
import torchmetrics
import torch.functional as F
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

from torch import nn
from torch.optim import Adam

plt.use('agg')


class EEGNet(pl.LightningModule):
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

        self.accuracy_train = torchmetrics.Precision(task="multiclass", average='macro', num_classes=3)
        self.accuracy_test = torchmetrics.Precision(task="multiclass", average='macro', num_classes=3)
        self.accuracy_val = torchmetrics.Precision(task="multiclass", average='macro', num_classes=3)

        self.f1 = torchmetrics.F1Score(task="multiclass", num_classes=3)
        self.confmat = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=3)
        self.roc = torchmetrics.ROC(task='multilabel', num_labels=3)

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

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        data, label = batch

        # get predictions
        output = self(data)

        # convert for loss calculation
        label = F.one_hot(label, num_classes=3)

        # calculate loss
        loss = F.binary_cross_entropy(output, label.to(torch.float32))
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
        label = F.one_hot(label, num_classes=3)

        # calculate loss
        loss = F.binary_cross_entropy(output, label.to(torch.float32))
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
        label = F.one_hot(label, num_classes=3)

        # calculate loss
        loss = F.binary_cross_entropy(output, label.to(torch.float32))
        self.log("validation_loss", loss)

        # calculate accuracy
        self.accuracy_val.update(output, label)
        self.log('validation_acc', self.accuracy_val)

        # calculate f1 score
        self.f1.update(output, label)
        self.log('train_f1', self.f1, on_epoch=True)

    def on_train_epoch_end(self):
        cm = self.confmat.compute().detach().cpu().numpy()

        fig, ax1 = plt.subplots(1)
        df_cm = pd.DataFrame(cm, index=[i for i in "012"],
                             columns=[i for i in "012"])
        sn.heatmap(df_cm, annot=True, ax=ax1)

        # add the confusion matrix to TensorBoard
        self.logger.experiment.add_figure("Confusion Matrix", fig, self.current_epoch)
