import torch
import torchmetrics
import torch.nn.functional as F
import pytorch_lightning as pl

from torch import nn
from torch.optim import Adam


class DeepConvNet(pl.LightningModule):
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

        self.accuracy_train = torchmetrics.Precision(task="multiclass", average='macro', num_classes=3)
        self.accuracy_test = torchmetrics.Precision(task="multiclass", average='macro', num_classes=3)
        self.accuracy_val = torchmetrics.Precision(task="multiclass", average='macro', num_classes=3)

        self.f1 = torchmetrics.F1Score(task="multiclass", num_classes=3)
        self.confmat = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=3)
        self.roc = torchmetrics.ROC(task='multilabel', num_labels=3)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        out = self.model(x)
        out = F.softmax(out, dim=-1)
        return out

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
