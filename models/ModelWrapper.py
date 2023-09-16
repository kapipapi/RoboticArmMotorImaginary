import torch
import torchmetrics
import seaborn as sn
import pandas as pd
import matplotlib

import matplotlib.pyplot as plt
import torch.nn.functional as F
import lightning.pytorch as pl
import numpy as np

from torch.nn.functional import cross_entropy, one_hot

from torch.optim import Adam

matplotlib.use('agg')


class ModelWrapper(pl.LightningModule):
    def __init__(self, n_classes):
        super().__init__()

        self.to(torch.float)

        self.n_classes = n_classes

        self.accuracy_train = torchmetrics.Precision(task="multiclass", average='macro', num_classes=n_classes)
        self.accuracy_test = torchmetrics.Precision(task="multiclass", average='macro', num_classes=n_classes)
        self.accuracy_val = torchmetrics.Precision(task="multiclass", average='macro', num_classes=n_classes)

        self.f1_train = torchmetrics.F1Score(task="multiclass", num_classes=n_classes)
        self.f1_test = torchmetrics.F1Score(task="multiclass", num_classes=n_classes)
        self.f1_val = torchmetrics.F1Score(task="multiclass", num_classes=n_classes)
        
        self.confmat = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=n_classes)
        self.roc = torchmetrics.ROC(task='multilabel', num_labels=n_classes)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        data, label_n = batch

        # convert to float
        data = data.to(torch.float)

        # get predictions
        output = self(data)

        # convert for loss calculation
        label = one_hot(label_n, num_classes=self.n_classes)

        # calculate loss
        loss = cross_entropy(label.to(torch.float32), output)
        self.log("train_loss", loss, on_step=True)

        # calculate accuracy
        self.accuracy_train.update(output, label_n)
        self.log('train_acc', self.accuracy_train, on_epoch=True)

        # calculate f1 score
        self.f1_train.update(output, label_n)
        self.log('train_f1', self.f1_train, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        data, label_n = batch

        # convert to float
        data = data.to(torch.float)

        # get predictions
        output = self(data)

        # convert for loss calculation
        label = one_hot(label_n, num_classes=self.n_classes)

        # calculate loss
        loss = cross_entropy(label.to(torch.float32), output)
        self.log("validation_loss", loss)

        # calculate accuracy
        self.accuracy_val.update(output, label_n)
        self.log('validation_acc', self.accuracy_val, on_epoch=True)

        # calculate f1 score
        self.f1_val.update(output, label_n)
        self.log('validation_f1', self.f1_val, on_epoch=True)

    def test_step(self, batch, batch_idx):
        data, label_n = batch

        # convert to float
        data = data.to(torch.float)

        # get predictions
        output = self(data)

        # convert for loss calculation
        label = one_hot(label_n, num_classes=self.n_classes)

        # calculate loss
        loss = cross_entropy(label.to(torch.float32), output)
        self.log("test_loss", loss)

        # calculate accuracy
        self.accuracy_test.update(output, label_n)
        self.log('test_acc', self.accuracy_test)

        # calculate f1 score
        self.f1_test.update(output, label_n)
        self.log('test_f1', self.f1_test)

        self.confmat.update(output, label_n)

    def on_test_epoch_end(self):
        fig, ax = self.confmat.plot()

        # Assuming you have a list of class names
        class_names = ["LEFT", "RIGHT", "RELAX", "FEET"][:self.n_classes]  # Modify this list as per your classes

        # Set the x and y axis labels
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)

        # Set x and y axis titles
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')

        # Set the title for the confusion matrix
        ax.set_title('Confusion Matrix')

        # Display the grid lines
        ax.grid(False)

        # Rotate the x tick labels for better visibility if needed
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")


        # add the confusion matrix to TensorBoard
        self.logger.experiment.add_figure("Confusion Matrix", fig, self.current_epoch)
