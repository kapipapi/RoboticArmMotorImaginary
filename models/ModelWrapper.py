import torch
import torchmetrics
import seaborn as sn
import pandas as pd
import matplotlib

import matplotlib.pyplot as plt
import torch.nn.functional as F
import lightning.pytorch as pl

from torch.nn.functional import cross_entropy, one_hot

from torch.optim import Adam

matplotlib.use('agg')


class ModelWrapper(pl.LightningModule):
    def __init__(self, n_classes):
        super().__init__()

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
        self.log('train_acc', self.accuracy_train)

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
        cm = self.confmat.compute().detach().cpu().numpy()

        import seaborn as sn
        import pandas as pd
        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt

        labels = ["LEFT", "RIGHT", "RELAX", "FEET"][:self.n_classes]

        fig, ax1 = plt.subplots(1)
        df_cm = pd.DataFrame(cm, index = labels,
                          columns = labels)
        sn.heatmap(df_cm, annot=True, ax=ax1)

        # add the confusion matrix to TensorBoard
        self.logger.experiment.add_figure("Confusion Matrix", fig, self.current_epoch)
