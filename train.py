import argparse
import pytorch_lightning as pl
from torch import manual_seed
from lightning.pytorch.loggers import TensorBoardLogger
from dataset import EEGDataset
from models.DeepConvNet import DeepConvNet
from models.EEGNet import EEGNet
from models.Transformer import Transformer
from utils.preprocessing import EEGDataProcessor
from torchvision.transforms import Compose
from torch.utils.data import random_split
from torch.utils.data import DataLoader


# TODO: Lightning DataModule + refactor
# TODO: K-fold shuffle
class EEGInception:
    pass


def train():
    manual_seed(42)
    pp = EEGDataProcessor()

    transforms = Compose([
        pp.correct_offset,
        pp.amplitude_conversion,
        pp.filter,
        pp.downsample,
        pp.normalize,
        pp.natural_logarithm,
    ])

    dataset = EEGDataset("./dataset/kuba", transforms)
    train_set, test_set, validation_set = random_split(dataset, [0.7, 0.2, 0.1])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=2, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=2)
    val_loader = DataLoader(validation_set, batch_size=args.batch_size, num_workers=2)

    models = {
        "EEGNet": EEGNet(),
        "EEGInception": EEGInception(),
        "DeepConvNet": DeepConvNet(),
        "Transformer": Transformer(),
    }

    logger = TensorBoardLogger("tb_logs", name=f"{args.model}_run")

    trainer = pl.Trainer(max_epochs=args.epochs, logger=logger)
    trainer.fit(models[args.model], train_loader, val_loader)
    trainer.test(models[args.model], test_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default="Transformer", help="Model to train")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size")
    parser.add_argument("--epochs", default=100, help="Number of Epochs")

    args = parser.parse_args()

    train()
