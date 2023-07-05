from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold

from dataset import EEGDataset
from utils.preprocessing import EEGDataProcessor
from torchvision.transforms import Compose


class KFoldDataModule(LightningDataModule):
    def __init__(
            self,
            data_dir: str = "./dataset/kuba",
            k: int = 1,  # fold number
            split_seed: int = 12345,  # split needs to be always the same for correct cross validation
            num_splits: int = 10,
            batch_size: int = 32,
            num_workers: int = 0,
            pin_memory: bool = False
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        # num_splits = 10 means our dataset will be split to 10 parts
        # so we train on 90% of the data and validate on 10%
        assert 1 <= self.k <= self.num_splits, "incorrect fold number"

        # data transformations
        pp = EEGDataProcessor()
        pp.DOWNSAMPLED_FREQ = 512
        self.transforms = Compose([
            pp.correct_offset,
            pp.filter,
            pp.downsample,
            pp.normalize,
        ])

        self.data_train: Dataset = None
        self.data_val: Dataset = None

    @property
    def num_classes(self) -> int:
        return 3

    def setup(self, stage=None):
        if not self.data_train and not self.data_val:
            dataset_full = EEGDataset(self.hparams.data_dir, self.transforms)

            # choose fold to train on
            kf = KFold(n_splits=self.hparams.num_splits, shuffle=True, random_state=self.hparams.split_seed)
            all_splits = [k for k in kf.split(dataset_full)]
            train_indexes, val_indexes = all_splits[self.hparams.k]
            train_indexes, val_indexes = train_indexes.tolist(), val_indexes.tolist()

            self.data_train, self.data_val = dataset_full[train_indexes], dataset_full[val_indexes]

    def train_dataloader(self):
        return DataLoader(dataset=self.data_train, batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          pin_memory=self.hparams.pin_memory, shuffle=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.data_val, batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          pin_memory=self.hparams.pin_memory)
