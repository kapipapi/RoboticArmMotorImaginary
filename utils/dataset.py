import os
import numpy as np
from torch.utils.data import Dataset


def load_paths(root_dir, n_classes):
    paths = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".npy"):
                data = np.load(os.path.join(root, file), allow_pickle=True).item()
                label = data["impulse_index"] - 1
                if label >= n_classes:
                    continue
                else:
                    paths.append(os.path.join(root, file))
    return paths


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


class EEGDataset(Dataset):
    def __init__(self, root_dir, transform=None, n_classes=3):
        self.transform = transform
        self.n_classes = n_classes
        self.paths = load_paths(root_dir, self.n_classes)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        data = np.load(self.paths[index], allow_pickle=True).item()

        signal = data["impulse_signal"]
        label = data["impulse_index"] - 1
        sample_rate = int(data["sample_rate"])

        # if label >= self.n_classes:
        #     return

        # shorten slice to even timing (4 second sample)
        end = sample_rate * 4

        # make sure signal is correct length
        if signal.shape[1] >= end:
            signal = signal[:, :end]
        else:
            new_signal = []
            for i, s in enumerate(signal):
                new_signal.append(np.pad(s, (0, end - signal.shape[1]), 'mean'))
            signal = np.array(new_signal)

        assert signal.shape[1] == end

        if self.transform:
            signal = self.transform(signal)

        return np.array(signal[:16]), label
