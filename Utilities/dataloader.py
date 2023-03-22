import os

import numpy as np
from torch.utils.data import Dataset


def load_paths(root_dir):
    paths = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".npy"):
                paths.append(os.path.join(root, file))

    return paths


class EEGDataLoader(Dataset):

    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.paths = load_paths(root_dir)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        data = np.load(self.paths[index], allow_pickle=True).item()

        signal = data["impulse_signal"]
        label = data["impulse_name"]
        sample_rate = data["sample_rate"]

        # shorten slice to even timing (4 second sample)
        end = sample_rate * 4
        signal = signal[:, :end]

        if self.transform:
            signal = self.transform(signal)

        return signal, label
