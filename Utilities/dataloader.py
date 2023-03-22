import os

import numpy as np
from torch.utils.data import Dataset


def load_paths(root_dir):
    paths = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".npy"):
                paths.append(os.path.join(root, file))

    print(len(paths))
    return paths


class EEGDataLoader(Dataset):

    def __init__(self, root_dir):
        self.paths = load_paths(root_dir)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        data = np.load(self.paths[index], allow_pickle=True).item()

        signal = data["impulse_signal"]
        label = data["impulse_name"]

        return signal, label
