# Author: Kacper Ledwosiński
# Based on Jupyter: plot_data.ipynb

import os

import numpy as np

from Utilities.converter import FileConverter

impulses_names = ["BREAK", "LEFT", "RIGHT", "RELAX"]


def split_file(filename):
    signals, markers = FileConverter().preconvert_file(filename)

    all_slices = []
    type_of_slice = None
    slicing = False
    slice_start_index = None
    for i in range(len(markers)):
        if i % 1000 == 0:
            print(i + 1, "/", len(markers), " " * 100, end="\r")

        m = markers[i]

        if not slicing:
            if m > 1:
                slicing = True
                slice_start_index = i
                type_of_slice = int(np.log2(m))
            else:
                continue

        else:
            if m == 1:
                current_slice = signals[:, slice_start_index:i]

                all_slices.append({
                    "impulse_name": impulses_names[type_of_slice],
                    "impulse_signal": current_slice,
                    "duration_s": (i - slice_start_index) / FileConverter.DATASET_FREQ
                })

                slicing = False
                slice_start_index = None
                type_of_slice = None
            else:
                continue

    bn = os.path.basename(filename)[:-4]
    savepath = os.path.join("dataset", bn)

    if not os.path.exists(savepath):
        os.makedirs(savepath)

    for i, impulse in enumerate(all_slices):
        data_filename = os.path.join(savepath, f"{i}.npy")
        np.save(data_filename, impulse)
