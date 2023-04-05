# Author: Kacper Ledwosiński
# Based on Jupyter: plot_data.ipynb

import os
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, buttord, lfilter, decimate, resample

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
                    "impulse_index": type_of_slice,
                    "impulse_signal": current_slice,
                    "sample_rate": FileConverter.DATASET_FREQ,
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


class EEGDataProcessor():
    DATASET_FREQ = 2048
    DOWNSAMPLED_FREQ = 256

    CLASSES_COUNT = 3

    LOW_PASS_FREQ_PB = 30
    LOW_PASS_FREQ_SB = 60
    HIGH_PASS_FREQ_PB = 6
    HIGH_PASS_FREQ_SB = 3

    MAX_LOSS_PB = 2
    MIN_ATT_SB = 6

    def __init__(self):
        f_ord, wn = buttord(self.LOW_PASS_FREQ_PB, self.LOW_PASS_FREQ_SB, self.MAX_LOSS_PB, self.MIN_ATT_SB, False, self.DATASET_FREQ)
        self.low_b, self.low_a = butter(f_ord, wn, 'lowpass', False, 'ba', self.DATASET_FREQ)

        f_ord, wn = buttord(self.HIGH_PASS_FREQ_PB, self.HIGH_PASS_FREQ_SB, self.MAX_LOSS_PB, self.MIN_ATT_SB, False, self.DATASET_FREQ)
        self.high_b, self.high_a = butter(f_ord, wn, 'highpass', False, 'ba', self.DATASET_FREQ)

    def filter(self, buffer):
        for c in range(buffer.shape[0]):
            buffer[c] = lfilter(self.low_b, self.low_a, buffer[c])
            buffer[c] = lfilter(self.high_b, self.high_a, buffer[c])
        return buffer

    def correct_offset(self, buffer):
        for c in range(buffer.shape[0]):
            buffer[c] = buffer[c] - np.mean(buffer[c])
        return buffer

    def downsample(self, buffer):
        sampling_factor = self.DATASET_FREQ / self.DOWNSAMPLED_FREQ
        downsampled_signal = resample(buffer, int(buffer.shape[-1] / sampling_factor), axis=-1)
        return downsampled_signal

    def normalize(self, buffer):
        mean = np.mean(buffer)
        std = np.std(buffer)
        z_score_signal = (buffer - mean) / std
        return z_score_signal

    def visualize_sample(self, buffer):
        plt.plot(buffer)

    def visualize_sample_freq(self, buffer):
        N = len(buffer)
        T = 1/self.DATASET_FREQ
        yf = torch.fft.fft(buffer)
        xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
        plt.plot(xf, 2.0/N * np.abs(yf[:N//2]))