# Author: Kacper Ledwosiński
# Based on Jupyter: plot_data.ipynb

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import butter, buttord, filtfilt, resample


class EEGDataProcessor:
    DATASET_FREQ = 2048
    DOWNSAMPLED_FREQ = 256

    CLASSES_COUNT = 3

    HIGH_PASS_BOTTOM = 3    # [Hz]
    HIGH_PASS_TOP = 6       # [Hz]

    LOW_PASS_BOTTOM = 30    # [Hz]
    LOW_PASS_TOP = 60       # [Hz]

    def __init__(self):
        N, Wn = buttord(
            [self.HIGH_PASS_TOP,    self.LOW_PASS_TOP], 
            [self.HIGH_PASS_BOTTOM, self.HIGH_PASS_BOTTOM], 
            3, 
            40, 
            False
        )
        self.b, self.a = butter(N, Wn, 'band', True)

    def forward(self, x, mean):
        x = self.remove_dc_component(x, mean)            # TODO: Fix function
        x = self.amplitude_conversion(x)
        x = self.filter(x)                      # TODO: Check with EEG
        x = self.downsample(x)
        x = self.normalize(x)
        # x = self.natural_logarithm(x)
        return x

    def remove_dc_component(self, buffer, dc_means):
        buffer_no_dc = np.copy(buffer)

        buffer_no_dc -= dc_means[:, None]
        buffer_no_dc *= 0.03125
        return buffer_no_dc

    def filter(self, buffer):
        for c in range(buffer.shape[0]):
            buffer[c] = filtfilt(self.b, self.a, buffer[c])
        return buffer

    def correct_offset(self, buffer):
        for c in range(buffer.shape[0]):
            buffer[c] = buffer[c] - np.mean(buffer[c])
        return buffer

    def downsample(self, buffer):
        freq = self.DOWNSAMPLED_FREQ
        sampling_factor = self.DATASET_FREQ / freq
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
        T = 1 / self.DATASET_FREQ
        yf = torch.fft.fft(buffer)
        xf = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)
        plt.plot(xf, 2.0 / N * np.abs(yf[:N // 2]))

    def natural_logarithm(self, buffer):
        return np.log(buffer)

    def amplitude_conversion(self, buffer):
        return buffer * 0.03125
