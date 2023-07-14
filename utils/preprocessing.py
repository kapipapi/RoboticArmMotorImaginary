# Author: Kacper Ledwosiński
# Based on Jupyter: plot_data.ipynb

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import butter, buttord, lfilter, resample


class EEGDataProcessor:
    DATASET_FREQ = 2048
    DOWNSAMPLED_FREQ = 512

    CLASSES_COUNT = 3

    LOW_PASS_FREQ_PB = 30
    LOW_PASS_FREQ_SB = 60
    HIGH_PASS_FREQ_PB = 6
    HIGH_PASS_FREQ_SB = 3

    MAX_LOSS_PB = 2
    MIN_ATT_SB = 6

    def __init__(self):
        f_ord, wn = buttord(self.LOW_PASS_FREQ_PB, self.LOW_PASS_FREQ_SB, self.MAX_LOSS_PB, self.MIN_ATT_SB, False,
                            self.DATASET_FREQ)
        self.low_b, self.low_a, *args = butter(f_ord, wn, 'lowpass', False, 'ba', self.DATASET_FREQ)

        f_ord, wn = buttord(self.HIGH_PASS_FREQ_PB, self.HIGH_PASS_FREQ_SB, self.MAX_LOSS_PB, self.MIN_ATT_SB, False,
                            self.DATASET_FREQ)
        self.high_b, self.high_a, *args = butter(f_ord, wn, 'highpass', False, 'ba', self.DATASET_FREQ)

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
            buffer[c] = lfilter(self.low_b, self.low_a, buffer[c])
            buffer[c] = lfilter(self.high_b, self.high_a, buffer[c])
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
