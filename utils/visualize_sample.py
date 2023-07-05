import matplotlib.pyplot as plt
import numpy as np

from utils.converter import FileConverter

data = FileConverter().preconvert_file("/home/administrator/Documents/dataset-eeg/sesja1_pawel_zaciskanie_dloni.bdf")

print("converted")

s0 = data[0]

samples = FileConverter.DATASET_FREQ * 6

s0_mean = np.convolve(s0, np.ones(samples)/samples, mode='valid')

fig, ax = plt.subplots()
ax.plot(s0_mean)
plt.show()
