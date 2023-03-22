import os

import numpy as np


class FileConverter:
    DATASET_FREQ = 2048
    CHANNELS_IN_FILE = 16 + 1  # with trigger
    HEADER_LENGTH = 256 * (CHANNELS_IN_FILE + 1)

    def preconvert_file(self, i_file_path):
        file_len_bytes = os.stat(i_file_path).st_size
        file_len_bytes_headless = file_len_bytes - self.HEADER_LENGTH

        channel_sections_count = file_len_bytes_headless // (self.CHANNELS_IN_FILE * self.DATASET_FREQ * 3)

        with open(i_file_path, 'rb') as f:
            data = f.read()
        data = np.frombuffer(data[self.HEADER_LENGTH:], dtype='<u1')

        samples = np.ndarray((self.CHANNELS_IN_FILE - 1,
                              self.DATASET_FREQ * channel_sections_count, 3), dtype='<u1')

        triggers = np.ndarray((1, self.DATASET_FREQ * channel_sections_count, 3), dtype='<u1')

        for sec in range(channel_sections_count):
            if sec % 50 == 0:
                print(sec + 1, "/", channel_sections_count, " " * 100, end="\r")
            for ch in range(self.CHANNELS_IN_FILE):
                for sam in range(self.DATASET_FREQ):
                    beg = sec * self.CHANNELS_IN_FILE * self.DATASET_FREQ * 3 + ch * self.DATASET_FREQ * 3 + sam * 3
                    if ch != self.CHANNELS_IN_FILE - 1:
                        samples[ch, sec * self.DATASET_FREQ + sam, :] = data[beg:beg + 3]
                    else:
                        triggers[0, sec * self.DATASET_FREQ + sam, :] = data[beg:beg + 3]

        raw_data = samples[:, :, 0].astype("int32") + samples[:, :, 1].astype("int32") * 256 + samples[:, :, 2].astype(
            "int32") * 256 * 256
        raw_data[raw_data > pow(2, 23)] -= pow(2, 24)

        markers = triggers[0, :, 1]

        raw_data = raw_data.astype('float32')
        raw_data -= 0.55 * (raw_data[6, :] + raw_data[8, :])  # referencing the signal

        return raw_data, markers
