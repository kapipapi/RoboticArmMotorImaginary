import os
import numpy as np


class FileConverter:
    DATASET_FREQ = 2048
    CHANNELS_IN_FILE = 16 + 1  # with trigger
    HEADER_LENGTH = 256 * (CHANNELS_IN_FILE + 1)

    impulses_names = ["BREAK", "LEFT", "RIGHT", "RELAX"]

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

        return np.array(raw_data), markers

    def split_file(self, filename):
        signals, markers = self.preconvert_file(filename)

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
                        "impulse_name": self.impulses_names[type_of_slice],
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


if __name__ == '__main__':
    converter = FileConverter()
    for i in range(1, 5):
        converter.split_file(f"../dataset/sesja{i}_pawel_zaciskanie_dloni.bdf")
        print(f"Finished converting file number {i}")

