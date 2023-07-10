import numpy as np

CHANNELS = 16  # BioSemi channels count
SAMPLES = 128  # BioSemi TCP samples per channel count
WORDS = CHANNELS * SAMPLES


def decode_data_from_bytes(raw_data):
    data_struct = np.zeros((CHANNELS, SAMPLES))

    # 32 bit unsigned words reorder
    raw_data_array = np.array(raw_data)
    raw_data_array = raw_data_array.reshape((WORDS, 3))
    raw_data_array = raw_data_array.astype("int32")
    raw_data_array = raw_data_array[:, 0].astype("int32") + \
                     raw_data_array[:, 1].astype("int32") * 256 + \
                     raw_data_array[:, 2].astype("int32") * 256 * 256
    raw_data_array[raw_data_array >= (1 << 23)] -= (1 << 24)

    for j in range(CHANNELS):
        for i in range(SAMPLES):
            data_struct[j, i] = raw_data_array[i * CHANNELS + j].astype('float32')

    # setting reference
    data_struct -= 0.55 * (data_struct[6, :] + data_struct[8, :])

    return data_struct
