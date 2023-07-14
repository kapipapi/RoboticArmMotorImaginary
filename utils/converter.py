import os
import numpy as np
from biosemipy import bdf


def split_file(filename, output_dir="../dataset"):
    impulses_names = ["BREAK", "LEFT", "RIGHT", "RELAX", "FEET"]

    # load file
    file = bdf.BDF(filename)
    
    # select 16 channels
    file.select_channels(['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16'])

    trigger_idx = np.array(np.log2(file.trig['val'] - 255) - 8, dtype=int)

    # create save filepath
    bn = os.path.basename(filename)[:-4]
    savepath = os.path.join(output_dir, bn)

    if not os.path.exists(savepath):
        os.makedirs(savepath)

    print("saving to:", savepath)

    # iterate through all triggers and slice data
    current_signal_id = 0
    for i in range(len(file.trig['idx'])-1):
        start = file.trig['idx'][i]
        end = file.trig['idx'][i+1]
        
        type_of_slice = trigger_idx[i]

        if type_of_slice > 0:
            current_slice = file.data[:,start:end]

            numpy_content = {
                "impulse_name": impulses_names[type_of_slice],
                "impulse_index": type_of_slice,
                "impulse_signal": current_slice,
                "sample_rate": file.freq,
            }

            data_filename = os.path.join(savepath, f"{current_signal_id}.npy")
            np.save(data_filename, numpy_content)
            current_signal_id += 1


if __name__ == '__main__':
    split_file(f"../dataset/pawel/sesja1_pawel_zaciskanie_dloni.bdf")
    print(f"Finished converting file number 1")
