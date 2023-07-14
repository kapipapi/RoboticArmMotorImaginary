import os
import numpy as np
from biosemipy import bdf


class FileConverter:
    DATASET_FREQ = 2048

    def split_file(self, filename, output_dir="../dataset"):
        data = bdf.BDF(filename)
        print(data.data)
        exit(0)

        all_slices = []
        type_of_slice = None
        slicing = False
        slice_start_index = None
        for i in range(len(markers)):
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
        savepath = os.path.join(output_dir, bn)

        if not os.path.exists(savepath):
            os.makedirs(savepath)
            
        for i, impulse in enumerate(all_slices):
            data_filename = os.path.join(savepath, f"{i}.npy")
            np.save(data_filename, impulse)


if __name__ == '__main__':
    converter = FileConverter()
    converter.split_file(f"../dataset/pawel/sesja1_pawel_zaciskanie_dloni.bdf")
    print(f"Finished converting file number 1")

