import os
import numpy as np
import matplotlib.pyplot as plt

for root, folders, files in os.walk("/home/lab218/Downloads/plots"):
    for file in files:
        if file.endswith(".csv"):
            filepath = os.path.join(root, file)
            print(filepath.split("/")[-2].split("_"))
            _, step, value = np.genfromtxt(filepath, dtype=float, delimiter=',', names=True, unpack=True) 
            
