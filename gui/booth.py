import time
import tkinter as tk

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class Booth:
    root: tk.Tk = None
    label: tk.Label = None
    buffer_length = None
    model = None

    canvas = None
    toolbar = None
    figure = None
    line = None
    bar = None

    def __init__(self):
        self.init_tk()
        self.update()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.mainloop()

    def init_tk(self):
        print("[!] Creating tkinter GUI")
        self.root = tk.Tk()

        self.root.geometry("1280x800")
        self.root.title("AV Emotion Recognition")

    def init_model(self):
        pass

    def init_capture(self):
        pass

    def update(self):
        pass

    def on_close(self):
        print("Ending processes")
        self.capture.stop()
        self.model.stop()
        del self.capture
        self.root.destroy()
        print("Processes closed")
