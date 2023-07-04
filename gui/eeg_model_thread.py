import threading
import time

import numpy as np
import torch


class ModelThread:

    def __init__(self, model: torch.nn.Module = None, device: torch.device = None):

        self.started = False
        self.thread = None

        self.device = device
        self.model = model

        self.load_model()

    def load_model(self):
        if self.device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        assert self.model is not None

    def update(self):
        pass

    def start(self):
        if self.started:
            return None
        self.started = True

        print("Model thread starting")

        self.thread = threading.Thread(
            target=self.update,
            args=()
        )
        self.thread.start()
        return self

    def stop(self):
        self.started = False
        self.thread.join()
