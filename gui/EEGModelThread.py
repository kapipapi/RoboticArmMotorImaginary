import threading
import torch

from gui import EEGThread
from utils.preprocessing import EEGDataProcessor


class EEGModelThread:

    def __init__(self, capture: EEGThread, model: torch.nn.Module = None, device: torch.device = None):

        self.capture = capture
        self.started = False
        self.thread = None

        self.preprocess = EEGDataProcessor()
        self.device = device
        self.model = model

        self.load_model()

        self.current_output = None

    def load_model(self):
        if self.device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        assert self.model is not None

    def preprocess_sample(self, x):
        return self.preprocess.forward(x)

    def classify_sample(self, x):
        self.model.eval()
        with torch.no_grad():
            return self.model(x)

    def update(self):
        while self.started:
            sample = self.capture.decode_tcp()

            if sample is not None:
                sample = self.preprocess_sample(sample)
                self.current_output = self.classify_sample(sample)
            else:
                print("[!] Waiting for the buffer to fill up")
            return self.current_output

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
