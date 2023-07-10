import threading
import torch
import numpy as np
from torch.utils.data import DataLoader

from gui import EEGThread
from models.EEGInception import EEGInception
from models.Transformer import Transformer
from utils.preprocessing import EEGDataProcessor


class EEGModelThread:

    def __init__(self, model: torch.nn.Module = None, device: torch.device = None):

        # self.capture = capture
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

    def get_eeg_tensor(self, signal):
        if signal.shape[1] >= end:
            signal = signal[:, :end]
        else:
            new_signal = []
            for i, s in enumerate(signal):
                new_signal.append(np.pad(s, (0, end - signal.shape[1]), 'mean'))
            signal = np.array(new_signal)

        assert signal.shape[1] == end

        signal = self.preprocess_sample(signal)
        signal = torch.unsqueeze(signal, 0)
        return signal

    def preprocess_sample(self, x):
        return self.preprocess.forward(x)

    def classify_sample(self, x):
        self.model.eval()
        with torch.no_grad():
            output = self.model(x)
            result = np.argmax(output.numpy())
            return result

    def update(self):
        while self.started:
            sample = self.capture.decode_tcp()

            if sample is not None:
                sample = self.get_eeg_tensor(sample)
                self.current_output = self.classify_sample(sample)
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


# Model Sanity test
# Needs to have matching downsample frequency
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Transformer()
    model.load_state_dict(torch.load("transformer.pt", map_location=device))
    model.to(device)
    sample = np.load('14.npy', allow_pickle=True).item()
    signal = sample["impulse_signal"]

    sample_rate = int(sample["sample_rate"])
    end = sample_rate * 4

    if signal.shape[1] >= end:
        signal = signal[:, :end]
    else:
        new_signal = []
        for i, s in enumerate(signal):
            new_signal.append(np.pad(s, (0, end - signal.shape[1]), 'mean'))
        signal = np.array(new_signal)

    assert signal.shape[1] == end

    processor = EEGDataProcessor()
    signal = processor.downsample(signal, 128)
    signal = torch.from_numpy(signal).float()
    signal = torch.unsqueeze(signal, 0)

    model.eval()
    with torch.no_grad():
        output = model(signal)
        result = np.argmax(output.numpy())
        print(f"Label number: {result}")

