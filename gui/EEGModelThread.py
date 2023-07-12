import threading
import torch
import numpy as np

from gui.EEGThread import EEGThread
from models.Transformer import Transformer
from utils.preprocessing import EEGDataProcessor

SAMPLE_RATE = 2048


class EEGModelThread:
    def __init__(self, capture: EEGThread, model: torch.nn.Module = None):
        self.capture = capture
        self.started = False
        self.thread = None

        self.preprocess = EEGDataProcessor()
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')
        self.model = model

        self.load_model()
        self.current_prediction = None

    def load_model(self):
        assert self.model is not None
        self.model.to(self.device)

    def get_eeg_tensor(self, sample):
        signal = sample[0]                          # TODO: Refactor
        mean = sample[1]
        end = SAMPLE_RATE * 4

        if signal.shape[1] >= end:
            signal = signal[:, :end]
        else:
            new_signal = []
            for i, s in enumerate(signal):
                new_signal.append(np.pad(s, (0, end - signal.shape[1]), 'mean'))
            signal = np.array(new_signal)

        assert signal.shape[1] == end

        tensor = self.preprocess_sample(signal, mean)
        tensor = torch.from_numpy(tensor).float()
        tensor = torch.unsqueeze(tensor, 0)
        return tensor

    def preprocess_sample(self, x, mean):
        return self.preprocess.forward(x, mean)

    def classify_sample(self, x):
        self.model.eval()
        with torch.no_grad():
            output = self.model(x)
            result = np.argmax(output.numpy())
            return result

    def update(self):
        while self.started:
            data = self.capture.read()
            if data is not None:
                sample = self.get_eeg_tensor(data)
                self.current_prediction = self.classify_sample(sample)

    def read(self):
        return self.current_prediction

    def start(self):
        if self.started:
            return None
        self.started = True

        print("[!] Model thread starting")

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
    model.to('cpu')
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
