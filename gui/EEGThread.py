import socket
import struct
import threading
from time import sleep

import numpy as np
from utils import utils
import matplotlib.pyplot as plt

from utils.preprocessing import EEGDataProcessor

TCP_PORT = 7230
TCP_BIOSEMI_ADDRESS = 'localhost'
TCP_BIOSEMI_PORT = 8888
CHANNELS = 16  # BioSemi channels count
SAMPLES = 128  # BioSemi TCP samples per channel count
WORDS = CHANNELS * SAMPLES
SERVER_BUFFER_LEN = 3200
MEAN_PERIOD_LEN = 8192


class EEGThread:
    def __init__(self):
        self.thread = None
        self.tcp_socket = None
        self.started = True
        self.read_lock = threading.Lock()
        self.preprocess = EEGDataProcessor()

        self.buffer = np.zeros((CHANNELS, SERVER_BUFFER_LEN))
        self.buffer_filled = 0

        self.sec_samp = 0
        self.received_data_struct_buffer = bytearray()

        self.latest_signal = None

    def initialize_socket(self):
        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self.tcp_socket.bind(("localhost", TCP_PORT))
        self.tcp_socket.connect((TCP_BIOSEMI_ADDRESS, TCP_BIOSEMI_PORT))

        return self.tcp_socket

    def decode_tcp(self):
        # Decoding the received packet from ActiView
        received_bytes = 0
        while received_bytes < WORDS * 3:
            received_data_struct_partial = self.tcp_socket.recv(WORDS * 3)
            received_bytes += len(received_data_struct_partial)
            self.received_data_struct_buffer += received_data_struct_partial
        received_data_struct = bytes(self.received_data_struct_buffer[:WORDS * 3])
        self.received_data_struct_buffer = self.received_data_struct_buffer[WORDS * 3:]
        raw_data = struct.unpack(str(WORDS * 3) + 'B', received_data_struct)
        decoded_data = utils.decode_data_from_bytes(raw_data)
        # decoded_data[CHANNELS-1, :] = np.bitwise_and(decoded_data[CHANNELS-1, :].astype(int), 2 ** 17 - 1)

        self.buffer = np.roll(self.buffer, -SAMPLES, axis=1)
        self.buffer[:, -SAMPLES:] = decoded_data

        if self.buffer_filled + SAMPLES < SERVER_BUFFER_LEN:
            self.buffer_filled += SAMPLES
            self.sec_samp += 1
        else:
            return self.buffer

    def update(self):
        while self.started:
            output = self.decode_tcp()
            if signal is not None:
                self.latest_signal = output
            else:
                print("[!] Waiting for the buffer to fill up")

    def start(self):
        if self.started:
            print('[!] Asynchronous capturing has already been started.')
            return None
        self.started = True

        self.thread = threading.Thread(
            target=self.update,
            args=()
        )
        self.thread.start()
        return self

    def stop(self):
        self.started = False
        self.thread.join()


def visualize_sample(sample):
    plt.plot(sample, color='magenta')
    plt.show()


if __name__ == '__main__':
    server = EEGThread()
    server.start()
    server.initialize_socket()
    processor = EEGDataProcessor()

    while True:
        signal = server.decode_tcp()
        if signal is not None:
            processor.forward(signal)
            for i in range(15):
                plt.plot(signal[i])
            plt.show()
