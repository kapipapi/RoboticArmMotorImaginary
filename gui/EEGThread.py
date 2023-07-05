import socket
import struct
import threading
import numpy as np
from utils import utils

TCP_PORT = 7230
TCP_BIOSEMI_ADDRESS = 'localhost'
TCP_BIOSEMI_PORT = 8188
CHANNELS = 16  # BioSemi channels count
SAMPLES = 128  # BioSemi TCP samples per channel count
WORDS = CHANNELS * SAMPLES
SERVER_BUFFER_LEN = 3200
MEAN_PERIOD_LEN = 8192


class EEGThread:
    def __init__(self):
        self.thread = None
        self.tcp_socket = None
        self.started = False
        self.read_lock = threading.Lock()
        self.received_data_struct_buffer = bytearray()

    def initialize_socket(self):
        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self.tcp_socket.bind(("localhost", TCP_PORT))
        self.tcp_socket.connect((TCP_BIOSEMI_ADDRESS, TCP_BIOSEMI_PORT))

        return self.tcp_socket

    def decode_tcp(self):
        buffer = np.zeros((CHANNELS, SERVER_BUFFER_LEN))
        buffer_filled = 0

        sec_samp = 0

        # Decoding the received packet from ActiView
        received_bytes = 0
        while received_bytes < WORDS * 3:
            received_data_struct_partial = self.tcp_socket.recv(WORDS * 3)
            received_bytes += len(received_data_struct_partial)
            self.received_data_struct_buffer += received_data_struct_partial
        received_data_struct = bytes(self.received_data_struct_buffer[:WORDS * 3])
        received_data_struct_buffer = self.received_data_struct_buffer[WORDS * 3:]
        raw_data = struct.unpack(str(WORDS * 3) + 'B', received_data_struct)
        decoded_data = utils.decode_data_from_bytes(raw_data)
        # decoded_data[CHANNELS-1, :] = np.bitwise_and(decoded_data[CHANNELS-1, :].astype(int), 2 ** 17 - 1)

        buffer = np.roll(buffer, -SAMPLES, axis=1)
        buffer[:, -SAMPLES:] = decoded_data

        if buffer_filled + SAMPLES < SERVER_BUFFER_LEN:
            buffer_filled += SAMPLES
            sec_samp += 1
        else:
            return buffer

    def update(self):
        pass

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
